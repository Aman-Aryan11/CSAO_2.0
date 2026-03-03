from __future__ import annotations

import json
import os
import time
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import src.ranking.train_neural_ranker as tnr


OUT = ROOT / "output" / "cycle3"
OUT.mkdir(parents=True, exist_ok=True)

TRIALS = int(os.getenv("CYCLE3_TRIALS", "4"))
EPOCHS = int(os.getenv("CYCLE3_EPOCHS", "5"))
TOP_K = int(os.getenv("CYCLE3_TOP_K", "10"))
MAX_TRAIN = int(os.getenv("CYCLE3_MAX_TRAIN_ROWS", "700000"))
MAX_VAL = int(os.getenv("CYCLE3_MAX_VAL_ROWS", "250000"))
MAX_TEST = int(os.getenv("CYCLE3_MAX_TEST_ROWS", "250000"))
DEVICE = os.getenv("CYCLE3_DEVICE", "cpu")
RESUME = os.getenv("CYCLE3_RESUME", "1").strip().lower() in {"1", "true", "yes"}


def log(msg: str) -> None:
    print(msg, flush=True)


def trial_space() -> List[Dict[str, Any]]:
    return [
        {"lr": 1e-3, "dropout": 0.2, "hidden_dims": [256, 128, 64], "pairwise_loss": "lambdarank", "hard_negative_mining": False},
        {"lr": 7e-4, "dropout": 0.25, "hidden_dims": [384, 192, 96], "pairwise_loss": "lambdarank", "hard_negative_mining": False},
        {"lr": 5e-4, "dropout": 0.2, "hidden_dims": [256, 128, 64], "pairwise_loss": "bpr", "hard_negative_mining": True},
        {"lr": 8e-4, "dropout": 0.3, "hidden_dims": [512, 256, 128], "pairwise_loss": "lambdarank", "hard_negative_mining": True},
    ][:TRIALS]


def build_key(df: pd.DataFrame, session_col: str = "session_id", item_col: str = "item_id") -> np.ndarray:
    return (
        df[session_col].to_numpy(dtype=np.uint64) << np.uint64(32)
    ) | df[item_col].to_numpy(dtype=np.uint64)


def fetch_test_meta(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch metadata columns for predicted test rows from processed test parquet."""
    path = ROOT / "data_pipeline" / "data" / "processed" / "test.parquet"
    if not path.exists():
        return pred_df.copy()

    keys = np.unique(build_key(pred_df))
    pf = pq.ParquetFile(path)
    parts: List[pd.DataFrame] = []

    cols = [
        "session_id",
        "candidate_item_id",
        "item_id",
        "label_added",
        "label",
        "meal_time",
        "user_segment",
        "candidate_category",
        "candidate_price",
        "cart_value",
    ]
    cols = [c for c in cols if c in pf.schema.names]

    for rb in pf.iter_batches(batch_size=250_000, columns=cols):
        b = rb.to_pandas()
        if b.empty:
            continue
        if "item_id" not in b.columns and "candidate_item_id" in b.columns:
            b = b.rename(columns={"candidate_item_id": "item_id"})
        if "label" not in b.columns and "label_added" in b.columns:
            b = b.rename(columns={"label_added": "label"})
        k = (
            b["session_id"].to_numpy(dtype=np.uint64) << np.uint64(32)
        ) | b["item_id"].to_numpy(dtype=np.uint64)
        idx = np.searchsorted(keys, k)
        valid = idx < keys.size
        mask = np.zeros(len(b), dtype=bool)
        mask[valid] = keys[idx[valid]] == k[valid]
        if mask.any():
            parts.append(
                b.loc[
                    mask,
                    [
                        c
                        for c in [
                            "session_id",
                            "item_id",
                            "meal_time",
                            "user_segment",
                            "candidate_category",
                            "candidate_price",
                            "cart_value",
                        ]
                        if c in b.columns
                    ],
                ].copy()
            )

    if not parts:
        return pred_df.copy()

    meta = pd.concat(parts, ignore_index=True)
    meta = meta.drop_duplicates(["session_id", "item_id"], keep="first")
    out = pred_df.merge(meta, on=["session_id", "item_id"], how="left")
    if "label_x" in out.columns and "label" not in out.columns:
        out["label"] = out["label_x"]
    return out


def evaluate_segment(pred: pd.DataFrame, top_k: int, model_name: str) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []

    def by_segment(col: str) -> None:
        for seg, g in pred.groupby(col, dropna=False):
            m = tnr.compute_metrics(g[["session_id", "item_id", "label", "score"]], top_k)
            out_rows.append({"model": model_name, "segment_type": col, "segment": str(seg), **m, "rows": int(len(g))})

    for c in ["meal_time", "user_segment", "candidate_category"]:
        if c in pred.columns:
            by_segment(c)

    if "cart_value" in pred.columns:
        x = pred.copy()
        x["cart_bucket"] = pd.qcut(x["cart_value"], q=4, duplicates="drop").astype(str)
        for seg, g in x.groupby("cart_bucket", dropna=False):
            m = tnr.compute_metrics(g[["session_id", "item_id", "label", "score"]], top_k)
            out_rows.append({"model": model_name, "segment_type": "cart_bucket", "segment": str(seg), **m, "rows": int(len(g))})

    return pd.DataFrame(out_rows)


def business_impact(pred: pd.DataFrame, top_k: int, model_name: str, seed: int = 42) -> Dict[str, Any]:
    x = pred.copy()
    if "candidate_price" not in x.columns:
        x["candidate_price"] = 0.0
    if "cart_value" not in x.columns:
        x["cart_value"] = 0.0

    top = x[x["rank"] <= top_k].copy()
    accept = top.groupby("session_id")["label"].max().mean()

    top["accepted_value"] = np.where(top["label"] == 1, top["candidate_price"].astype(np.float32), 0.0)
    model_val = top.groupby("session_id")["accepted_value"].sum()

    rng = np.random.default_rng(seed)
    r = x[["session_id", "item_id", "label", "candidate_price", "cart_value"]].copy()
    r["rnd"] = rng.random(len(r))
    r = r.sort_values(["session_id", "rnd"], kind="mergesort")
    r["rank"] = r.groupby("session_id", sort=False).cumcount().add(1)
    r = r[r["rank"] <= top_k].copy()
    r["accepted_value"] = np.where(r["label"] == 1, r["candidate_price"].astype(np.float32), 0.0)
    rand_val = r.groupby("session_id")["accepted_value"].sum()

    all_sid = model_val.index.union(rand_val.index)
    m = model_val.reindex(all_sid, fill_value=0.0)
    rv = rand_val.reindex(all_sid, fill_value=0.0)

    abs_up = float(m.mean() - rv.mean())
    rel_up = float(abs_up / max(float(rv.mean()), 1e-6))

    cart = x.groupby("session_id")["cart_value"].first().reindex(all_sid).fillna(0.0)
    aov_lift = float((abs_up / max(float(cart.mean()), 1e-6)) * 100.0)

    return {
        "model": model_name,
        "top_k": top_k,
        "acceptance_at_k": float(accept),
        "avg_addon_value_model": float(m.mean()),
        "avg_addon_value_random": float(rv.mean()),
        "absolute_addon_uplift": abs_up,
        "relative_addon_uplift_vs_random": rel_up,
        "projected_aov_lift_percent": aov_lift,
    }


def run_trial(base: tnr.TrainConfig, params: Dict[str, Any], trial_id: int, device: str) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    cfg = replace(
        base,
        num_epochs=EPOCHS,
        top_k=TOP_K,
        lr=float(params["lr"]),
        dropout=float(params["dropout"]),
        hidden_dims=list(params["hidden_dims"]),
        pairwise_loss=str(params["pairwise_loss"]),
        hard_negative_mining=bool(params["hard_negative_mining"]),
        max_train_rows=MAX_TRAIN,
        max_val_rows=MAX_VAL,
        max_test_rows=MAX_TEST,
        device=device,
        seed=base.seed + trial_id,
    )

    tnr.set_seed(cfg.seed)
    train_loader, val_loader, test_loader, feature_spec, source = tnr.load_datasets(cfg)
    model = tnr.build_model(cfg, feature_spec, device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_val = -1.0
    best_state = None
    trial_log: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for ep in range(cfg.num_epochs):
        tr_loss = tnr.train_one_epoch(model, train_loader, optimizer, None, cfg, device)
        val_metrics, _ = tnr.evaluate_model(model, val_loader, device, cfg.top_k)
        ndcg = float(val_metrics.get(f"NDCG@{cfg.top_k}", 0.0))
        scheduler.step(ndcg)
        trial_log.append({"trial": trial_id, "epoch": ep, "train_loss": tr_loss, **val_metrics})
        log(f"[trial {trial_id}] epoch={ep} loss={tr_loss:.5f} val_ndcg@{cfg.top_k}={ndcg:.5f} val_auc={val_metrics.get('AUC')}")
        if ndcg > best_val:
            best_val = ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, pred_test = tnr.evaluate_model(model, test_loader, device, cfg.top_k)
    runtime = float(time.perf_counter() - t0)

    result = {
        "trial": trial_id,
        "runtime_sec": runtime,
        "params": params,
        "source": source,
        "feature_spec": asdict(feature_spec),
        "val_best_ndcg": best_val,
        "test_metrics": test_metrics,
    }
    return result, asdict(cfg), pred_test


def load_cycle2_baseline() -> Dict[str, Any]:
    p = ROOT / "output" / "cycle2" / "cycle2_summary.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def main() -> None:
    start = time.perf_counter()
    base_cfg = tnr.load_config()
    device = DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    trials = trial_space()
    trial_results: List[Dict[str, Any]] = []
    train_logs: List[Dict[str, Any]] = []
    preds_by_trial: Dict[int, pd.DataFrame] = {}

    best_idx = -1
    best_score = -1.0

    for i, p in enumerate(trials, start=1):
        trial_json = OUT / f"trial_{i}_result.json"
        trial_pred = OUT / f"trial_{i}_test_predictions.parquet"

        if RESUME and trial_json.exists() and trial_pred.exists():
            loaded = json.loads(trial_json.read_text())
            trial_results.append(loaded)
            preds_by_trial[i] = pd.read_parquet(trial_pred)
            log(f"[resume] loaded trial {i} from saved artifacts")
        else:
            result, used_cfg, pred_test = run_trial(base_cfg, p, i, device)
            merged_result = {**result, "config": used_cfg}
            trial_results.append(merged_result)
            preds_by_trial[i] = pred_test
            trial_json.write_text(json.dumps(merged_result, indent=2, default=float))
            pred_test.to_parquet(trial_pred, index=False)

        pd.DataFrame(
            [
                {
                    "trial": r["trial"],
                    "val_best_ndcg": r["val_best_ndcg"],
                    "test_ndcg": r["test_metrics"].get(f"NDCG@{TOP_K}", 0.0),
                    "test_auc": r["test_metrics"].get("AUC", float("nan")),
                    "runtime_sec": r["runtime_sec"],
                    "params": json.dumps(r["params"], sort_keys=True),
                }
                for r in trial_results
            ]
        ).to_csv(OUT / "neural_tuning_results.partial.csv", index=False)

        for ep_row in []:
            train_logs.append(ep_row)

        score = float(result["val_best_ndcg"])
        if score > best_score:
            best_score = score
            best_idx = i

    res_df = pd.DataFrame(
        [
            {
                "trial": r["trial"],
                "val_best_ndcg": r["val_best_ndcg"],
                "test_ndcg": r["test_metrics"].get(f"NDCG@{TOP_K}", 0.0),
                "test_auc": r["test_metrics"].get("AUC", float("nan")),
                "runtime_sec": r["runtime_sec"],
                "params": json.dumps(r["params"], sort_keys=True),
            }
            for r in trial_results
        ]
    )
    res_df.to_csv(OUT / "neural_tuning_results.csv", index=False)

    best_pred = preds_by_trial[best_idx].copy()
    best_pred = best_pred.sort_values(["session_id", "score", "item_id"], ascending=[True, False, True], kind="mergesort")
    best_pred["rank"] = best_pred.groupby("session_id", sort=False).cumcount().add(1).astype(np.int32)
    best_pred["model"] = "NeuralRanker"
    best_pred.to_parquet(OUT / "test_predictions_neural.parquet", index=False)

    best_pred_meta = fetch_test_meta(best_pred)
    seg_df = evaluate_segment(best_pred_meta, TOP_K, "NeuralRanker")
    seg_df.to_csv(OUT / "segment_performance_neural.csv", index=False)

    impact = business_impact(best_pred_meta, TOP_K, "NeuralRanker")
    (OUT / "business_impact_neural.json").write_text(json.dumps(impact, indent=2))

    # Compare with cycle2 baseline
    c2 = load_cycle2_baseline()
    baseline_metrics = c2.get("metrics_test", {}).get(c2.get("best_model", ""), {})
    neural_metrics = trial_results[best_idx - 1]["test_metrics"]

    cmp = {
        "top_k": TOP_K,
        "best_neural_trial": best_idx,
        "neural_metrics_test": neural_metrics,
        "cycle2_best_model": c2.get("best_model"),
        "cycle2_best_metrics_test": baseline_metrics,
        "delta_ndcg": float(neural_metrics.get(f"NDCG@{TOP_K}", 0.0) - float(baseline_metrics.get(f"NDCG@{TOP_K}", 0.0) if baseline_metrics else 0.0)),
        "delta_auc": float((neural_metrics.get("AUC", float("nan")) or 0.0) - float(baseline_metrics.get("AUC", 0.0) if baseline_metrics else 0.0)),
        "runtime_sec": float(time.perf_counter() - start),
    }
    (OUT / "comparison_vs_cycle2.json").write_text(json.dumps(cmp, indent=2, default=float))

    summary = {
        "trials": trial_results,
        "best_trial": best_idx,
        "best_val_ndcg": best_score,
        "comparison": cmp,
        "impact": impact,
    }
    (OUT / "cycle3_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    md = [
        "# Cycle 3 Neural Ranking Summary",
        "",
        f"- Best neural trial: {best_idx}",
        f"- Neural test NDCG@{TOP_K}: {neural_metrics.get(f'NDCG@{TOP_K}', float('nan')):.4f}",
        f"- Neural test AUC: {neural_metrics.get('AUC', float('nan')):.4f}",
        f"- Cycle2 best model: {c2.get('best_model', 'NA')}",
        f"- Cycle2 test NDCG@{TOP_K}: {float(baseline_metrics.get(f'NDCG@{TOP_K}', float('nan')) if baseline_metrics else float('nan')):.4f}",
        f"- Delta NDCG: {cmp['delta_ndcg']:.4f}",
        "",
        "## Recommendation",
        "- Promote neural to champion only if delta NDCG is positive and stable across segments.",
        "- Otherwise keep LightGBM as champion and neural as challenger with further tuning.",
    ]
    (OUT / "cycle3_report.md").write_text("\n".join(md))

    log(f"[done] cycle3 best_trial={best_idx} ndcg={best_score:.4f} artifacts={OUT}")


if __name__ == "__main__":
    main()
