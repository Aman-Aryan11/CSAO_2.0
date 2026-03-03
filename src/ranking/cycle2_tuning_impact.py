from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score


# -------------------------
# Paths / Config
# -------------------------
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data_pipeline" / "data" / "processed"
OUT = ROOT / "output" / "cycle2"
OUT.mkdir(parents=True, exist_ok=True)

SEED = int(os.getenv("CYCLE2_SEED", "42"))
TRAIN_ROWS = int(os.getenv("CYCLE2_TRAIN_ROWS", "1200000"))
VAL_ROWS = int(os.getenv("CYCLE2_VAL_ROWS", "350000"))
TEST_ROWS = int(os.getenv("CYCLE2_TEST_ROWS", "350000"))
TRIALS_LGBM = int(os.getenv("CYCLE2_TRIALS_LGBM", "4"))
TRIALS_XGB = int(os.getenv("CYCLE2_TRIALS_XGB", "4"))
TOP_K = int(os.getenv("CYCLE2_TOP_K", "10"))


def log(msg: str) -> None:
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_parquet_spread(path: Path, max_rows: int) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    nrg = pf.metadata.num_row_groups
    total_rows = int(pf.metadata.num_rows or 0)

    if max_rows <= 0 or total_rows <= max_rows or nrg <= 0:
        return pf.read().to_pandas()

    avg_rows = max(total_rows // nrg, 1)
    target_rgs = max(int(np.ceil(max_rows / avg_rows)), 1)
    target_rgs = min(target_rgs + 2, nrg)
    rg_idx = np.unique(np.linspace(0, nrg - 1, num=target_rgs, dtype=int))

    parts: List[pd.DataFrame] = []
    taken = 0
    for i in rg_idx:
        b = pf.read_row_group(int(i)).to_pandas()
        if b.empty:
            continue
        need = max_rows - taken
        if need <= 0:
            break
        if len(b) > need:
            b = b.iloc[:need].copy()
        parts.append(b)
        taken += len(b)
        if taken >= max_rows:
            break

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_session_aware_sample(path: Path, target_rows: int) -> pd.DataFrame:
    """Sample full sessions from parquet to preserve within-session label structure."""
    pf = pq.ParquetFile(path)
    if target_rows <= 0:
        return pf.read().to_pandas()

    # Phase 1: gather candidate sessions from spread session_id scan.
    session_scan_rows = max(target_rows * 2, 200_000)
    sess_df = load_parquet_spread(path, session_scan_rows)
    if sess_df.empty or "session_id" not in sess_df.columns:
        return pd.DataFrame()
    uniq_sessions = sess_df["session_id"].drop_duplicates().to_numpy(dtype=np.int64)
    if uniq_sessions.size == 0:
        return pd.DataFrame()

    desired_sessions = max(int(target_rows // 18), 1)
    if uniq_sessions.size > desired_sessions:
        rng = np.random.default_rng(SEED)
        chosen = rng.choice(uniq_sessions, size=desired_sessions, replace=False)
    else:
        chosen = uniq_sessions
    chosen_set = set(int(x) for x in chosen.tolist())

    # Phase 2: stream full parquet and keep rows for selected sessions.
    parts: List[pd.DataFrame] = []
    for rb in pf.iter_batches(batch_size=250_000):
        b = rb.to_pandas()
        if b.empty:
            continue
        take = b[b["session_id"].isin(chosen_set)]
        if not take.empty:
            parts.append(take)
        # Stop once we have enough and at least most sessions represented.
        if parts:
            rows_now = sum(len(x) for x in parts)
            if rows_now >= target_rows:
                # keep one more batch chance for coverage, then break
                break

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out


def group_sizes(session_ids: np.ndarray) -> np.ndarray:
    _, c = np.unique(session_ids, return_counts=True)
    return c.astype(np.int32)


def evaluate_ranking(y_true: np.ndarray, session_ids: np.ndarray, scores: np.ndarray, k: int = 10) -> Dict[str, float]:
    df = pd.DataFrame({"session_id": session_ids.astype(np.int64), "y": y_true.astype(np.int32), "s": scores.astype(np.float32)})
    df = df.sort_values(["session_id", "s"], ascending=[True, False], kind="mergesort")

    ndcgs: List[float] = []
    aps: List[float] = []
    mrrs: List[float] = []
    pks: List[float] = []
    rks: List[float] = []

    for _, g in df.groupby("session_id", sort=False):
        rel = g["y"].to_numpy(dtype=np.int32)
        if rel.size == 0:
            continue
        cutoff = min(k, rel.size)
        rel_k = rel[:cutoff]
        total_rel = int(rel.sum())

        tp_k = int(rel_k.sum())
        pks.append(tp_k / float(max(cutoff, 1)))
        rks.append(0.0 if total_rel == 0 else tp_k / float(total_rel))

        pos_idx = np.flatnonzero(rel)
        mrrs.append(0.0 if pos_idx.size == 0 else 1.0 / float(pos_idx[0] + 1))

        if total_rel == 0:
            aps.append(0.0)
        else:
            csum = np.cumsum(rel)
            hits = np.flatnonzero(rel)
            prec_at_hits = csum[hits] / (hits + 1.0)
            aps.append(float(prec_at_hits.sum() / total_rel))

        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
        dcg = float((rel_k * discounts).sum())
        ideal = np.sort(rel)[::-1][:cutoff]
        idcg = float((ideal * discounts).sum())
        ndcgs.append(0.0 if idcg <= 0 else dcg / idcg)

    out = {
        f"NDCG@{k}": float(np.mean(ndcgs) if ndcgs else 0.0),
        "MAP": float(np.mean(aps) if aps else 0.0),
        "MRR": float(np.mean(mrrs) if mrrs else 0.0),
        f"Precision@{k}": float(np.mean(pks) if pks else 0.0),
        f"Recall@{k}": float(np.mean(rks) if rks else 0.0),
    }
    uniq = np.unique(y_true)
    out["AUC"] = float(roc_auc_score(y_true, scores)) if uniq.size > 1 else float("nan")
    return out


@dataclass
class Prep:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_cols: List[str]
    cat_cols: List[str]


def prepare_data() -> Prep:
    t0 = time.perf_counter()
    train = load_session_aware_sample(PROCESSED / "train.parquet", TRAIN_ROWS)
    val = load_session_aware_sample(PROCESSED / "val.parquet", VAL_ROWS)
    test = load_session_aware_sample(PROCESSED / "test.parquet", TEST_ROWS)

    # Normalize column names
    for df in (train, val, test):
        if "candidate_item_id" in df.columns and "item_id" not in df.columns:
            df.rename(columns={"candidate_item_id": "item_id"}, inplace=True)
        if "label_added" in df.columns and "label" not in df.columns:
            df.rename(columns={"label_added": "label"}, inplace=True)

    # Keep one row per (session,item)
    def dedupe(x: pd.DataFrame) -> pd.DataFrame:
        y = x.sort_values(["session_id", "item_id", "label"], ascending=[True, True, False], kind="mergesort")
        return y.drop_duplicates(["session_id", "item_id"], keep="first").reset_index(drop=True)

    train = dedupe(train)
    val = dedupe(val)
    test = dedupe(test)

    drop_cols = {"label", "label_added", "timestamp", "split", "rank", "score", "model"}
    feature_cols = [c for c in train.columns if c not in drop_cols and c not in {"session_id", "item_id", "candidate_item_id"}]

    cat_cols = [
        c
        for c in feature_cols
        if str(train[c].dtype) in {"object", "string", "category", "bool"}
        or c.endswith("_category")
        or c.endswith("_segment")
        or c == "meal_time"
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Encode categoricals from train vocab
    for c in cat_cols:
        tr = train[c].astype("string").fillna("__MISSING__")
        vocab = {v: i for i, v in enumerate(pd.Index(tr.unique()))}
        train[c] = tr.map(vocab).fillna(-1).astype(np.int32)
        val[c] = val[c].astype("string").fillna("__MISSING__").map(vocab).fillna(-1).astype(np.int32)
        test[c] = test[c].astype("string").fillna("__MISSING__").map(vocab).fillna(-1).astype(np.int32)

    # Numeric fill
    med = train[num_cols].median(numeric_only=True) if num_cols else pd.Series(dtype=np.float32)
    if num_cols:
        train[num_cols] = train[num_cols].fillna(med).astype(np.float32)
        val[num_cols] = val[num_cols].fillna(med).astype(np.float32)
        test[num_cols] = test[num_cols].fillna(med).astype(np.float32)

    for c in cat_cols:
        train[c] = train[c].astype(np.float32)
        val[c] = val[c].astype(np.float32)
        test[c] = test[c].astype(np.float32)

    feature_cols = num_cols + cat_cols

    log(f"[prep] train={len(train)} val={len(val)} test={len(test)} features={len(feature_cols)} cats={len(cat_cols)} runtime_sec={time.perf_counter()-t0:.2f}")
    log(f"[prep] pos_rate train={train['label'].mean():.4f} val={val['label'].mean():.4f} test={test['label'].mean():.4f}")

    return Prep(train=train, val=val, test=test, feature_cols=feature_cols, cat_cols=cat_cols)


def random_lgbm_params(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [TOP_K],
        "feature_pre_filter": False,
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
        "num_leaves": int(rng.choice([31, 63, 127])),
        "max_depth": int(rng.choice([-1, 8, 10, 12])),
        "feature_fraction": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        "bagging_fraction": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        "bagging_freq": 1,
        "min_data_in_leaf": int(rng.choice([20, 40, 80, 120])),
        "lambda_l1": float(rng.choice([0.0, 0.1, 0.5])),
        "lambda_l2": float(rng.choice([0.1, 1.0, 3.0, 5.0])),
        "seed": SEED,
        "verbosity": -1,
        "num_threads": 8,
    }


def random_xgb_params(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "objective": "rank:ndcg",
        "eval_metric": f"ndcg@{TOP_K}",
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
        "max_depth": int(rng.choice([4, 6, 8, 10])),
        "min_child_weight": float(rng.choice([1.0, 3.0, 5.0])),
        "subsample": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        "colsample_bytree": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        "reg_alpha": float(rng.choice([0.0, 0.1, 0.5])),
        "reg_lambda": float(rng.choice([0.5, 1.0, 3.0, 5.0])),
        "tree_method": "hist",
        "n_jobs": 8,
        "random_state": SEED,
    }


def tune_lgbm(prep: Prep, trials: int) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    import lightgbm as lgb

    Xtr = prep.train[prep.feature_cols]
    Xva = prep.val[prep.feature_cols]
    ytr = prep.train["label"].to_numpy(np.int32)
    yva = prep.val["label"].to_numpy(np.int32)
    gtr = group_sizes(prep.train["session_id"].to_numpy(np.int64))
    gva = group_sizes(prep.val["session_id"].to_numpy(np.int64))

    dtr = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=prep.feature_cols, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, group=gva, feature_name=prep.feature_cols, reference=dtr, free_raw_data=False)

    rng = np.random.default_rng(SEED)
    rows: List[Dict[str, Any]] = []
    best_model = None
    best = {"score": -1.0, "trial": -1, "params": None, "best_iteration": None}

    for i in range(1, trials + 1):
        params = random_lgbm_params(rng)
        t0 = time.perf_counter()
        model = lgb.train(
            params=params,
            train_set=dtr,
            valid_sets=[dva],
            valid_names=["val"],
            num_boost_round=700,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        val_scores = model.predict(Xva, num_iteration=model.best_iteration)
        met = evaluate_ranking(yva, prep.val["session_id"].to_numpy(np.int64), val_scores, k=TOP_K)
        ndcg = float(met[f"NDCG@{TOP_K}"])
        row = {
            "model": "LightGBM",
            "trial": i,
            "ndcg_val": ndcg,
            "auc_val": float(met["AUC"]),
            "runtime_sec": float(time.perf_counter() - t0),
            "best_iteration": int(model.best_iteration or 0),
            "params": json.dumps(params, sort_keys=True),
        }
        rows.append(row)
        log(f"[tune][lgbm] trial={i} ndcg@{TOP_K}={ndcg:.6f} auc={met['AUC']:.6f}")
        if ndcg > best["score"]:
            best = {"score": ndcg, "trial": i, "params": params, "best_iteration": int(model.best_iteration or 0)}
            best_model = model

    return best_model, best, pd.DataFrame(rows)


def tune_xgb(prep: Prep, trials: int) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    import xgboost as xgb

    Xtr = prep.train[prep.feature_cols]
    Xva = prep.val[prep.feature_cols]
    ytr = prep.train["label"].to_numpy(np.int32)
    yva = prep.val["label"].to_numpy(np.int32)
    gtr = group_sizes(prep.train["session_id"].to_numpy(np.int64)).tolist()
    gva = group_sizes(prep.val["session_id"].to_numpy(np.int64)).tolist()

    rng = np.random.default_rng(SEED + 17)
    rows: List[Dict[str, Any]] = []
    best_model = None
    best = {"score": -1.0, "trial": -1, "params": None}

    for i in range(1, trials + 1):
        params = random_xgb_params(rng)
        t0 = time.perf_counter()
        model = xgb.XGBRanker(**params, n_estimators=700)
        model.fit(Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva], verbose=False)
        val_scores = model.predict(Xva)
        met = evaluate_ranking(yva, prep.val["session_id"].to_numpy(np.int64), val_scores, k=TOP_K)
        ndcg = float(met[f"NDCG@{TOP_K}"])
        row = {
            "model": "XGBoost",
            "trial": i,
            "ndcg_val": ndcg,
            "auc_val": float(met["AUC"]),
            "runtime_sec": float(time.perf_counter() - t0),
            "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
            "params": json.dumps(params, sort_keys=True),
        }
        rows.append(row)
        log(f"[tune][xgb] trial={i} ndcg@{TOP_K}={ndcg:.6f} auc={met['AUC']:.6f}")
        if ndcg > best["score"]:
            best = {"score": ndcg, "trial": i, "params": params}
            best_model = model

    return best_model, best, pd.DataFrame(rows)


def build_pred_df(model: Any, model_name: str, split: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    s = model.predict(split[feature_cols])
    pred = pd.DataFrame(
        {
            "session_id": split["session_id"].to_numpy(np.int64),
            "item_id": split["item_id"].to_numpy(np.int64),
            "label": split["label"].to_numpy(np.int8),
            "score": np.asarray(s, dtype=np.float32),
            "model": model_name,
        }
    )
    pred = pred.sort_values(["session_id", "score", "item_id"], ascending=[True, False, True], kind="mergesort")
    pred["rank"] = pred.groupby("session_id", sort=False).cumcount().add(1).astype(np.int32)
    return pred


def segment_metrics(pred: pd.DataFrame, base: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df = pred.merge(
        base[["session_id", "item_id", "meal_time", "user_segment", "cart_value", "candidate_category"]],
        on=["session_id", "item_id"],
        how="left",
    )

    def calc(by: str) -> pd.DataFrame:
        out: List[Dict[str, Any]] = []
        for k, g in df.groupby(by, dropna=False):
            m = evaluate_ranking(g["label"].to_numpy(np.int32), g["session_id"].to_numpy(np.int64), g["score"].to_numpy(np.float32), TOP_K)
            out.append({"model": model_name, "segment_type": by, "segment": str(k), **m, "rows": int(len(g))})
        return pd.DataFrame(out)

    df["cart_bucket"] = pd.qcut(df["cart_value"], q=4, duplicates="drop").astype(str)

    seg = pd.concat(
        [
            calc("meal_time"),
            calc("user_segment"),
            calc("candidate_category"),
            calc("cart_bucket"),
        ],
        ignore_index=True,
    )
    return seg


def business_impact(pred: pd.DataFrame, base: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    df = pred.merge(base[["session_id", "item_id", "candidate_price", "cart_value", "label"]], on=["session_id", "item_id", "label"], how="left")
    topk = df[df["rank"] <= TOP_K].copy()

    # Acceptance@K: at least one positive in top-K per session
    hit = topk.groupby("session_id")["label"].max()
    acceptance_at_k = float(hit.mean())

    # Realized add-on value per session (sum price of positive items in top-K)
    topk["accepted_value"] = np.where(topk["label"] == 1, topk["candidate_price"].astype(np.float32), 0.0)
    model_value_per_session = topk.groupby("session_id")["accepted_value"].sum()

    # Random baseline by session using same K
    rng = np.random.default_rng(SEED)
    rand = base[["session_id", "item_id", "label", "candidate_price", "cart_value"]].copy()
    rand["r"] = rng.random(len(rand))
    rand = rand.sort_values(["session_id", "r"], kind="mergesort")
    rand["rank"] = rand.groupby("session_id", sort=False).cumcount().add(1)
    rand = rand[rand["rank"] <= TOP_K].copy()
    rand["accepted_value"] = np.where(rand["label"] == 1, rand["candidate_price"].astype(np.float32), 0.0)
    rand_value_per_session = rand.groupby("session_id")["accepted_value"].sum()

    # Align sessions
    all_sid = model_value_per_session.index.union(rand_value_per_session.index)
    m = model_value_per_session.reindex(all_sid, fill_value=0.0)
    r = rand_value_per_session.reindex(all_sid, fill_value=0.0)

    avg_model_addon = float(m.mean())
    avg_random_addon = float(r.mean())
    abs_uplift = avg_model_addon - avg_random_addon
    rel_uplift = float(abs_uplift / max(avg_random_addon, 1e-6))

    cart_map = base.groupby("session_id")["cart_value"].first().reindex(all_sid).fillna(0.0)
    aov_lift_pct = float((abs_uplift / max(float(cart_map.mean()), 1e-6)) * 100.0)

    return {
        "model": model_name,
        "top_k": TOP_K,
        "acceptance_at_k": acceptance_at_k,
        "avg_addon_value_model": avg_model_addon,
        "avg_addon_value_random": avg_random_addon,
        "absolute_addon_uplift": abs_uplift,
        "relative_addon_uplift_vs_random": rel_uplift,
        "projected_aov_lift_percent": aov_lift_pct,
    }


def deployment_notes(best_model_name: str, metrics: Dict[str, float]) -> str:
    return (
        "# Cycle 2 Deployment Notes\n\n"
        f"Selected baseline: **{best_model_name}**\n\n"
        "## Why this model\n"
        f"- Best holdout NDCG@{TOP_K}: {metrics.get(f'NDCG@{TOP_K}', float('nan')):.4f}\n"
        f"- Holdout AUC: {metrics.get('AUC', float('nan')):.4f}\n\n"
        "## Recommended deployment strategy\n"
        "1. Start with shadow mode for 3-7 days against current heuristic baseline.\n"
        "2. Run 10-20% traffic A/B test with guardrails on conversion, latency, and bounce.\n"
        "3. Retrain weekly; refresh candidates daily or intra-day depending on catalog volatility.\n"
        "4. Monitor by segments: meal_time, user_segment, candidate_category, cart_value buckets.\n"
        "5. Add fallback policy for cold-start sessions/users/items.\n\n"
        "## Scalability checks\n"
        "- Keep candidate pool bounded (top-K retrieval) before ranking.\n"
        "- Use batch scoring for offline; cache top recommendations for high-traffic cohorts.\n"
        "- Track feature drift and label delay explicitly in monitoring.\n"
    )


def main() -> None:
    set_seed(SEED)
    t0 = time.perf_counter()
    prep = prepare_data()

    # Tune
    lgbm_model, lgbm_best, lgbm_trials = tune_lgbm(prep, TRIALS_LGBM)
    xgb_model, xgb_best, xgb_trials = tune_xgb(prep, TRIALS_XGB)

    tuning = pd.concat([lgbm_trials, xgb_trials], ignore_index=True)
    tuning.to_csv(OUT / "tuning_results.csv", index=False)

    # Evaluate both on test
    pred_lgbm_test = build_pred_df(lgbm_model, "LightGBM", prep.test, prep.feature_cols)
    pred_xgb_test = build_pred_df(xgb_model, "XGBoost", prep.test, prep.feature_cols)

    met_lgbm = evaluate_ranking(
        pred_lgbm_test["label"].to_numpy(np.int32),
        pred_lgbm_test["session_id"].to_numpy(np.int64),
        pred_lgbm_test["score"].to_numpy(np.float32),
        TOP_K,
    )
    met_xgb = evaluate_ranking(
        pred_xgb_test["label"].to_numpy(np.int32),
        pred_xgb_test["session_id"].to_numpy(np.int64),
        pred_xgb_test["score"].to_numpy(np.float32),
        TOP_K,
    )

    best_name = "LightGBM" if met_lgbm[f"NDCG@{TOP_K}"] >= met_xgb[f"NDCG@{TOP_K}"] else "XGBoost"
    best_pred = pred_lgbm_test if best_name == "LightGBM" else pred_xgb_test
    best_metrics = met_lgbm if best_name == "LightGBM" else met_xgb

    # Save predictions
    pred_lgbm_test.to_parquet(OUT / "test_predictions_lightgbm.parquet", index=False)
    pred_xgb_test.to_parquet(OUT / "test_predictions_xgboost.parquet", index=False)

    # Segment analysis + business impact on best model
    seg = segment_metrics(best_pred, prep.test, best_name)
    seg.to_csv(OUT / "segment_performance.csv", index=False)

    impact = business_impact(best_pred, prep.test, best_name)
    (OUT / "business_impact_summary.json").write_text(json.dumps(impact, indent=2))

    summary = {
        "seed": SEED,
        "top_k": TOP_K,
        "rows": {
            "train": int(len(prep.train)),
            "val": int(len(prep.val)),
            "test": int(len(prep.test)),
        },
        "features": {
            "count": len(prep.feature_cols),
            "categorical": prep.cat_cols,
        },
        "best_model": best_name,
        "metrics_test": {
            "LightGBM": met_lgbm,
            "XGBoost": met_xgb,
        },
        "best_trial": {
            "LightGBM": lgbm_best,
            "XGBoost": xgb_best,
        },
        "runtime_sec": float(time.perf_counter() - t0),
    }

    (OUT / "cycle2_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (OUT / "deployment_recommendations.md").write_text(deployment_notes(best_name, best_metrics))

    log(f"[done] best_model={best_name} NDCG@{TOP_K}={best_metrics[f'NDCG@{TOP_K}']:.4f} AUC={best_metrics['AUC']:.4f}")
    log(f"[done] artifacts_dir={OUT}")


if __name__ == "__main__":
    main()
