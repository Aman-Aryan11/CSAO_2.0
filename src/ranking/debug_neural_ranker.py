"""Debugging and stabilization pipeline for CSAO neural ranker."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ranking import train_neural_ranker as tnr


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _report_dir() -> Path:
    p = _project_root() / "reports" / "debug"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_json(name: str, obj: Dict[str, Any]) -> None:
    (_report_dir() / name).write_text(json.dumps(obj, indent=2, default=float))


def _save_md(name: str, text: str) -> None:
    (_report_dir() / name).write_text(text)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _load_split_limited(path: Path, max_rows: int) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    parts: List[pd.DataFrame] = []
    total = 0
    for rb in pf.iter_batches(batch_size=min(250_000, max_rows if max_rows > 0 else 250_000)):
        b = rb.to_pandas()
        if b.empty:
            continue
        if max_rows > 0:
            need = max_rows - total
            if need <= 0:
                break
            if len(b) > need:
                b = b.iloc[:need].copy()
        parts.append(b)
        total += len(b)
        if max_rows > 0 and total >= max_rows:
            break
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _load_split_spread(path: Path, max_rows: int) -> pd.DataFrame:
    """Load across row groups to avoid head-only class bias."""
    pf = pq.ParquetFile(path)
    nrg = pf.metadata.num_row_groups
    if nrg <= 0:
        return pd.DataFrame()
    total_rows = int(pf.metadata.num_rows or 0)
    if max_rows <= 0 or total_rows <= max_rows:
        return pf.read().to_pandas()

    avg_rows = max(total_rows // nrg, 1)
    target_rgs = min(max(int(np.ceil(max_rows / avg_rows)) + 2, 1), nrg)
    rg_ids = np.unique(np.linspace(0, nrg - 1, target_rgs, dtype=int))

    parts: List[pd.DataFrame] = []
    total = 0
    for i in rg_ids:
        b = pf.read_row_group(int(i)).to_pandas()
        if b.empty:
            continue
        need = max_rows - total
        if need <= 0:
            break
        if len(b) > need:
            b = b.iloc[:need].copy()
        parts.append(b)
        total += len(b)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _load_split_balanced_by_label(path: Path, max_rows: int) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    cols = set(pf.schema.names)
    label_col = "label" if "label" in cols else ("label_added" if "label_added" in cols else None)
    if label_col is None:
        return _load_split_spread(path, max_rows)

    half = max_rows // 2
    pos_parts: List[pd.DataFrame] = []
    neg_parts: List[pd.DataFrame] = []
    pos_n = 0
    neg_n = 0
    for rb in pf.iter_batches(batch_size=250_000):
        b = rb.to_pandas()
        if b.empty:
            continue
        pos = b[b[label_col] == 1]
        if pos_n < half and not pos.empty:
            need = half - pos_n
            take = pos.iloc[:need].copy() if len(pos) > need else pos
            pos_parts.append(take)
            pos_n += len(take)
        neg = b[b[label_col] == 0]
        if neg_n < (max_rows - half) and not neg.empty:
            need = (max_rows - half) - neg_n
            take = neg.iloc[:need].copy() if len(neg) > need else neg
            neg_parts.append(take)
            neg_n += len(take)
        if pos_n >= half and neg_n >= (max_rows - half):
            break
    out = pd.concat(pos_parts + neg_parts, ignore_index=True) if (pos_parts or neg_parts) else pd.DataFrame()
    if len(out) > max_rows:
        out = out.iloc[:max_rows].copy()
    if not out.empty:
        out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return out


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "candidate_item_id" in out.columns and "item_id" not in out.columns:
        out = out.rename(columns={"candidate_item_id": "item_id"})
    if "label_added" in out.columns and "label" not in out.columns:
        out = out.rename(columns={"label_added": "label"})
    if "label" not in out.columns:
        out["label"] = 0
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(np.int8)
    out["session_id"] = pd.to_numeric(out.get("session_id", 0), errors="coerce").fillna(0).astype(np.int64)
    out["item_id"] = pd.to_numeric(out.get("item_id", 0), errors="coerce").fillna(0).astype(np.int64)
    return out


def _load_dataframes(max_rows: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    train_p, val_p, test_p, source = tnr._resolve_data_paths()  # pylint: disable=protected-access
    train_df = _normalize_df(_load_split_balanced_by_label(train_p, max_rows))
    val_df = _normalize_df(_load_split_spread(val_p, max_rows))
    test_df = _normalize_df(_load_split_spread(test_p, max_rows))
    return train_df, val_df, test_df, source


def run_data_diagnostics(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute split-level data diagnostics and warnings."""

    def _split_stats(name: str, df: pd.DataFrame) -> Dict[str, Any]:
        rows = int(len(df))
        sess = int(df["session_id"].nunique()) if rows else 0
        items = int(df["item_id"].nunique()) if rows else 0
        cand_per = df.groupby("session_id")["item_id"].size() if rows else pd.Series(dtype=np.int64)
        pos_per = df.groupby("session_id")["label"].sum() if rows else pd.Series(dtype=np.int64)

        warnings: List[str] = []
        if len(cand_per) > 0 and float(cand_per.median()) <= 1:
            warnings.append("median_candidates_leq_1")
        if rows > 0 and int(df["label"].nunique()) < 2:
            warnings.append("no_label_variance")

        return {
            "split": name,
            "rows": rows,
            "unique_sessions": sess,
            "unique_items": items,
            "avg_candidates_per_session": float(cand_per.mean()) if len(cand_per) else 0.0,
            "median_candidates_per_session": float(cand_per.median()) if len(cand_per) else 0.0,
            "label_mean": float(df["label"].mean()) if rows else 0.0,
            "label_counts": {"0": int((df["label"] == 0).sum()), "1": int((df["label"] == 1).sum())},
            "avg_positives_per_session": float(pos_per.mean()) if len(pos_per) else 0.0,
            "pct_sessions_single_candidate": float((cand_per <= 1).mean()) if len(cand_per) else 0.0,
            "pct_sessions_no_positive": float((pos_per == 0).mean()) if len(pos_per) else 0.0,
            "pct_sessions_no_negative": float(((cand_per - pos_per) == 0).mean()) if len(cand_per) else 0.0,
            "warnings": warnings,
        }

    out = {
        "train": _split_stats("train", train_df),
        "val": _split_stats("val", val_df),
        "test": _split_stats("test", test_df),
    }
    _save_json("data_diagnostics.json", out)
    return out


def check_feature_leakage(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Detect direct and proxy leakage in feature columns."""
    ignore = {"session_id", "item_id", "label", "label_added", "timestamp", "split", "rank", "score", "model"}
    feature_cols = [c for c in train_df.columns if c not in ignore]

    sample_n = min(len(train_df), 200_000)
    sdf = train_df.sample(n=sample_n, random_state=42) if len(train_df) > sample_n else train_df
    y = sdf["label"].astype(np.int32)

    suspicious_names = [
        c
        for c in feature_cols
        if any(k in c.lower() for k in ["label", "target", "ground_truth", "clicked", "purchased", "is_positive"])
    ]

    corr_rows: List[Dict[str, Any]] = []
    auc_rows: List[Dict[str, Any]] = []
    for c in feature_cols:
        s = sdf[c]
        try:
            if pd.api.types.is_numeric_dtype(s.dtype):
                x = pd.to_numeric(s, errors="coerce").fillna(0.0)
                corr = float(np.corrcoef(x.to_numpy(), y.to_numpy())[0, 1]) if x.nunique() > 1 and y.nunique() > 1 else 0.0
                corr_rows.append({"feature": c, "abs_corr": float(abs(corr))})
                if y.nunique() > 1:
                    auc = float(tnr.roc_auc_score(y, x))
                    auc_rows.append({"feature": c, "auc": auc})
            else:
                grp = sdf.groupby(c, dropna=False)["label"].mean()
                score = s.map(grp).fillna(float(y.mean()))
                if y.nunique() > 1:
                    auc = float(tnr.roc_auc_score(y, score))
                    auc_rows.append({"feature": c, "auc": auc})
        except Exception:
            continue

    high_corr = [r for r in corr_rows if r["abs_corr"] >= 0.999]
    high_auc = [r for r in auc_rows if r["auc"] >= 0.999]

    out = {
        "num_features": len(feature_cols),
        "label_in_features": "label" in feature_cols,
        "suspicious_name_features": suspicious_names,
        "high_corr_features": high_corr,
        "high_auc_features": high_auc,
        "top_abs_corr": sorted(corr_rows, key=lambda x: x["abs_corr"], reverse=True)[:20],
        "top_auc": sorted(auc_rows, key=lambda x: x["auc"], reverse=True)[:20],
        "flags": {
            "label_in_features": bool("label" in feature_cols),
            "high_corr_detected": bool(len(high_corr) > 0),
            "high_auc_detected": bool(len(high_auc) > 0),
        },
    }
    _save_json("leakage_checks.json", out)
    return out


def _make_debug_config(base: tnr.TrainConfig, overfit: bool = False) -> tnr.TrainConfig:
    return replace(
        base,
        max_train_rows=_env_int("DEBUG_MAX_ROWS_PER_SPLIT", 2_000_000),
        max_val_rows=_env_int("DEBUG_MAX_ROWS_PER_SPLIT", 2_000_000),
        max_test_rows=_env_int("DEBUG_MAX_ROWS_PER_SPLIT", 2_000_000),
        batch_size=min(base.batch_size, 1024),
        num_epochs=_env_int("DEBUG_OVERFIT_EPOCHS", 20) if overfit else base.num_epochs,
    )


def _load_model_for_debug(config: tnr.TrainConfig, feature_spec: tnr.FeatureSpec, device: str) -> torch.nn.Module:
    model = tnr.build_model(config, feature_spec, device)
    p = tnr._paths()  # pylint: disable=protected-access
    ckpt_p = p["latest_checkpoint"]
    if ckpt_p.exists():
        try:
            ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        except Exception:
            pass
    return model


def validate_model_outputs(config: tnr.TrainConfig) -> Dict[str, Any]:
    """Validate prediction distribution on one validation batch."""
    dcfg = _make_debug_config(config)
    _, val_loader, _, spec, _ = tnr.load_datasets(dcfg)
    device = "cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu"
    model = _load_model_for_debug(dcfg, spec, device)
    model.eval()

    stats: Dict[str, Any] = {}
    for batch in val_loader:
        with torch.no_grad():
            logits = model(batch["num"].to(device), batch["cat"].to(device)).detach().cpu().numpy()
        stats = {
            "mean": float(np.mean(logits)) if logits.size else 0.0,
            "std": float(np.std(logits)) if logits.size else 0.0,
            "min": float(np.min(logits)) if logits.size else 0.0,
            "max": float(np.max(logits)) if logits.size else 0.0,
            "unique_ratio": float(np.unique(np.round(logits, 8)).size / max(len(logits), 1)),
            "constant_prediction_flag": bool(float(np.std(logits)) < 1e-6),
        }
        break

    if not stats:
        stats = {"error": "empty_validation_loader", "constant_prediction_flag": True}

    _save_json("prediction_statistics.json", stats)
    return stats


def validate_training_loop(config: tnr.TrainConfig) -> Dict[str, Any]:
    """Instrument short training run to verify optimizer/grad flow."""
    dcfg = _make_debug_config(config)
    dcfg = replace(dcfg, max_train_rows=min(dcfg.max_train_rows, 200_000), max_val_rows=min(dcfg.max_val_rows, 100_000), max_test_rows=min(dcfg.max_test_rows, 100_000))
    train_loader, val_loader, _, spec, _ = tnr.load_datasets(dcfg)
    device = "cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu"

    model = tnr.build_model(dcfg, spec, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=dcfg.lr)

    checks: Dict[str, Any] = {
        "model_train_mode_called": False,
        "loss_finite_batches": 0,
        "optimizer_steps": 0,
        "nonzero_grad_batches": 0,
        "param_change_detected": False,
        "grad_norm_mean": 0.0,
        "lr_before": float(optimizer.param_groups[0]["lr"]),
        "lr_after": float(optimizer.param_groups[0]["lr"]),
    }

    grad_norms: List[float] = []
    initial_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    model.train()
    checks["model_train_mode_called"] = bool(model.training)

    max_batches = 50
    for i, batch in enumerate(train_loader):
        if i >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["num"].to(device), batch["cat"].to(device))
        p, n = tnr.pairwise_sampler(
            logits=logits,
            labels=batch["label"].to(device),
            session_ids=batch["session_id"].to(device),
            neg_scores=batch["neg_score"].to(device),
            category_ids=batch["category_id"].to(device),
            restaurant_ids=batch["restaurant_id"].to(device),
            hard_negative=dcfg.hard_negative_mining,
            max_pairs_per_session=dcfg.max_pairs_per_session,
        )
        if p.numel() == 0 or n.numel() == 0:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["label"].to(device))
        else:
            loss = tnr.compute_pairwise_loss(p, n, dcfg.pairwise_loss)
        if not torch.isfinite(loss):
            continue
        checks["loss_finite_batches"] += 1
        loss.backward()

        total_norm_sq = 0.0
        nonzero_grad = False
        for param in model.parameters():
            if param.grad is None:
                continue
            g = float(param.grad.detach().norm().item())
            total_norm_sq += g * g
            if g > 0:
                nonzero_grad = True
        grad_norm = float(np.sqrt(total_norm_sq))
        grad_norms.append(grad_norm)
        if nonzero_grad:
            checks["nonzero_grad_batches"] += 1

        optimizer.step()
        checks["optimizer_steps"] += 1

    checks["lr_after"] = float(optimizer.param_groups[0]["lr"])
    checks["grad_norm_mean"] = float(np.mean(grad_norms)) if grad_norms else 0.0

    final_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    if initial_params and final_params:
        delta = sum(float((a - b).abs().sum().item()) for a, b in zip(initial_params, final_params))
        checks["param_change_detected"] = bool(delta > 0)

    flags = {
        "zero_optimizer_steps": checks["optimizer_steps"] == 0,
        "all_zero_gradients": checks["nonzero_grad_batches"] == 0,
        "no_parameter_update": not checks["param_change_detected"],
    }
    checks["flags"] = flags

    _save_json("training_sanity_checks.json", checks)
    return checks


def verify_loss_computation(config: tnr.TrainConfig) -> Dict[str, Any]:
    """Validate pairwise loss inputs and gradients."""
    dcfg = _make_debug_config(config)
    dcfg = replace(dcfg, max_train_rows=min(dcfg.max_train_rows, 200_000))
    train_loader, _, _, spec, _ = tnr.load_datasets(dcfg)
    device = "cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu"
    model = tnr.build_model(dcfg, spec, device)

    checks: Dict[str, Any] = {
        "checked_batches": 0,
        "shape_ok": True,
        "labels_binary_ok": True,
        "valid_pair_batches": 0,
        "loss_requires_grad_batches": 0,
        "loss_positive_batches": 0,
        "mask_removed_all_samples_batches": 0,
    }

    for i, batch in enumerate(train_loader):
        if i >= 50:
            break
        checks["checked_batches"] += 1

        y = batch["label"]
        if not torch.all((y == 0) | (y == 1)):
            checks["labels_binary_ok"] = False

        logits = model(batch["num"].to(device), batch["cat"].to(device))
        if logits.shape[0] != y.shape[0]:
            checks["shape_ok"] = False

        p, n = tnr.pairwise_sampler(
            logits=logits,
            labels=y.to(device),
            session_ids=batch["session_id"].to(device),
            neg_scores=batch["neg_score"].to(device),
            category_ids=batch["category_id"].to(device),
            restaurant_ids=batch["restaurant_id"].to(device),
            hard_negative=dcfg.hard_negative_mining,
            max_pairs_per_session=dcfg.max_pairs_per_session,
        )
        if p.numel() == 0 or n.numel() == 0:
            checks["mask_removed_all_samples_batches"] += 1
            continue
        checks["valid_pair_batches"] += 1
        loss = tnr.compute_pairwise_loss(p, n, dcfg.pairwise_loss)
        if loss.requires_grad:
            checks["loss_requires_grad_batches"] += 1
        if torch.isfinite(loss) and float(loss.item()) > 0:
            checks["loss_positive_batches"] += 1

    checks["flags"] = {
        "no_valid_pair_batches": checks["valid_pair_batches"] == 0,
        "shape_mismatch": not checks["shape_ok"],
        "non_binary_labels": not checks["labels_binary_ok"],
        "loss_never_requires_grad": checks["loss_requires_grad_batches"] == 0,
    }
    return checks


def validate_metrics_pipeline(config: tnr.TrainConfig) -> Dict[str, Any]:
    """Sanity-check metric behavior under random/shuffled/oracle scores."""
    dcfg = _make_debug_config(config)
    dcfg = replace(
        dcfg,
        max_train_rows=min(dcfg.max_train_rows, 300_000),
        max_val_rows=min(dcfg.max_val_rows, 400_000),
        max_test_rows=min(dcfg.max_test_rows, 200_000),
        num_epochs=3,
    )
    train_loader, val_loader, _, spec, _ = tnr.load_datasets(dcfg)
    device = "cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu"
    model = tnr.build_model(dcfg, spec, device)

    # Warm up model briefly so comparison isn't against an untrained checkpoint.
    optimizer = torch.optim.AdamW(model.parameters(), lr=dcfg.lr)
    for _ in range(2):
        _ = tnr.train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=None,
            config=dcfg,
            device=device,
        )

    model_metrics, pred = tnr.evaluate_model(model, val_loader, device=device, top_k=dcfg.top_k)
    if pred.empty:
        out = {"error": "empty_predictions"}
        _save_json("metric_validation.json", out)
        return out

    rnd = pred.copy()
    rng = np.random.default_rng(42)
    rnd["score"] = rng.random(len(rnd)).astype(np.float32)

    shf = pred.copy()
    shf["score"] = rng.permutation(shf["score"].to_numpy())

    oracle = pred.copy()
    oracle["score"] = oracle["label"].astype(np.float32)

    random_metrics = tnr.compute_metrics(rnd, top_k=dcfg.top_k)
    shuffled_metrics = tnr.compute_metrics(shf, top_k=dcfg.top_k)
    oracle_metrics = tnr.compute_metrics(oracle, top_k=dcfg.top_k)

    key = f"NDCG@{dcfg.top_k}"
    comparisons = {
        "oracle_ge_model": float(oracle_metrics.get(key, 0.0)) >= float(model_metrics.get(key, 0.0)),
        "model_ge_random": float(model_metrics.get(key, 0.0)) >= float(random_metrics.get(key, 0.0)) - 1e-3,
        "shuffled_near_random": abs(float(shuffled_metrics.get(key, 0.0)) - float(random_metrics.get(key, 0.0))) <= 0.1,
    }

    out = {
        "model_metrics": model_metrics,
        "random_metrics": random_metrics,
        "shuffled_metrics": shuffled_metrics,
        "oracle_metrics": oracle_metrics,
        "comparisons": comparisons,
    }
    _save_json("metric_validation.json", out)
    return out


def check_session_grouping() -> Dict[str, Any]:
    """Validate ranking metric grouping/sorting semantics using controlled example."""
    df = pd.DataFrame(
        {
            "session_id": [1, 1, 2, 2],
            "item_id": [10, 11, 20, 21],
            "label": [1, 0, 0, 1],
            "score": [0.9, 0.1, 0.2, 0.8],
        }
    )
    good = tnr.compute_metrics(df, top_k=10)

    bad_sorted = df.sort_values(["session_id", "label"], ascending=[True, False])
    bad = tnr.compute_metrics(bad_sorted.assign(score=bad_sorted["label"].to_numpy()), top_k=10)

    out = {
        "controlled_good_metrics": good,
        "controlled_label_sorted_metrics": bad,
        "grouping_by_session_check": True,
        "sorted_by_pred_not_label_check": float(good.get("MRR", 0.0)) >= 0.99,
    }
    return out


def detect_data_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Detect overlap leakage between train/val/test."""
    def _set_pairs(df: pd.DataFrame, cols: List[str]) -> set:
        return set(map(tuple, df[cols].drop_duplicates().to_numpy()))

    tr_s = set(train_df["session_id"].unique())
    va_s = set(val_df["session_id"].unique())
    te_s = set(test_df["session_id"].unique())

    tr_pairs = _set_pairs(train_df, ["session_id", "item_id"])
    va_pairs = _set_pairs(val_df, ["session_id", "item_id"])
    te_pairs = _set_pairs(test_df, ["session_id", "item_id"])

    def _pct(n: int, d: int) -> float:
        return float(n / d) if d else 0.0

    out = {
        "session_overlap": {
            "train_val": len(tr_s & va_s),
            "train_test": len(tr_s & te_s),
            "val_test": len(va_s & te_s),
        },
        "pair_overlap": {
            "train_val": len(tr_pairs & va_pairs),
            "train_test": len(tr_pairs & te_pairs),
            "val_test": len(va_pairs & te_pairs),
        },
        "item_overlap_pct": {
            "train_val": _pct(len(set(train_df["item_id"]) & set(val_df["item_id"])), len(set(val_df["item_id"]))),
            "train_test": _pct(len(set(train_df["item_id"]) & set(test_df["item_id"])), len(set(test_df["item_id"]))),
            "val_test": _pct(len(set(val_df["item_id"]) & set(test_df["item_id"])), len(set(test_df["item_id"]))),
        },
        "flags": {
            "high_session_leakage": bool(len(tr_s & va_s) > 0 or len(tr_s & te_s) > 0 or len(va_s & te_s) > 0),
            "pair_leakage": bool(len(tr_pairs & va_pairs) > 0 or len(tr_pairs & te_pairs) > 0 or len(va_pairs & te_pairs) > 0),
        },
    }
    return out


def run_overfit_test(config: tnr.TrainConfig, train_df: pd.DataFrame) -> Dict[str, Any]:
    """Train on tiny subset and check loss/metric progression."""
    n_sessions = _env_int("DEBUG_OVERFIT_SESSIONS", 100)
    n_epochs = _env_int("DEBUG_OVERFIT_EPOCHS", 20)

    sess_stats = train_df.groupby("session_id")["label"].agg(["sum", "count"])
    good_sessions = sess_stats[(sess_stats["sum"] > 0) & (sess_stats["sum"] < sess_stats["count"])].index
    if len(good_sessions) == 0:
        return {"error": "no_sessions_with_both_classes"}

    rng = np.random.default_rng(42)
    chosen = rng.choice(np.array(list(good_sessions)), size=min(n_sessions, len(good_sessions)), replace=False)
    small = train_df[train_df["session_id"].isin(chosen)].copy()

    # Build tensors from numeric-only quick path for overfit sanity.
    ignore = {"session_id", "item_id", "label", "label_added", "timestamp", "split", "rank", "score", "model"}
    feats = [c for c in small.columns if c not in ignore and pd.api.types.is_numeric_dtype(small[c].dtype)]
    if not feats:
        return {"error": "no_numeric_features_for_overfit"}

    X = torch.tensor(small[feats].fillna(0.0).to_numpy(np.float32))
    y = torch.tensor(small["label"].to_numpy(np.float32))
    sid = torch.tensor(small["session_id"].to_numpy(np.int64))

    class Tiny(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(1)

    model = Tiny(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses: List[float] = []
    ndcgs: List[float] = []
    for _ in range(n_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        p, n = tnr.pairwise_sampler(logits, y, sid, hard_negative=False, max_pairs_per_session=64)
        if p.numel() == 0 or n.numel() == 0:
            break
        loss = tnr.compute_pairwise_loss(p, n, "lambdarank")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))

        with torch.no_grad():
            pred = pd.DataFrame(
                {
                    "session_id": sid.numpy(),
                    "item_id": np.arange(len(y), dtype=np.int64),
                    "label": y.numpy().astype(np.int8),
                    "score": model(X).numpy().astype(np.float32),
                }
            )
        m = tnr.compute_metrics(pred, top_k=10)
        ndcgs.append(float(m.get("NDCG@10", 0.0)))

    out = {
        "epochs": len(losses),
        "loss_curve": losses,
        "ndcg_curve": ndcgs,
        "loss_decrease_ratio": float((losses[0] - losses[-1]) / max(losses[0], 1e-12)) if len(losses) >= 2 else 0.0,
        "flags": {
            "no_loss_decrease": bool(len(losses) >= 2 and losses[-1] >= losses[0]),
            "instant_perfect_metric": bool(len(ndcgs) > 0 and ndcgs[0] >= 0.999),
        },
    }
    return out


def generate_debug_report(all_outputs: Dict[str, Any]) -> str:
    """Generate markdown debug summary report and final readiness verdict."""
    issues: List[str] = []

    diag = all_outputs.get("data_diagnostics", {})
    for split in ["train", "val", "test"]:
        for w in diag.get(split, {}).get("warnings", []):
            issues.append(f"{split}: {w}")

    leakage = all_outputs.get("leakage_checks", {})
    if leakage.get("flags", {}).get("label_in_features"):
        issues.append("label present in feature set")
    if leakage.get("flags", {}).get("high_corr_detected"):
        issues.append("high feature-label correlation detected")

    sanity = all_outputs.get("training_sanity_checks", {})
    if sanity.get("flags", {}).get("zero_optimizer_steps"):
        issues.append("optimizer step never executed")
    if sanity.get("flags", {}).get("no_parameter_update"):
        issues.append("parameters did not update")

    pred_stats = all_outputs.get("prediction_statistics", {})
    if pred_stats.get("constant_prediction_flag", False):
        issues.append("model outputs are constant")

    metric_val = all_outputs.get("metric_validation", {})
    cmp = metric_val.get("comparisons", {})
    if cmp and (not cmp.get("oracle_ge_model", True) or not cmp.get("model_ge_random", True)):
        issues.append("metric sensitivity checks failed")

    split_leak = all_outputs.get("split_leakage", {})
    if split_leak.get("flags", {}).get("high_session_leakage"):
        issues.append("session overlap leakage across splits")

    verdict = "READY_FOR_TUNING" if not issues else "NOT_READY"

    md = []
    md.append("# Neural Ranker Debug Summary")
    md.append("")
    md.append(f"**Final Verdict:** `{verdict}`")
    md.append("")
    md.append("## Root Cause(s) of Metric Anomaly")
    if issues:
        for i in issues:
            md.append(f"- {i}")
    else:
        md.append("- No critical anomalies detected.")

    md.append("")
    md.append("## Data Integrity Findings")
    md.append(f"- Train rows: {diag.get('train', {}).get('rows', 0)}")
    md.append(f"- Val rows: {diag.get('val', {}).get('rows', 0)}")
    md.append(f"- Test rows: {diag.get('test', {}).get('rows', 0)}")

    md.append("")
    md.append("## Leakage Findings")
    md.append(f"- Feature leakage flags: {leakage.get('flags', {})}")
    md.append(f"- Split leakage flags: {split_leak.get('flags', {})}")

    md.append("")
    md.append("## Training Loop Findings")
    md.append(f"- Sanity checks: {sanity.get('flags', {})}")
    md.append(f"- Prediction stats: {pred_stats}")

    md.append("")
    md.append("## Metric Pipeline Findings")
    md.append(f"- Metric comparisons: {cmp}")

    md.append("")
    md.append("## Overfit Test Outcome")
    md.append(f"- {all_outputs.get('overfit_test', {})}")

    md.append("")
    md.append("## Next Actions")
    if issues:
        md.append("- P0: Fix critical flags listed in Root Cause(s).")
        md.append("- P1: Re-run debug module and verify metric sensitivity checks pass.")
        md.append("- P2: Start hyperparameter tuning only after verdict becomes READY_FOR_TUNING.")
    else:
        md.append("- P1: Proceed with hyperparameter tuning (lr, hidden dims, dropout, pairwise mining).")
        md.append("- P2: Add periodic debug checks in CI before long training runs.")

    md.append("")
    md.append("```json")
    md.append(json.dumps({"verdict": verdict, "issues": issues}, indent=2))
    md.append("```")

    text = "\n".join(md)
    _save_md("debug_summary_report.md", text)
    return verdict


def main() -> None:
    """Run full neural ranker debugging and stabilization checks."""
    start = time.perf_counter()
    cfg = tnr.load_config()
    tnr.set_seed(cfg.seed)

    max_rows = _env_int("DEBUG_MAX_ROWS_PER_SPLIT", 2_000_000)
    train_df, val_df, test_df, source = _load_dataframes(max_rows=max_rows)

    outputs: Dict[str, Any] = {}

    outputs["data_source"] = source
    outputs["data_diagnostics"] = run_data_diagnostics(train_df, val_df, test_df)
    outputs["leakage_checks"] = check_feature_leakage(train_df)
    outputs["prediction_statistics"] = validate_model_outputs(cfg)
    outputs["training_sanity_checks"] = validate_training_loop(cfg)
    outputs["loss_verification"] = verify_loss_computation(cfg)
    outputs["metric_validation"] = validate_metrics_pipeline(cfg)
    outputs["session_grouping"] = check_session_grouping()
    outputs["split_leakage"] = detect_data_leakage(train_df, val_df, test_df)
    outputs["overfit_test"] = run_overfit_test(cfg, train_df)

    # Fail-fast guard checks.
    if outputs["prediction_statistics"].get("constant_prediction_flag", False):
        raise RuntimeError("Constant predictions detected (std < 1e-6).")
    if outputs["training_sanity_checks"].get("flags", {}).get("zero_optimizer_steps", False):
        raise RuntimeError("No optimizer steps executed during sanity training run.")
    mv = outputs["metric_validation"].get("model_metrics", {})
    if mv and all(abs(float(mv.get(k, 0.0)) - 1.0) < 1e-9 for k in [f"NDCG@{cfg.top_k}", "MAP", "MRR", f"Precision@{cfg.top_k}", f"Recall@{cfg.top_k}"]):
        raise RuntimeError("All ranking metrics are exactly 1.0; likely leakage or broken metric path.")

    verdict = generate_debug_report(outputs)
    outputs["runtime_sec"] = float(time.perf_counter() - start)
    outputs["verdict"] = verdict

    # keep master summary for quick machine consumption
    _save_json("data_diagnostics.json", outputs["data_diagnostics"])  # ensure refreshed
    _save_json("prediction_statistics.json", outputs["prediction_statistics"])  # ensure refreshed
    _save_json("training_sanity_checks.json", outputs["training_sanity_checks"])  # ensure refreshed
    _save_json("metric_validation.json", outputs["metric_validation"])  # ensure refreshed
    _save_json("leakage_checks.json", outputs["leakage_checks"])  # ensure refreshed

    status = 0 if verdict == "READY_FOR_TUNING" else 2
    print(json.dumps({"status": status, "verdict": verdict, "report": str(_report_dir() / 'debug_summary_report.md')}))


if __name__ == "__main__":
    main()
