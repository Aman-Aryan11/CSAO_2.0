"""Production-grade error analysis pipeline for CSAO ranking models."""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.api.types import CategoricalDtype
from sklearn.metrics import roc_auc_score

try:
    from data_pipeline.config import RANDOM_SEED, LOGGER  # type: ignore
except Exception:
    try:
        from config import RANDOM_SEED, LOGGER  # type: ignore
    except Exception:
        RANDOM_SEED = 42
        LOGGER = logging.getLogger("error_analysis")
        if not LOGGER.handlers:
            LOGGER.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            LOGGER.addHandler(handler)
            LOGGER.propagate = False

try:
    from data_pipeline.config import TOP_K_EVAL as _TOP_K_EVAL  # type: ignore
except Exception:
    _TOP_K_EVAL = 10

TOP_K_EVAL = int(_TOP_K_EVAL) if _TOP_K_EVAL is not None else 10


# Runtime controls (safe defaults for large-scale stability)
MAX_ROWS_VAL = int(os.getenv("ERROR_ANALYSIS_MAX_ROWS_VAL", "2000000"))
MAX_ROWS_TEST = int(os.getenv("ERROR_ANALYSIS_MAX_ROWS_TEST", "2000000"))
ANALYSIS_SAMPLE_SIZE = int(os.getenv("ERROR_ANALYSIS_SAMPLE_SIZE", "0"))
DIAG_MAX_ROWS = int(os.getenv("ERROR_ANALYSIS_DIAG_MAX_ROWS", "1000000"))
PRED_BATCH_SIZE = int(os.getenv("ERROR_ANALYSIS_PRED_BATCH_SIZE", "500000"))


def _log(msg: str, *args: Any) -> None:
    LOGGER.info(msg, *args)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_featured_dir() -> Path:
    root = _project_root()
    options = [root / "data_pipeline" / "data" / "featured", root / "data" / "featured"]
    for p in options:
        if p.exists():
            return p
    return options[0]


def _resolve_processed_dir() -> Path:
    return _project_root() / "data_pipeline" / "data" / "processed"


def _resolve_model_dir() -> Path:
    root = _project_root()
    options = [root / "models" / "ranking", root / "output" / "baseline_output"]
    for p in options:
        if p.exists():
            return p
    return options[1]


def _report_dir() -> Path:
    return _project_root() / "output" / "error_analysis"


def _paths() -> Dict[str, Path]:
    fd = _resolve_featured_dir()
    pd_dir = _resolve_processed_dir()
    md = _resolve_model_dir()
    rd = _report_dir()
    return {
        "featured_val": fd / "val_ranking_features.parquet",
        "featured_test": fd / "test_ranking_features.parquet",
        "processed_val": pd_dir / "val.parquet",
        "processed_test": pd_dir / "test.parquet",
        "lgbm_model": md / "lightgbm_model.pkl",
        "xgb_model": md / "xgboost_model.pkl",
        "pred_val": rd / "predictions_val.parquet",
        "pred_test": rd / "predictions_test.parquet",
        "feature_importance": rd / "feature_importance.csv",
        "segment_metrics": rd / "segment_metrics.csv",
        "error_cases": rd / "error_cases.parquet",
        "calibration_curve": rd / "calibration_curve.csv",
        "summary": rd / "summary_report.json",
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024.0 * 1024.0))


def _safe_qcut(series: pd.Series, q: int, labels: Optional[Sequence[str]] = None) -> pd.Series:
    if series.nunique(dropna=True) < 2:
        return pd.Series(["single_bin"] * len(series), index=series.index, dtype="string")
    return pd.qcut(series, q=q, labels=labels, duplicates="drop").astype("string")


def _load_parquet_limited(path: Path, max_rows: int) -> pd.DataFrame:
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
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _load_parquet_spread(path: Path, max_rows: int) -> pd.DataFrame:
    """Load up to max_rows by spreading across parquet row groups."""
    pf = pq.ParquetFile(path)
    nrg = pf.metadata.num_row_groups
    if nrg <= 0:
        return pd.DataFrame()
    total_rows = int(pf.metadata.num_rows or 0)
    if max_rows <= 0 or total_rows <= max_rows:
        return pf.read().to_pandas()

    avg_rows = max(total_rows // nrg, 1)
    target_rgs = max(int(np.ceil(max_rows / avg_rows)), 1)
    target_rgs = min(target_rgs + 2, nrg)
    rg_idx = np.linspace(0, nrg - 1, num=target_rgs, dtype=int)
    rg_idx = np.unique(rg_idx)

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


def _load_processed_balanced(path: Path, max_rows: int) -> pd.DataFrame:
    """Load processed split with roughly balanced positive/negative labels."""
    pf = pq.ParquetFile(path)
    cols = set(pf.schema.names)
    label_col = "label" if "label" in cols else ("label_added" if "label_added" in cols else None)
    if label_col is None:
        return _load_parquet_limited(path, max_rows)

    half = max_rows // 2 if max_rows > 0 else 0
    pos_parts: List[pd.DataFrame] = []
    neg_parts: List[pd.DataFrame] = []
    pos_n = 0
    neg_n = 0

    for rb in pf.iter_batches(batch_size=250_000):
        b = rb.to_pandas()
        if b.empty:
            continue

        pos = b[b[label_col] == 1]
        if max_rows <= 0:
            pos_parts.append(pos)
        elif pos_n < half and not pos.empty:
            need = half - pos_n
            take = pos.iloc[:need].copy() if len(pos) > need else pos
            pos_parts.append(take)
            pos_n += len(take)

        neg = b[b[label_col] == 0]
        if max_rows <= 0:
            neg_parts.append(neg)
        elif neg_n < (max_rows - half) and not neg.empty:
            need = (max_rows - half) - neg_n
            take = neg.iloc[:need].copy() if len(neg) > need else neg
            neg_parts.append(take)
            neg_n += len(take)

        if max_rows > 0 and pos_n >= half and neg_n >= (max_rows - half):
            break

    parts = pos_parts + neg_parts
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if max_rows > 0 and len(out) > max_rows:
        out = out.iloc[:max_rows].copy()
    out = out.sample(frac=1.0, random_state=int(RANDOM_SEED)).reset_index(drop=True)
    return out


def _load_processed_natural(path: Path, max_rows: int) -> pd.DataFrame:
    """Load processed split preserving natural label distribution."""
    return _load_parquet_spread(path, max_rows)


def _normalize_featured(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "label_added" in out.columns and "label" not in out.columns:
        out = out.rename(columns={"label_added": "label"})
    required = {"session_id", "item_id", "label"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Missing required columns in featured data: {sorted(missing)}")
    out["label"] = out["label"].astype(np.int8)
    return out


def _normalize_processed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "candidate_item_id" in out.columns and "item_id" not in out.columns:
        out = out.rename(columns={"candidate_item_id": "item_id"})
    if "label_added" in out.columns and "label" not in out.columns:
        out = out.rename(columns={"label_added": "label"})
    required = {"session_id", "item_id", "label"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Missing required columns in processed data: {sorted(missing)}")
    out["label"] = out["label"].astype(np.int8)
    return out


def _sample_rows(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if n_rows <= 0 or len(df) <= n_rows:
        return df
    return df.sample(n=n_rows, random_state=int(RANDOM_SEED)).reset_index(drop=True)


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
    """Load val/test datasets with fallback to processed labels when featured labels are single-class."""
    paths = _paths()

    for key in ("featured_val", "featured_test"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing dataset: {paths[key]}")

    val_df = _normalize_featured(_load_parquet_limited(paths["featured_val"], MAX_ROWS_VAL))
    test_df = _normalize_featured(_load_parquet_limited(paths["featured_test"], MAX_ROWS_TEST))
    data_source = "featured"

    val_pos = int(val_df["label"].sum())
    test_pos = int(test_df["label"].sum())

    # Fallback to processed if featured labels are all zeros or single-class.
    if (val_pos == 0 and test_pos == 0) or (val_df["label"].nunique() < 2 or test_df["label"].nunique() < 2):
        _log("featured labels are single-class; switching to processed val/test for valid metrics")
        if not paths["processed_val"].exists() or not paths["processed_test"].exists():
            _log("processed fallback files not found; proceeding with featured datasets")
        else:
            use_balanced = os.getenv("ERROR_ANALYSIS_PROCESSED_BALANCED", "0").strip().lower() in {"1", "true", "yes"}
            loader = _load_processed_balanced if use_balanced else _load_processed_natural
            val_df = _normalize_processed(loader(paths["processed_val"], MAX_ROWS_VAL))
            test_df = _normalize_processed(loader(paths["processed_test"], MAX_ROWS_TEST))
            data_source = "processed"

    if ANALYSIS_SAMPLE_SIZE > 0:
        val_df = _sample_rows(val_df, ANALYSIS_SAMPLE_SIZE)
        test_df = _sample_rows(test_df, ANALYSIS_SAMPLE_SIZE)

    drop_cols = {"label", "session_id", "item_id", "timestamp", "split", "rank", "score", "model"}
    shared_features = sorted([c for c in val_df.columns if c in test_df.columns and c not in drop_cols])

    _log(
        "data_source=%s val rows=%d mem_mb=%.2f label_pos=%d",
        data_source,
        len(val_df),
        _memory_mb(val_df),
        int(val_df["label"].sum()),
    )
    _log(
        "data_source=%s test rows=%d mem_mb=%.2f label_pos=%d",
        data_source,
        len(test_df),
        _memory_mb(test_df),
        int(test_df["label"].sum()),
    )
    _log("shared feature columns=%d", len(shared_features))

    return val_df, test_df, shared_features, data_source


def _load_pickle_model(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_models() -> Dict[str, Any]:
    paths = _paths()
    lgbm_model = _load_pickle_model(paths["lgbm_model"])
    xgb_model = _load_pickle_model(paths["xgb_model"])
    _log("loaded LightGBM model: %s", paths["lgbm_model"])
    _log("loaded XGBoost model: %s", paths["xgb_model"])
    return {"LightGBM": lgbm_model, "XGBoost": xgb_model}


def _model_feature_names(model: Any) -> Optional[List[str]]:
    try:
        if hasattr(model, "feature_name"):
            names = model.feature_name()
            if names:
                return list(names)
    except Exception:
        pass
    try:
        if hasattr(model, "feature_name_"):
            names = model.feature_name_
            if names is not None:
                return list(names)
    except Exception:
        pass
    try:
        if hasattr(model, "get_booster"):
            names = model.get_booster().feature_names
            if names:
                return list(names)
    except Exception:
        pass
    return None


def _prepare_feature_matrix(df: pd.DataFrame, available_features: List[str], model: Any) -> Tuple[pd.DataFrame, List[str]]:
    model_features = _model_feature_names(model)
    cols = list(model_features) if model_features else list(available_features)
    if not cols:
        raise ValueError("No usable feature columns available for inference.")

    x = pd.DataFrame(index=df.index)
    for c in cols:
        x[c] = df[c] if c in df.columns else 0.0

    for c in cols:
        dt = x[c].dtype
        if pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt) or isinstance(dt, CategoricalDtype):
            x[c] = x[c].astype("category").cat.codes.astype(np.int32)
        elif pd.api.types.is_bool_dtype(dt):
            x[c] = x[c].astype(np.int8)

    x = x.replace([np.inf, -np.inf], np.nan)
    for c in cols:
        if x[c].isna().any():
            if pd.api.types.is_numeric_dtype(x[c].dtype):
                x[c] = x[c].fillna(float(x[c].median() if x[c].notna().any() else 0.0))
            else:
                x[c] = x[c].fillna(0)

    return x, cols


def _predict_in_batches(model: Any, x: pd.DataFrame, model_name: str, split: str) -> np.ndarray:
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float32)
    if n <= PRED_BATCH_SIZE:
        return np.asarray(model.predict(x), dtype=np.float32)

    chunks: List[np.ndarray] = []
    start = 0
    while start < n:
        end = min(start + PRED_BATCH_SIZE, n)
        batch_scores = np.asarray(model.predict(x.iloc[start:end]), dtype=np.float32)
        chunks.append(batch_scores)
        _log("%s %s prediction progress: %d/%d", model_name, split, end, n)
        start = end
    return np.concatenate(chunks)


def predict_scores(model: Any, model_name: str, df: pd.DataFrame, feature_cols: List[str], split: str) -> pd.DataFrame:
    start = time.perf_counter()
    x, used_cols = _prepare_feature_matrix(df, feature_cols, model)

    if not hasattr(model, "predict"):
        raise ValueError(f"Model {model_name} does not support predict().")

    scores = _predict_in_batches(model, x, model_name, split)

    pred = pd.DataFrame(
        {
            "session_id": df["session_id"].to_numpy(dtype=np.int64),
            "item_id": df["item_id"].to_numpy(dtype=np.int64),
            "label": df["label"].to_numpy(dtype=np.int8),
            "score": scores,
            "model": model_name,
            "split": split,
        }
    )
    pred.sort_values(["session_id", "score", "item_id"], ascending=[True, False, True], kind="mergesort", inplace=True)
    pred["rank"] = pred.groupby("session_id", sort=False).cumcount().add(1).astype(np.int32)

    _log(
        "%s %s prediction rows=%d features_used=%d score_mean=%.6f score_std=%.6f time_sec=%.2f",
        model_name,
        split,
        len(pred),
        len(used_cols),
        float(pred["score"].mean() if len(pred) else 0.0),
        float(pred["score"].std(ddof=0) if len(pred) else 0.0),
        time.perf_counter() - start,
    )
    return pred


def _ranking_metrics_from_pred(pred: pd.DataFrame, k: int = 10) -> Dict[str, float]:
    if pred.empty:
        return {f"NDCG@{k}": 0.0, "MAP": 0.0, "MRR": 0.0, f"Precision@{k}": 0.0, f"Recall@{k}": 0.0, "AUC": float("nan")}

    y_true = pred["label"].to_numpy(dtype=np.int32)
    scores = pred["score"].to_numpy(dtype=np.float32)
    session_ids = pred["session_id"].to_numpy(dtype=np.int64)

    order = np.lexsort((pred["item_id"].to_numpy(dtype=np.int64), -scores, session_ids))
    y = y_true[order]
    sids = session_ids[order]
    _, starts, counts = np.unique(sids, return_index=True, return_counts=True)

    ndcgs: List[float] = []
    aps: List[float] = []
    mrrs: List[float] = []
    pks: List[float] = []
    rks: List[float] = []

    for st, ct in zip(starts, counts):
        rel = y[st:st + ct]
        total_rel = int(rel.sum())
        cutoff = int(min(k, rel.size))
        rel_k = rel[:cutoff]

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
            aps.append(float((csum[hits] / (hits + 1.0)).sum() / total_rel))

        if cutoff == 0:
            ndcgs.append(0.0)
        else:
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

    out["AUC"] = float(roc_auc_score(y_true, scores)) if np.unique(y_true).size > 1 else float("nan")
    return out


def compute_global_metrics(pred_val: pd.DataFrame, pred_test: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_name, df in (("val", pred_val), ("test", pred_test)):
        for model_name, mdf in df.groupby("model", sort=False):
            metrics = _ranking_metrics_from_pred(mdf, k=k)
            row = {"model": model_name, "split": split_name, **metrics}
            rows.append(row)
            _log("%s %s metrics: %s", model_name, split_name, json.dumps(row, default=float))
    return pd.DataFrame(rows)


def compute_feature_importance(models: Dict[str, Any], feature_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    lgbm = models.get("LightGBM")
    if lgbm is not None:
        try:
            gain = lgbm.feature_importance(importance_type="gain")
            split = lgbm.feature_importance(importance_type="split")
            names = _model_feature_names(lgbm) or feature_cols
            for f, g, s in zip(names, gain, split):
                rows.append({"model": "LightGBM", "feature": f, "importance_gain": float(g), "importance_split": float(s)})
        except Exception as e:
            _log("LightGBM importance extraction failed: %s", str(e))

    xgb = models.get("XGBoost")
    if xgb is not None:
        try:
            gain_map = xgb.get_booster().get_score(importance_type="gain")
            names = _model_feature_names(xgb) or feature_cols
            for i, f in enumerate(names):
                rows.append({"model": "XGBoost", "feature": f, "importance_gain": float(gain_map.get(f, gain_map.get(f"f{i}", 0.0))), "importance_split": np.nan})
        except Exception as e:
            _log("XGBoost importance extraction failed: %s", str(e))

    fi = pd.DataFrame(rows)
    if not fi.empty:
        fi.sort_values(["model", "importance_gain"], ascending=[True, False], inplace=True)
    return fi


def _subset_for_diagnostics(pred_test: pd.DataFrame, test_df: pd.DataFrame, max_rows: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if max_rows <= 0 or len(pred_test) <= max_rows:
        return pred_test, test_df
    keep_sessions = pred_test["session_id"].drop_duplicates().sample(
        n=min(max_rows, pred_test["session_id"].nunique()), random_state=int(RANDOM_SEED)
    )
    p = pred_test[pred_test["session_id"].isin(keep_sessions)]
    if len(p) > max_rows:
        p = p.sample(n=max_rows, random_state=int(RANDOM_SEED))
    t = test_df[test_df["session_id"].isin(p["session_id"].unique())]
    _log("diagnostic subset rows: pred=%d test=%d", len(p), len(t))
    return p, t


def _assign_segment_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "candidate_source_count" in out.columns:
        out["candidate_source_bucket"] = pd.cut(
            out["candidate_source_count"].fillna(0), bins=[-1, 0, 1, 2, 3, 1000], labels=["0", "1", "2", "3", "4+"]
        ).astype("string")
    else:
        out["candidate_source_bucket"] = "unknown"

    if "item_count" in out.columns:
        out["session_length_bucket"] = pd.cut(
            out["item_count"].fillna(0), bins=[-1, 1, 3, 5, 10, 1000], labels=["1", "2-3", "4-5", "6-10", "11+"]
        ).astype("string")
    else:
        out["session_length_bucket"] = "unknown"

    if "candidate_popularity" in out.columns:
        out["item_popularity_bucket"] = _safe_qcut(out["candidate_popularity"], q=5)
    else:
        out["item_popularity_bucket"] = _safe_qcut(out.groupby("item_id")["item_id"].transform("count"), q=5)

    out["score_decile"] = _safe_qcut(out["score"], q=10)
    return out


def segment_analysis(pred_test: pd.DataFrame, test_df: pd.DataFrame, model_name: str, k: int = 10) -> pd.DataFrame:
    m = pred_test[pred_test["model"] == model_name]
    if m.empty:
        return pd.DataFrame()

    keep_cols = [c for c in ["session_id", "item_id", "candidate_source_count", "item_count", "candidate_popularity"] if c in test_df.columns]
    merged = m.merge(test_df[keep_cols], on=["session_id", "item_id"], how="left")
    merged = _assign_segment_buckets(merged)

    rows: List[Dict[str, Any]] = []
    for seg_col in ["candidate_source_bucket", "session_length_bucket", "item_popularity_bucket", "score_decile"]:
        for seg_val, sdf in merged.groupby(seg_col, dropna=False, sort=False):
            metrics = _ranking_metrics_from_pred(sdf[["session_id", "item_id", "label", "score"]], k=k)
            rows.append({"model": model_name, "segment_type": seg_col, "segment_value": str(seg_val), "rows": int(len(sdf)), "positive_rate": float(sdf["label"].mean()), **metrics})

    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["segment_type", f"NDCG@{k}", "rows"], ascending=[True, True, False], inplace=True)
    return out


def extract_error_cases(pred_test: pd.DataFrame, test_df: pd.DataFrame, model_name: str, top_n: int = 500) -> pd.DataFrame:
    m = pred_test[pred_test["model"] == model_name].copy()
    if m.empty:
        return pd.DataFrame()

    feature_cols = [c for c in ["candidate_source_count", "item_count", "candidate_popularity", "final_rank"] if c in test_df.columns]
    m = m.merge(test_df[["session_id", "item_id", *feature_cols]], on=["session_id", "item_id"], how="left")

    fp = m[m["label"] == 0].nlargest(top_n, "score")
    fp = fp.assign(error_type="false_positive")

    fn = m[m["label"] == 1].nsmallest(top_n, "score")
    fn = fn.assign(error_type="false_negative")

    err = pd.concat([fp, fn], ignore_index=True)
    err.sort_values(["error_type", "score"], ascending=[True, False], inplace=True)
    return err


def calibration_analysis(pred_df: pd.DataFrame, model_name: str, bins: int = 10) -> pd.DataFrame:
    m = pred_df[pred_df["model"] == model_name]
    if m.empty:
        return pd.DataFrame()
    w = m[["score", "label"]].copy()
    w["score_bin"] = _safe_qcut(w["score"], q=bins)
    cal = w.groupby("score_bin", dropna=False, sort=False).agg(count=("label", "size"), mean_pred=("score", "mean"), mean_actual=("label", "mean")).reset_index()
    cal.insert(0, "model", model_name)
    return cal


def retrieval_diagnosis(test_df: pd.DataFrame, pred_test: pd.DataFrame, model_name: str, k: int = 10) -> Dict[str, float]:
    m = pred_test[pred_test["model"] == model_name]
    positives = test_df[test_df["label"] == 1][["session_id", "item_id"]].drop_duplicates()
    total_pos = float(len(positives))
    if m.empty or total_pos == 0:
        return {"candidate_pool_recall": 0.0, "missing_positive_pct": 100.0, f"ranking_recall@{k}": 0.0}

    candidates = test_df[["session_id", "item_id"]].drop_duplicates()
    covered = positives.merge(candidates, on=["session_id", "item_id"], how="inner")
    pool_recall = float(len(covered) / total_pos)

    topk = m[m["rank"] <= k][["session_id", "item_id"]]
    topk_hits = positives.merge(topk, on=["session_id", "item_id"], how="inner")
    rank_recall = float(len(topk_hits) / total_pos)

    return {
        "candidate_pool_recall": pool_recall,
        "missing_positive_pct": float((1.0 - pool_recall) * 100.0),
        f"ranking_recall@{k}": rank_recall,
    }


def generate_summary_report(global_metrics: pd.DataFrame, segment_metrics: pd.DataFrame, retrieval_diag: Dict[str, float], feature_importance: pd.DataFrame, k: int, runtime_sec: float, data_source: str) -> Dict[str, Any]:
    ndcg_col = f"NDCG@{k}"
    best_model = None
    if not global_metrics.empty and ndcg_col in global_metrics.columns:
        val_rows = global_metrics[global_metrics["split"] == "val"]
        if not val_rows.empty:
            best_model = str(val_rows.sort_values(ndcg_col, ascending=False).iloc[0]["model"])

    worst_segments: List[Dict[str, Any]] = []
    if not segment_metrics.empty and ndcg_col in segment_metrics.columns:
        worst_segments = segment_metrics.nsmallest(5, ndcg_col).to_dict("records")

    top_features: Dict[str, List[Dict[str, Any]]] = {}
    if not feature_importance.empty:
        for model_name, g in feature_importance.groupby("model", sort=False):
            top_features[str(model_name)] = g.nlargest(10, "importance_gain")[["feature", "importance_gain"]].to_dict("records")

    return {
        "runtime_sec": float(runtime_sec),
        "data_source": data_source,
        "top_k_eval": int(k),
        "best_model": best_model,
        "global_metrics": global_metrics.to_dict("records"),
        "retrieval_diagnosis": retrieval_diag,
        "worst_segments": worst_segments,
        "top_features": top_features,
        "notes": [
            "If label positives are zero in featured data, processed fallback is used.",
            "Diagnostics run on a capped subset for memory stability.",
        ],
    }


def save_outputs(pred_val: pd.DataFrame, pred_test: pd.DataFrame, feature_importance: pd.DataFrame, segment_metrics: pd.DataFrame, error_cases: pd.DataFrame, calibration_curve: pd.DataFrame, summary_report: Dict[str, Any]) -> None:
    paths = _paths()
    out_dir = _report_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_val.to_parquet(paths["pred_val"], index=False)
    pred_test.to_parquet(paths["pred_test"], index=False)
    feature_importance.to_csv(paths["feature_importance"], index=False)
    segment_metrics.to_csv(paths["segment_metrics"], index=False)
    error_cases.to_parquet(paths["error_cases"], index=False)
    calibration_curve.to_csv(paths["calibration_curve"], index=False)
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, default=float)

    _log("saved predictions_val: %s", paths["pred_val"])
    _log("saved predictions_test: %s", paths["pred_test"])
    _log("saved feature_importance: %s", paths["feature_importance"])
    _log("saved segment_metrics: %s", paths["segment_metrics"])
    _log("saved error_cases: %s", paths["error_cases"])
    _log("saved calibration_curve: %s", paths["calibration_curve"])
    _log("saved summary_report: %s", paths["summary"])


def main() -> None:
    start = time.perf_counter()
    _set_seed(int(RANDOM_SEED))

    val_df, test_df, feature_cols, data_source = load_datasets()
    models = load_models()

    pred_val_parts: List[pd.DataFrame] = []
    pred_test_parts: List[pd.DataFrame] = []
    for name, model in models.items():
        pred_val_parts.append(predict_scores(model, name, val_df, feature_cols, split="val"))
        pred_test_parts.append(predict_scores(model, name, test_df, feature_cols, split="test"))

    pred_val = pd.concat(pred_val_parts, ignore_index=True)
    pred_test = pd.concat(pred_test_parts, ignore_index=True)

    global_metrics = compute_global_metrics(pred_val, pred_test, k=TOP_K_EVAL)

    best_model = "LightGBM"
    ndcg_col = f"NDCG@{TOP_K_EVAL}"
    if not global_metrics.empty and ndcg_col in global_metrics.columns:
        val_rows = global_metrics[global_metrics["split"] == "val"]
        if not val_rows.empty:
            best_model = str(val_rows.sort_values(ndcg_col, ascending=False).iloc[0]["model"])

    fi = compute_feature_importance(models, feature_cols)

    pred_diag, test_diag = _subset_for_diagnostics(pred_test, test_df, DIAG_MAX_ROWS)
    seg = segment_analysis(pred_diag, test_diag, model_name=best_model, k=TOP_K_EVAL)
    err = extract_error_cases(pred_diag, test_diag, model_name=best_model, top_n=500)
    cal = calibration_analysis(pred_diag, model_name=best_model, bins=10)
    retrieval = retrieval_diagnosis(test_diag, pred_diag, model_name=best_model, k=TOP_K_EVAL)

    summary = generate_summary_report(
        global_metrics=global_metrics,
        segment_metrics=seg,
        retrieval_diag=retrieval,
        feature_importance=fi,
        k=TOP_K_EVAL,
        runtime_sec=time.perf_counter() - start,
        data_source=data_source,
    )

    save_outputs(pred_val, pred_test, fi, seg, err, cal, summary)
    _log("analysis runtime_sec=%.2f", time.perf_counter() - start)


if __name__ == "__main__":
    main()
