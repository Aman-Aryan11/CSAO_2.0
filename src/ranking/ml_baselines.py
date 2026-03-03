"""Gradient-boosted ranking baselines (LightGBM, XGBoost) for CSAO."""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

# ============================================================
# CONFIG (compatibility-safe)
# ============================================================

try:
    from data_pipeline.config import (  # type: ignore
        RANDOM_SEED,
        LOGGER,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        LGBM_PARAMS,
        XGB_PARAMS,
        TOP_K,
    )
except Exception:
    try:
        from config import (  # type: ignore
            RANDOM_SEED,
            LOGGER,
            TRAIN_RATIO,
            VAL_RATIO,
            TEST_RATIO,
            LGBM_PARAMS,
            XGB_PARAMS,
            TOP_K,
        )
    except Exception:
        RANDOM_SEED = 42
        TRAIN_RATIO = 0.7
        VAL_RATIO = 0.15
        TEST_RATIO = 0.15
        TOP_K = 10
        LOGGER = logging.getLogger("ml_baselines")
        if not LOGGER.handlers:
            LOGGER.setLevel(logging.INFO)
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            LOGGER.addHandler(h)
            LOGGER.propagate = False
        LGBM_PARAMS = {}
        XGB_PARAMS = {}


# ============================================================
# DATA CONTAINERS
# ============================================================


@dataclass
class SplitData:
    name: str
    df: pd.DataFrame
    y: np.ndarray
    session_ids: np.ndarray
    item_ids: np.ndarray
    group_sizes: np.ndarray


@dataclass
class PreparedData:
    train: SplitData
    val: SplitData
    test: SplitData
    feature_cols: List[str]
    categorical_cols: List[str]


# ============================================================
# PATHS
# ============================================================


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _featured_dir() -> Path:
    return _project_root() / "data_pipeline" / "data" / "featured"


def _processed_dir() -> Path:
    return _project_root() / "data_pipeline" / "data" / "processed"


def _candidates_dir() -> Path:
    return _project_root() / "data_pipeline" / "data" / "candidates"


def _artifact_dir() -> Path:
    return _project_root() / "output" / "baseline_output"


def _state_path() -> Path:
    return _artifact_dir() / "checkpoint_state.json"


def _file_paths() -> Dict[str, Path]:
    fd = _artifact_dir()
    return {
        "train": _featured_dir() / "train_ranking_features.parquet",
        "val": _featured_dir() / "val_ranking_features.parquet",
        "test": _featured_dir() / "test_ranking_features.parquet",
        "processed_train": _processed_dir() / "train.parquet",
        "processed_val": _processed_dir() / "val.parquet",
        "processed_test": _processed_dir() / "test.parquet",
        "train_pred": fd / "train_predictions.parquet",
        "val_pred": fd / "val_predictions.parquet",
        "test_pred": fd / "test_predictions.parquet",
        "train_pred_lgbm": fd / "train_predictions_lightgbm.parquet",
        "val_pred_lgbm": fd / "val_predictions_lightgbm.parquet",
        "test_pred_lgbm": fd / "test_predictions_lightgbm.parquet",
        "train_pred_xgb": fd / "train_predictions_xgboost.parquet",
        "val_pred_xgb": fd / "val_predictions_xgboost.parquet",
        "test_pred_xgb": fd / "test_predictions_xgboost.parquet",
        "eval_json": _artifact_dir() / "evaluation_summary.json",
        "eval_csv": _artifact_dir() / "evaluation_summary.csv",
        "lgbm_fi": _artifact_dir() / "feature_importance_lightgbm.csv",
        "xgb_fi": _artifact_dir() / "feature_importance_xgboost.csv",
        "lgbm_model": _artifact_dir() / "lightgbm_model.pkl",
        "xgb_model": _artifact_dir() / "xgboost_model.pkl",
        "ranking_dataset": _processed_dir() / "ranking_dataset.parquet",
        "candidates_merged": _candidates_dir() / "candidates_merged.parquet",
    }


# ============================================================
# HELPERS
# ============================================================


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _log(msg: str, *args: Any) -> None:
    LOGGER.info(msg, *args)


def _load_state() -> Dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, sort_keys=True))


def _set_state_flag(state: Dict[str, Any], key: str, value: bool = True) -> None:
    state[key] = bool(value)
    _save_state(state)


def _build_positive_key_index(ranking_path: Path) -> np.ndarray:
    """Build sorted uint64 key index of positive (session_id, candidate_item_id)."""
    if not ranking_path.exists():
        return np.array([], dtype=np.uint64)

    pf = pq.ParquetFile(ranking_path)
    key_parts: List[np.ndarray] = []
    label_col = "label_added"

    cols = set(pf.schema.names)
    if "label" in cols:
        label_col = "label"
    elif "label_added" in cols:
        label_col = "label_added"
    else:
        return np.array([], dtype=np.uint64)

    item_col = "candidate_item_id" if "candidate_item_id" in cols else ("item_id" if "item_id" in cols else None)
    if item_col is None or "session_id" not in cols:
        return np.array([], dtype=np.uint64)

    for rb in pf.iter_batches(batch_size=500_000, columns=["session_id", item_col, label_col]):
        b = rb.to_pandas()
        pos = b[b[label_col] == 1]
        if pos.empty:
            continue
        sess = pos["session_id"].to_numpy(dtype=np.uint64)
        itm = pos[item_col].to_numpy(dtype=np.uint64)
        key_parts.append((sess << np.uint64(32)) | itm)

    if not key_parts:
        return np.array([], dtype=np.uint64)
    return np.unique(np.concatenate(key_parts))


def _relabel_split_from_positive_index(df: pd.DataFrame, target_col: str, pos_keys: np.ndarray) -> pd.DataFrame:
    if pos_keys.size == 0:
        return df
    keys = (df["session_id"].to_numpy(dtype=np.uint64) << np.uint64(32)) | df["item_id"].to_numpy(dtype=np.uint64)
    idx = np.searchsorted(pos_keys, keys)
    valid = idx < pos_keys.size
    is_pos = np.zeros(len(df), dtype=np.int32)
    is_pos[valid] = (pos_keys[idx[valid]] == keys[valid]).astype(np.int32)
    df[target_col] = is_pos
    return df


def _memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024 ** 2))


def _row_cap(split: str) -> int:
    defaults = {"train": 2_000_000, "val": 500_000, "test": 500_000}
    env_key = f"ML_BASELINE_MAX_ROWS_{split.upper()}"
    try:
        return int(os.getenv(env_key, str(defaults[split])))
    except Exception:
        return defaults[split]


def _target_col(df: pd.DataFrame) -> str:
    if "label" in df.columns:
        return "label"
    if "label_added" in df.columns:
        return "label_added"
    raise ValueError("Target column not found. Expected one of: ['label', 'label_added']")


def _group_sizes(session_ids: np.ndarray) -> np.ndarray:
    _, counts = np.unique(session_ids, return_counts=True)
    return counts.astype(np.int32)


def _sort_for_ranking(df: pd.DataFrame, session_col: str = "session_id") -> pd.DataFrame:
    return df.sort_values([session_col], kind="mergesort").reset_index(drop=True)


def _rank_within_session(df: pd.DataFrame, score_col: str) -> pd.Series:
    ordered = df.sort_values(["session_id", score_col, "item_id"], ascending=[True, False, True], kind="mergesort")
    rank = ordered.groupby("session_id", sort=False).cumcount().add(1).astype(np.int32)
    out = pd.Series(index=ordered.index, data=rank.values)
    return out.reindex(df.index)


def _load_split_limited(path: Path, max_rows: int, batch_size: int = 250_000) -> pd.DataFrame:
    """Load at most max_rows from parquet using streaming batches."""
    pf = pq.ParquetFile(path)
    parts: List[pd.DataFrame] = []
    total = 0

    for rb in pf.iter_batches(batch_size=batch_size):
        b = rb.to_pandas()
        if b.empty:
            continue
        need = max_rows - total
        if need <= 0:
            break
        if len(b) > need:
            b = b.iloc[:need].copy()
        parts.append(b)
        total += len(b)
        if total >= max_rows:
            break

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _load_split_spread(path: Path, max_rows: int) -> pd.DataFrame:
    """Load up to max_rows by spreading reads across parquet row groups."""
    pf = pq.ParquetFile(path)
    nrg = pf.metadata.num_row_groups
    if nrg <= 0:
        return pd.DataFrame()

    total_rows = int(pf.metadata.num_rows or 0)
    if total_rows <= max_rows:
        return pf.read().to_pandas()

    avg_rows = max(total_rows // nrg, 1)
    target_rgs = max(int(np.ceil(max_rows / avg_rows)), 1)
    target_rgs = min(target_rgs + 2, nrg)

    rg_idx = np.linspace(0, nrg - 1, num=target_rgs, dtype=int)
    rg_idx = np.unique(rg_idx)

    parts: List[pd.DataFrame] = []
    total = 0
    for i in rg_idx:
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
        if total >= max_rows:
            break

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _load_split_balanced_by_label(path: Path, max_rows: int) -> pd.DataFrame:
    """Load up to max_rows with approximately balanced labels from a processed split."""
    pf = pq.ParquetFile(path)
    cols = set(pf.schema.names)
    label_col = "label" if "label" in cols else ("label_added" if "label_added" in cols else None)
    if label_col is None:
        return _load_split_limited(path, max_rows=max_rows)

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
        if not pos.empty and pos_n < half:
            need = half - pos_n
            take = pos.iloc[:need].copy() if len(pos) > need else pos
            pos_parts.append(take)
            pos_n += len(take)

        neg = b[b[label_col] == 0]
        if not neg.empty and neg_n < (max_rows - half):
            need = (max_rows - half) - neg_n
            take = neg.iloc[:need].copy() if len(neg) > need else neg
            neg_parts.append(take)
            neg_n += len(take)

        if pos_n >= half and neg_n >= (max_rows - half):
            break

    out_parts = pos_parts + neg_parts
    if not out_parts:
        return pd.DataFrame()
    out = pd.concat(out_parts, ignore_index=True)
    if len(out) > max_rows:
        out = out.iloc[:max_rows].copy()

    # Deterministic shuffle to avoid label blocks from sorted source parquet.
    out = out.sample(frac=1.0, random_state=int(RANDOM_SEED)).reset_index(drop=True)
    return out


def _load_split_natural(path: Path, max_rows: int) -> pd.DataFrame:
    """Load up to max_rows while preserving natural class/session distribution."""
    # Spread sampling avoids head-bias while keeping original label ratio.
    return _load_split_spread(path, max_rows=max_rows)


# ============================================================
# REQUIRED FUNCTIONS
# ============================================================


def load_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test ranking feature datasets."""
    paths = _file_paths()

    for key in ("train", "val", "test"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing feature dataset: {paths[key]}")

    train_cap = _row_cap("train")
    val_cap = _row_cap("val")
    test_cap = _row_cap("test")

    use_spread = os.getenv("ML_BASELINE_SPREAD_SAMPLE", "1").strip().lower() not in {"0", "false", "no"}
    if use_spread:
        train_df = _load_split_spread(paths["train"], max_rows=train_cap)
        val_df = _load_split_spread(paths["val"], max_rows=val_cap)
        test_df = _load_split_spread(paths["test"], max_rows=test_cap)
    else:
        train_df = _load_split_limited(paths["train"], max_rows=train_cap)
        val_df = _load_split_limited(paths["val"], max_rows=val_cap)
        test_df = _load_split_limited(paths["test"], max_rows=test_cap)

    _log("loaded train rows=%d (cap=%d) mem_mb=%.2f", len(train_df), train_cap, _memory_mb(train_df))
    _log("loaded val rows=%d (cap=%d) mem_mb=%.2f", len(val_df), val_cap, _memory_mb(val_df))
    _log("loaded test rows=%d (cap=%d) mem_mb=%.2f", len(test_df), test_cap, _memory_mb(test_df))

    return train_df, val_df, test_df


def _normalize_processed_split(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize processed ranking split schema to baseline expected columns."""
    out = df.copy()
    if "candidate_item_id" in out.columns and "item_id" not in out.columns:
        out = out.rename(columns={"candidate_item_id": "item_id"})
    if "label_added" in out.columns and "label" not in out.columns:
        out = out.rename(columns={"label_added": "label"})
    return out


def _load_processed_fallback() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed train/val/test splits as a fallback labeled source."""
    paths = _file_paths()
    for key in ("processed_train", "processed_val", "processed_test"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing processed fallback dataset: {paths[key]}")

    train_cap = _row_cap("train")
    val_cap = _row_cap("val")
    test_cap = _row_cap("test")

    use_balanced = os.getenv("ML_BASELINE_PROCESSED_BALANCED", "0").strip().lower() in {"1", "true", "yes"}
    loader = _load_split_balanced_by_label if use_balanced else _load_split_natural

    train_df = _normalize_processed_split(loader(paths["processed_train"], max_rows=train_cap))
    val_df = _normalize_processed_split(loader(paths["processed_val"], max_rows=val_cap))
    test_df = _normalize_processed_split(loader(paths["processed_test"], max_rows=test_cap))

    _log(
        "processed fallback loaded train=%d val=%d test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def preprocess_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> PreparedData:
    """Preprocess ranking features for LightGBM/XGBoost training."""
    target = _target_col(train_df)
    paths = _file_paths()

    required = {"session_id", "item_id", target}
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    # If any split is single-class, switch directly to processed fallback.
    train_classes = np.unique(train_df[target].to_numpy()).size
    val_target = _target_col(val_df)
    test_target = _target_col(test_df)
    val_classes = np.unique(val_df[val_target].to_numpy()).size
    test_classes = np.unique(test_df[test_target].to_numpy()).size
    if train_classes < 2 or val_classes < 2 or test_classes < 2:
        _log(
            "single-class labels detected in featured split(s) train=%d val=%d test=%d; switching to processed fallback",
            train_classes,
            val_classes,
            test_classes,
        )
        train_df, val_df, test_df = _load_processed_fallback()
        target = _target_col(train_df)
        _log(
            "processed label distribution: train_pos=%d val_pos=%d test_pos=%d",
            int((train_df[target] == 1).sum()),
            int((val_df[target] == 1).sum()),
            int((test_df[target] == 1).sum()),
        )

    drop_cols = {
        target,
        "label",
        "label_added",
        "timestamp",
        "split",
        "rank",
        "score",
        "model",
    }
    if os.getenv("ML_BASELINE_DROP_LEAKY_FEATURES", "1").strip().lower() not in {"0", "false", "no"}:
        drop_cols.update({"final_rank", "candidate_source_count", "source_count_ratio"})

    feature_cols = [
        c
        for c in train_df.columns
        if c not in drop_cols and c not in {"session_id", "item_id"}
    ]

    categorical_cols = [
        c
        for c in feature_cols
        if str(train_df[c].dtype) in {"object", "string", "category", "bool"}
        or c.endswith("_category")
        or c.endswith("_segment")
        or c == "meal_time"
    ]

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    tr = _sort_for_ranking(train_df)
    va = _sort_for_ranking(val_df)
    te = _sort_for_ranking(test_df)

    # Remove duplicate session-item rows to prevent inflated/easy metrics.
    def _dedupe_session_item(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
        if "session_id" not in df.columns or "item_id" not in df.columns:
            return df
        x = df.copy()
        x = x.sort_values(["session_id", "item_id", y_col], ascending=[True, True, False], kind="mergesort")
        x = x.drop_duplicates(subset=["session_id", "item_id"], keep="first")
        return x

    tr = _dedupe_session_item(tr, target)
    va = _dedupe_session_item(va, target)
    te = _dedupe_session_item(te, target)

    # Encode categoricals using train vocabulary.
    for c in categorical_cols:
        tr_c = tr[c].astype("string").fillna("__MISSING__")
        cats = pd.Index(tr_c.unique())
        cat_to_id = {v: i for i, v in enumerate(cats)}

        tr[c] = tr_c.map(cat_to_id).fillna(-1).astype(np.int32)
        va[c] = va[c].astype("string").fillna("__MISSING__").map(cat_to_id).fillna(-1).astype(np.int32)
        te[c] = te[c].astype("string").fillna("__MISSING__").map(cat_to_id).fillna(-1).astype(np.int32)

    # Fill numeric with train medians.
    if numeric_cols:
        medians = tr[numeric_cols].median(numeric_only=True)
        tr[numeric_cols] = tr[numeric_cols].fillna(medians).astype(np.float32)
        va[numeric_cols] = va[numeric_cols].fillna(medians).astype(np.float32)
        te[numeric_cols] = te[numeric_cols].fillna(medians).astype(np.float32)

    # Cast category ints to float for unified matrices (xgboost safe).
    for c in categorical_cols:
        tr[c] = tr[c].astype(np.float32)
        va[c] = va[c].astype(np.float32)
        te[c] = te[c].astype(np.float32)

    feature_cols = numeric_cols + categorical_cols

    # Auto-prune leakage-like features if they are near-deterministic with labels.
    leak_thresh = float(os.getenv("ML_BASELINE_LEAK_CORR_THRESHOLD", "0.995"))
    if leak_thresh > 0.0 and feature_cols:
        y_train = tr[target].to_numpy(dtype=np.float64)
        keep_cols: List[str] = []
        dropped_corr: List[Tuple[str, float]] = []
        y_std = float(np.std(y_train))
        for c in feature_cols:
            x = pd.to_numeric(tr[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            x_std = float(np.std(x))
            if x_std <= 1e-12 or y_std <= 1e-12:
                keep_cols.append(c)
                continue
            corr = float(np.corrcoef(x, y_train)[0, 1])
            if np.isfinite(corr) and abs(corr) >= leak_thresh:
                dropped_corr.append((c, corr))
            else:
                keep_cols.append(c)
        if dropped_corr:
            for c, corr in sorted(dropped_corr, key=lambda t: abs(t[1]), reverse=True):
                _log("dropping leakage-like feature by corr threshold: %s corr=%.6f", c, corr)
            feature_cols = keep_cols

    train_split = SplitData(
        name="train",
        df=tr,
        y=tr[target].to_numpy(dtype=np.int32),
        session_ids=tr["session_id"].to_numpy(dtype=np.int64),
        item_ids=tr["item_id"].to_numpy(dtype=np.int64),
        group_sizes=_group_sizes(tr["session_id"].to_numpy(dtype=np.int64)),
    )
    val_split = SplitData(
        name="val",
        df=va,
        y=va[target].to_numpy(dtype=np.int32),
        session_ids=va["session_id"].to_numpy(dtype=np.int64),
        item_ids=va["item_id"].to_numpy(dtype=np.int64),
        group_sizes=_group_sizes(va["session_id"].to_numpy(dtype=np.int64)),
    )
    test_split = SplitData(
        name="test",
        df=te,
        y=te[target].to_numpy(dtype=np.int32),
        session_ids=te["session_id"].to_numpy(dtype=np.int64),
        item_ids=te["item_id"].to_numpy(dtype=np.int64),
        group_sizes=_group_sizes(te["session_id"].to_numpy(dtype=np.int64)),
    )

    _log("feature columns=%d categorical=%d", len(feature_cols), len([c for c in categorical_cols if c in feature_cols]))

    return PreparedData(
        train=train_split,
        val=val_split,
        test=test_split,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
    )


def train_lgbm(data: PreparedData) -> Any:
    """Train LightGBM ranking model with lambdarank objective."""
    try:
        import lightgbm as lgb
    except Exception as e:
        raise ImportError("lightgbm is required for train_lgbm") from e

    defaults = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "num_threads": 8,
        "seed": int(RANDOM_SEED),
        "verbosity": -1,
        "n_estimators": 800,
    }
    params = {**defaults, **(LGBM_PARAMS if isinstance(LGBM_PARAMS, dict) else {})}

    X_tr = data.train.df[data.feature_cols]
    X_va = data.val.df[data.feature_cols]

    dtrain = lgb.Dataset(
        X_tr,
        label=data.train.y,
        group=data.train.group_sizes,
        feature_name=data.feature_cols,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_va,
        label=data.val.y,
        group=data.val.group_sizes,
        feature_name=data.feature_cols,
        reference=dtrain,
        free_raw_data=False,
    )

    start = time.perf_counter()
    model = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dval],
        valid_names=["val"],
        num_boost_round=int(params.get("n_estimators", 2000)),
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    _log("LightGBM training_time_sec=%.2f best_iteration=%s", time.perf_counter() - start, str(model.best_iteration))
    return model


def train_xgb(data: PreparedData) -> Any:
    """Train XGBoost ranking model with rank:ndcg objective."""
    try:
        import xgboost as xgb
    except Exception as e:
        raise ImportError("xgboost is required for train_xgb") from e

    defaults = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_estimators": 800,
        "n_jobs": 8,
        "random_state": int(RANDOM_SEED),
    }
    params = {**defaults, **(XGB_PARAMS if isinstance(XGB_PARAMS, dict) else {})}

    model = xgb.XGBRanker(**params)

    X_tr = data.train.df[data.feature_cols]
    X_va = data.val.df[data.feature_cols]

    start = time.perf_counter()
    model.fit(
        X_tr,
        data.train.y,
        group=data.train.group_sizes.tolist(),
        eval_set=[(X_va, data.val.y)],
        eval_group=[data.val.group_sizes.tolist()],
        verbose=False,
    )
    _log("XGBoost training_time_sec=%.2f", time.perf_counter() - start)
    return model


def generate_predictions(model: Any, split: SplitData, feature_cols: List[str], model_name: str) -> pd.DataFrame:
    """Generate model scores and per-session ranks for a dataset split."""
    X = split.df[feature_cols]

    start = time.perf_counter()
    if hasattr(model, "predict"):
        scores = model.predict(X)
    else:
        raise ValueError(f"Model {model_name} does not expose predict()")

    pred = pd.DataFrame(
        {
            "session_id": split.session_ids,
            "item_id": split.item_ids,
            "score": np.asarray(scores, dtype=np.float32),
            "model": model_name,
        }
    )

    pred = pred.sort_values(["session_id", "score", "item_id"], ascending=[True, False, True], kind="mergesort")
    pred["rank"] = pred.groupby("session_id", sort=False).cumcount().add(1).astype(np.int32)

    _log("%s prediction_time_sec=%.2f rows=%d", model_name, time.perf_counter() - start, len(pred))
    return pred


def evaluate_ranking(y_true: np.ndarray, session_ids: np.ndarray, scores: np.ndarray, k: int = 10) -> Dict[str, float]:
    """Compute ranking metrics: NDCG@K, MAP, MRR, AUC, Precision@K, Recall@K."""
    k = int(k)
    eval_df = pd.DataFrame(
        {
            "session_id": session_ids.astype(np.int64),
            "y": y_true.astype(np.int32),
            "s": scores.astype(np.float32),
        }
    )
    eval_df = eval_df.sort_values(["session_id", "s"], ascending=[True, False], kind="mergesort")

    ndcgs: List[float] = []
    aps: List[float] = []
    mrrs: List[float] = []
    pks: List[float] = []
    rks: List[float] = []

    for _, g in eval_df.groupby("session_id", sort=False):
        rel = g["y"].to_numpy(dtype=np.int32)
        if rel.size == 0:
            continue

        total_rel = int(rel.sum())
        cutoff = min(k, rel.size)
        rel_k = rel[:cutoff]

        # Precision@K / Recall@K
        tp_k = int(rel_k.sum())
        pks.append(tp_k / float(max(cutoff, 1)))
        rks.append(0.0 if total_rel == 0 else tp_k / float(total_rel))

        # MRR
        pos_idx = np.flatnonzero(rel)
        mrrs.append(0.0 if pos_idx.size == 0 else 1.0 / float(pos_idx[0] + 1))

        # AP
        if total_rel == 0:
            aps.append(0.0)
        else:
            csum = np.cumsum(rel)
            hits = np.flatnonzero(rel)
            prec_at_hits = csum[hits] / (hits + 1.0)
            aps.append(float(prec_at_hits.sum() / total_rel))

        # NDCG@K
        if cutoff == 0:
            ndcgs.append(0.0)
        else:
            discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
            dcg = float((rel_k * discounts).sum())
            ideal = np.sort(rel)[::-1][:cutoff]
            idcg = float((ideal * discounts).sum())
            ndcgs.append(0.0 if idcg <= 0 else dcg / idcg)

    metrics: Dict[str, float] = {
        f"NDCG@{k}": float(np.mean(ndcgs) if ndcgs else 0.0),
        "MAP": float(np.mean(aps) if aps else 0.0),
        "MRR": float(np.mean(mrrs) if mrrs else 0.0),
        f"Precision@{k}": float(np.mean(pks) if pks else 0.0),
        f"Recall@{k}": float(np.mean(rks) if rks else 0.0),
    }

    unique_labels = np.unique(y_true)
    if unique_labels.size > 1:
        metrics["AUC"] = float(roc_auc_score(y_true, scores))
    else:
        metrics["AUC"] = float("nan")

    return metrics


def save_predictions(train_pred: pd.DataFrame, val_pred: pd.DataFrame, test_pred: pd.DataFrame) -> None:
    """Persist train/val/test predictions parquet files."""
    paths = _file_paths()
    for key in ("train_pred", "val_pred", "test_pred"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)

    train_pred.to_parquet(paths["train_pred"], index=False)
    val_pred.to_parquet(paths["val_pred"], index=False)
    test_pred.to_parquet(paths["test_pred"], index=False)

    _log("saved train predictions: %s", paths["train_pred"])
    _log("saved val predictions: %s", paths["val_pred"])
    _log("saved test predictions: %s", paths["test_pred"])


def _align_labels_to_predictions(pred: pd.DataFrame, split: SplitData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align labels to prediction rows using (session_id, item_id) keys."""
    gold = split.df[["session_id", "item_id"]].copy()
    gold["label"] = split.y.astype(np.int32)
    gold["dup_rank"] = gold.groupby(["session_id", "item_id"], sort=False).cumcount().astype(np.int32)

    pred_keys = pred[["session_id", "item_id", "score"]].copy()
    pred_keys["dup_rank"] = pred_keys.groupby(["session_id", "item_id"], sort=False).cumcount().astype(np.int32)

    aligned = pred_keys.merge(
        gold,
        on=["session_id", "item_id", "dup_rank"],
        how="left",
        validate="one_to_one",
    )
    labels = aligned["label"].fillna(0).to_numpy(dtype=np.int32)
    sess = aligned["session_id"].to_numpy(dtype=np.int64)
    scores = aligned["score"].to_numpy(dtype=np.float32)
    return labels, sess, scores


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Train LightGBM/XGBoost baselines, evaluate, and save outputs."""
    _set_seed(int(RANDOM_SEED))
    run_start = time.perf_counter()
    paths = _file_paths()
    _artifact_dir().mkdir(parents=True, exist_ok=True)
    state = _load_state()
    state_version = 3
    if state.get("version") != state_version:
        state = {"version": state_version}
        _save_state(state)
        _log("checkpoint state reset for version=%d", state_version)

    train_df, val_df, test_df = load_features()
    data = preprocess_features(train_df, val_df, test_df)
    data_signature = {
        "train_rows": int(len(data.train.df)),
        "val_rows": int(len(data.val.df)),
        "test_rows": int(len(data.test.df)),
        "train_pos": int(data.train.y.sum()),
        "val_pos": int(data.val.y.sum()),
        "test_pos": int(data.test.y.sum()),
        "feature_cols": list(data.feature_cols),
        "spread_sample": os.getenv("ML_BASELINE_SPREAD_SAMPLE", "1"),
        "drop_leaky_features": os.getenv("ML_BASELINE_DROP_LEAKY_FEATURES", "1"),
        "leak_corr_threshold": os.getenv("ML_BASELINE_LEAK_CORR_THRESHOLD", "0.995"),
    }

    same_signature = state.get("data_signature") == data_signature
    if not same_signature:
        _log("data signature changed; invalidating model/prediction/eval checkpoints")
        for key in (
            "lgbm_trained",
            "xgb_trained",
            "lgbm_pred_done",
            "xgb_pred_done",
            "merged_pred_done",
            "eval_done",
        ):
            state[key] = False
        state["data_signature"] = data_signature
        _save_state(state)

    # -----------------------------
    # Train/load LightGBM checkpoint
    # -----------------------------
    if paths["lgbm_model"].exists() and state.get("lgbm_trained", False):
        with open(paths["lgbm_model"], "rb") as f:
            lgbm_model = pickle.load(f)
        _log("loaded LightGBM checkpoint: %s", paths["lgbm_model"])
    else:
        lgbm_model = train_lgbm(data)
        with open(paths["lgbm_model"], "wb") as f:
            pickle.dump(lgbm_model, f)
        _set_state_flag(state, "lgbm_trained", True)
        _log("saved LightGBM checkpoint: %s", paths["lgbm_model"])

    # -----------------------------
    # Train/load XGBoost checkpoint
    # -----------------------------
    if paths["xgb_model"].exists() and state.get("xgb_trained", False):
        with open(paths["xgb_model"], "rb") as f:
            xgb_model = pickle.load(f)
        _log("loaded XGBoost checkpoint: %s", paths["xgb_model"])
    else:
        xgb_model = train_xgb(data)
        with open(paths["xgb_model"], "wb") as f:
            pickle.dump(xgb_model, f)
        _set_state_flag(state, "xgb_trained", True)
        _log("saved XGBoost checkpoint: %s", paths["xgb_model"])

    # -----------------------------
    # Prediction checkpoints
    # -----------------------------
    if paths["train_pred_lgbm"].exists() and paths["val_pred_lgbm"].exists() and paths["test_pred_lgbm"].exists() and state.get("lgbm_pred_done", False):
        pred_train_lgbm = pd.read_parquet(paths["train_pred_lgbm"])
        pred_val_lgbm = pd.read_parquet(paths["val_pred_lgbm"])
        pred_test_lgbm = pd.read_parquet(paths["test_pred_lgbm"])
        lgbm_ckpt_ok = (
            len(pred_train_lgbm) == len(data.train.df)
            and len(pred_val_lgbm) == len(data.val.df)
            and len(pred_test_lgbm) == len(data.test.df)
        )
        if lgbm_ckpt_ok:
            _log("loaded LightGBM prediction checkpoints")
        else:
            _log("LightGBM prediction checkpoint size mismatch; regenerating predictions")
            pred_train_lgbm = generate_predictions(lgbm_model, data.train, data.feature_cols, "LightGBM")
            pred_val_lgbm = generate_predictions(lgbm_model, data.val, data.feature_cols, "LightGBM")
            pred_test_lgbm = generate_predictions(lgbm_model, data.test, data.feature_cols, "LightGBM")
            pred_train_lgbm.to_parquet(paths["train_pred_lgbm"], index=False)
            pred_val_lgbm.to_parquet(paths["val_pred_lgbm"], index=False)
            pred_test_lgbm.to_parquet(paths["test_pred_lgbm"], index=False)
    else:
        pred_train_lgbm = generate_predictions(lgbm_model, data.train, data.feature_cols, "LightGBM")
        pred_val_lgbm = generate_predictions(lgbm_model, data.val, data.feature_cols, "LightGBM")
        pred_test_lgbm = generate_predictions(lgbm_model, data.test, data.feature_cols, "LightGBM")
        pred_train_lgbm.to_parquet(paths["train_pred_lgbm"], index=False)
        pred_val_lgbm.to_parquet(paths["val_pred_lgbm"], index=False)
        pred_test_lgbm.to_parquet(paths["test_pred_lgbm"], index=False)
        _set_state_flag(state, "lgbm_pred_done", True)
        _log("saved LightGBM prediction checkpoints")

    if paths["train_pred_xgb"].exists() and paths["val_pred_xgb"].exists() and paths["test_pred_xgb"].exists() and state.get("xgb_pred_done", False):
        pred_train_xgb = pd.read_parquet(paths["train_pred_xgb"])
        pred_val_xgb = pd.read_parquet(paths["val_pred_xgb"])
        pred_test_xgb = pd.read_parquet(paths["test_pred_xgb"])
        xgb_ckpt_ok = (
            len(pred_train_xgb) == len(data.train.df)
            and len(pred_val_xgb) == len(data.val.df)
            and len(pred_test_xgb) == len(data.test.df)
        )
        if xgb_ckpt_ok:
            _log("loaded XGBoost prediction checkpoints")
        else:
            _log("XGBoost prediction checkpoint size mismatch; regenerating predictions")
            pred_train_xgb = generate_predictions(xgb_model, data.train, data.feature_cols, "XGBoost")
            pred_val_xgb = generate_predictions(xgb_model, data.val, data.feature_cols, "XGBoost")
            pred_test_xgb = generate_predictions(xgb_model, data.test, data.feature_cols, "XGBoost")
            pred_train_xgb.to_parquet(paths["train_pred_xgb"], index=False)
            pred_val_xgb.to_parquet(paths["val_pred_xgb"], index=False)
            pred_test_xgb.to_parquet(paths["test_pred_xgb"], index=False)
    else:
        pred_train_xgb = generate_predictions(xgb_model, data.train, data.feature_cols, "XGBoost")
        pred_val_xgb = generate_predictions(xgb_model, data.val, data.feature_cols, "XGBoost")
        pred_test_xgb = generate_predictions(xgb_model, data.test, data.feature_cols, "XGBoost")
        pred_train_xgb.to_parquet(paths["train_pred_xgb"], index=False)
        pred_val_xgb.to_parquet(paths["val_pred_xgb"], index=False)
        pred_test_xgb.to_parquet(paths["test_pred_xgb"], index=False)
        _set_state_flag(state, "xgb_pred_done", True)
        _log("saved XGBoost prediction checkpoints")

    train_pred = pd.concat([pred_train_lgbm, pred_train_xgb], ignore_index=True)
    val_pred = pd.concat([pred_val_lgbm, pred_val_xgb], ignore_index=True)
    test_pred = pd.concat([pred_test_lgbm, pred_test_xgb], ignore_index=True)

    if not (paths["train_pred"].exists() and paths["val_pred"].exists() and paths["test_pred"].exists() and state.get("merged_pred_done", False)):
        save_predictions(train_pred, val_pred, test_pred)
        _set_state_flag(state, "merged_pred_done", True)
    else:
        _log("merged prediction outputs already exist; skipping overwrite")

    # Evaluation on val/test.
    k = int(TOP_K if TOP_K is not None else 10)

    eval_rows: List[Dict[str, Any]] = []

    for model_name, pred_val, pred_test in (
        ("LightGBM", pred_val_lgbm, pred_test_lgbm),
        ("XGBoost", pred_val_xgb, pred_test_xgb),
    ):
        val_y, val_sid, val_scores = _align_labels_to_predictions(pred_val, data.val)
        test_y, test_sid, test_scores = _align_labels_to_predictions(pred_test, data.test)

        val_metrics = evaluate_ranking(
            y_true=val_y,
            session_ids=val_sid,
            scores=val_scores,
            k=k,
        )
        test_metrics = evaluate_ranking(
            y_true=test_y,
            session_ids=test_sid,
            scores=test_scores,
            k=k,
        )

        row_val = {"model": model_name, "split": "val", **val_metrics}
        row_test = {"model": model_name, "split": "test", **test_metrics}
        eval_rows.extend([row_val, row_test])

        _log("%s val metrics: %s", model_name, json.dumps(row_val, default=float))
        _log("%s test metrics: %s", model_name, json.dumps(row_test, default=float))

    if not (paths["eval_csv"].exists() and paths["eval_json"].exists() and state.get("eval_done", False)):
        eval_df = pd.DataFrame(eval_rows)
        eval_df.to_csv(paths["eval_csv"], index=False)
        with open(paths["eval_json"], "w", encoding="utf-8") as f:
            json.dump(eval_rows, f, indent=2, default=float)
        _set_state_flag(state, "eval_done", True)
    else:
        _log("evaluation outputs already exist; skipping overwrite")

    # Feature importances.
    try:
        fi_lgbm = pd.DataFrame(
            {
                "feature": data.feature_cols,
                "importance_gain": lgbm_model.feature_importance(importance_type="gain"),
                "importance_split": lgbm_model.feature_importance(importance_type="split"),
            }
        ).sort_values("importance_gain", ascending=False)
        fi_lgbm.to_csv(paths["lgbm_fi"], index=False)
    except Exception as e:
        _log("failed to save LightGBM feature importance: %s", str(e))

    try:
        xgb_gain = xgb_model.get_booster().get_score(importance_type="gain")
        fi_xgb = pd.DataFrame(
            {
                "feature": data.feature_cols,
                "importance_gain": [
                    float(xgb_gain.get(col, xgb_gain.get(f"f{i}", 0.0)))
                    for i, col in enumerate(data.feature_cols)
                ],
            }
        ).sort_values("importance_gain", ascending=False)
        fi_xgb.to_csv(paths["xgb_fi"], index=False)
    except Exception as e:
        _log("failed to save XGBoost feature importance: %s", str(e))

    _set_state_flag(state, "run_completed", True)
    _log("checkpoint state: %s", _state_path())
    _log("evaluation csv: %s", paths["eval_csv"])
    _log("evaluation json: %s", paths["eval_json"])
    _log("runtime_sec: %.2f", time.perf_counter() - run_start)


if __name__ == "__main__":
    main()
