"""Production-grade neural pairwise ranker training pipeline for CSAO."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from pandas.api.types import CategoricalDtype
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, IterableDataset


# ============================================================
# Config
# ============================================================

try:
    from data_pipeline import config as cfg  # type: ignore
except Exception:
    try:
        import config as cfg  # type: ignore
    except Exception:
        cfg = object()  # type: ignore


def _cfg(name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def _env(name: str, default: Any) -> Any:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(default, int):
        try:
            return int(raw)
        except Exception:
            return default
    if isinstance(default, float):
        try:
            return float(raw)
        except Exception:
            return default
    return raw


@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    num_epochs: int
    top_k: int
    patience: int
    seed: int
    device: str
    hidden_dims: List[int]
    dropout: float
    use_batch_norm: bool
    mixed_precision: bool
    checkpoint_freq: int
    gradient_clip_norm: float
    pairwise_loss: str
    hard_negative_mining: bool
    parquet_batch_rows: int
    max_train_rows: int
    max_val_rows: int
    max_test_rows: int
    max_pairs_per_session: int
    cat_hash_dim: int
    embed_dim: int


def load_config() -> TrainConfig:
    """Load training configuration from project config with safe defaults."""
    seed = int(_env("NEURAL_SEED", int(_cfg("SEED", _cfg("RANDOM_SEED", 42)))))
    batch_size = int(_env("NEURAL_BATCH_SIZE", int(_cfg("BATCH_SIZE", 2048))))
    lr = float(_env("NEURAL_LR", float(_cfg("LR", _cfg("LEARNING_RATE", 1e-3)))))
    num_epochs = int(_env("NEURAL_NUM_EPOCHS", int(_cfg("NUM_EPOCHS", 20))))
    top_k = int(_env("NEURAL_TOP_K", int(_cfg("TOP_K", _cfg("TOP_K_EVAL", 10)))))
    patience = int(_env("NEURAL_PATIENCE", int(_cfg("PATIENCE", 6))))
    device = str(_env("NEURAL_DEVICE", str(_cfg("DEVICE", "auto"))))

    hidden_dims = list(_cfg("NEURAL_HIDDEN_DIMS", [256, 128, 64]))
    hidden_dims_env = os.getenv("NEURAL_HIDDEN_DIMS")
    if hidden_dims_env:
        try:
            hidden_dims = [int(x.strip()) for x in hidden_dims_env.split(",") if x.strip()]
        except Exception:
            pass
    dropout = float(_env("NEURAL_DROPOUT", float(_cfg("NEURAL_DROPOUT", 0.2))))
    use_batch_norm = bool(_env("NEURAL_BATCH_NORM", bool(_cfg("NEURAL_BATCH_NORM", True))))
    mixed_precision = bool(_env("NEURAL_MIXED_PRECISION", bool(_cfg("MIXED_PRECISION", True))))

    checkpoint_freq = int(_env("NEURAL_CHECKPOINT_FREQ", int(_cfg("CHECKPOINT_FREQ", 1))))
    gradient_clip_norm = float(_env("NEURAL_GRADIENT_CLIP_NORM", float(_cfg("GRADIENT_CLIP_NORM", 1.0))))
    pairwise_loss = str(_env("NEURAL_PAIRWISE_LOSS", str(_cfg("PAIRWISE_LOSS", "lambdarank")))).lower()
    hard_negative_mining = bool(_env("NEURAL_HARD_NEGATIVE_MINING", bool(_cfg("HARD_NEGATIVE_MINING", False))))

    parquet_batch_rows = int(_env("NEURAL_PARQUET_BATCH_ROWS", int(_cfg("PARQUET_BATCH_ROWS", 100_000))))
    max_train_rows = int(_cfg("MAX_TRAIN_ROWS", int(os.getenv("NEURAL_MAX_TRAIN_ROWS", "5000000"))))
    max_val_rows = int(_cfg("MAX_VAL_ROWS", int(os.getenv("NEURAL_MAX_VAL_ROWS", "1500000"))))
    max_test_rows = int(_cfg("MAX_TEST_ROWS", int(os.getenv("NEURAL_MAX_TEST_ROWS", "1500000"))))

    max_pairs_per_session = int(_env("NEURAL_MAX_PAIRS_PER_SESSION", int(_cfg("MAX_PAIRS_PER_SESSION", 128))))
    cat_hash_dim = int(_env("NEURAL_CAT_HASH_DIM", int(_cfg("CAT_HASH_DIM", 20000))))
    embed_dim = int(_env("NEURAL_EMBED_DIM", int(_cfg("EMBED_DIM", 16))))

    return TrainConfig(
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        top_k=top_k,
        patience=patience,
        seed=seed,
        device=device,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        mixed_precision=mixed_precision,
        checkpoint_freq=checkpoint_freq,
        gradient_clip_norm=gradient_clip_norm,
        pairwise_loss=pairwise_loss,
        hard_negative_mining=hard_negative_mining,
        parquet_batch_rows=parquet_batch_rows,
        max_train_rows=max_train_rows,
        max_val_rows=max_val_rows,
        max_test_rows=max_test_rows,
        max_pairs_per_session=max_pairs_per_session,
        cat_hash_dim=cat_hash_dim,
        embed_dim=embed_dim,
    )


def _get_logger() -> logging.Logger:
    try:
        logger = getattr(cfg, "LOGGER")
        if logger is not None:
            return logger
    except Exception:
        pass

    logger = logging.getLogger("train_neural_ranker")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(h)
        logger.propagate = False
    return logger


LOGGER = _get_logger()


def _log(msg: str, *args: Any) -> None:
    LOGGER.info(msg, *args)


# ============================================================
# Paths and feature metadata
# ============================================================

@dataclass
class FeatureSpec:
    numeric_cols: List[str]
    categorical_cols: List[str]
    session_col: str
    item_col: str
    label_col: str
    renames: Dict[str, str]
    neg_score_col: Optional[str]
    category_col: Optional[str]
    restaurant_col: Optional[str]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _paths() -> Dict[str, Path]:
    root = _project_root()
    featured = root / "data_pipeline" / "data" / "featured"
    processed = root / "data_pipeline" / "data" / "processed"
    model_dir = root / "models" / "neural_ranker"
    ckpt_dir = model_dir / "checkpoints"
    return {
        "featured_train": featured / "train_ranking_features.parquet",
        "featured_val": featured / "val_ranking_features.parquet",
        "featured_test": featured / "test_ranking_features.parquet",
        "processed_train": processed / "train.parquet",
        "processed_val": processed / "val.parquet",
        "processed_test": processed / "test.parquet",
        "model_dir": model_dir,
        "ckpt_dir": ckpt_dir,
        "best_model": model_dir / "best_model.pt",
        "latest_checkpoint": ckpt_dir / "latest_checkpoint.pt",
        "training_log": model_dir / "training_log.json",
    }


def _schema_names(path: Path) -> List[str]:
    return list(pq.ParquetFile(path).schema.names)


def _find_label_col(cols: Sequence[str]) -> str:
    if "label" in cols:
        return "label"
    if "label_added" in cols:
        return "label_added"
    raise ValueError("No label column found")


def _has_positive(path: Path, label_col: str, scan_rows: int = 1_000_000) -> bool:
    pf = pq.ParquetFile(path)
    seen = 0
    for rb in pf.iter_batches(batch_size=min(250_000, scan_rows), columns=[label_col]):
        b = rb.to_pandas()
        if (b[label_col] == 1).any():
            return True
        seen += len(b)
        if seen >= scan_rows:
            break
    return False


def _has_both_classes(path: Path, label_col: str) -> bool:
    pf = pq.ParquetFile(path)
    has_pos = False
    has_neg = False
    for rb in pf.iter_batches(batch_size=250_000, columns=[label_col]):
        b = rb.to_pandas()
        if not has_pos and (b[label_col] == 1).any():
            has_pos = True
        if not has_neg and (b[label_col] == 0).any():
            has_neg = True
        if has_pos and has_neg:
            return True
    return False


def _resolve_data_paths() -> Tuple[Path, Path, Path, str]:
    p = _paths()
    ftrain, fval, ftest = p["featured_train"], p["featured_val"], p["featured_test"]
    for q in (ftrain, fval, ftest):
        if not q.exists():
            raise FileNotFoundError(f"Missing dataset: {q}")

    ltrain = _find_label_col(_schema_names(ftrain))
    lval = _find_label_col(_schema_names(fval))
    ltest = _find_label_col(_schema_names(ftest))
    if _has_both_classes(ftrain, ltrain) and _has_both_classes(fval, lval) and _has_both_classes(ftest, ltest):
        return ftrain, fval, ftest, "featured"

    ptrain, pval, ptest = p["processed_train"], p["processed_val"], p["processed_test"]
    for q in (ptrain, pval, ptest):
        if not q.exists():
            raise FileNotFoundError(f"Featured labels are single-class and processed fallback is missing: {q}")
    return ptrain, pval, ptest, "processed"


def _infer_feature_spec(train_path: Path, val_path: Path, test_path: Path) -> FeatureSpec:
    cols = sorted(set(_schema_names(train_path)).intersection(_schema_names(val_path)).intersection(_schema_names(test_path)))

    renames: Dict[str, str] = {}
    item_col = "item_id" if "item_id" in cols else "candidate_item_id"
    if item_col == "candidate_item_id":
        renames[item_col] = "item_id"

    label_col = _find_label_col(cols)
    if label_col != "label":
        renames[label_col] = "label"

    session_col = "session_id"

    drop_cols = {
        "session_id",
        "item_id",
        "candidate_item_id",
        "label",
        "label_added",
        "timestamp",
        "split",
        "rank",
        "score",
        "model",
    }
    candidate_cols = [c for c in cols if c not in drop_cols]

    sample_pf = pq.ParquetFile(train_path)
    sample_rb = next(sample_pf.iter_batches(batch_size=5000, columns=[c for c in candidate_cols if c in sample_pf.schema.names]))
    sample_df = sample_rb.to_pandas()

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for c in candidate_cols:
        dt = sample_df[c].dtype if c in sample_df.columns else np.dtype("float64")
        if (
            pd.api.types.is_object_dtype(dt)
            or pd.api.types.is_string_dtype(dt)
            or isinstance(dt, CategoricalDtype)
            or pd.api.types.is_bool_dtype(dt)
            or c.endswith("_category")
            or c.endswith("_segment")
            or c in {"meal_time", "candidate_category", "user_segment"}
        ):
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)

    neg_score_col = None
    for c in ["cf_score", "similarity_score", "final_rank", "candidate_popularity"]:
        if c in candidate_cols:
            neg_score_col = c
            break

    category_col = "candidate_category" if "candidate_category" in candidate_cols else None
    restaurant_col = "restaurant_id" if "restaurant_id" in candidate_cols else None

    return FeatureSpec(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        session_col=session_col,
        item_col=item_col,
        label_col=label_col,
        renames=renames,
        neg_score_col=neg_score_col,
        category_col=category_col,
        restaurant_col=restaurant_col,
    )


# ============================================================
# Determinism
# ============================================================

def set_seed(seed: int) -> None:
    """Set deterministic random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset and loader
# ============================================================

class StreamingParquetDataset(IterableDataset):
    """Memory-efficient iterable dataset from parquet batches."""

    def __init__(
        self,
        path: Path,
        feature_spec: FeatureSpec,
        batch_size: int,
        parquet_batch_rows: int,
        max_rows: int,
        seed: int,
        shuffle_chunks: bool,
        cat_hash_dim: int,
        balance_labels: bool = False,
        require_mixed_sessions: bool = False,
    ) -> None:
        super().__init__()
        self.path = path
        self.feature_spec = feature_spec
        self.batch_size = batch_size
        self.parquet_batch_rows = parquet_batch_rows
        self.max_rows = max_rows
        self.seed = seed
        self.shuffle_chunks = shuffle_chunks
        self.cat_hash_dim = cat_hash_dim
        self.balance_labels = balance_labels
        self.require_mixed_sessions = require_mixed_sessions

    @staticmethod
    def _hash_series(col: pd.Series, mod: int) -> np.ndarray:
        h = pd.util.hash_pandas_object(col.astype("string"), index=False).to_numpy(dtype=np.uint64)
        return (h % np.uint64(mod)).astype(np.int64)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_spec.renames:
            df = df.rename(columns=self.feature_spec.renames)

        needed = [
            "session_id",
            "item_id",
            "label",
            *self.feature_spec.numeric_cols,
            *self.feature_spec.categorical_cols,
        ]
        if self.feature_spec.neg_score_col is not None:
            needed.append(self.feature_spec.neg_score_col)
        if self.feature_spec.category_col is not None:
            needed.append(self.feature_spec.category_col)
        if self.feature_spec.restaurant_col is not None:
            needed.append(self.feature_spec.restaurant_col)

        # Deduplicate requested columns while preserving order to avoid duplicate-column frames.
        needed = list(dict.fromkeys(needed))
        present = [c for c in needed if c in df.columns]
        out = df[present].copy()
        if out.columns.duplicated().any():
            out = out.loc[:, ~out.columns.duplicated()].copy()

        if "session_id" not in out.columns:
            out["session_id"] = 0
        if "item_id" not in out.columns:
            out["item_id"] = 0
        if "label" not in out.columns:
            out["label"] = 0

        out["session_id"] = pd.to_numeric(_as_series(out["session_id"]), errors="coerce").fillna(0).astype(np.int64)
        out["item_id"] = pd.to_numeric(_as_series(out["item_id"]), errors="coerce").fillna(0).astype(np.int64)
        out["label"] = pd.to_numeric(_as_series(out["label"]), errors="coerce").fillna(0).astype(np.float32)

        for c in self.feature_spec.numeric_cols:
            if c not in out.columns:
                out[c] = 0.0
            out[c] = pd.to_numeric(_as_series(out[c]), errors="coerce").fillna(0.0).astype(np.float32)

        for c in self.feature_spec.categorical_cols:
            if c not in out.columns:
                out[c] = "__MISSING__"
            out[c] = _as_series(out[c]).astype("string").fillna("__MISSING__")

        if self.feature_spec.neg_score_col is not None:
            c = self.feature_spec.neg_score_col
            if c not in out.columns:
                out[c] = 0.0
            out[c] = pd.to_numeric(_as_series(out[c]), errors="coerce").fillna(0.0).astype(np.float32)

        if self.feature_spec.category_col is not None and self.feature_spec.category_col in out.columns:
            out[self.feature_spec.category_col] = _as_series(out[self.feature_spec.category_col]).astype("string").fillna("__MISSING__")

        if self.feature_spec.restaurant_col is not None and self.feature_spec.restaurant_col in out.columns:
            out[self.feature_spec.restaurant_col] = pd.to_numeric(
                _as_series(out[self.feature_spec.restaurant_col]), errors="coerce"
            ).fillna(-1).astype(np.int64)

        return out

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        pf = pq.ParquetFile(self.path)
        cols_needed = set([self.feature_spec.session_col, self.feature_spec.item_col, self.feature_spec.label_col])
        cols_needed.update(self.feature_spec.numeric_cols)
        cols_needed.update(self.feature_spec.categorical_cols)
        if self.feature_spec.neg_score_col is not None:
            cols_needed.add(self.feature_spec.neg_score_col)
        if self.feature_spec.category_col is not None:
            cols_needed.add(self.feature_spec.category_col)
        if self.feature_spec.restaurant_col is not None:
            cols_needed.add(self.feature_spec.restaurant_col)

        cols = [c for c in pf.schema.names if c in cols_needed]

        use_spread = str(_env("NEURAL_SPREAD_SAMPLE", True)).lower() not in {"0", "false", "no"}
        session_aware = str(_env("NEURAL_SESSION_AWARE_SAMPLE", True)).lower() not in {"0", "false", "no"}
        chosen_sessions: Optional[set[int]] = None

        if self.max_rows > 0 and use_spread and session_aware and pf.metadata.num_row_groups > 0:
            total_rows = int(pf.metadata.num_rows or 0)
            nrg = pf.metadata.num_row_groups
            probe_rows = max(self.max_rows * 2, 200_000)
            avg_rows = max(total_rows // max(nrg, 1), 1)
            probe_rgs = max(int(np.ceil(probe_rows / avg_rows)), 1)
            probe_rgs = min(probe_rgs + 2, nrg)
            rg_probe_idx = np.unique(np.linspace(0, nrg - 1, num=probe_rgs, dtype=int))

            sid_parts: List[np.ndarray] = []
            for i in rg_probe_idx:
                t = pf.read_row_group(int(i), columns=["session_id"])
                s = t.column("session_id").to_numpy(zero_copy_only=False)
                sid_parts.append(np.asarray(s, dtype=np.int64))
            if sid_parts:
                sid_all = np.unique(np.concatenate(sid_parts))
                # Estimate sessions needed from avg items/session observed in probe.
                approx_items_per_sess = max(float(probe_rows) / max(float(len(sid_all)), 1.0), 1.0)
                target_sessions = max(int(self.max_rows / approx_items_per_sess), 1)
                rng_s = np.random.default_rng(self.seed + 1337)
                if len(sid_all) > target_sessions:
                    sid_take = rng_s.choice(sid_all, size=target_sessions, replace=False)
                else:
                    sid_take = sid_all
                chosen_sessions = set(int(x) for x in sid_take.tolist())

        batch_stream: Iterator[Any]
        if self.max_rows > 0 and use_spread and not session_aware and pf.metadata.num_row_groups > 0 and int(pf.metadata.num_rows or 0) > self.max_rows:
            total_rows = int(pf.metadata.num_rows or 0)
            nrg = pf.metadata.num_row_groups
            avg_rows = max(total_rows // nrg, 1)
            target_rgs = max(int(np.ceil(self.max_rows / avg_rows)), 1)
            target_rgs = min(target_rgs + 2, nrg)
            rg_idx = np.unique(np.linspace(0, nrg - 1, num=target_rgs, dtype=int))

            def _rg_batches() -> Iterator[Any]:
                for i in rg_idx:
                    t = pf.read_row_group(int(i), columns=cols)
                    for rb2 in t.to_batches(max_chunksize=self.parquet_batch_rows):
                        yield rb2

            batch_stream = _rg_batches()
        else:
            batch_stream = pf.iter_batches(batch_size=self.parquet_batch_rows, columns=cols)
        rows_seen = 0
        pos_seen = 0
        neg_seen = 0
        target_pos = self.max_rows // 2 if self.max_rows > 0 else 0
        target_neg = self.max_rows - target_pos if self.max_rows > 0 else 0
        rng = np.random.default_rng(self.seed)

        for rb in batch_stream:
            df = rb.to_pandas()
            if df.empty:
                continue

            if chosen_sessions is not None:
                df = df[df["session_id"].isin(chosen_sessions)]
                if df.empty:
                    continue

            if self.max_rows > 0 and not self.balance_labels:
                remaining = self.max_rows - rows_seen
                if remaining <= 0:
                    break
                if len(df) > remaining:
                    df = df.iloc[:remaining].copy()

            df = self._transform(df)

            if self.require_mixed_sessions and not df.empty:
                sess_stats = df.groupby("session_id")["label"].agg(["sum", "count"])
                keep_sessions = sess_stats[(sess_stats["sum"] > 0) & (sess_stats["sum"] < sess_stats["count"])].index
                if len(keep_sessions) == 0:
                    continue
                df = df[df["session_id"].isin(keep_sessions)].reset_index(drop=True)
                if df.empty:
                    continue

            if self.max_rows > 0 and self.balance_labels:
                pos_df = df[df["label"] > 0.5]
                neg_df = df[df["label"] <= 0.5]

                take_parts: List[pd.DataFrame] = []
                if target_pos > pos_seen and not pos_df.empty:
                    need_pos = target_pos - pos_seen
                    take_pos = pos_df.iloc[:need_pos].copy() if len(pos_df) > need_pos else pos_df
                    take_parts.append(take_pos)
                    pos_seen += len(take_pos)

                if target_neg > neg_seen and not neg_df.empty:
                    need_neg = target_neg - neg_seen
                    take_neg = neg_df.iloc[:need_neg].copy() if len(neg_df) > need_neg else neg_df
                    take_parts.append(take_neg)
                    neg_seen += len(take_neg)

                if not take_parts:
                    if pos_seen >= target_pos and neg_seen >= target_neg:
                        break
                    continue
                df = pd.concat(take_parts, ignore_index=True)
                rows_seen = pos_seen + neg_seen
            else:
                rows_seen += len(df)

            # Keep sessions intact inside batches to preserve pairwise positives/negatives.
            df = df.sort_values("session_id", kind="mergesort").reset_index(drop=True)
            sess = df["session_id"].to_numpy(dtype=np.int64, copy=False)
            if len(sess) == 0:
                continue
            boundaries = np.flatnonzero(sess[1:] != sess[:-1]) + 1
            starts = np.concatenate(([0], boundaries, [len(df)]))

            group_ids = np.arange(len(starts) - 1)
            if self.shuffle_chunks:
                rng.shuffle(group_ids)

            batch_parts: List[pd.DataFrame] = []
            batch_rows = 0

            def _emit_batch(batch_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
                num_np = np.array(batch_df[self.feature_spec.numeric_cols].to_numpy(dtype=np.float32, copy=False), copy=True)
                if self.feature_spec.categorical_cols:
                    cat_arrs = [self._hash_series(batch_df[c], self.cat_hash_dim) for c in self.feature_spec.categorical_cols]
                    cat_np = np.array(np.stack(cat_arrs, axis=1).astype(np.int64, copy=False), copy=True)
                else:
                    cat_np = np.zeros((len(batch_df), 0), dtype=np.int64)

                label_np = np.array(batch_df["label"].to_numpy(dtype=np.float32, copy=False), copy=True)
                sid_np = np.array(batch_df["session_id"].to_numpy(dtype=np.int64, copy=False), copy=True)
                iid_np = np.array(batch_df["item_id"].to_numpy(dtype=np.int64, copy=False), copy=True)

                if self.feature_spec.neg_score_col is not None and self.feature_spec.neg_score_col in batch_df.columns:
                    neg_score_np = np.array(batch_df[self.feature_spec.neg_score_col].to_numpy(dtype=np.float32, copy=False), copy=True)
                else:
                    neg_score_np = np.zeros(len(batch_df), dtype=np.float32)

                if self.feature_spec.category_col is not None and self.feature_spec.category_col in batch_df.columns:
                    cat_pref_np = self._hash_series(batch_df[self.feature_spec.category_col], self.cat_hash_dim)
                else:
                    cat_pref_np = np.full(len(batch_df), -1, dtype=np.int64)

                if self.feature_spec.restaurant_col is not None and self.feature_spec.restaurant_col in batch_df.columns:
                    rest_np = np.array(batch_df[self.feature_spec.restaurant_col].to_numpy(dtype=np.int64, copy=False), copy=True)
                else:
                    rest_np = np.full(len(batch_df), -1, dtype=np.int64)

                return {
                    "num": torch.from_numpy(num_np),
                    "cat": torch.from_numpy(cat_np),
                    "label": torch.from_numpy(label_np),
                    "session_id": torch.from_numpy(sid_np),
                    "item_id": torch.from_numpy(iid_np),
                    "neg_score": torch.from_numpy(neg_score_np),
                    "category_id": torch.from_numpy(cat_pref_np),
                    "restaurant_id": torch.from_numpy(rest_np),
                }

            for gid in group_ids:
                st, ed = int(starts[gid]), int(starts[gid + 1])
                gdf = df.iloc[st:ed]
                glen = len(gdf)

                if batch_rows > 0 and batch_rows + glen > self.batch_size:
                    batch_df = pd.concat(batch_parts, ignore_index=True)
                    yield _emit_batch(batch_df)
                    batch_parts = []
                    batch_rows = 0

                batch_parts.append(gdf)
                batch_rows += glen

            if batch_parts:
                batch_df = pd.concat(batch_parts, ignore_index=True)
                yield _emit_batch(batch_df)

            if self.max_rows > 0 and rows_seen >= self.max_rows:
                break


def load_datasets(config: TrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader, FeatureSpec, str]:
    """Load train/val/test datasets as streaming DataLoaders."""
    train_path, val_path, test_path, source = _resolve_data_paths()
    spec = _infer_feature_spec(train_path, val_path, test_path)
    train_max = config.max_train_rows
    val_max = config.max_val_rows
    test_max = config.max_test_rows

    train_ds = StreamingParquetDataset(
        path=train_path,
        feature_spec=spec,
        batch_size=config.batch_size,
        parquet_batch_rows=config.parquet_batch_rows,
        max_rows=train_max,
        seed=config.seed,
        shuffle_chunks=True,
        cat_hash_dim=config.cat_hash_dim,
        balance_labels=False,
        require_mixed_sessions=False,
    )
    val_ds = StreamingParquetDataset(
        path=val_path,
        feature_spec=spec,
        batch_size=config.batch_size,
        parquet_batch_rows=config.parquet_batch_rows,
        max_rows=val_max,
        seed=config.seed,
        shuffle_chunks=False,
        cat_hash_dim=config.cat_hash_dim,
        balance_labels=False,
        require_mixed_sessions=False,
    )
    test_ds = StreamingParquetDataset(
        path=test_path,
        feature_spec=spec,
        batch_size=config.batch_size,
        parquet_batch_rows=config.parquet_batch_rows,
        max_rows=test_max,
        seed=config.seed,
        shuffle_chunks=False,
        cat_hash_dim=config.cat_hash_dim,
        balance_labels=False,
        require_mixed_sessions=False,
    )

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=None, num_workers=0)

    return train_loader, val_loader, test_loader, spec, source


# ============================================================
# Model
# ============================================================

class MLPNeuralRanker(nn.Module):
    """Configurable MLP-based ranker."""

    def __init__(
        self,
        num_numeric: int,
        num_categorical: int,
        cat_hash_dim: int,
        embed_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        self.num_categorical = num_categorical

        if num_categorical > 0:
            self.embeddings = nn.ModuleList([nn.Embedding(cat_hash_dim, embed_dim) for _ in range(num_categorical)])
            emb_out_dim = num_categorical * embed_dim
        else:
            self.embeddings = nn.ModuleList()
            emb_out_dim = 0

        input_dim = num_numeric + emb_out_dim
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h

        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)

    def forward(self, num_x: torch.Tensor, cat_x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = [num_x]
        if self.num_categorical > 0 and cat_x.numel() > 0:
            emb_list = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
            feats.append(torch.cat(emb_list, dim=1))

        x = torch.cat(feats, dim=1)
        h = self.backbone(x)
        return self.out(h).squeeze(1)


def build_model(config: TrainConfig, feature_spec: FeatureSpec, device: str) -> nn.Module:
    """Build neural ranker model."""
    model = MLPNeuralRanker(
        num_numeric=len(feature_spec.numeric_cols),
        num_categorical=len(feature_spec.categorical_cols),
        cat_hash_dim=config.cat_hash_dim,
        embed_dim=config.embed_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        use_batch_norm=config.use_batch_norm,
    )
    return model.to(device)


# ============================================================
# Pairwise loss and sampling
# ============================================================

def pairwise_sampler(
    logits: torch.Tensor,
    labels: torch.Tensor,
    session_ids: torch.Tensor,
    neg_scores: Optional[torch.Tensor] = None,
    category_ids: Optional[torch.Tensor] = None,
    restaurant_ids: Optional[torch.Tensor] = None,
    hard_negative: bool = False,
    max_pairs_per_session: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate positive-negative logit pairs within each session."""
    uniq_sessions = torch.unique(session_ids)

    pos_all: List[torch.Tensor] = []
    neg_all: List[torch.Tensor] = []

    for sid in uniq_sessions:
        idx = torch.nonzero(session_ids == sid, as_tuple=False).squeeze(1)
        if idx.numel() < 2:
            continue

        lbl = labels[idx]
        pos_idx = idx[lbl > 0.5]
        neg_idx = idx[lbl <= 0.5]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue

        if hard_negative and neg_scores is not None:
            ns = neg_scores[neg_idx]
            order = torch.argsort(ns, descending=True)
            neg_idx = neg_idx[order]

        max_neg = min(neg_idx.numel(), 4)
        if max_neg == 0:
            continue
        sel_neg = neg_idx[:max_neg]

        pos_rep = pos_idx.repeat_interleave(max_neg)
        neg_rep = sel_neg.repeat(pos_idx.numel())

        if hard_negative and category_ids is not None and restaurant_ids is not None:
            cat_match = category_ids[pos_rep] == category_ids[neg_rep]
            rest_match = restaurant_ids[pos_rep] == restaurant_ids[neg_rep]
            keep = cat_match | rest_match
            if keep.any():
                pos_rep = pos_rep[keep]
                neg_rep = neg_rep[keep]

        if pos_rep.numel() == 0:
            continue

        if pos_rep.numel() > max_pairs_per_session:
            pos_rep = pos_rep[:max_pairs_per_session]
            neg_rep = neg_rep[:max_pairs_per_session]

        pos_all.append(logits[pos_rep])
        neg_all.append(logits[neg_rep])

    if not pos_all:
        empty = torch.empty(0, device=logits.device, dtype=logits.dtype)
        return empty, empty

    return torch.cat(pos_all), torch.cat(neg_all)


def compute_pairwise_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Compute pairwise ranking loss (LambdaRank-style or BPR)."""
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return torch.tensor(0.0, device=pos_scores.device if pos_scores.numel() else neg_scores.device, dtype=torch.float32)

    diff = pos_scores - neg_scores
    if loss_type == "bpr":
        return -torch.log(torch.sigmoid(diff) + 1e-12).mean()

    return torch.nn.functional.softplus(-diff).mean()


# ============================================================
# Training and evaluation
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    config: TrainConfig,
    device: str,
) -> float:
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    steps = 0

    for i, batch in enumerate(loader, start=1):
        num = batch["num"].to(device, non_blocking=True)
        cat = batch["cat"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        session_ids = batch["session_id"].to(device, non_blocking=True)
        neg_scores = batch["neg_score"].to(device, non_blocking=True)
        category_ids = batch["category_id"].to(device, non_blocking=True)
        restaurant_ids = batch["restaurant_id"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and config.mixed_precision and device.startswith("cuda")

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(num, cat)
                p, n = pairwise_sampler(
                    logits=logits,
                    labels=labels,
                    session_ids=session_ids,
                    neg_scores=neg_scores,
                    category_ids=category_ids,
                    restaurant_ids=restaurant_ids,
                    hard_negative=config.hard_negative_mining,
                    max_pairs_per_session=config.max_pairs_per_session,
                )
                if p.numel() == 0 or n.numel() == 0:
                    # Fallback for batches lacking valid within-session pairs.
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                else:
                    loss = compute_pairwise_loss(p, n, config.pairwise_loss)

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
        else:
            logits = model(num, cat)
            p, n = pairwise_sampler(
                logits=logits,
                labels=labels,
                session_ids=session_ids,
                neg_scores=neg_scores,
                category_ids=category_ids,
                restaurant_ids=restaurant_ids,
                hard_negative=config.hard_negative_mining,
                max_pairs_per_session=config.max_pairs_per_session,
            )
            if p.numel() == 0 or n.numel() == 0:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            else:
                loss = compute_pairwise_loss(p, n, config.pairwise_loss)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()

        if torch.isfinite(loss):
            total_loss += float(loss.detach().item())
            steps += 1

        if i % 200 == 0:
            _log("train progress batch=%d avg_loss=%.6f", i, total_loss / max(steps, 1))

    if steps == 0:
        raise RuntimeError("No optimizer steps executed in this epoch (no valid pos-neg pairs found).")
    return total_loss / float(steps)


def compute_metrics(pred_df: pd.DataFrame, top_k: int) -> Dict[str, float]:
    """Compute session-grouped ranking metrics."""
    if pred_df.empty:
        return {
            f"NDCG@{top_k}": 0.0,
            "MAP": 0.0,
            "MRR": 0.0,
            "AUC": float("nan"),
            f"Precision@{top_k}": 0.0,
            f"Recall@{top_k}": 0.0,
        }

    y_true = pred_df["label"].to_numpy(dtype=np.int32)
    scores = pred_df["score"].to_numpy(dtype=np.float32)
    sid = pred_df["session_id"].to_numpy(dtype=np.int64)
    iid = pred_df["item_id"].to_numpy(dtype=np.int64)

    order = np.lexsort((iid, -scores, sid))
    y = y_true[order]
    s = scores[order]
    sid_sorted = sid[order]

    _, starts, counts = np.unique(sid_sorted, return_index=True, return_counts=True)

    ndcgs: List[float] = []
    aps: List[float] = []
    mrrs: List[float] = []
    pks: List[float] = []
    rks: List[float] = []
    eligible_sessions = 0

    for st, ct in zip(starts, counts):
        rel = y[st:st + ct]
        total_rel = int(rel.sum())
        # Skip degenerate sessions that cannot provide ranking signal.
        if total_rel == 0 or total_rel == ct:
            continue
        eligible_sessions += 1
        cutoff = int(min(top_k, rel.size))
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

    metrics = {
        f"NDCG@{top_k}": float(np.mean(ndcgs) if ndcgs else 0.0),
        "MAP": float(np.mean(aps) if aps else 0.0),
        "MRR": float(np.mean(mrrs) if mrrs else 0.0),
        f"Precision@{top_k}": float(np.mean(pks) if pks else 0.0),
        f"Recall@{top_k}": float(np.mean(rks) if rks else 0.0),
        "eligible_sessions": float(eligible_sessions),
    }

    metrics["AUC"] = float(roc_auc_score(y_true, s)) if np.unique(y_true).size > 1 else float("nan")
    return metrics


def evaluate_model(model: nn.Module, loader: DataLoader, device: str, top_k: int) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Run inference and compute ranking metrics."""
    model.eval()
    parts: List[pd.DataFrame] = []

    with torch.no_grad():
        for batch in loader:
            num = batch["num"].to(device, non_blocking=True)
            cat = batch["cat"].to(device, non_blocking=True)
            logits = model(num, cat)

            part = pd.DataFrame(
                {
                    "session_id": batch["session_id"].cpu().numpy().astype(np.int64),
                    "item_id": batch["item_id"].cpu().numpy().astype(np.int64),
                    "label": batch["label"].cpu().numpy().astype(np.int8),
                    "score": logits.cpu().numpy().astype(np.float32),
                }
            )
            parts.append(part)

    pred_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["session_id", "item_id", "label", "score"])
    pred_df.sort_values(["session_id", "score", "item_id"], ascending=[True, False, True], kind="mergesort", inplace=True)
    metrics = compute_metrics(pred_df, top_k=top_k)
    return metrics, pred_df


# ============================================================
# Checkpoint management
# ============================================================

def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _as_series(obj: Any) -> pd.Series:
    """Convert a pandas object selection to a guaranteed 1-D Series."""
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return pd.Series(obj)


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    best_metric: float,
    config: TrainConfig,
    feature_spec: FeatureSpec,
    source: str,
    training_state: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save full checkpoint and update latest checkpoint pointer."""
    p = _paths()
    p["ckpt_dir"].mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "config": asdict(config),
        "feature_spec": asdict(feature_spec),
        "source": source,
        "rng_state": _rng_state(),
        "training_state": training_state or {},
    }

    path = p["ckpt_dir"] / f"epoch_{epoch}.pt"
    torch.save(ckpt, path)
    shutil.copy2(path, p["latest_checkpoint"])
    return path


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str,
) -> Tuple[int, float, int]:
    """Load latest checkpoint if available and return next epoch + best metric."""
    p = _paths()
    latest = p["latest_checkpoint"]
    if not latest.exists():
        return 0, float("-inf"), 0

    try:
        # PyTorch >=2.6 defaults to weights_only=True, but we store optimizer/scheduler/RNG states.
        ckpt = torch.load(latest, map_location=device, weights_only=False)
    except Exception as e:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        bad_path = latest.with_suffix(f".corrupt_{ts}.pt")
        try:
            shutil.move(str(latest), str(bad_path))
            _log("Unreadable latest checkpoint moved to %s", bad_path)
        except Exception:
            _log("Unreadable latest checkpoint could not be moved: %s", latest)
        _log("Checkpoint load failed, starting fresh training: %s", str(e))
        return 0, float("-inf"), 0

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    _set_rng_state(ckpt.get("rng_state", {}))

    next_epoch = int(ckpt.get("epoch", 0)) + 1
    best_metric = float(ckpt.get("best_metric", float("-inf")))
    training_state = ckpt.get("training_state", {}) or {}
    patience_counter = int(training_state.get("patience_counter", 0))
    _log(
        "Resumed training from %s (next_epoch=%d, best_ndcg=%.6f, patience_counter=%d)",
        latest,
        next_epoch,
        best_metric,
        patience_counter,
    )
    return next_epoch, best_metric, patience_counter


def save_best_model(model: nn.Module, config: TrainConfig, feature_spec: FeatureSpec, source: str, best_metric: float) -> None:
    """Save best model artifact."""
    p = _paths()
    p["model_dir"].mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "feature_spec": asdict(feature_spec),
        "source": source,
        "best_metric": best_metric,
    }
    torch.save(payload, p["best_model"])


# ============================================================
# Training loop
# ============================================================

def _resolve_device(config: TrainConfig) -> str:
    d = config.device.lower()
    if d == "cpu":
        return "cpu"
    if d.startswith("cuda") and torch.cuda.is_available():
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_training_log() -> List[Dict[str, Any]]:
    p = _paths()["training_log"]
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _save_training_log(logs: List[Dict[str, Any]]) -> None:
    p = _paths()["training_log"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(logs, indent=2, default=float))


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    config: TrainConfig,
    feature_spec: FeatureSpec,
    source: str,
    device: str,
) -> Dict[str, Any]:
    """Main training loop with checkpointing, resume, metrics, and early stopping."""
    scaler: Optional[torch.cuda.amp.GradScaler]
    if config.mixed_precision and device.startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    start_epoch, best_metric, patience_counter = load_checkpoint(model, optimizer, scheduler, device=device)
    logs = _load_training_log()
    runtime_start = time.perf_counter()
    last_epoch = max(start_epoch - 1, 0)

    try:
        for epoch in range(start_epoch, config.num_epochs):
            last_epoch = epoch
            epoch_start = time.perf_counter()

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                device=device,
            )

            val_metrics, _ = evaluate_model(model, val_loader, device=device, top_k=config.top_k)
            test_metrics, _ = evaluate_model(model, test_loader, device=device, top_k=config.top_k)

            val_ndcg = float(val_metrics.get(f"NDCG@{config.top_k}", 0.0))
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_ndcg)
                else:
                    scheduler.step()

            improved = val_ndcg > best_metric
            if improved:
                best_metric = val_ndcg
                patience_counter = 0
                save_best_model(model, config, feature_spec, source, best_metric)
            else:
                patience_counter += 1

            lr = float(optimizer.param_groups[0]["lr"])
            epoch_time = float(time.perf_counter() - epoch_start)

            log_row: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "learning_rate": lr,
                "epoch_time_sec": epoch_time,
                "best_metric": float(best_metric),
                "improved": bool(improved),
            }
            logs.append(log_row)
            _save_training_log(logs)

            ckpt_path = save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=best_metric,
                config=config,
                feature_spec=feature_spec,
                source=source,
                training_state={"patience_counter": patience_counter, "logs_len": len(logs)},
            )

            _log(
                "epoch=%d train_loss=%.6f val_ndcg@%d=%.6f val_map=%.6f val_mrr=%.6f val_auc=%s lr=%.6g time_sec=%.2f checkpoint=%s",
                epoch,
                train_loss,
                config.top_k,
                val_ndcg,
                float(val_metrics.get("MAP", 0.0)),
                float(val_metrics.get("MRR", 0.0)),
                str(val_metrics.get("AUC", float("nan"))),
                lr,
                epoch_time,
                ckpt_path,
            )

            if patience_counter >= config.patience:
                _log("Early stopping triggered (patience=%d)", config.patience)
                break
    except KeyboardInterrupt:
        _log("Training interrupted. Saving recovery checkpoint at epoch=%d", last_epoch)
        save_checkpoint(
            epoch=last_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_metric,
            config=config,
            feature_spec=feature_spec,
            source=source,
            training_state={"patience_counter": patience_counter, "logs_len": len(logs)},
        )
        _save_training_log(logs)
        raise
    except Exception as e:
        _log("Training loop crashed. Saving recovery checkpoint at epoch=%d err=%s", last_epoch, str(e))
        save_checkpoint(
            epoch=last_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_metric,
            config=config,
            feature_spec=feature_spec,
            source=source,
            training_state={"patience_counter": patience_counter, "logs_len": len(logs), "error": str(e)},
        )
        _save_training_log(logs)
        raise

    total_runtime = float(time.perf_counter() - runtime_start)
    return {
        "best_metric": float(best_metric),
        "total_runtime_sec": total_runtime,
        "epochs_logged": len(logs),
    }


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    """Run full training pipeline."""
    config = load_config()
    set_seed(config.seed)

    p = _paths()
    p["model_dir"].mkdir(parents=True, exist_ok=True)
    p["ckpt_dir"].mkdir(parents=True, exist_ok=True)
    if _env("NEURAL_RESET_CHECKPOINT", False):
        for q in [p["latest_checkpoint"], p["best_model"], p["training_log"]]:
            try:
                if q.exists():
                    q.unlink()
            except Exception:
                pass
        for ck in p["ckpt_dir"].glob("epoch_*.pt"):
            try:
                ck.unlink()
            except Exception:
                pass
        _log("Checkpoint state reset via NEURAL_RESET_CHECKPOINT=1")

    device = _resolve_device(config)
    _log("Device: %s", device)

    train_loader, val_loader, test_loader, feature_spec, source = load_datasets(config)
    _log(
        "Feature count: numeric=%d categorical=%d source=%s",
        len(feature_spec.numeric_cols),
        len(feature_spec.categorical_cols),
        source,
    )

    model = build_model(config, feature_spec, device=device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    try:
        summary = train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            feature_spec=feature_spec,
            source=source,
            device=device,
        )
        _log("Training complete: %s", json.dumps(summary, default=float))

    except KeyboardInterrupt:
        _log("Training interrupted by user.")
        raise

    except Exception as e:
        _log("Training crashed: %s", str(e))
        raise


if __name__ == "__main__":
    main()
