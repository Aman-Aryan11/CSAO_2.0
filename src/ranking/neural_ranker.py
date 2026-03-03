"""Neural ranking training module for CSAO with robust checkpointing and resume support."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, IterableDataset

# ------------------------------------------------------------
# Config loading
# ------------------------------------------------------------
try:
    from data_pipeline.config import LOGGER as CFG_LOGGER  # type: ignore
except Exception:
    CFG_LOGGER = None

try:
    from data_pipeline import config as cfg  # type: ignore
except Exception:
    try:
        import config as cfg  # type: ignore
    except Exception:
        cfg = object()  # type: ignore


def _cfg(name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


RANDOM_SEED = int(_cfg("RANDOM_SEED", 42))
BATCH_SIZE = int(_cfg("BATCH_SIZE", 2048))
LEARNING_RATE = float(_cfg("LEARNING_RATE", 1e-3))
NUM_EPOCHS = int(_cfg("NUM_EPOCHS", 10))
CHECKPOINT_FREQ = int(_cfg("CHECKPOINT_FREQ", 1))
EARLY_STOPPING_PATIENCE = int(_cfg("EARLY_STOPPING_PATIENCE", 3))
TOP_K_EVAL = int(_cfg("TOP_K_EVAL", 10))
LOSS_TYPE = str(_cfg("NEURAL_LOSS", "bce")).lower()
WEIGHT_DECAY = float(_cfg("WEIGHT_DECAY", 1e-5))
HIDDEN_DIMS = list(_cfg("NEURAL_HIDDEN_DIMS", [256, 128, 64]))
DROPOUT = float(_cfg("NEURAL_DROPOUT", 0.2))
CHECKPOINT_SCAN_ROWS = int(_cfg("CHECKPOINT_SCAN_ROWS", 1_000_000))
MAX_TRAIN_ROWS = int(_cfg("MAX_TRAIN_ROWS", int(os.getenv("NEURAL_MAX_TRAIN_ROWS", "5000000"))))
MAX_VAL_ROWS = int(_cfg("MAX_VAL_ROWS", int(os.getenv("NEURAL_MAX_VAL_ROWS", "1500000"))))
MAX_TEST_ROWS = int(_cfg("MAX_TEST_ROWS", int(os.getenv("NEURAL_MAX_TEST_ROWS", "1500000"))))
PARQUET_BATCH_ROWS = int(_cfg("PARQUET_BATCH_ROWS", int(os.getenv("NEURAL_PARQUET_BATCH_ROWS", "100000"))))
CAT_HASH_DIM = int(_cfg("CAT_HASH_DIM", 20000))
EMBED_DIM = int(_cfg("EMBED_DIM", 16))
USE_MIXED_PRECISION = bool(_cfg("USE_MIXED_PRECISION", True))

if CFG_LOGGER is not None:
    LOGGER = CFG_LOGGER
else:
    LOGGER = logging.getLogger("neural_ranker")
    if not LOGGER.handlers:
        LOGGER.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        LOGGER.addHandler(h)
        LOGGER.propagate = False


def _log(msg: str, *args: Any) -> None:
    LOGGER.info(msg, *args)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _featured_dir() -> Path:
    return _project_root() / "data_pipeline" / "data" / "featured"


def _processed_dir() -> Path:
    return _project_root() / "data_pipeline" / "data" / "processed"


def _model_dir() -> Path:
    return _project_root() / "models" / "neural"


def _checkpoint_dir() -> Path:
    return _model_dir() / "checkpoints"


def _paths() -> Dict[str, Path]:
    fd = _featured_dir()
    pd_dir = _processed_dir()
    md = _model_dir()
    cd = _checkpoint_dir()
    return {
        "featured_train": fd / "train_ranking_features.parquet",
        "featured_val": fd / "val_ranking_features.parquet",
        "featured_test": fd / "test_ranking_features.parquet",
        "processed_train": pd_dir / "train.parquet",
        "processed_val": pd_dir / "val.parquet",
        "processed_test": pd_dir / "test.parquet",
        "checkpoint_dir": cd,
        "best_model": md / "best_model.pt",
        "final_model": md / "final_model.pt",
        "training_log": md / "training_log.json",
        "val_predictions": md / "val_predictions.parquet",
    }


# ------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------

@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    checkpoint_freq: int
    early_stopping_patience: int
    top_k_eval: int
    loss_type: str
    weight_decay: float
    hidden_dims: List[int]
    dropout: float
    cat_hash_dim: int
    embed_dim: int
    max_train_rows: int
    max_val_rows: int
    max_test_rows: int
    parquet_batch_rows: int
    use_mixed_precision: bool
    device: str


@dataclass
class FeatureSpec:
    numeric_cols: List[str]
    categorical_cols: List[str]
    session_col: str
    item_col: str
    label_col: str
    renames: Dict[str, str]


@dataclass
class DataPaths:
    train_path: Path
    val_path: Path
    test_path: Path
    source: str


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    cfg_dev = str(_cfg("DEVICE", "auto")).lower()
    if cfg_dev == "cpu":
        return "cpu"
    if cfg_dev.startswith("cuda") and torch.cuda.is_available():
        return cfg_dev
    return "cuda" if torch.cuda.is_available() else "cpu"


def _schema_names(path: Path) -> List[str]:
    return list(pq.ParquetFile(path).schema.names)


def _find_label_col(cols: Sequence[str]) -> str:
    if "label" in cols:
        return "label"
    if "label_added" in cols:
        return "label_added"
    raise ValueError("No label column found in dataset.")


def _positive_exists(path: Path, label_col: str, scan_rows: int) -> bool:
    pf = pq.ParquetFile(path)
    seen = 0
    for rb in pf.iter_batches(batch_size=min(250_000, scan_rows), columns=[label_col]):
        b = rb.to_pandas()
        seen += len(b)
        if (b[label_col] == 1).any():
            return True
        if seen >= scan_rows:
            break
    return False


def resolve_data_paths() -> DataPaths:
    p = _paths()

    for k in ("featured_train", "featured_val", "featured_test"):
        if not p[k].exists():
            raise FileNotFoundError(f"Missing dataset: {p[k]}")

    ftrain_cols = _schema_names(p["featured_train"])
    fval_cols = _schema_names(p["featured_val"])
    ftest_cols = _schema_names(p["featured_test"])

    flabel = _find_label_col(ftrain_cols)
    featured_has_pos = (
        _positive_exists(p["featured_train"], flabel, CHECKPOINT_SCAN_ROWS)
        or _positive_exists(p["featured_val"], _find_label_col(fval_cols), CHECKPOINT_SCAN_ROWS)
        or _positive_exists(p["featured_test"], _find_label_col(ftest_cols), CHECKPOINT_SCAN_ROWS)
    )

    if featured_has_pos:
        _log("Using featured splits for neural training")
        return DataPaths(
            train_path=p["featured_train"],
            val_path=p["featured_val"],
            test_path=p["featured_test"],
            source="featured",
        )

    for k in ("processed_train", "processed_val", "processed_test"):
        if not p[k].exists():
            raise FileNotFoundError(f"featured labels are single-class and processed fallback missing: {p[k]}")

    _log("Featured labels are single-class; using processed splits fallback")
    return DataPaths(
        train_path=p["processed_train"],
        val_path=p["processed_val"],
        test_path=p["processed_test"],
        source="processed",
    )


def infer_feature_spec(paths: DataPaths) -> FeatureSpec:
    cols_train = set(_schema_names(paths.train_path))
    cols_val = set(_schema_names(paths.val_path))
    cols_test = set(_schema_names(paths.test_path))
    shared = sorted(cols_train.intersection(cols_val).intersection(cols_test))

    renames: Dict[str, str] = {}
    item_col = "item_id"
    if "item_id" not in shared and "candidate_item_id" in shared:
        item_col = "candidate_item_id"
        renames["candidate_item_id"] = "item_id"

    label_col = _find_label_col(shared)
    if label_col != "label":
        renames[label_col] = "label"

    session_col = "session_id"
    if session_col not in shared:
        raise ValueError("session_id column missing from shared schema.")

    exclude = {
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

    cand_cols = [c for c in shared if c not in exclude]

    # infer from train sample
    sample_pf = pq.ParquetFile(paths.train_path)
    sample_rb = next(sample_pf.iter_batches(batch_size=5000, columns=[c for c in shared if c in sample_pf.schema.names]))
    sample_df = sample_rb.to_pandas()

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for c in cand_cols:
        dt = sample_df[c].dtype if c in sample_df.columns else np.dtype("float64")
        if (
            pd.api.types.is_object_dtype(dt)
            or pd.api.types.is_string_dtype(dt)
            or isinstance(dt, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dt)
            or c.endswith("_segment")
            or c.endswith("_category")
            or c in {"meal_time"}
        ):
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)

    _log("Feature spec: numeric=%d categorical=%d", len(numeric_cols), len(categorical_cols))
    return FeatureSpec(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        session_col=session_col,
        item_col=item_col,
        label_col=label_col,
        renames=renames,
    )


# ------------------------------------------------------------
# Dataset (streaming batches)
# ------------------------------------------------------------

class ParquetBatchDataset(IterableDataset):
    """Iterable dataset that streams parquet and yields tensor mini-batches."""

    def __init__(
        self,
        path: Path,
        feature_spec: FeatureSpec,
        batch_size: int,
        parquet_batch_rows: int,
        max_rows: int,
        seed: int,
        shuffle_batches: bool,
    ) -> None:
        super().__init__()
        self.path = path
        self.feature_spec = feature_spec
        self.batch_size = batch_size
        self.parquet_batch_rows = parquet_batch_rows
        self.max_rows = max_rows
        self.seed = seed
        self.shuffle_batches = shuffle_batches

    def _transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_spec.renames:
            df = df.rename(columns=self.feature_spec.renames)

        needed = [
            "session_id",
            "item_id",
            "label",
            *self.feature_spec.numeric_cols,
            *self.feature_spec.categorical_cols,
        ]
        present = [c for c in needed if c in df.columns]
        out = df[present].copy()

        if "label" not in out.columns:
            raise ValueError("Transformed batch missing label column.")
        if "session_id" not in out.columns:
            out["session_id"] = 0
        if "item_id" not in out.columns:
            out["item_id"] = 0

        for c in self.feature_spec.numeric_cols:
            if c not in out.columns:
                out[c] = 0.0
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(np.float32)

        for c in self.feature_spec.categorical_cols:
            if c not in out.columns:
                out[c] = "__MISSING__"
            out[c] = out[c].astype("string").fillna("__MISSING__")

        out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(np.float32)
        out["session_id"] = pd.to_numeric(out["session_id"], errors="coerce").fillna(0).astype(np.int64)
        out["item_id"] = pd.to_numeric(out["item_id"], errors="coerce").fillna(0).astype(np.int64)

        return out

    @staticmethod
    def _hash_cat(col: pd.Series, mod: int) -> np.ndarray:
        hashed = pd.util.hash_pandas_object(col, index=False).to_numpy(dtype=np.uint64)
        return (hashed % np.uint64(mod)).astype(np.int64)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        pf = pq.ParquetFile(self.path)

        cols_needed = set([self.feature_spec.session_col, self.feature_spec.item_col, self.feature_spec.label_col])
        cols_needed.update(self.feature_spec.numeric_cols)
        cols_needed.update(self.feature_spec.categorical_cols)
        cols = [c for c in pf.schema.names if c in cols_needed]

        rows_seen = 0
        rng = np.random.default_rng(self.seed)

        for rb in pf.iter_batches(batch_size=self.parquet_batch_rows, columns=cols):
            df = rb.to_pandas()
            if df.empty:
                continue

            if self.max_rows > 0:
                remaining = self.max_rows - rows_seen
                if remaining <= 0:
                    break
                if len(df) > remaining:
                    df = df.iloc[:remaining].copy()

            rows_seen += len(df)

            df = self._transform_frame(df)
            if self.shuffle_batches:
                df = df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)

            num_np = df[self.feature_spec.numeric_cols].to_numpy(dtype=np.float32, copy=False)

            if self.feature_spec.categorical_cols:
                cat_arrays = [self._hash_cat(df[c], CAT_HASH_DIM) for c in self.feature_spec.categorical_cols]
                cat_np = np.stack(cat_arrays, axis=1).astype(np.int64, copy=False)
            else:
                cat_np = np.zeros((len(df), 0), dtype=np.int64)

            y_np = df["label"].to_numpy(dtype=np.float32, copy=False)
            sid_np = df["session_id"].to_numpy(dtype=np.int64, copy=False)
            iid_np = df["item_id"].to_numpy(dtype=np.int64, copy=False)

            n = len(df)
            for st in range(0, n, self.batch_size):
                ed = min(st + self.batch_size, n)
                yield {
                    "num": torch.from_numpy(num_np[st:ed]),
                    "cat": torch.from_numpy(cat_np[st:ed]),
                    "label": torch.from_numpy(y_np[st:ed]),
                    "session_id": torch.from_numpy(sid_np[st:ed]),
                    "item_id": torch.from_numpy(iid_np[st:ed]),
                }

            if self.max_rows > 0 and rows_seen >= self.max_rows:
                break


def create_loader(
    path: Path,
    feature_spec: FeatureSpec,
    batch_size: int,
    parquet_batch_rows: int,
    max_rows: int,
    seed: int,
    shuffle_batches: bool,
) -> DataLoader:
    dataset = ParquetBatchDataset(
        path=path,
        feature_spec=feature_spec,
        batch_size=batch_size,
        parquet_batch_rows=parquet_batch_rows,
        max_rows=max_rows,
        seed=seed,
        shuffle_batches=shuffle_batches,
    )
    return DataLoader(dataset, batch_size=None, num_workers=0)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class NeuralRanker(nn.Module):
    """Wide & Deep neural ranker."""

    def __init__(
        self,
        num_numeric: int,
        num_categorical: int,
        cat_hash_dim: int,
        embed_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical

        if num_categorical > 0:
            self.embeddings = nn.ModuleList([nn.Embedding(cat_hash_dim, embed_dim) for _ in range(num_categorical)])
            emb_out_dim = num_categorical * embed_dim
        else:
            self.embeddings = nn.ModuleList()
            emb_out_dim = 0

        in_dim = num_numeric + emb_out_dim

        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(prev, 1)

        self.wide = nn.Linear(in_dim, 1)

    def forward(self, num_x: torch.Tensor, cat_x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = [num_x]
        if self.num_categorical > 0 and cat_x.numel() > 0:
            embs: List[torch.Tensor] = []
            for i, emb in enumerate(self.embeddings):
                embs.append(emb(cat_x[:, i]))
            feats.append(torch.cat(embs, dim=1))

        x = torch.cat(feats, dim=1)
        deep_h = self.deep(x)
        deep_score = self.deep_out(deep_h)
        wide_score = self.wide(x)
        return (deep_score + wide_score).squeeze(1)


# ------------------------------------------------------------
# Losses and metrics
# ------------------------------------------------------------

def bpr_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pos_mask = labels > 0.5
    neg_mask = labels <= 0.5
    pos = logits[pos_mask]
    neg = logits[neg_mask]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    n = min(pos.numel(), neg.numel())
    pos = pos[:n]
    neg = neg[:n]
    return -torch.log(torch.sigmoid(pos - neg) + 1e-12).mean()


def lambdarank_pairwise_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pos_mask = labels > 0.5
    neg_mask = labels <= 0.5
    pos = logits[pos_mask]
    neg = logits[neg_mask]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return torch.nn.functional.softplus(-diff).mean()


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "bpr":
        return bpr_loss(logits, labels)
    if loss_type in {"lambdarank", "lambda"}:
        return lambdarank_pairwise_loss(logits, labels)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


def ranking_metrics(df: pd.DataFrame, k: int) -> Dict[str, float]:
    if df.empty:
        return {f"NDCG@{k}": 0.0, "MAP": 0.0, "MRR": 0.0, "AUC": float("nan")}

    y_true = df["label"].to_numpy(dtype=np.int32)
    scores = df["score"].to_numpy(dtype=np.float32)
    sid = df["session_id"].to_numpy(dtype=np.int64)
    iid = df["item_id"].to_numpy(dtype=np.int64)

    order = np.lexsort((iid, -scores, sid))
    y = y_true[order]
    s = scores[order]
    sid_sorted = sid[order]

    _, starts, counts = np.unique(sid_sorted, return_index=True, return_counts=True)

    ndcgs: List[float] = []
    aps: List[float] = []
    mrrs: List[float] = []

    for st, ct in zip(starts, counts):
        rel = y[st:st + ct]
        total_rel = int(rel.sum())
        cutoff = int(min(k, rel.size))

        if cutoff <= 0:
            ndcgs.append(0.0)
            aps.append(0.0)
            mrrs.append(0.0)
            continue

        rel_k = rel[:cutoff]
        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
        dcg = float((rel_k * discounts).sum())
        ideal = np.sort(rel)[::-1][:cutoff]
        idcg = float((ideal * discounts).sum())
        ndcgs.append(0.0 if idcg <= 0 else dcg / idcg)

        if total_rel == 0:
            aps.append(0.0)
        else:
            csum = np.cumsum(rel)
            hits = np.flatnonzero(rel)
            aps.append(float((csum[hits] / (hits + 1.0)).sum() / total_rel))

        pos_idx = np.flatnonzero(rel)
        mrrs.append(0.0 if pos_idx.size == 0 else 1.0 / float(pos_idx[0] + 1))

    metrics = {
        f"NDCG@{k}": float(np.mean(ndcgs) if ndcgs else 0.0),
        "MAP": float(np.mean(aps) if aps else 0.0),
        "MRR": float(np.mean(mrrs) if mrrs else 0.0),
    }
    metrics["AUC"] = float(roc_auc_score(y_true, s)) if np.unique(y_true).size > 1 else float("nan")
    return metrics


# ------------------------------------------------------------
# Checkpoint manager
# ------------------------------------------------------------

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, best_model_path: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.best_model_path = best_model_path
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

    def _ckpt_path(self, epoch: int) -> Path:
        return self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

    def latest_checkpoint(self) -> Optional[Path]:
        files = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not files:
            return None
        def _extract_epoch(p: Path) -> int:
            stem = p.stem
            try:
                return int(stem.split("_")[-1])
            except Exception:
                return -1
        files = sorted(files, key=_extract_epoch)
        return files[-1]

    @staticmethod
    def _rng_state() -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
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

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[ReduceLROnPlateau],
        best_metric: float,
        config: Dict[str, Any],
        is_best: bool,
        val_metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "config": config,
            "rng_state": self._rng_state(),
            "val_metrics": val_metrics or {},
        }
        path = self._ckpt_path(epoch)
        torch.save(checkpoint, path)
        _log("Saved checkpoint: %s", path)

        if is_best:
            torch.save(checkpoint, self.best_model_path)
            _log("Updated best model: %s", self.best_model_path)

        return path

    def load_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[ReduceLROnPlateau],
        device: str,
    ) -> Tuple[int, float]:
        latest = self.latest_checkpoint()
        if latest is None:
            return 0, float("-inf")

        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self._set_rng_state(ckpt.get("rng_state", {}))

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", float("-inf")))
        _log("Resumed from checkpoint %s | start_epoch=%d best_ndcg=%.6f", latest, start_epoch, best_metric)
        return start_epoch, best_metric


# ------------------------------------------------------------
# Training and validation loops
# ------------------------------------------------------------

def train_one_epoch(
    model: NeuralRanker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    loss_type: str,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
) -> float:
    model.train()
    loss_sum = 0.0
    n_batches = 0

    for i, batch in enumerate(loader, start=1):
        num = batch["num"].to(device, non_blocking=True)
        cat = batch["cat"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and device.startswith("cuda") and USE_MIXED_PRECISION
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(num, cat)
                loss = compute_loss(logits, y, loss_type)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(num, cat)
            loss = compute_loss(logits, y, loss_type)
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.detach().item())
        n_batches += 1

        if i % 200 == 0:
            _log("epoch=%d train_batch=%d avg_loss=%.6f", epoch, i, loss_sum / max(n_batches, 1))

    return loss_sum / max(n_batches, 1)


def validate(
    model: NeuralRanker,
    loader: DataLoader,
    device: str,
    k: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()

    preds: List[pd.DataFrame] = []
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
            preds.append(part)

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=["session_id", "item_id", "label", "score"])
    pred_df.sort_values(["session_id", "score", "item_id"], ascending=[True, False, True], inplace=True, kind="mergesort")
    metrics = ranking_metrics(pred_df, k=k)
    return metrics, pred_df


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()
    set_seed(RANDOM_SEED)

    p = _paths()
    p["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    p["best_model"].parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    cfg_obj = TrainConfig(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        checkpoint_freq=CHECKPOINT_FREQ,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        top_k_eval=TOP_K_EVAL,
        loss_type=LOSS_TYPE,
        weight_decay=WEIGHT_DECAY,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        cat_hash_dim=CAT_HASH_DIM,
        embed_dim=EMBED_DIM,
        max_train_rows=MAX_TRAIN_ROWS,
        max_val_rows=MAX_VAL_ROWS,
        max_test_rows=MAX_TEST_ROWS,
        parquet_batch_rows=PARQUET_BATCH_ROWS,
        use_mixed_precision=USE_MIXED_PRECISION,
        device=device,
    )

    data_paths = resolve_data_paths()
    feature_spec = infer_feature_spec(data_paths)

    _log("Data source: %s", data_paths.source)
    _log("Training device: %s", device)

    train_loader = create_loader(
        data_paths.train_path,
        feature_spec,
        batch_size=cfg_obj.batch_size,
        parquet_batch_rows=cfg_obj.parquet_batch_rows,
        max_rows=cfg_obj.max_train_rows,
        seed=RANDOM_SEED,
        shuffle_batches=True,
    )
    val_loader = create_loader(
        data_paths.val_path,
        feature_spec,
        batch_size=cfg_obj.batch_size,
        parquet_batch_rows=cfg_obj.parquet_batch_rows,
        max_rows=cfg_obj.max_val_rows,
        seed=RANDOM_SEED,
        shuffle_batches=False,
    )

    model = NeuralRanker(
        num_numeric=len(feature_spec.numeric_cols),
        num_categorical=len(feature_spec.categorical_cols),
        cat_hash_dim=cfg_obj.cat_hash_dim,
        embed_dim=cfg_obj.embed_dim,
        hidden_dims=cfg_obj.hidden_dims,
        dropout=cfg_obj.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg_obj.learning_rate, weight_decay=cfg_obj.weight_decay)
    scheduler: Optional[ReduceLROnPlateau] = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    scaler: Optional[torch.cuda.amp.GradScaler]
    if cfg_obj.use_mixed_precision and device.startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    ckpt_mgr = CheckpointManager(p["checkpoint_dir"], p["best_model"])
    start_epoch, best_ndcg = ckpt_mgr.load_latest(model, optimizer, scheduler, device=device)

    history: List[Dict[str, Any]] = []
    if p["training_log"].exists():
        try:
            history = json.loads(p["training_log"].read_text())
        except Exception:
            history = []

    no_improve_epochs = 0

    try:
        for epoch in range(start_epoch, cfg_obj.num_epochs):
            ep_start = time.perf_counter()
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                loss_type=cfg_obj.loss_type,
                scaler=scaler,
                epoch=epoch,
            )

            val_metrics, val_pred = validate(model, val_loader, device=device, k=cfg_obj.top_k_eval)
            val_ndcg = float(val_metrics.get(f"NDCG@{cfg_obj.top_k_eval}", 0.0))

            if scheduler is not None:
                scheduler.step(val_ndcg)

            lr = float(optimizer.param_groups[0]["lr"])
            epoch_log = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_metrics": val_metrics,
                "lr": lr,
                "epoch_time_sec": float(time.perf_counter() - ep_start),
            }
            history.append(epoch_log)

            _log(
                "epoch=%d train_loss=%.6f val_ndcg@%d=%.6f val_map=%.6f val_mrr=%.6f val_auc=%s lr=%.6g",
                epoch,
                train_loss,
                cfg_obj.top_k_eval,
                val_ndcg,
                float(val_metrics.get("MAP", 0.0)),
                float(val_metrics.get("MRR", 0.0)),
                str(val_metrics.get("AUC", float("nan"))),
                lr,
            )

            improved = val_ndcg > best_ndcg
            if improved:
                best_ndcg = val_ndcg
                no_improve_epochs = 0
                val_pred.to_parquet(p["val_predictions"], index=False)
                _log("Saved validation predictions: %s", p["val_predictions"])
            else:
                no_improve_epochs += 1

            should_ckpt = improved or ((epoch + 1) % cfg_obj.checkpoint_freq == 0)
            if should_ckpt:
                ckpt_mgr.save(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_ndcg,
                    config={
                        "train_config": asdict(cfg_obj),
                        "feature_spec": asdict(feature_spec),
                        "data_source": data_paths.source,
                    },
                    is_best=improved,
                    val_metrics=val_metrics,
                )

            p["training_log"].write_text(json.dumps(history, indent=2, default=float))

            if no_improve_epochs >= cfg_obj.early_stopping_patience:
                _log("Early stopping triggered at epoch=%d", epoch)
                break

    except KeyboardInterrupt:
        _log("Training interrupted by user. Saving recovery checkpoint.")
        ckpt_mgr.save(
            epoch=max(start_epoch, len(history)),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_ndcg,
            config={
                "train_config": asdict(cfg_obj),
                "feature_spec": asdict(feature_spec),
                "data_source": data_paths.source,
            },
            is_best=False,
            val_metrics={},
        )
        p["training_log"].write_text(json.dumps(history, indent=2, default=float))
        raise
    except Exception as e:
        _log("Training crashed: %s", str(e))
        ckpt_mgr.save(
            epoch=max(start_epoch, len(history)),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_ndcg,
            config={
                "train_config": asdict(cfg_obj),
                "feature_spec": asdict(feature_spec),
                "data_source": data_paths.source,
                "crash": str(e),
            },
            is_best=False,
            val_metrics={},
        )
        p["training_log"].write_text(json.dumps(history, indent=2, default=float))
        raise

    final_payload = {
        "model_state_dict": model.state_dict(),
        "best_ndcg": best_ndcg,
        "train_config": asdict(cfg_obj),
        "feature_spec": asdict(feature_spec),
        "data_source": data_paths.source,
    }
    torch.save(final_payload, p["final_model"])
    _log("Saved final model: %s", p["final_model"])

    if not p["training_log"].exists():
        p["training_log"].write_text(json.dumps(history, indent=2, default=float))

    _log("Total runtime_sec=%.2f", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
