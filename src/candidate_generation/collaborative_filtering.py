"""Collaborative Filtering candidate retrieval using implicit-feedback ALS."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse as sps

# ============================================================
# CONFIG (compatibility-safe)
# ============================================================

try:
    from data_pipeline.config import (  # type: ignore
        CF_FACTORS,
        CF_ITERATIONS,
        CF_REGULARIZATION,
        CF_TOP_K,
        CF_ALPHA,
        RANDOM_SEED,
        LOGGER,
        PROCESSED_DATA_PATHS,
    )
except Exception:
    try:
        from config import (  # type: ignore
            CF_FACTORS,
            CF_ITERATIONS,
            CF_REGULARIZATION,
            CF_TOP_K,
            CF_ALPHA,
            RANDOM_SEED,
            LOGGER,
            PROCESSED_DATA_PATHS,
        )
    except Exception:
        RANDOM_SEED = 42
        CF_FACTORS = 64
        CF_ITERATIONS = 20
        CF_REGULARIZATION = 0.01
        CF_TOP_K = 100
        CF_ALPHA = 40.0
        LOGGER = logging.getLogger("collaborative_filtering")
        if not LOGGER.handlers:
            LOGGER.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            LOGGER.addHandler(handler)
            LOGGER.propagate = False

        project_root = Path(__file__).resolve().parents[2]
        processed_dir = project_root / "data_pipeline" / "data" / "processed"
        PROCESSED_DATA_PATHS = {
            "interactions": processed_dir / "interactions.parquet",
        }


# ============================================================
# HELPERS
# ============================================================


def _path(key: str) -> Path:
    if key == "cf_candidates":
        interactions_base = Path(
            PROCESSED_DATA_PATHS.get(
                "interactions",
                Path(__file__).resolve().parents[2]
                / "data_pipeline"
                / "data"
                / "processed"
                / "interactions.parquet",
            )
        ).parent
        return interactions_base.parent / "candidates" / "cf_candidates.parquet"

    if key in PROCESSED_DATA_PATHS:
        return Path(PROCESSED_DATA_PATHS[key])

    interactions_base = Path(
        PROCESSED_DATA_PATHS.get(
            "interactions",
            Path(__file__).resolve().parents[2]
            / "data_pipeline"
            / "data"
            / "processed"
            / "interactions.parquet",
        )
    ).parent
    return interactions_base / f"{key}.parquet"


def _checkpoint_paths() -> Tuple[Path, Path]:
    out_path = _path("cf_candidates")
    return out_path.with_suffix(".state.json"), out_path.with_name("cf_candidates_chunks")


def _load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {}


def _save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))


def _memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024 ** 2))


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _looks_like_col_indices(arr: np.ndarray, n_items: int) -> bool:
    if arr.size == 0:
        return True
    if not np.issubdtype(arr.dtype, np.integer):
        return False
    return int(arr.min()) >= 0 and int(arr.max()) < int(n_items)


@dataclass
class IdMappings:
    row_col: str
    row_ids: np.ndarray
    item_ids: np.ndarray


@dataclass
class NumpyALSModel:
    user_factors: np.ndarray
    item_factors: np.ndarray


# ============================================================
# REQUIRED FUNCTIONS
# ============================================================


def load_interactions() -> pd.DataFrame:
    """Load interactions needed for collaborative filtering."""
    path = _path("interactions")
    if not path.exists():
        raise FileNotFoundError(f"interactions file not found: {path}")

    LOGGER.info("Loading interactions from %s", path)
    df = pd.read_parquet(path, columns=["session_id", "item_id", "interaction_weight", "timestamp"])
    df = df.dropna(subset=["session_id", "item_id"])

    row_col = "session_id"
    if "user_id" in df.columns:
        row_col = "user_id"

    df[row_col] = df[row_col].astype(np.int64)
    df["item_id"] = df["item_id"].astype(np.int64)
    if "interaction_weight" not in df.columns:
        df["interaction_weight"] = 1.0
    df["interaction_weight"] = df["interaction_weight"].astype(np.float32)

    LOGGER.info("interactions rows: %d", len(df))
    LOGGER.info("interactions memory: %.2f MB", _memory_mb(df))
    return df


def build_sparse_matrix(df: pd.DataFrame) -> Tuple[sps.csr_matrix, IdMappings]:
    """Build CSR sparse matrix from interactions and return ID mappings."""
    row_col = "session_id" if "session_id" in df.columns else "user_id"

    agg = (
        df[[row_col, "item_id", "interaction_weight"]]
        .groupby([row_col, "item_id"], observed=True, sort=False)["interaction_weight"]
        .sum()
        .reset_index()
    )

    row_codes, row_ids = pd.factorize(agg[row_col], sort=True)
    item_codes, item_ids = pd.factorize(agg["item_id"], sort=True)

    values = agg["interaction_weight"].to_numpy(dtype=np.float32)
    matrix = sps.coo_matrix(
        (values, (row_codes.astype(np.int64), item_codes.astype(np.int64))),
        shape=(len(row_ids), len(item_ids)),
        dtype=np.float32,
    ).tocsr()

    nnz = matrix.nnz
    total = matrix.shape[0] * matrix.shape[1]
    sparsity = 1.0 - (nnz / float(max(total, 1)))

    LOGGER.info("matrix shape: %s", matrix.shape)
    LOGGER.info("matrix nnz: %d", nnz)
    LOGGER.info("matrix sparsity: %.8f", sparsity)

    return matrix, IdMappings(row_col=row_col, row_ids=row_ids.to_numpy(dtype=np.int64), item_ids=item_ids.to_numpy(dtype=np.int64))


def _train_implicit_als(matrix: sps.csr_matrix) -> Any:
    import implicit  # type: ignore

    factors = _as_int(globals().get("CF_FACTORS", 64), 64)
    iterations = _as_int(globals().get("CF_ITERATIONS", 20), 20)
    reg = _as_float(globals().get("CF_REGULARIZATION", 0.01), 0.01)
    alpha = _as_float(globals().get("CF_ALPHA", 40.0), 40.0)

    conf = (matrix * alpha).astype(np.float32)

    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=reg,
        iterations=iterations,
        random_state=_as_int(globals().get("RANDOM_SEED", 42), 42),
        calculate_training_loss=False,
        use_gpu=False,
    )
    # Train with user-item orientation to match recommend(userid, user_items).
    model.fit(conf)
    model._cf_backend = "implicit"  # type: ignore[attr-defined]
    return model


def _train_numpy_als(matrix: sps.csr_matrix) -> NumpyALSModel:
    factors = _as_int(globals().get("CF_FACTORS", 64), 64)
    iterations = _as_int(globals().get("CF_ITERATIONS", 20), 20)
    reg = _as_float(globals().get("CF_REGULARIZATION", 0.01), 0.01)
    alpha = _as_float(globals().get("CF_ALPHA", 40.0), 40.0)
    seed = _as_int(globals().get("RANDOM_SEED", 42), 42)

    rng = np.random.default_rng(seed)

    n_users, n_items = matrix.shape
    f = factors

    user_factors = (rng.normal(0.0, 0.01, size=(n_users, f))).astype(np.float32)
    item_factors = (rng.normal(0.0, 0.01, size=(n_items, f))).astype(np.float32)

    csr = matrix.tocsr()
    csc = matrix.tocsc()
    eye = np.eye(f, dtype=np.float32)

    for _ in range(iterations):
        yty = item_factors.T @ item_factors

        for u in range(n_users):
            start, end = csr.indptr[u], csr.indptr[u + 1]
            idx = csr.indices[start:end]
            if idx.size == 0:
                continue
            vals = csr.data[start:end].astype(np.float32)
            conf = 1.0 + alpha * vals
            y_u = item_factors[idx]
            cu_i = conf - 1.0

            a = yty + (y_u.T * cu_i) @ y_u + reg * eye
            b = (y_u.T * conf) @ np.ones_like(conf)
            user_factors[u] = np.linalg.solve(a, b)

        xtx = user_factors.T @ user_factors

        for i in range(n_items):
            start, end = csc.indptr[i], csc.indptr[i + 1]
            idx = csc.indices[start:end]
            if idx.size == 0:
                continue
            vals = csc.data[start:end].astype(np.float32)
            conf = 1.0 + alpha * vals
            x_i = user_factors[idx]
            cu_i = conf - 1.0

            a = xtx + (x_i.T * cu_i) @ x_i + reg * eye
            b = (x_i.T * conf) @ np.ones_like(conf)
            item_factors[i] = np.linalg.solve(a, b)

    return NumpyALSModel(user_factors=user_factors, item_factors=item_factors)


def train_als(matrix: sps.csr_matrix) -> Any:
    """Train ALS model using implicit, with deterministic numpy fallback."""
    LOGGER.info("Training ALS model...")
    t0 = time.perf_counter()

    np.random.seed(_as_int(globals().get("RANDOM_SEED", 42), 42))

    model: Any
    try:
        model = _train_implicit_als(matrix)
        backend = "implicit"
    except Exception as e:
        LOGGER.warning("implicit ALS unavailable, using numpy fallback: %s", e)
        model = _train_numpy_als(matrix)
        setattr(model, "_cf_backend", "numpy")
        backend = "numpy"

    dt = time.perf_counter() - t0
    LOGGER.info("ALS backend: %s", backend)
    LOGGER.info("training_time_sec: %.2f", dt)
    return model


def generate_candidates(
    model: Any,
    matrix: sps.csr_matrix,
    id_mappings: IdMappings,
) -> Dict[str, Any]:
    """Generate top-N candidate items per session/user in batches."""
    top_k = _as_int(globals().get("CF_TOP_K", 100), 100)
    top_k = max(1, min(top_k, matrix.shape[1]))

    n_rows = matrix.shape[0]
    batch_size = 2048
    n_items = matrix.shape[1]

    state_path, chunks_dir = _checkpoint_paths()
    chunks_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(state_path)
    next_start = _as_int(state.get("next_start", 0), 0)
    if next_start < 0 or next_start > n_rows:
        next_start = 0

    total_rows = _as_int(state.get("total_rows", 0), 0)
    max_score = _as_float(state.get("max_score", float("-inf")), float("-inf"))

    if next_start > 0:
        LOGGER.info("Resuming candidate generation from row index %d", next_start)

    backend = getattr(model, "_cf_backend", "implicit")

    if backend == "implicit":
        for start in range(next_start, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            user_batch = np.arange(start, end, dtype=np.int32)

            try:
                item_ids_idx, scores = model.recommend(
                    userid=user_batch,
                    user_items=matrix[user_batch],
                    N=top_k,
                    filter_already_liked_items=True,
                    recalculate_user=False,
                )
            except TypeError:
                item_ids_idx, scores = model.recommend(
                    user_batch,
                    matrix[user_batch],
                    N=top_k,
                    filter_already_liked_items=True,
                )

            item_ids_idx = np.asarray(item_ids_idx)
            scores = np.asarray(scores)

            # Compatibility handling across implicit versions:
            # - Some versions can return (scores, ids) ordering.
            # - Some stacks may surface item ids directly instead of column indices.
            if not _looks_like_col_indices(item_ids_idx, n_items) and _looks_like_col_indices(scores, n_items):
                item_ids_idx, scores = scores, item_ids_idx

            flat_idx = item_ids_idx.reshape(-1)
            flat_scores = scores.reshape(-1).astype(np.float32)

            if _looks_like_col_indices(flat_idx.astype(np.int64, copy=False), n_items):
                flat_item_ids = id_mappings.item_ids[flat_idx.astype(np.int64, copy=False)]
            else:
                flat_item_ids = flat_idx.astype(np.int64, copy=False)

            ranks = np.tile(np.arange(1, top_k + 1, dtype=np.int16), (end - start, 1))
            part = pd.DataFrame(
                {
                    "session_id": np.repeat(id_mappings.row_ids[start:end], top_k),
                    "item_id": flat_item_ids,
                    "score": flat_scores,
                    "rank": ranks.reshape(-1),
                    "model": "ALS",
                }
            )
            part = part[(part["item_id"] > 0) & np.isfinite(part["score"])]
            part = format_candidates(part)

            chunk_path = chunks_dir / f"chunk_{start:010d}_{end:010d}.parquet"
            part.to_parquet(chunk_path, index=False)

            total_rows += len(part)
            if not part.empty:
                max_score = max(max_score, float(part["score"].max()))

            _save_state(
                state_path,
                {
                    "next_start": end,
                    "total_rows": total_rows,
                    "max_score": max_score,
                    "batch_size": batch_size,
                    "top_k": top_k,
                    "backend": backend,
                },
            )
            LOGGER.info("candidate generation progress: %d/%d rows", end, n_rows)
    else:
        user_factors = model.user_factors
        item_factors = model.item_factors

        for start in range(next_start, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            rows = end - start

            scores = user_factors[start:end] @ item_factors.T
            interacted = matrix[start:end].tocoo()
            scores[interacted.row, interacted.col] = -np.inf

            k = min(top_k, n_items)
            idx_part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            score_part = np.take_along_axis(scores, idx_part, axis=1)

            order = np.argsort(-score_part, axis=1)
            sorted_idx = np.take_along_axis(idx_part, order, axis=1)
            sorted_scores = np.take_along_axis(score_part, order, axis=1)

            ranks = np.tile(np.arange(1, k + 1, dtype=np.int16), (rows, 1))
            part = pd.DataFrame(
                {
                    "session_id": np.repeat(id_mappings.row_ids[start:end], k),
                    "item_id": id_mappings.item_ids[sorted_idx.reshape(-1)],
                    "score": sorted_scores.reshape(-1).astype(np.float32),
                    "rank": ranks.reshape(-1),
                    "model": "ALS",
                }
            )
            part = part[(part["item_id"] > 0) & np.isfinite(part["score"])]
            part = format_candidates(part)

            chunk_path = chunks_dir / f"chunk_{start:010d}_{end:010d}.parquet"
            part.to_parquet(chunk_path, index=False)

            total_rows += len(part)
            if not part.empty:
                max_score = max(max_score, float(part["score"].max()))

            _save_state(
                state_path,
                {
                    "next_start": end,
                    "total_rows": total_rows,
                    "max_score": max_score,
                    "batch_size": batch_size,
                    "top_k": top_k,
                    "backend": backend,
                },
            )
            LOGGER.info("candidate generation progress: %d/%d rows", end, n_rows)

    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    return {
        "chunk_files": chunk_files,
        "total_rows": total_rows,
        "max_score": 0.0 if max_score == float("-inf") else max_score,
        "n_rows": n_rows,
        "state_path": state_path,
        "chunks_dir": chunks_dir,
    }


def format_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Ensure schema, dtypes, and deterministic ordering."""
    if candidates.empty:
        return pd.DataFrame(columns=["session_id", "item_id", "score", "rank", "model"])

    out = candidates[["session_id", "item_id", "score", "rank", "model"]].copy()
    out["session_id"] = out["session_id"].astype(np.int64)
    out["item_id"] = out["item_id"].astype(np.int64)
    out["score"] = out["score"].astype(np.float32)
    out["rank"] = out["rank"].astype(np.int16)
    out["model"] = "ALS"

    out = out.sort_values(["session_id", "rank", "item_id"], ascending=[True, True, True])
    return out


def save_candidates(candidates: pd.DataFrame) -> Path:
    """Persist collaborative filtering candidate set to parquet."""
    out_path = _path("cf_candidates")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(out_path, index=False)
    LOGGER.info("Saved candidates to %s", out_path)
    return out_path


def _save_candidates_from_chunks(gen_result: Dict[str, Any]) -> Path:
    out_path = _path("cf_candidates")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    chunk_files: List[Path] = list(gen_result.get("chunk_files", []))
    chunk_files = sorted(chunk_files)
    if not chunk_files:
        empty = pd.DataFrame(columns=["session_id", "item_id", "score", "rank", "model"])
        empty.to_parquet(out_path, index=False)
        return out_path

    writer: pq.ParquetWriter | None = None
    try:
        for chunk_file in chunk_files:
            table = pq.read_table(chunk_file)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    LOGGER.info("Saved candidates to %s", out_path)
    return out_path


def main() -> None:
    """Run collaborative filtering retrieval pipeline end-to-end."""
    t0 = time.perf_counter()

    df = load_interactions()
    matrix, id_mappings = build_sparse_matrix(df)
    model = train_als(matrix)

    gen_result = generate_candidates(model, matrix, id_mappings)
    out_path = _save_candidates_from_chunks(gen_result)

    total_rows = _as_int(gen_result.get("total_rows", 0), 0)
    n_rows = _as_int(gen_result.get("n_rows", matrix.shape[0]), matrix.shape[0])
    avg_candidates = total_rows / float(max(n_rows, 1))
    max_score = _as_float(gen_result.get("max_score", 0.0), 0.0)

    # Cleanup checkpoints after successful finalization.
    state_path: Path = gen_result["state_path"]
    chunks_dir: Path = gen_result["chunks_dir"]
    if state_path.exists():
        state_path.unlink()
    for chunk_file in chunks_dir.glob("chunk_*.parquet"):
        chunk_file.unlink()
    chunks_dir.rmdir()

    LOGGER.info("avg candidates per session: %.2f", float(avg_candidates))
    LOGGER.info("max score: %.6f", max_score)
    LOGGER.info("output_path: %s", out_path)
    LOGGER.info("runtime_sec: %.2f", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
