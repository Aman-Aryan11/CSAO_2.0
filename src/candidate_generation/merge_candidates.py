"""Merge retrieval candidates from multiple sources into a unified pool."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================
# CONFIG (compatibility-safe)
# ============================================================

try:
    from data_pipeline.config import MERGE_TOP_K, RANDOM_SEED, LOGGER, PROCESSED_DATA_PATHS  # type: ignore
except Exception:
    try:
        from config import MERGE_TOP_K, RANDOM_SEED, LOGGER, PROCESSED_DATA_PATHS  # type: ignore
    except Exception:
        MERGE_TOP_K = 300
        RANDOM_SEED = 42
        LOGGER = logging.getLogger("merge_candidates")
        if not LOGGER.handlers:
            LOGGER.setLevel(logging.INFO)
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            LOGGER.addHandler(h)
            LOGGER.propagate = False

        project_root = Path(__file__).resolve().parents[2]
        processed_dir = project_root / "data_pipeline" / "data" / "processed"
        candidates_dir = project_root / "data_pipeline" / "data" / "candidates"
        PROCESSED_DATA_PATHS = {
            "item_similarity": candidates_dir / "item_similarity.parquet",
            "cf_candidates": candidates_dir / "cf_candidates.parquet",
            "interactions": processed_dir / "interactions.parquet",
            "candidates_merged": candidates_dir / "candidates_merged.parquet",
        }


# ============================================================
# HELPERS
# ============================================================


def _path(key: str) -> Path:
    if key in PROCESSED_DATA_PATHS:
        p = Path(PROCESSED_DATA_PATHS[key])
        if key in {"item_similarity", "cf_candidates", "candidates_merged"} and p.parent.name == "processed":
            return p.parent.parent / "candidates" / p.name
        return p

    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data_pipeline" / "data" / "processed"
    candidates_dir = project_root / "data_pipeline" / "data" / "candidates"

    if key == "interactions":
        return processed_dir / "interactions.parquet"
    if key == "item_similarity":
        return candidates_dir / "item_similarity.parquet"
    if key == "cf_candidates":
        return candidates_dir / "cf_candidates.parquet"
    if key == "candidates_merged":
        return candidates_dir / "candidates_merged.parquet"
    return processed_dir / f"{key}.parquet"


def _resolve_input(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


def _memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024 ** 2))


def _to_int32(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(np.int32)


def _to_float32(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(np.float32)


def _session_seed_items(session_ids: np.ndarray, interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized session -> seed items mapping via merge (no Python loops)."""
    if session_ids.size == 0:
        return pd.DataFrame(columns=["session_id", "seed_item_id"])
    sess_df = pd.DataFrame({"session_id": session_ids.astype(np.int32)})
    out = sess_df.merge(interactions_df, on="session_id", how="inner")
    if out.empty:
        return pd.DataFrame(columns=["session_id", "seed_item_id"])
    out = out.rename(columns={"item_id": "seed_item_id"})[["session_id", "seed_item_id"]]
    out = out.drop_duplicates()
    return out


def _append_parquet_chunk(writer: Optional[pq.ParquetWriter], df: pd.DataFrame, out_path: Path) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
    writer.write_table(table)
    return writer


# ============================================================
# REQUIRED FUNCTIONS
# ============================================================


def load_candidates() -> Dict[str, Any]:
    """Load candidate sources and build artifacts for memory-safe fusion."""
    sim_primary = _path("item_similarity")
    cf_primary = _path("cf_candidates")

    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data_pipeline" / "data" / "processed"

    sim_path = _resolve_input(sim_primary, processed_dir / "item_similarity.parquet")
    cf_path = _resolve_input(cf_primary, processed_dir / "cf_candidates.parquet")

    if not sim_path.exists():
        raise FileNotFoundError(f"item_similarity file not found: {sim_path}")
    if not cf_path.exists():
        raise FileNotFoundError(f"cf_candidates file not found: {cf_path}")

    interactions_path = _path("interactions")
    if not interactions_path.exists():
        interactions_path = processed_dir / "interactions.parquet"
    if not interactions_path.exists():
        raise FileNotFoundError(f"interactions file not found: {interactions_path}")

    interactions_df = pd.read_parquet(interactions_path, columns=["session_id", "item_id"])
    interactions_df = interactions_df.drop_duplicates()
    interactions_df["session_id"] = _to_int32(interactions_df["session_id"])
    interactions_df["item_id"] = _to_int32(interactions_df["item_id"])
    sim_raw = pd.read_parquet(sim_path)

    sim_mode = "session"
    sim_session_df: Optional[pd.DataFrame] = None
    sim_edges_df: Optional[pd.DataFrame] = None

    if {"session_id", "item_id"}.issubset(sim_raw.columns):
        if "similarity_score" not in sim_raw.columns and "score" in sim_raw.columns:
            sim_raw = sim_raw.rename(columns={"score": "similarity_score"})
        if "similarity_score" not in sim_raw.columns:
            raise ValueError("item_similarity session-level file requires similarity_score")

        sim_session_df = sim_raw[["session_id", "item_id", "similarity_score"]].copy()
        sim_session_df["session_id"] = _to_int32(sim_session_df["session_id"])
        sim_session_df["item_id"] = _to_int32(sim_session_df["item_id"])
        sim_session_df["similarity_score"] = _to_float32(sim_session_df["similarity_score"])
    elif {"item_id", "similar_item_id"}.issubset(sim_raw.columns):
        sim_mode = "edge"
        if "similarity_score" not in sim_raw.columns and "score" in sim_raw.columns:
            sim_raw = sim_raw.rename(columns={"score": "similarity_score"})
        if "similarity_score" not in sim_raw.columns:
            raise ValueError("item_similarity edge file requires similarity_score")

        edges = sim_raw[["item_id", "similar_item_id", "similarity_score"]].copy()

        # Reduce edge volume: if multiple metrics exist in file, prefer cosine only.
        if "similarity_type" in sim_raw.columns:
            cosine_mask = sim_raw["similarity_type"].astype(str).str.lower().eq("cosine")
            if cosine_mask.any():
                edges = sim_raw.loc[cosine_mask, ["item_id", "similar_item_id", "similarity_score"]].copy()

        edges["item_id"] = _to_int32(edges["item_id"])
        edges["similar_item_id"] = _to_int32(edges["similar_item_id"])
        edges["similarity_score"] = _to_float32(edges["similarity_score"])

        edges = edges.sort_values(["item_id", "similarity_score", "similar_item_id"], ascending=[True, False, True])
        # Keep a bounded neighborhood per seed item to control memory/compute.
        edges = edges.groupby("item_id", observed=True, sort=False).head(50)

        sim_edges_df = edges.rename(columns={"item_id": "seed_item_id", "similar_item_id": "item_id"})
    else:
        raise ValueError(
            "item_similarity must contain either ['session_id','item_id','similarity_score'] "
            "or ['item_id','similar_item_id','similarity_score']."
        )

    cf_pf = pq.ParquetFile(cf_path)

    LOGGER.info("item_similarity mode: %s", sim_mode)
    if sim_session_df is not None:
        LOGGER.info("item_similarity rows: %d", len(sim_session_df))
        LOGGER.info("item_similarity memory: %.2f MB", _memory_mb(sim_session_df))
    if sim_edges_df is not None:
        LOGGER.info("item_similarity edges rows (trimmed): %d", len(sim_edges_df))
        LOGGER.info("item_similarity edges memory: %.2f MB", _memory_mb(sim_edges_df))

    LOGGER.info("cf_candidates rows: %d", cf_pf.metadata.num_rows)
    LOGGER.info("interactions rows (dedup): %d", len(interactions_df))

    return {
        "sim_mode": sim_mode,
        "sim_session_df": sim_session_df,
        "sim_edges_df": sim_edges_df,
        "cf_path": cf_path,
        "cf_pf": cf_pf,
        "interactions_df": interactions_df,
    }


def standardize_scores(sim_df: pd.DataFrame, cf_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure consistent score column names and source flags."""
    sim = sim_df[["session_id", "item_id", "similarity_score"]].copy()
    cf = cf_df[["session_id", "item_id", "score"]].rename(columns={"score": "cf_score"}).copy()

    sim["source_item_similarity"] = np.int8(1)
    cf["source_cf"] = np.int8(1)

    return sim, cf


def merge_sources(sim_df: pd.DataFrame, cf_df: pd.DataFrame) -> pd.DataFrame:
    """Outer join candidate sources on (session_id, item_id), preserving all candidates."""
    merged = sim_df.merge(cf_df, on=["session_id", "item_id"], how="outer")

    merged["similarity_score"] = merged["similarity_score"].fillna(0.0).astype(np.float32)
    merged["cf_score"] = merged["cf_score"].fillna(0.0).astype(np.float32)
    merged["source_item_similarity"] = merged["source_item_similarity"].fillna(0).astype(np.int8)
    merged["source_cf"] = merged["source_cf"].fillna(0).astype(np.int8)

    merged = merged.drop_duplicates(subset=["session_id", "item_id"], keep="first")
    return merged


def add_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived source features used for fusion ranking."""
    df["candidate_source_count"] = (df["source_item_similarity"] + df["source_cf"]).astype(np.int8)
    return df


def filter_seen_items(df: pd.DataFrame, interactions_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Optionally remove session-item pairs already seen in interactions."""
    if interactions_df is None or interactions_df.empty:
        return df

    seen = interactions_df.assign(_seen=np.int8(1))
    merged = df.merge(seen, on=["session_id", "item_id"], how="left")
    out = merged[merged["_seen"].isna()].drop(columns=["_seen"])
    return out


def select_top_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Select top-K candidates per session and assign final rank."""
    if df.empty:
        out = df.copy()
        out["final_rank"] = np.int16([])
        return out

    ordered = df.sort_values(
        ["session_id", "candidate_source_count", "similarity_score", "cf_score", "item_id"],
        ascending=[True, False, False, False, True],
    )

    top = ordered.groupby("session_id", observed=True, sort=False).head(int(k)).copy()
    top["final_rank"] = top.groupby("session_id", observed=True, sort=False).cumcount().add(1).astype(np.int16)
    return top


def save_candidates(df: pd.DataFrame) -> Path:
    """Save merged candidate pool for downstream ranking."""
    out_path = _path("candidates_merged")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    LOGGER.info("Saved merged candidates: %s", out_path)
    return out_path


def _sim_for_sessions(artifacts: Dict[str, Any], session_ids: np.ndarray) -> pd.DataFrame:
    sim_mode = artifacts["sim_mode"]

    if sim_mode == "session":
        sim_session_df = artifacts["sim_session_df"]
        sess_df = pd.DataFrame({"session_id": session_ids})
        out = sess_df.merge(sim_session_df, on="session_id", how="left")
        out = out.dropna(subset=["item_id"])
        out["session_id"] = _to_int32(out["session_id"])
        out["item_id"] = _to_int32(out["item_id"])
        out["similarity_score"] = _to_float32(out["similarity_score"])
        return out[["session_id", "item_id", "similarity_score"]]

    sim_edges_df = artifacts["sim_edges_df"]
    interactions_df = artifacts["interactions_df"]
    seed_items_df = _session_seed_items(session_ids, interactions_df)
    if seed_items_df.empty:
        return pd.DataFrame(columns=["session_id", "item_id", "similarity_score"])

    sim_expanded = seed_items_df.merge(sim_edges_df, on="seed_item_id", how="inner")
    if sim_expanded.empty:
        return pd.DataFrame(columns=["session_id", "item_id", "similarity_score"])

    sim_agg = (
        sim_expanded.groupby(["session_id", "item_id"], observed=True, sort=False)["similarity_score"]
        .max()
        .reset_index()
    )
    sim_agg["session_id"] = _to_int32(sim_agg["session_id"])
    sim_agg["item_id"] = _to_int32(sim_agg["item_id"])
    sim_agg["similarity_score"] = _to_float32(sim_agg["similarity_score"])
    return sim_agg


def _process_cf_chunk(cf_chunk: pd.DataFrame, artifacts: Dict[str, Any], top_k: int) -> Tuple[pd.DataFrame, int]:
    cf_chunk = cf_chunk[["session_id", "item_id", "score"]].copy()
    cf_chunk["session_id"] = _to_int32(cf_chunk["session_id"])
    cf_chunk["item_id"] = _to_int32(cf_chunk["item_id"])
    cf_chunk["score"] = _to_float32(cf_chunk["score"])

    cf_chunk = (
        cf_chunk.groupby(["session_id", "item_id"], observed=True, sort=False)["score"]
        .max()
        .reset_index()
    )

    session_ids = cf_chunk["session_id"].drop_duplicates().to_numpy(dtype=np.int32)
    sim_chunk = _sim_for_sessions(artifacts, session_ids)

    sim_std, cf_std = standardize_scores(sim_chunk, cf_chunk)
    merged = merge_sources(sim_std, cf_std)
    merged = add_source_features(merged)

    total_after_merge = len(merged)

    seen_subset = pd.DataFrame(columns=["session_id", "item_id"])
    if artifacts["sim_mode"] == "edge":
        # Use exact seen set for this session batch.
        seen_subset = _session_seed_items(session_ids, artifacts["interactions_df"]).rename(columns={"seed_item_id": "item_id"})
    else:
        # Session mode fallback: derive seen from global interactions through merge.
        interactions_df = artifacts["interactions_df"]
        seen_subset = pd.DataFrame({"session_id": session_ids}).merge(interactions_df, on="session_id", how="left")
        seen_subset = seen_subset.dropna(subset=["item_id"])

    seen_subset["session_id"] = _to_int32(seen_subset["session_id"])
    seen_subset["item_id"] = _to_int32(seen_subset["item_id"])
    seen_subset = seen_subset.drop_duplicates()

    merged = filter_seen_items(merged, seen_subset)
    merged = select_top_k(merged, top_k)

    cols = [
        "session_id",
        "item_id",
        "similarity_score",
        "cf_score",
        "source_item_similarity",
        "source_cf",
        "candidate_source_count",
        "final_rank",
    ]
    merged = merged[cols]

    merged["session_id"] = _to_int32(merged["session_id"])
    merged["item_id"] = _to_int32(merged["item_id"])
    merged["similarity_score"] = _to_float32(merged["similarity_score"])
    merged["cf_score"] = _to_float32(merged["cf_score"])
    merged["source_item_similarity"] = merged["source_item_similarity"].astype(np.int8)
    merged["source_cf"] = merged["source_cf"].astype(np.int8)
    merged["candidate_source_count"] = merged["candidate_source_count"].astype(np.int8)
    merged["final_rank"] = merged["final_rank"].astype(np.int16)

    return merged, total_after_merge


def main() -> None:
    """Run candidate fusion pipeline end-to-end."""
    start = time.perf_counter()
    np.random.seed(int(RANDOM_SEED))

    artifacts = load_candidates()
    cf_pf: pq.ParquetFile = artifacts["cf_pf"]

    sim_rows = 0
    if artifacts["sim_mode"] == "session":
        sim_rows = len(artifacts["sim_session_df"])
    else:
        sim_rows = len(artifacts["sim_edges_df"])

    total_before = int(sim_rows + cf_pf.metadata.num_rows)

    out_path = _path("candidates_merged")
    if out_path.exists():
        out_path.unlink()

    writer: Optional[pq.ParquetWriter] = None
    top_k = int(MERGE_TOP_K)
    batch_size = 1_000_000

    total_after_merge = 0
    total_final_rows = 0
    total_sessions = 0
    max_candidates = 0
    batches_done = 0

    carry = pd.DataFrame(columns=["session_id", "item_id", "score"])

    for rb in cf_pf.iter_batches(batch_size=batch_size, columns=["session_id", "item_id", "score"]):
        batches_done += 1
        batch_df = rb.to_pandas()
        if batch_df.empty:
            continue

        batch_df["session_id"] = _to_int32(batch_df["session_id"])
        batch_df["item_id"] = _to_int32(batch_df["item_id"])
        batch_df["score"] = _to_float32(batch_df["score"])

        if not carry.empty:
            batch_df = pd.concat([carry, batch_df], ignore_index=True)
            carry = pd.DataFrame(columns=["session_id", "item_id", "score"])

        batch_df = batch_df.sort_values(["session_id", "item_id"], ascending=[True, True])
        last_session = int(batch_df["session_id"].iloc[-1])

        process_mask = batch_df["session_id"] != last_session
        process_df = batch_df.loc[process_mask]
        carry = batch_df.loc[~process_mask]

        if process_df.empty:
            continue

        out_chunk, merged_rows = _process_cf_chunk(process_df, artifacts, top_k)
        total_after_merge += merged_rows

        if not out_chunk.empty:
            writer = _append_parquet_chunk(writer, out_chunk, out_path)
            total_final_rows += len(out_chunk)
            cnt = out_chunk.groupby("session_id", observed=True).size()
            total_sessions += len(cnt)
            if len(cnt) > 0:
                max_candidates = max(max_candidates, int(cnt.max()))

        LOGGER.info(
            "merge progress: batches=%d total_after_merge=%d total_final=%d elapsed_sec=%.1f",
            batches_done,
            total_after_merge,
            total_final_rows,
            time.perf_counter() - start,
        )

    if not carry.empty:
        out_chunk, merged_rows = _process_cf_chunk(carry, artifacts, top_k)
        total_after_merge += merged_rows

        if not out_chunk.empty:
            writer = _append_parquet_chunk(writer, out_chunk, out_path)
            total_final_rows += len(out_chunk)
            cnt = out_chunk.groupby("session_id", observed=True).size()
            total_sessions += len(cnt)
            if len(cnt) > 0:
                max_candidates = max(max_candidates, int(cnt.max()))

    if writer is not None:
        writer.close()
    else:
        # Write empty file with expected schema.
        empty = pd.DataFrame(
            columns=[
                "session_id",
                "item_id",
                "similarity_score",
                "cf_score",
                "source_item_similarity",
                "source_cf",
                "candidate_source_count",
                "final_rank",
            ]
        )
        save_candidates(empty)

    avg_candidates = total_final_rows / float(max(total_sessions, 1))

    LOGGER.info("total candidate rows before merge: %d", total_before)
    LOGGER.info("total candidate rows after merge: %d", total_after_merge)
    LOGGER.info("avg candidates per session: %.2f", avg_candidates)
    LOGGER.info("max candidates per session: %d", max_candidates)
    LOGGER.info("runtime_sec: %.2f", time.perf_counter() - start)


if __name__ == "__main__":
    main()
