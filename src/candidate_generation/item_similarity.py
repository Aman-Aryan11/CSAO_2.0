"""Item-to-item similarity retrieval for CSAO add-on recommendation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ============================================================
# CONFIG (compatibility-safe)
# ============================================================

try:
    from data_pipeline.config import (  # type: ignore
        ITEM_SIM_TOP_K,
        SIMILARITY_METRICS,
        LOGGER,
        RANDOM_SEED,
        PROCESSED_DATA_PATHS,
    )
except Exception:
    try:
        from config import (  # type: ignore
            ITEM_SIM_TOP_K,
            SIMILARITY_METRICS,
            LOGGER,
            RANDOM_SEED,
            PROCESSED_DATA_PATHS,
        )
    except Exception:
        RANDOM_SEED = 42
        ITEM_SIM_TOP_K = 50
        SIMILARITY_METRICS = ["cosine", "lift", "conditional_probability"]
        LOGGER = logging.getLogger("item_similarity")
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
            "interactions": processed_dir / "interactions.parquet",
            "item_similarity": candidates_dir / "item_similarity.parquet",
        }


# ============================================================
# HELPERS
# ============================================================


def _processed_path(key: str) -> Path:
    if key == "item_similarity":
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
        return interactions_base.parent / "candidates" / "item_similarity.parquet"
    if key in PROCESSED_DATA_PATHS:
        return Path(PROCESSED_DATA_PATHS[key])
    base = Path(PROCESSED_DATA_PATHS.get("interactions", Path(__file__).resolve().parents[2] / "data_pipeline" / "data" / "processed" / "interactions.parquet")).parent
    return base / f"{key}.parquet"


def _log_memory(df: pd.DataFrame, label: str) -> None:
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    LOGGER.info("%s memory: %.2f MB", label, mb)


def _normalize_metrics(metrics: List[str]) -> List[str]:
    supported = {"cosine", "lift", "conditional_probability"}
    out = []
    for m in metrics:
        mm = str(m).strip().lower()
        if mm in supported:
            out.append(mm)
    return out or ["cosine", "lift", "conditional_probability"]


# ============================================================
# REQUIRED FUNCTIONS
# ============================================================


def load_interactions() -> pd.DataFrame:
    """Load minimal interactions needed for item similarity computation."""
    path = _processed_path("interactions")
    if not path.exists():
        raise FileNotFoundError(f"interactions file not found: {path}")

    LOGGER.info("Loading interactions from %s", path)
    df = pd.read_parquet(path, columns=["session_id", "item_id", "interaction_weight", "timestamp"])
    df = df.dropna(subset=["session_id", "item_id"]).copy()
    df["session_id"] = df["session_id"].astype(np.int64)
    df["item_id"] = df["item_id"].astype(np.int64)
    _log_memory(df, "interactions")
    return df


def compute_item_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-item session frequency used by similarity metrics."""
    dedup = df[["session_id", "item_id"]].drop_duplicates()
    freq = dedup.groupby("item_id", observed=True).size().rename("item_freq").reset_index()
    freq["item_freq"] = freq["item_freq"].astype(np.int64)
    LOGGER.info("Computed item frequencies for %d items", len(freq))
    return freq


def compute_cooccurrence(df: pd.DataFrame) -> Dict[str, object]:
    """Compute item co-occurrence counts via sparse session-item matrix multiplication."""
    LOGGER.info("Computing item co-occurrence...")

    dedup = df[["session_id", "item_id"]].drop_duplicates()
    n_sessions = int(dedup["session_id"].nunique())

    session_codes, session_uniques = pd.factorize(dedup["session_id"], sort=True)
    item_codes, item_uniques = pd.factorize(dedup["item_id"], sort=True)

    try:
        from scipy.sparse import coo_matrix  # type: ignore

        data = np.ones(len(dedup), dtype=np.uint8)
        mat = coo_matrix(
            (data, (session_codes, item_codes)),
            shape=(len(session_uniques), len(item_uniques)),
            dtype=np.uint8,
        ).tocsr()

        item_item = (mat.T @ mat).tocoo()
        mask = item_item.row != item_item.col

        pairs = pd.DataFrame(
            {
                "item_id": item_uniques[item_item.row[mask]].astype(np.int64),
                "similar_item_id": item_uniques[item_item.col[mask]].astype(np.int64),
                "cooccurrence_count": item_item.data[mask].astype(np.int64),
            }
        )
    except Exception:
        LOGGER.warning("scipy not available; using pandas fallback for co-occurrence")
        pairs = dedup.merge(dedup, on="session_id", suffixes=("", "_r"))
        pairs = pairs[pairs["item_id"] != pairs["item_id_r"]]
        pairs = (
            pairs.groupby(["item_id", "item_id_r"], observed=True)
            .size()
            .rename("cooccurrence_count")
            .reset_index()
            .rename(columns={"item_id_r": "similar_item_id"})
        )
        pairs["item_id"] = pairs["item_id"].astype(np.int64)
        pairs["similar_item_id"] = pairs["similar_item_id"].astype(np.int64)
        pairs["cooccurrence_count"] = pairs["cooccurrence_count"].astype(np.int64)

    LOGGER.info("Co-occurrence pairs: %d", len(pairs))
    _log_memory(pairs, "cooccurrence_pairs")
    return {"pairs": pairs, "n_sessions": n_sessions}


def compute_similarity(co_matrix: Dict[str, object], freq_df: pd.DataFrame) -> pd.DataFrame:
    """Compute similarity scores for configured metrics."""
    pairs = co_matrix["pairs"]  # type: ignore[index]
    n_sessions = float(co_matrix["n_sessions"])  # type: ignore[index]

    f = freq_df.rename(columns={"item_freq": "freq_i"})
    sim = pairs.merge(f, on="item_id", how="left")
    f2 = freq_df.rename(columns={"item_id": "similar_item_id", "item_freq": "freq_j"})
    sim = sim.merge(f2, on="similar_item_id", how="left")

    sim["freq_i"] = sim["freq_i"].fillna(1).astype(np.float64)
    sim["freq_j"] = sim["freq_j"].fillna(1).astype(np.float64)
    c = sim["cooccurrence_count"].to_numpy(dtype=np.float64)
    fi = sim["freq_i"].to_numpy(dtype=np.float64)
    fj = sim["freq_j"].to_numpy(dtype=np.float64)

    metrics = _normalize_metrics(list(SIMILARITY_METRICS))
    out_frames = []

    if "cosine" in metrics:
        cosine = c / np.sqrt(fi * fj)
        x = sim[["item_id", "similar_item_id", "cooccurrence_count"]].copy()
        x["similarity_score"] = cosine.astype(np.float32)
        x["similarity_type"] = "cosine"
        out_frames.append(x)

    if "lift" in metrics:
        p_ij = c / max(n_sessions, 1.0)
        p_i = fi / max(n_sessions, 1.0)
        p_j = fj / max(n_sessions, 1.0)
        lift = p_ij / np.maximum(p_i * p_j, 1e-12)
        x = sim[["item_id", "similar_item_id", "cooccurrence_count"]].copy()
        x["similarity_score"] = lift.astype(np.float32)
        x["similarity_type"] = "lift"
        out_frames.append(x)

    if "conditional_probability" in metrics:
        cp = c / np.maximum(fi, 1e-12)
        x = sim[["item_id", "similar_item_id", "cooccurrence_count"]].copy()
        x["similarity_score"] = cp.astype(np.float32)
        x["similarity_type"] = "conditional_probability"
        out_frames.append(x)

    if not out_frames:
        return pd.DataFrame(
            columns=[
                "item_id",
                "similar_item_id",
                "similarity_score",
                "cooccurrence_count",
                "similarity_type",
                "similarity_norm",
            ]
        )

    sim_df = pd.concat(out_frames, ignore_index=True)
    sim_df = sim_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["similarity_score"])
    return sim_df


def select_top_k(sim_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep top-K neighbors per item per similarity type and add normalized score."""
    if sim_df.empty:
        return sim_df

    sim_df = sim_df.sort_values(
        ["similarity_type", "item_id", "similarity_score", "cooccurrence_count", "similar_item_id"],
        ascending=[True, True, False, False, True],
    )

    topk = (
        sim_df.groupby(["similarity_type", "item_id"], observed=True, sort=False)
        .head(int(k))
        .copy()
    )

    max_per_item = topk.groupby(["similarity_type", "item_id"], observed=True)["similarity_score"].transform("max")
    topk["similarity_norm"] = (topk["similarity_score"] / np.maximum(max_per_item, 1e-12)).astype(np.float32)

    cols = [
        "item_id",
        "similar_item_id",
        "similarity_score",
        "cooccurrence_count",
        "similarity_type",
        "similarity_norm",
    ]
    return topk[cols]


def save_similarity(sim_df: pd.DataFrame) -> Path:
    """Persist retrieval-ready item similarity edges to parquet."""
    out_path = _processed_path("item_similarity")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_parquet(out_path, index=False)
    LOGGER.info("Saved item similarity to %s", out_path)
    return out_path


def main() -> None:
    """Run item-to-item similarity computation and save top-K edges."""
    start = time.perf_counter()
    np.random.seed(int(RANDOM_SEED))

    df = load_interactions()
    freq_df = compute_item_frequencies(df)
    co = compute_cooccurrence(df)

    sim_df = compute_similarity(co, freq_df)
    sim_df = select_top_k(sim_df, k=int(ITEM_SIM_TOP_K))
    save_similarity(sim_df)

    total_pairs = int(len(sim_df))
    avg_neighbors = (
        sim_df.groupby(["similarity_type", "item_id"], observed=True).size().mean()
        if not sim_df.empty
        else 0.0
    )
    max_sim = float(sim_df["similarity_score"].max()) if not sim_df.empty else 0.0
    runtime = time.perf_counter() - start

    LOGGER.info("total item pairs: %d", total_pairs)
    LOGGER.info("avg neighbors per item: %.2f", float(avg_neighbors))
    LOGGER.info("max similarity: %.6f", max_sim)
    LOGGER.info("runtime_sec: %.2f", runtime)


if __name__ == "__main__":
    main()
