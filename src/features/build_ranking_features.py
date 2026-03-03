"""Build ranking-ready feature datasets from merged retrieval candidates."""

from __future__ import annotations

import json
import logging
import os
import shutil
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
    from data_pipeline.config import (  # type: ignore
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        RANDOM_SEED,
        FEATURE_TOP_K,
        LOGGER,
        PROCESSED_DATA_PATHS,
    )
except Exception:
    try:
        from config import (  # type: ignore
            TRAIN_RATIO,
            VAL_RATIO,
            TEST_RATIO,
            RANDOM_SEED,
            FEATURE_TOP_K,
            LOGGER,
            PROCESSED_DATA_PATHS,
        )
    except Exception:
        TRAIN_RATIO = 0.7
        VAL_RATIO = 0.15
        TEST_RATIO = 0.15
        RANDOM_SEED = 42
        FEATURE_TOP_K = None
        LOGGER = logging.getLogger("build_ranking_features")
        if not LOGGER.handlers:
            LOGGER.setLevel(logging.INFO)
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            LOGGER.addHandler(h)
            LOGGER.propagate = False
        PROCESSED_DATA_PATHS = {}


# ============================================================
# PATH HELPERS
# ============================================================


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _processed_dir() -> Path:
    return _project_root() / "data_pipeline" / "data" / "processed"


def _raw_dir() -> Path:
    return _project_root() / "data_pipeline" / "data" / "raw"


def _featured_dir() -> Path:
    return _project_root() / "data_pipeline" / "data" / "featured"


def _candidates_dir() -> Path:
    return _project_root() / "data_pipeline" / "data" / "candidates"


def _path(key: str) -> Path:
    if key in PROCESSED_DATA_PATHS:
        p = Path(PROCESSED_DATA_PATHS[key])
        if key == "candidates_merged" and p.parent.name == "processed":
            return p.parent.parent / "candidates" / p.name
        return p

    if key == "candidates_merged":
        return _candidates_dir() / "candidates_merged.parquet"
    if key == "interactions":
        return _processed_dir() / "interactions.parquet"
    if key == "train_features":
        return _featured_dir() / "train_ranking_features.parquet"
    if key == "val_features":
        return _featured_dir() / "val_ranking_features.parquet"
    if key == "test_features":
        return _featured_dir() / "test_ranking_features.parquet"
    return _processed_dir() / f"{key}.parquet"


def _resolve_existing(paths: Tuple[Path, ...]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _to_int32(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(np.int32)


def _to_float32(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(np.float32)


def _fast_int32(s: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.astype(np.int32)
    return _to_int32(s)


def _fast_float32(s: pd.Series) -> pd.Series:
    if pd.api.types.is_float_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype):
        return s.astype(np.float32)
    return _to_float32(s)


def _memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024 ** 2))


def _log(msg: str, *args: Any) -> None:
    LOGGER.info(msg, *args)
    try:
        text = msg % args if args else msg
        print(text, flush=True)
    except Exception:
        pass


def _ckpt_dir() -> Path:
    return _featured_dir() / ".ranking_features_ckpt"


def _ckpt_state_path() -> Path:
    return _ckpt_dir() / "state.json"


def _ckpt_chunk_dir(split: str) -> Path:
    return _ckpt_dir() / split


def _load_state() -> Dict[str, Any]:
    p = _ckpt_state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    p = _ckpt_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, sort_keys=True))


def _append_chunk(split: str, batch_no: int, df: pd.DataFrame) -> None:
    if df.empty:
        return
    d = _ckpt_chunk_dir(split)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"batch_{batch_no:07d}.parquet"
    df.to_parquet(p, index=False)


def _merge_chunks_to_output(split: str, out_path: Path) -> int:
    d = _ckpt_chunk_dir(split)
    files = sorted(d.glob("batch_*.parquet")) if d.exists() else []

    if out_path.exists():
        out_path.unlink()

    if not files:
        pd.DataFrame().to_parquet(out_path, index=False)
        return 0

    writer: Optional[pq.ParquetWriter] = None
    rows = 0
    try:
        for f in files:
            t = pq.read_table(f)
            rows += t.num_rows
            if writer is None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(out_path, t.schema, compression="snappy")
            writer.write_table(t)
    finally:
        if writer is not None:
            writer.close()
    return rows


# ============================================================
# REQUIRED FUNCTIONS
# ============================================================


def load_merged_candidates() -> pq.ParquetFile:
    """Load merged candidate parquet file handle for chunked processing."""
    candidates_path = _resolve_existing(
        (
            _path("candidates_merged"),
            _processed_dir() / "candidates_merged.parquet",
            _candidates_dir() / "candidates_merged.parquet",
        )
    )
    if candidates_path is None:
        raise FileNotFoundError("candidates_merged.parquet not found in candidates/ or processed/")

    pf = pq.ParquetFile(candidates_path)
    _log("candidates_merged path: %s", candidates_path)
    _log("candidates_merged rows: %d", pf.metadata.num_rows)
    return pf


def compute_session_features(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute or load session-level features with time context."""
    sessions_path = _raw_dir() / "sessions.parquet"

    if sessions_path.exists():
        s = pd.read_parquet(
            sessions_path,
            columns=[
                "session_id",
                "user_id",
                "restaurant_id",
                "timestamp",
                "hour",
                "day_of_week",
                "meal_time",
                "cart_value",
                "item_count",
            ],
        )
        s["session_id"] = _to_int32(s["session_id"])
        s["user_id"] = _to_int32(s["user_id"])
        s["restaurant_id"] = _to_int32(s["restaurant_id"])
        s["hour"] = _to_int32(s["hour"])
        s["day_of_week"] = _to_int32(s["day_of_week"])
        s["cart_value"] = _to_float32(s["cart_value"])
        s["item_count"] = _to_int32(s["item_count"])
        s["timestamp"] = pd.to_datetime(s["timestamp"], errors="coerce")
        s["meal_time"] = s["meal_time"].astype("string").fillna("unknown")
        s["is_weekend"] = (s["day_of_week"] >= 5).astype(np.int8)
        return s

    # Fallback derived from interactions only.
    x = interactions_df[["session_id", "user_id", "item_id", "interaction_weight", "timestamp"]].copy()
    x["session_id"] = _to_int32(x["session_id"])
    x["user_id"] = _to_int32(x["user_id"])
    x["item_id"] = _to_int32(x["item_id"])
    x["interaction_weight"] = _to_float32(x["interaction_weight"])
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")

    agg = x.groupby("session_id", observed=True, sort=False).agg(
        user_id=("user_id", "first"),
        timestamp=("timestamp", "max"),
        item_count=("item_id", "nunique"),
        cart_value=("interaction_weight", "sum"),
    )
    agg = agg.reset_index()

    agg["restaurant_id"] = np.int32(0)
    agg["hour"] = agg["timestamp"].dt.hour.fillna(0).astype(np.int32)
    agg["day_of_week"] = agg["timestamp"].dt.dayofweek.fillna(0).astype(np.int32)

    hour = agg["hour"].to_numpy(dtype=np.int32)
    meal = np.where(
        (hour >= 7) & (hour <= 10),
        "breakfast",
        np.where(
            (hour >= 12) & (hour <= 15),
            "lunch",
            np.where(
                (hour >= 16) & (hour <= 18),
                "snack",
                np.where((hour >= 19) & (hour <= 22), "dinner", "late_night"),
            ),
        ),
    )
    agg["meal_time"] = pd.Series(meal, dtype="string")
    agg["is_weekend"] = (agg["day_of_week"] >= 5).astype(np.int8)
    agg["cart_value"] = _to_float32(agg["cart_value"])
    agg["item_count"] = _to_int32(agg["item_count"])

    return agg[
        [
            "session_id",
            "user_id",
            "restaurant_id",
            "timestamp",
            "hour",
            "day_of_week",
            "meal_time",
            "cart_value",
            "item_count",
            "is_weekend",
        ]
    ]


def compute_user_features(interactions_df: pd.DataFrame, session_df: pd.DataFrame) -> pd.DataFrame:
    """Compute or load user-level features."""
    users_path = _raw_dir() / "users.parquet"
    if users_path.exists():
        u = pd.read_parquet(
            users_path,
            columns=["user_id", "user_segment", "avg_order_value", "order_frequency", "recency_days"],
        )
        u["user_id"] = _to_int32(u["user_id"])
        u["user_segment"] = u["user_segment"].astype("string").fillna("unknown")
        u["avg_order_value"] = _to_float32(u["avg_order_value"])
        u["order_frequency"] = _to_float32(u["order_frequency"])
        u["recency_days"] = _to_float32(u["recency_days"])
        return u

    # Fallback derive from sessions/interactions.
    s = session_df[["user_id", "session_id", "cart_value", "timestamp"]].copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"], errors="coerce")

    u = s.groupby("user_id", observed=True, sort=False).agg(
        order_frequency=("session_id", "count"),
        avg_order_value=("cart_value", "mean"),
        last_session_ts=("timestamp", "max"),
    )
    u = u.reset_index()

    now_ts = pd.to_datetime(s["timestamp"].max(), errors="coerce")
    if pd.isna(now_ts):
        u["recency_days"] = np.float32(0)
    else:
        u["recency_days"] = ((now_ts - u["last_session_ts"]).dt.total_seconds() / 86400.0).fillna(0.0).astype(np.float32)

    u["user_segment"] = "unknown"
    u["avg_order_value"] = _to_float32(u["avg_order_value"])
    u["order_frequency"] = _to_float32(u["order_frequency"])
    u["recency_days"] = _to_float32(u["recency_days"])

    return u[["user_id", "user_segment", "avg_order_value", "order_frequency", "recency_days"]]


def compute_candidate_features(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute or load candidate/item metadata features."""
    items_path = _raw_dir() / "items.parquet"

    if items_path.exists():
        i = pd.read_parquet(
            items_path,
            columns=["item_id", "restaurant_id", "category", "price", "popularity_score"],
        )
        i["item_id"] = _to_int32(i["item_id"])
        i["restaurant_id"] = _to_int32(i["restaurant_id"])
        i["candidate_price"] = _to_float32(i["price"])
        i["candidate_category"] = i["category"].astype("string").fillna("unknown")
        i["candidate_popularity"] = _to_float32(i["popularity_score"])
        return i[["item_id", "restaurant_id", "candidate_category", "candidate_price", "candidate_popularity"]]

    # Fallback from interactions popularity only.
    pop = interactions_df[["item_id"]].copy()
    pop["item_id"] = _to_int32(pop["item_id"])
    pop = pop.groupby("item_id", observed=True, sort=False).size().rename("cnt").reset_index()
    cnt = pop["cnt"].to_numpy(dtype=np.float32)
    if cnt.size == 0:
        pop["candidate_popularity"] = np.float32(0.0)
    else:
        pop["candidate_popularity"] = (cnt / max(float(cnt.max()), 1.0)).astype(np.float32)

    pop["restaurant_id"] = np.int32(0)
    pop["candidate_category"] = "unknown"
    pop["candidate_price"] = np.float32(0.0)

    return pop[["item_id", "restaurant_id", "candidate_category", "candidate_price", "candidate_popularity"]]


def compute_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute session-user-candidate cross features."""
    cart_avg = df["cart_value"].to_numpy(dtype=np.float32) / np.maximum(df["item_count"].to_numpy(dtype=np.float32), 1.0)
    df["price_diff_from_cart_avg"] = (df["candidate_price"].to_numpy(dtype=np.float32) - cart_avg).astype(np.float32)

    # Approximate category match via user_segment-aware preference fallback.
    # If preferred cuisine is unavailable, use unknown comparison.
    if "preferred_cuisine" in df.columns:
        df["category_match_flag"] = (df["candidate_category"].astype(str) == df["preferred_cuisine"].astype(str)).astype(np.int8)
    else:
        df["category_match_flag"] = np.int8(0)

    df["restaurant_match_flag"] = (
        df["restaurant_id"].to_numpy(dtype=np.int32)
        == df["candidate_restaurant_id"].to_numpy(dtype=np.int32)
    ).astype(np.int8)

    df["source_count_ratio"] = (df["candidate_source_count"].to_numpy(dtype=np.float32) / 2.0).astype(np.float32)
    return df


def create_label(df: pd.DataFrame, purchased_keys: np.ndarray) -> pd.DataFrame:
    """Create binary label from session-item purchase membership."""
    keys = (df["session_id"].to_numpy(dtype=np.uint64) << np.uint64(32)) | df["item_id"].to_numpy(dtype=np.uint64)
    idx = np.searchsorted(purchased_keys, keys)
    valid = idx < purchased_keys.size
    label = np.zeros(len(df), dtype=np.int8)
    label[valid] = (purchased_keys[idx[valid]] == keys[valid]).astype(np.int8)
    df["label_added"] = label
    return df


def _add_missing_positive_candidates(
    candidates: pd.DataFrame,
    purchased_lookup: pd.DataFrame,
    session_anchor: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """Inject purchased session-item pairs missing from retrieval candidates."""
    if candidates.empty or purchased_lookup.empty:
        return candidates, 0

    session_ids = candidates["session_id"].drop_duplicates().to_numpy(dtype=np.int32)
    if session_ids.size == 0:
        return candidates, 0

    take_ids = purchased_lookup.index.intersection(session_ids, sort=False)
    if len(take_ids) == 0:
        return candidates, 0

    positives = purchased_lookup.loc[take_ids, ["session_id", "item_id"]].reset_index(drop=True)
    if positives.empty:
        return candidates, 0

    cand_keys = (
        (candidates["session_id"].to_numpy(dtype=np.uint64) << np.uint64(32))
        | candidates["item_id"].to_numpy(dtype=np.uint64)
    )
    cand_keys = np.unique(cand_keys)

    pos_keys = (
        (positives["session_id"].to_numpy(dtype=np.uint64) << np.uint64(32))
        | positives["item_id"].to_numpy(dtype=np.uint64)
    )
    idx = np.searchsorted(cand_keys, pos_keys)
    present = np.zeros(len(pos_keys), dtype=bool)
    valid = idx < cand_keys.size
    present[valid] = cand_keys[idx[valid]] == pos_keys[valid]
    positives = positives[~present]
    if positives.empty:
        return candidates, 0

    injected = positives.merge(session_anchor, on="session_id", how="left")
    injected["session_id"] = injected["session_id"].astype(np.int32)
    injected["item_id"] = injected["item_id"].astype(np.int32)
    injected["similarity_score"] = _to_float32(injected["similarity_score"])
    injected["cf_score"] = _to_float32(injected["cf_score"])
    injected["source_item_similarity"] = _to_int32(injected["source_item_similarity"]).astype(np.int8)
    injected["source_cf"] = _to_int32(injected["source_cf"]).astype(np.int8)
    injected["candidate_source_count"] = _to_int32(injected["candidate_source_count"]).astype(np.int8)
    injected["final_rank"] = _to_int32(injected["final_rank"]).astype(np.int16)

    out = pd.concat([candidates, injected], axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=["session_id", "item_id"], keep="first")
    return out, int(len(injected))


def time_split_dataset(session_df: pd.DataFrame) -> pd.DataFrame:
    """Create session-level time split map: train/val/test."""
    s = session_df[["session_id", "timestamp"]].copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"], errors="coerce")

    if s["timestamp"].notna().any():
        q_train = s["timestamp"].quantile(float(TRAIN_RATIO))
        q_val = s["timestamp"].quantile(float(TRAIN_RATIO + VAL_RATIO))

        split = np.where(
            s["timestamp"] <= q_train,
            "train",
            np.where(s["timestamp"] <= q_val, "val", "test"),
        )
    else:
        # deterministic fallback by session_id ordering
        s = s.sort_values("session_id")
        n = len(s)
        t_end = int(n * float(TRAIN_RATIO))
        v_end = int(n * float(TRAIN_RATIO + VAL_RATIO))
        split = np.empty(n, dtype=object)
        split[:t_end] = "train"
        split[t_end:v_end] = "val"
        split[v_end:] = "test"
        s["split"] = split
        return s[["session_id", "split"]]

    s["split"] = split
    return s[["session_id", "split"]]


def save_features(
    writer: Optional[pq.ParquetWriter],
    df: pd.DataFrame,
    out_path: Path,
) -> pq.ParquetWriter:
    """Append feature chunk to parquet using a persistent writer."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
    writer.write_table(table)
    return writer


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Run end-to-end ranking feature engineering pipeline."""
    start = time.perf_counter()
    np.random.seed(int(RANDOM_SEED))

    candidates_pf = load_merged_candidates()

    interactions_path = _path("interactions")
    if not interactions_path.exists():
        raise FileNotFoundError(f"interactions.parquet not found: {interactions_path}")

    interactions_df = pd.read_parquet(
        interactions_path,
        columns=["session_id", "item_id", "interaction_weight", "timestamp", "user_id"],
    )
    interactions_df["session_id"] = _to_int32(interactions_df["session_id"])
    interactions_df["item_id"] = _to_int32(interactions_df["item_id"])
    interactions_df["user_id"] = _to_int32(interactions_df["user_id"])
    _log("interactions loaded rows=%d mem_mb=%.2f", len(interactions_df), _memory_mb(interactions_df))

    _log("building session features...")
    session_df = compute_session_features(interactions_df)
    _log("session features ready rows=%d mem_mb=%.2f", len(session_df), _memory_mb(session_df))
    _log("building user features...")
    user_df = compute_user_features(interactions_df, session_df)
    _log("user features ready rows=%d mem_mb=%.2f", len(user_df), _memory_mb(user_df))
    _log("building candidate/item features...")
    item_df = compute_candidate_features(interactions_df)
    _log("candidate/item features ready rows=%d mem_mb=%.2f", len(item_df), _memory_mb(item_df))

    # Optional preferred cuisine for category match proxy.
    users_path = _raw_dir() / "users.parquet"
    if users_path.exists():
        user_pref = pd.read_parquet(users_path, columns=["user_id", "preferred_cuisine"])
        user_pref["user_id"] = _to_int32(user_pref["user_id"])
        user_pref["preferred_cuisine"] = user_pref["preferred_cuisine"].astype("string").fillna("unknown")
        user_df = user_df.merge(user_pref, on="user_id", how="left")
        user_df["preferred_cuisine"] = user_df["preferred_cuisine"].astype("string").fillna("unknown")

    split_map = time_split_dataset(session_df)

    # Build label membership index (session,item) from interactions.
    purchased = interactions_df[["session_id", "item_id"]].drop_duplicates()
    purchased["session_id"] = _to_int32(purchased["session_id"])
    purchased["item_id"] = _to_int32(purchased["item_id"])
    inject_scope = os.getenv("RANKING_FEATURES_INJECT_POS_SCOPE", "all").strip().lower()
    if inject_scope == "train":
        purchased_with_split = purchased.merge(split_map, on="session_id", how="left")
        purchased_work = purchased_with_split[purchased_with_split["split"] == "train"][["session_id", "item_id"]]
    else:
        purchased_work = purchased
    purchased_lookup = purchased_work.set_index("session_id", drop=False)
    purchased_keys = (
        (purchased["session_id"].to_numpy(dtype=np.uint64) << np.uint64(32))
        | purchased["item_id"].to_numpy(dtype=np.uint64)
    )
    purchased_keys = np.unique(purchased_keys)

    # Index small lookup tables for fast merge.
    session_lookup = session_df.set_index("session_id")
    user_lookup = user_df.set_index("user_id")
    item_lookup = item_df.rename(columns={"restaurant_id": "candidate_restaurant_id"}).set_index("item_id")
    split_lookup = split_map.set_index("session_id")

    train_path = _path("train_features")
    val_path = _path("val_features")
    test_path = _path("test_features")

    # Checkpoint/resume state
    state = _load_state()
    resumed = bool(state)
    last_completed_batch = int(state.get("last_completed_batch", 0))

    if resumed:
        _log("resuming from checkpoint: last_completed_batch=%d", last_completed_batch)
    else:
        # fresh run: clear old checkpoint and outputs
        if _ckpt_dir().exists():
            shutil.rmtree(_ckpt_dir())
        for p in (train_path, val_path, test_path):
            if p.exists():
                p.unlink()

    feature_top_k = None if FEATURE_TOP_K in (None, "", 0) else int(FEATURE_TOP_K)
    if feature_top_k is None:
        feature_top_k = 100
    else:
        feature_top_k = min(feature_top_k, 100)
    _log("using feature_top_k=%d", feature_top_k)
    _log("positive injection scope=%s pairs=%d", inject_scope, len(purchased_work))

    total_rows = int(state.get("total_rows", 0))
    pos_rows = int(state.get("pos_rows", 0))
    sessions_total = int(session_df["session_id"].nunique())
    min_price = float(state.get("min_price", np.inf))
    max_price = float(state.get("max_price", -np.inf))
    min_sim = float(state.get("min_sim", np.inf))
    max_sim = float(state.get("max_sim", -np.inf))
    min_cf = float(state.get("min_cf", np.inf))
    max_cf = float(state.get("max_cf", -np.inf))

    batch_size_env = os.getenv("RANKING_FEATURES_BATCH_SIZE", "").strip()
    batch_size = int(batch_size_env) if batch_size_env.isdigit() and int(batch_size_env) > 0 else 100_000
    max_batches_env = os.getenv("RANKING_FEATURES_MAX_BATCHES", "").strip()
    max_batches = int(max_batches_env) if max_batches_env.isdigit() and int(max_batches_env) > 0 else 0
    _log("using batch_size=%d", batch_size)
    if max_batches > 0:
        _log("debug max_batches=%d", max_batches)
    batches = int(state.get("batches", 0))
    processed_candidates = int(state.get("processed_candidates", 0))
    total_candidates = int(candidates_pf.metadata.num_rows)
    injected_positive_rows = int(state.get("injected_positive_rows", 0))

    cols_in = [
        "session_id",
        "item_id",
        "similarity_score",
        "cf_score",
        "source_item_similarity",
        "source_cf",
        "candidate_source_count",
        "final_rank",
    ]

    for rb in candidates_pf.iter_batches(batch_size=batch_size, columns=cols_in):
        if max_batches > 0 and batches >= max_batches:
            _log("stopping at max_batches=%d", max_batches)
            break
        batches += 1
        if batches <= last_completed_batch:
            continue
        c = rb.to_pandas()
        if c.empty:
            _save_state(
                {
                    "last_completed_batch": batches,
                    "batches": batches,
                    "processed_candidates": processed_candidates,
                    "total_rows": total_rows,
                    "pos_rows": pos_rows,
                    "min_price": min_price,
                    "max_price": max_price,
                    "min_sim": min_sim,
                    "max_sim": max_sim,
                    "min_cf": min_cf,
                    "max_cf": max_cf,
                    "feature_top_k": feature_top_k,
                    "batch_size": batch_size,
                    "injected_positive_rows": injected_positive_rows,
                }
            )
            continue
        processed_candidates += len(c)

        c["session_id"] = _fast_int32(c["session_id"])
        c["item_id"] = _fast_int32(c["item_id"])
        c["similarity_score"] = _fast_float32(c["similarity_score"])
        c["cf_score"] = _fast_float32(c["cf_score"])
        c["candidate_source_count"] = _fast_int32(c["candidate_source_count"]).astype(np.int8)
        c["final_rank"] = _fast_int32(c["final_rank"]).astype(np.int16)
        c["source_item_similarity"] = _fast_int32(c["source_item_similarity"]).astype(np.int8)
        c["source_cf"] = _fast_int32(c["source_cf"]).astype(np.int8)

        if feature_top_k is not None:
            c = c[c["final_rank"] <= int(feature_top_k)]
            if c.empty:
                continue
        session_anchor = (
            c.sort_values(["session_id", "final_rank", "item_id"], ascending=[True, True, True], kind="mergesort")
            .groupby("session_id", sort=False, as_index=False)
            .first()[[
                "session_id",
                "similarity_score",
                "cf_score",
                "source_item_similarity",
                "source_cf",
                "candidate_source_count",
                "final_rank",
            ]]
        )
        c, injected_now = _add_missing_positive_candidates(c, purchased_lookup, session_anchor)
        injected_positive_rows += int(injected_now)

        # Join session, user, candidate metadata.
        c = c.join(session_lookup, on="session_id", how="left", rsuffix="_session")
        if "user_id_session" in c.columns:
            c["user_id"] = c["user_id_session"]
            c = c.drop(columns=["user_id_session"])

        c = c.join(user_lookup, on="user_id", how="left", rsuffix="_user")
        c = c.join(item_lookup, on="item_id", how="left")
        c = c.join(split_lookup, on="session_id", how="left")

        # Fill missing defaults for robustness.
        c["cart_value"] = _to_float32(c.get("cart_value", 0.0))
        c["item_count"] = _to_int32(c.get("item_count", 0)).replace(0, 1)
        c["hour"] = _to_int32(c.get("hour", 0))
        c["day_of_week"] = _to_int32(c.get("day_of_week", 0))
        c["meal_time"] = c.get("meal_time", pd.Series(index=c.index, dtype="string")).astype("string").fillna("unknown")
        c["is_weekend"] = (c["day_of_week"] >= 5).astype(np.int8)

        c["user_segment"] = c.get("user_segment", pd.Series(index=c.index, dtype="string")).astype("string").fillna("unknown")
        c["avg_order_value"] = _to_float32(c.get("avg_order_value", 0.0))
        c["order_frequency"] = _to_float32(c.get("order_frequency", 0.0))
        c["recency_days"] = _to_float32(c.get("recency_days", 0.0))

        c["candidate_price"] = _to_float32(c.get("candidate_price", 0.0))
        c["candidate_category"] = c.get("candidate_category", pd.Series(index=c.index, dtype="string")).astype("string").fillna("unknown")
        c["candidate_popularity"] = _to_float32(c.get("candidate_popularity", 0.0))
        c["candidate_restaurant_id"] = _to_int32(c.get("candidate_restaurant_id", 0))
        c["restaurant_id"] = _to_int32(c.get("restaurant_id", 0))

        c = create_label(c, purchased_keys)
        c["label"] = c["label_added"].astype(np.int8)
        c = compute_cross_features(c)

        c["split"] = c["split"].astype("string").fillna("train")

        out_cols = [
            "session_id",
            "user_id",
            "item_id",
            "cart_value",
            "item_count",
            "hour",
            "day_of_week",
            "meal_time",
            "is_weekend",
            "user_segment",
            "avg_order_value",
            "order_frequency",
            "recency_days",
            "candidate_price",
            "candidate_category",
            "candidate_popularity",
            "similarity_score",
            "cf_score",
            "price_diff_from_cart_avg",
            "category_match_flag",
            "restaurant_match_flag",
            "source_count_ratio",
            "label_added",
            "label",
            "timestamp",
            "final_rank",
            "candidate_source_count",
        ]

        c = c[out_cols + ["split"]]

        # update stats
        total_rows += len(c)
        pos_rows += int(c["label_added"].sum())

        if len(c) > 0:
            min_price = min(min_price, float(c["candidate_price"].min()))
            max_price = max(max_price, float(c["candidate_price"].max()))
            min_sim = min(min_sim, float(c["similarity_score"].min()))
            max_sim = max(max_sim, float(c["similarity_score"].max()))
            min_cf = min(min_cf, float(c["cf_score"].min()))
            max_cf = max(max_cf, float(c["cf_score"].max()))

        train_chunk = c[c["split"] == "train"].drop(columns=["split"])
        val_chunk = c[c["split"] == "val"].drop(columns=["split"])
        test_chunk = c[c["split"] == "test"].drop(columns=["split"])

        _append_chunk("train", batches, train_chunk)
        _append_chunk("val", batches, val_chunk)
        _append_chunk("test", batches, test_chunk)

        _save_state(
            {
                "last_completed_batch": batches,
                "batches": batches,
                "processed_candidates": processed_candidates,
                "total_rows": total_rows,
                "pos_rows": pos_rows,
                "min_price": min_price,
                "max_price": max_price,
                "min_sim": min_sim,
                "max_sim": max_sim,
                "min_cf": min_cf,
                "max_cf": max_cf,
                "feature_top_k": feature_top_k,
                "batch_size": batch_size,
                "injected_positive_rows": injected_positive_rows,
            }
        )

        elapsed = time.perf_counter() - start
        rate = processed_candidates / max(elapsed, 1e-6)
        remaining = max(total_candidates - processed_candidates, 0)
        eta = remaining / max(rate, 1e-6)
        _log(
            "progress batch=%d processed=%d/%d (%.2f%%) feature_rows=%d pos_rate=%.4f mem_mb=%.2f rows_per_sec=%.0f eta_sec=%.0f",
            batches,
            processed_candidates,
            total_candidates,
            100.0 * processed_candidates / max(total_candidates, 1),
            total_rows,
            pos_rows / max(total_rows, 1),
            _memory_mb(c),
            rate,
            eta,
        )

    train_rows = _merge_chunks_to_output("train", train_path)
    val_rows = _merge_chunks_to_output("val", val_path)
    test_rows = _merge_chunks_to_output("test", test_path)

    # cleanup checkpoint only after successful final merge
    if _ckpt_dir().exists():
        shutil.rmtree(_ckpt_dir())

    avg_candidates = total_rows / float(max(sessions_total, 1))
    pos_rate = pos_rows / float(max(total_rows, 1))

    _log("total rows: %d", total_rows)
    _log("positive rate: %.6f", pos_rate)
    _log("avg candidates per session: %.2f", avg_candidates)
    _log("candidate_price range: [%.4f, %.4f]", 0.0 if min_price is np.inf else min_price, 0.0 if max_price is -np.inf else max_price)
    _log("similarity_score range: [%.6f, %.6f]", 0.0 if min_sim is np.inf else min_sim, 0.0 if max_sim is -np.inf else max_sim)
    _log("cf_score range: [%.6f, %.6f]", 0.0 if min_cf is np.inf else min_cf, 0.0 if max_cf is -np.inf else max_cf)
    _log("train output: %s", train_path)
    _log("val output: %s", val_path)
    _log("test output: %s", test_path)
    _log("train rows: %d", train_rows)
    _log("val rows: %d", val_rows)
    _log("test rows: %d", test_rows)
    _log("injected positive candidates: %d", injected_positive_rows)
    _log("runtime_sec: %.2f", time.perf_counter() - start)


if __name__ == "__main__":
    main()
