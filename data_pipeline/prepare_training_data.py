"""Prepare model-ready training datasets for CSAO recommendation system."""

from __future__ import annotations

from inspect import signature
from pathlib import Path
from typing import Dict, Iterator, Tuple
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ============================================================
# CONFIG IMPORTS (Compatibility-safe)
# ============================================================

try:
    from config import (
        RANDOM_SEED,
        RAW_DATA_PATHS,
        PROCESSED_DATA_PATHS,
        NEGATIVE_SAMPLING_CONFIG,
        CHUNK_SIZES,
        LOGGER,
    )
except ImportError:
    from config import (  # type: ignore
        RANDOM_SEED,
        USERS_PATH,
        RESTAURANTS_PATH,
        ITEMS_PATH,
        SESSIONS_PATH,
        SESSION_ITEMS_PATH,
        TRAIN_PATH,
        VAL_PATH,
        TEST_PATH,
        SESSIONS_CHUNK_SIZE,
        MAX_POSITIVES_PER_SESSION,
    )
    from utils import get_logger  # type: ignore

    raw_dir = Path(USERS_PATH).parent
    processed_dir = Path(TRAIN_PATH).parent

    RAW_DATA_PATHS = {
        "users": USERS_PATH,
        "restaurants": RESTAURANTS_PATH,
        "items": ITEMS_PATH,
        "sessions": SESSIONS_PATH,
        "session_items": SESSION_ITEMS_PATH,
    }

    PROCESSED_DATA_PATHS = {
        "interactions": processed_dir / "interactions.parquet",
        "user_item_matrix": processed_dir / "user_item_matrix.parquet",
        "ranking_dataset": processed_dir / "ranking_dataset.parquet",
        "train": TRAIN_PATH,
        "val": VAL_PATH,
        "test": TEST_PATH,
        "positive_examples": processed_dir / "positive_examples.parquet",
        "negative_examples": processed_dir / "negative_examples.parquet",
    }

    NEGATIVE_SAMPLING_CONFIG = {
        "negatives_per_positive": 5,
        "popularity_power": 0.75,
        "same_restaurant_prob": 0.35,
        "category_sampling_prob": 0.45,
        "max_positives_per_session": MAX_POSITIVES_PER_SESSION,
    }

    CHUNK_SIZES = {
        "session_items": SESSIONS_CHUNK_SIZE,
        "interactions": SESSIONS_CHUNK_SIZE,
        "ranking": 300_000,
    }

    LOGGER = get_logger(__name__)


# ============================================================
# UTILS IMPORTS (Compatibility-safe)
# ============================================================

try:
    from utils import (
        set_random_seed,
        log_step,
        save_parquet,
        append_parquet,
        chunk_generator,
        memory_usage,
    )
except ImportError:
    from utils import set_random_seed, save_parquet, append_parquet, chunk_range  # type: ignore

    def log_step(msg: str) -> None:
        LOGGER.info(msg)

    def chunk_generator(total: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
        return chunk_range(total, chunk_size)

    def memory_usage(df: pd.DataFrame) -> float:
        return float(df.memory_usage(deep=True).sum() / (1024 ** 2))


# ============================================================
# HELPERS
# ============================================================


def _path(group: str, key: str) -> Path:
    mapping = RAW_DATA_PATHS if group == "raw" else PROCESSED_DATA_PATHS
    return Path(mapping[key])


def _append(df: pd.DataFrame, path: Path, is_first_chunk: bool) -> None:
    sig = signature(append_parquet)
    if "is_first_chunk" in sig.parameters:
        append_parquet(df, path, is_first_chunk=is_first_chunk)
    else:
        if is_first_chunk and path.exists():
            path.unlink()
        append_parquet(df, path)


def _chunk_size(key: str, default: int = 100_000) -> int:
    value = CHUNK_SIZES.get(key, default)
    return int(value) if int(value) > 0 else default


def _rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _cfg_int(name: str, default: int) -> int:
    val = os.getenv(name, "").strip()
    if not val:
        return int(default)
    try:
        return int(val)
    except Exception:
        return int(default)


def _cfg_float(name: str, default: float) -> float:
    val = os.getenv(name, "").strip()
    if not val:
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def _session_category_mode(interactions_path: Path, item_category_map: pd.Series) -> pd.Series:
    """Compute dominant category per session using chunked interaction scan."""
    pf = pq.ParquetFile(interactions_path)
    all_counts = []

    for batch in pf.iter_batches(batch_size=_chunk_size("interactions", 200_000)):
        x = batch.to_pandas()[["session_id", "item_id"]]
        x["candidate_category"] = x["item_id"].map(item_category_map)
        grp = x.groupby(["session_id", "candidate_category"], observed=True).size().rename("cnt")
        all_counts.append(grp)

    if not all_counts:
        return pd.Series(dtype=object)

    merged = pd.concat(all_counts).groupby(level=[0, 1]).sum().reset_index()
    merged = merged.sort_values(["session_id", "cnt"], ascending=[True, False])
    mode_df = merged.drop_duplicates("session_id", keep="first")
    return mode_df.set_index("session_id")["candidate_category"]


def _build_cart_key_index(interactions_path: Path) -> np.ndarray:
    """Build sorted unique session-item keys for fast cart membership checks."""
    pf = pq.ParquetFile(interactions_path)
    key_chunks = []

    for batch in pf.iter_batches(batch_size=_chunk_size("interactions", 300_000)):
        x = batch.to_pandas()[["session_id", "item_id"]]
        keys = (
            x["session_id"].to_numpy(dtype=np.uint64) << np.uint64(32)
        ) | x["item_id"].to_numpy(dtype=np.uint64)
        key_chunks.append(keys)

    if not key_chunks:
        return np.array([], dtype=np.uint64)

    return np.unique(np.concatenate(key_chunks))


def _in_cart_mask(session_ids: np.ndarray, item_ids: np.ndarray, cart_keys: np.ndarray) -> np.ndarray:
    if cart_keys.size == 0:
        return np.zeros(len(session_ids), dtype=bool)
    keys = (session_ids.astype(np.uint64) << np.uint64(32)) | item_ids.astype(np.uint64)
    idx = np.searchsorted(cart_keys, keys)
    valid = idx < cart_keys.size
    out = np.zeros(len(keys), dtype=bool)
    out[valid] = cart_keys[idx[valid]] == keys[valid]
    return out


# ============================================================
# CORE PIPELINE FUNCTIONS
# ============================================================


def load_static_tables() -> Dict[str, pd.DataFrame]:
    """Load static tables needed for feature engineering and sampling."""
    log_step("Loading static tables...")

    users = pd.read_parquet(
        _path("raw", "users"),
        columns=[
            "user_id",
            "user_segment",
            "order_frequency",
            "avg_order_value",
            "recency_days",
            "preferred_cuisine",
        ],
    )
    restaurants = pd.read_parquet(_path("raw", "restaurants"))
    items = pd.read_parquet(
        _path("raw", "items"),
        columns=["item_id", "restaurant_id", "category", "price", "popularity_score"],
    )
    sessions = pd.read_parquet(
        _path("raw", "sessions"),
        columns=[
            "session_id",
            "user_id",
            "restaurant_id",
            "cart_value",
            "item_count",
            "hour",
            "day_of_week",
            "meal_time",
            "timestamp",
        ],
    )

    log_step(
        f"Static memory MB | users={memory_usage(users):.1f}, items={memory_usage(items):.1f}, sessions={memory_usage(sessions):.1f}"
    )

    return {
        "users": users,
        "restaurants": restaurants,
        "items": items,
        "sessions": sessions,
    }


def build_interactions(static: Dict[str, pd.DataFrame]) -> Path:
    """Build interactions.parquet from session_items joined with session context."""
    log_step("Building interactions dataset...")

    sessions = static["sessions"][["session_id", "user_id", "timestamp"]].set_index("session_id")
    out_path = _path("processed", "interactions")

    if out_path.exists():
        out_path.unlink()

    source = pq.ParquetFile(_path("raw", "session_items"))
    first_chunk = True
    total_rows = 0

    for batch in source.iter_batches(batch_size=_chunk_size("session_items", 300_000)):
        chunk = batch.to_pandas()[["session_id", "item_id", "quantity", "add_sequence"]]

        sess_meta = sessions.reindex(chunk["session_id"].to_numpy())
        chunk["user_id"] = sess_meta["user_id"].to_numpy()
        chunk["timestamp"] = sess_meta["timestamp"].to_numpy()

        chunk["interaction_weight"] = (
            chunk["quantity"].astype(np.float32)
            + (0.2 / np.maximum(chunk["add_sequence"].astype(np.float32), 1.0))
        ).astype(np.float32)

        out = chunk[
            [
                "session_id",
                "user_id",
                "item_id",
                "quantity",
                "interaction_weight",
                "timestamp",
            ]
        ]

        _append(out, out_path, is_first_chunk=first_chunk)
        first_chunk = False

        total_rows += len(out)
        log_step(f"interactions progress: {total_rows:,} rows")

    log_step(f"interactions rows: {_rows(out_path):,}")
    return out_path


def build_user_item_matrix(interactions_path: Path, static: Dict[str, pd.DataFrame]) -> Path:
    """Aggregate interactions into user-item implicit matrix."""
    log_step("Building user-item matrix...")

    out_path = _path("processed", "user_item_matrix")
    if out_path.exists():
        out_path.unlink()

    pf = pq.ParquetFile(interactions_path)
    agg = pd.Series(dtype=np.float64)

    for batch in pf.iter_batches(batch_size=_chunk_size("interactions", 300_000)):
        x = batch.to_pandas()[["user_id", "item_id", "interaction_weight"]]
        keys = (
            x["user_id"].to_numpy(dtype=np.uint64) << np.uint64(32)
        ) | x["item_id"].to_numpy(dtype=np.uint64)

        chunk_sum = pd.Series(x["interaction_weight"].to_numpy(dtype=np.float64), index=keys).groupby(level=0).sum()
        agg = chunk_sum if agg.empty else agg.add(chunk_sum, fill_value=0.0)

    if agg.empty:
        save_parquet(pd.DataFrame(columns=["user_id", "item_id", "interaction_weight"]), out_path)
        return out_path

    keys = agg.index.to_numpy(dtype=np.uint64)
    matrix = pd.DataFrame(
        {
            "user_id": (keys >> np.uint64(32)).astype(np.int64),
            "item_id": (keys & np.uint64(0xFFFFFFFF)).astype(np.int64),
            "interaction_weight": agg.to_numpy(dtype=np.float32),
        }
    )

    save_parquet(matrix, out_path)

    users_n = static["users"]["user_id"].nunique()
    items_n = static["items"]["item_id"].nunique()
    density = len(matrix) / float(max(users_n * items_n, 1))
    log_step(f"user_item_matrix rows: {len(matrix):,}")
    log_step(f"matrix_density: {density:.8f}")
    return out_path


def generate_positive_examples(
    interactions_path: Path,
    static: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Select positive examples per session and prepare auxiliary lookup artifacts."""
    log_step("Generating positive examples...")

    max_pos_default = int(
        NEGATIVE_SAMPLING_CONFIG.get(
            "max_positives_per_session",
            NEGATIVE_SAMPLING_CONFIG.get("max_positives", 3),
        )
    )
    max_pos = _cfg_int("PREP_MAX_POSITIVES_PER_SESSION", max_pos_default)

    interactions = pd.read_parquet(interactions_path)
    interactions = interactions.sort_values(
        ["session_id", "interaction_weight", "quantity", "item_id"],
        ascending=[True, False, False, True],
    )
    positives = interactions.groupby("session_id", sort=False).head(max_pos).copy()
    positives.rename(columns={"item_id": "candidate_item_id"}, inplace=True)

    sessions = static["sessions"].set_index("session_id")
    users = static["users"].set_index("user_id")
    items = static["items"].set_index("item_id")

    positives = positives.join(
        sessions[
            [
                "user_id",
                "restaurant_id",
                "cart_value",
                "item_count",
                "hour",
                "day_of_week",
                "meal_time",
                "timestamp",
            ]
        ],
        on="session_id",
        rsuffix="_session",
    )

    if "user_id_session" in positives.columns:
        positives["user_id"] = positives["user_id_session"]
        positives.drop(columns=["user_id_session"], inplace=True)

    positives = positives.join(
        users[["user_segment", "order_frequency", "avg_order_value", "recency_days"]],
        on="user_id",
    )

    positives = positives.join(
        items[["price", "category", "popularity_score", "restaurant_id"]],
        on="candidate_item_id",
        rsuffix="_item",
    )

    positives.rename(
        columns={
            "price": "candidate_price",
            "category": "candidate_category",
            "popularity_score": "candidate_popularity",
            "restaurant_id_item": "candidate_restaurant_id",
        },
        inplace=True,
    )

    if "candidate_restaurant_id" not in positives.columns:
        positives["candidate_restaurant_id"] = positives["restaurant_id"]

    item_category_map = static["items"].set_index("item_id")["category"]
    session_category_mode = _session_category_mode(interactions_path, item_category_map)
    positives["session_primary_category"] = positives["session_id"].map(session_category_mode)

    cart_avg = positives["cart_value"].to_numpy(dtype=np.float32) / np.maximum(
        positives["item_count"].to_numpy(dtype=np.float32), 1.0
    )
    positives["price_diff_from_cart_avg"] = (
        positives["candidate_price"].to_numpy(dtype=np.float32) - cart_avg
    ).astype(np.float32)

    positives["category_match_flag"] = (
        positives["candidate_category"].astype(str)
        == positives["session_primary_category"].astype(str)
    ).astype(np.int8)

    positives["restaurant_match_flag"] = (
        positives["candidate_restaurant_id"].to_numpy(dtype=np.int64)
        == positives["restaurant_id"].to_numpy(dtype=np.int64)
    ).astype(np.int8)

    positives["label_added"] = np.int8(1)

    keep_cols = [
        "session_id",
        "user_id",
        "restaurant_id",
        "cart_value",
        "item_count",
        "hour",
        "day_of_week",
        "meal_time",
        "user_segment",
        "order_frequency",
        "avg_order_value",
        "recency_days",
        "candidate_item_id",
        "candidate_price",
        "candidate_category",
        "candidate_popularity",
        "price_diff_from_cart_avg",
        "category_match_flag",
        "restaurant_match_flag",
        "label_added",
        "timestamp",
    ]

    positives = positives[keep_cols].copy()
    cart_keys = _build_cart_key_index(interactions_path)

    log_step(f"positive_examples rows: {len(positives):,}")
    return positives, session_category_mode, cart_keys


def sample_negative_examples(
    positives: pd.DataFrame,
    static: Dict[str, pd.DataFrame],
    session_category_mode: pd.Series,
    cart_keys: np.ndarray,
) -> pd.DataFrame:
    """Sample negative candidates per positive with popularity, restaurant, and category bias."""
    log_step("Sampling negative examples...")

    neg_per_pos = _cfg_int("PREP_NEGATIVES_PER_POSITIVE", int(NEGATIVE_SAMPLING_CONFIG.get("negatives_per_positive", 5)))
    pop_power = _cfg_float("PREP_POPULARITY_POWER", float(NEGATIVE_SAMPLING_CONFIG.get("popularity_power", 0.75)))
    same_rest_p = _cfg_float("PREP_SAME_RESTAURANT_PROB", float(NEGATIVE_SAMPLING_CONFIG.get("same_restaurant_prob", 0.35)))
    cat_p = _cfg_float("PREP_CATEGORY_SAMPLING_PROB", float(NEGATIVE_SAMPLING_CONFIG.get("category_sampling_prob", 0.45)))

    items = static["items"].copy()
    items["w"] = np.power(np.clip(items["popularity_score"].to_numpy(dtype=np.float64), 1e-6, None), pop_power)

    global_ids = items["item_id"].to_numpy(dtype=np.int64)
    global_probs = items["w"].to_numpy(dtype=np.float64)
    global_probs = global_probs / global_probs.sum()

    by_rest = {
        int(rid): (
            g["item_id"].to_numpy(dtype=np.int64),
            (g["w"].to_numpy(dtype=np.float64) / g["w"].sum()) if g["w"].sum() > 0 else None,
        )
        for rid, g in items.groupby("restaurant_id", sort=False)
    }

    by_cat = {
        str(cat): (
            g["item_id"].to_numpy(dtype=np.int64),
            (g["w"].to_numpy(dtype=np.float64) / g["w"].sum()) if g["w"].sum() > 0 else None,
        )
        for cat, g in items.groupby("category", sort=False)
    }

    item_feat = (
        items.set_index("item_id")[["price", "category", "popularity_score", "restaurant_id"]]
        .rename(columns={"restaurant_id": "candidate_restaurant_id"})
    )

    n_pos = len(positives)
    if n_pos == 0 or neg_per_pos <= 0:
        return pd.DataFrame(columns=positives.columns)

    rep_idx = np.repeat(np.arange(n_pos, dtype=np.int64), neg_per_pos)
    neg = positives.iloc[rep_idx].copy()
    neg["label_added"] = np.int8(0)

    neg_session = neg["session_id"].to_numpy(dtype=np.int64)
    neg_rest = neg["restaurant_id"].to_numpy(dtype=np.int64)
    neg_primary_cat = neg["session_id"].map(session_category_mode).astype(str).to_numpy()

    sampled_ids = np.random.choice(global_ids, size=len(neg), p=global_probs)

    same_rest_mask = np.random.rand(len(neg)) < same_rest_p
    cat_mask = (~same_rest_mask) & (np.random.rand(len(neg)) < cat_p)

    if np.any(same_rest_mask):
        same_idx = np.where(same_rest_mask)[0]
        same_rest_vals = neg_rest[same_idx]
        for rid in np.unique(same_rest_vals):
            idx = same_idx[same_rest_vals == rid]
            pool = by_rest.get(int(rid))
            if pool is None:
                continue
            ids, probs = pool
            sampled_ids[idx] = np.random.choice(ids, size=len(idx), p=probs)

    if np.any(cat_mask):
        cat_idx = np.where(cat_mask)[0]
        cat_vals = neg_primary_cat[cat_idx]
        for cat in np.unique(cat_vals):
            idx = cat_idx[cat_vals == cat]
            pool = by_cat.get(str(cat))
            if pool is None:
                continue
            ids, probs = pool
            sampled_ids[idx] = np.random.choice(ids, size=len(idx), p=probs)

    # Ensure negatives are not in cart; retry unresolved rows a few times.
    unresolved = _in_cart_mask(neg_session, sampled_ids, cart_keys)
    retries = 0
    while np.any(unresolved) and retries < 5:
        sampled_ids[unresolved] = np.random.choice(global_ids, size=int(unresolved.sum()), p=global_probs)
        unresolved = _in_cart_mask(neg_session, sampled_ids, cart_keys)
        retries += 1

    neg["candidate_item_id"] = sampled_ids

    neg = neg.drop(
        columns=[
            "candidate_price",
            "candidate_category",
            "candidate_popularity",
            "price_diff_from_cart_avg",
            "category_match_flag",
            "restaurant_match_flag",
        ]
    )

    neg = neg.join(item_feat, on="candidate_item_id")
    neg.rename(
        columns={
            "price": "candidate_price",
            "category": "candidate_category",
            "popularity_score": "candidate_popularity",
        },
        inplace=True,
    )

    cart_avg = neg["cart_value"].to_numpy(dtype=np.float32) / np.maximum(
        neg["item_count"].to_numpy(dtype=np.float32), 1.0
    )
    neg["price_diff_from_cart_avg"] = (
        neg["candidate_price"].to_numpy(dtype=np.float32) - cart_avg
    ).astype(np.float32)

    neg["category_match_flag"] = (
        neg["candidate_category"].astype(str)
        == neg["session_id"].map(session_category_mode).astype(str)
    ).astype(np.int8)

    neg["restaurant_match_flag"] = (
        neg["candidate_restaurant_id"].to_numpy(dtype=np.int64)
        == neg["restaurant_id"].to_numpy(dtype=np.int64)
    ).astype(np.int8)

    # Keep one row per (session, candidate_item) to avoid duplicate easy negatives.
    neg = (
        neg.sort_values(["session_id", "candidate_item_id", "candidate_popularity"], ascending=[True, True, False])
        .drop_duplicates(subset=["session_id", "candidate_item_id"], keep="first")
        .reset_index(drop=True)
    )

    keep_cols = [
        "session_id",
        "user_id",
        "restaurant_id",
        "cart_value",
        "item_count",
        "hour",
        "day_of_week",
        "meal_time",
        "user_segment",
        "order_frequency",
        "avg_order_value",
        "recency_days",
        "candidate_item_id",
        "candidate_price",
        "candidate_category",
        "candidate_popularity",
        "price_diff_from_cart_avg",
        "category_match_flag",
        "restaurant_match_flag",
        "label_added",
        "timestamp",
    ]

    neg = neg[keep_cols]
    log_step(f"negative_examples rows: {len(neg):,}")
    return neg


def build_ranking_dataset(positives: pd.DataFrame, negatives: pd.DataFrame) -> Path:
    """Combine positives and negatives into ranking_dataset.parquet."""
    log_step("Building ranking dataset...")

    out_path = _path("processed", "ranking_dataset")
    if out_path.exists():
        out_path.unlink()

    first_chunk = True
    chunk = _chunk_size("ranking", 300_000)

    for start, size in chunk_generator(len(positives), chunk):
        part = positives.iloc[start : start + size]
        _append(part, out_path, is_first_chunk=first_chunk)
        first_chunk = False

    for start, size in chunk_generator(len(negatives), chunk):
        part = negatives.iloc[start : start + size]
        _append(part, out_path, is_first_chunk=False)

    log_step(f"ranking_dataset rows: {_rows(out_path):,}")
    return out_path


def time_split_dataset(ranking_path: Path) -> Tuple[Path, Path, Path]:
    """Time-based split: 70% train, 15% val, 15% test by timestamp."""
    log_step("Splitting ranking dataset into train/val/test...")

    train_path = _path("processed", "train")
    val_path = _path("processed", "val")
    test_path = _path("processed", "test")

    for p in (train_path, val_path, test_path):
        if p.exists():
            p.unlink()

    ts = pd.read_parquet(ranking_path, columns=["timestamp"])
    q70 = ts["timestamp"].quantile(0.70)
    q85 = ts["timestamp"].quantile(0.85)

    pf = pq.ParquetFile(ranking_path)
    first_train = True
    first_val = True
    first_test = True

    for batch in pf.iter_batches(batch_size=_chunk_size("ranking", 400_000)):
        x = batch.to_pandas()

        train_mask = x["timestamp"] <= q70
        val_mask = (x["timestamp"] > q70) & (x["timestamp"] <= q85)
        test_mask = x["timestamp"] > q85

        if train_mask.any():
            _append(x.loc[train_mask], train_path, is_first_chunk=first_train)
            first_train = False
        if val_mask.any():
            _append(x.loc[val_mask], val_path, is_first_chunk=first_val)
            first_val = False
        if test_mask.any():
            _append(x.loc[test_mask], test_path, is_first_chunk=first_test)
            first_test = False

    return train_path, val_path, test_path


def process_sessions() -> None:
    """Orchestrate full processing from raw events to split ranking datasets."""
    static = load_static_tables()

    interactions_path = build_interactions(static)
    matrix_path = build_user_item_matrix(interactions_path, static)

    positives, session_mode, cart_keys = generate_positive_examples(interactions_path, static)
    negatives = sample_negative_examples(positives, static, session_mode, cart_keys)

    ranking_path = build_ranking_dataset(positives, negatives)
    train_path, val_path, test_path = time_split_dataset(ranking_path)

    # Metrics
    total_interactions = _rows(interactions_path)
    users_n = static["users"]["user_id"].nunique()
    items_n = static["items"]["item_id"].nunique()
    matrix_rows = _rows(matrix_path)
    density = matrix_rows / float(max(users_n * items_n, 1))

    total_rank = _rows(ranking_path)
    pos_rate = (len(positives) / float(max(total_rank, 1)))
    session_n = max(ranking_path.exists() and pd.read_parquet(ranking_path, columns=["session_id"])["session_id"].nunique(), 1)
    avg_candidates = total_rank / float(session_n)

    train_n = _rows(train_path)
    val_n = _rows(val_path)
    test_n = _rows(test_path)

    log_step("=== Training Data Summary ===")
    log_step(f"total_interactions: {total_interactions:,}")
    log_step(f"matrix_density: {density:.8f}")
    log_step(f"positive_rate: {pos_rate:.4f}")
    log_step(f"avg_candidates_per_session: {avg_candidates:.2f}")
    log_step(f"train_rows: {train_n:,}")
    log_step(f"val_rows: {val_n:,}")
    log_step(f"test_rows: {test_n:,}")


def main() -> None:
    """Entry point for training data preparation pipeline."""
    set_random_seed(int(RANDOM_SEED))
    log_step("Starting prepare_training_data pipeline")
    process_sessions()
    log_step("prepare_training_data pipeline completed")


if __name__ == "__main__":
    main()
