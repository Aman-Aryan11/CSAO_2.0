"""
Production-grade synthetic raw data generation for CSAO recommendation system.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from inspect import signature
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ============================================================
# CONFIG IMPORTS (Compatibility-safe)
# ============================================================

try:
    from config import RANDOM_SEED, DATA_SCALE, RAW_DATA_PATHS, CHUNK_SIZES, LOGGER, SKIP_EXISTING
except ImportError:
    from config import (  # type: ignore
        RANDOM_SEED,
        DATA_SCALE,
        USERS_PATH,
        RESTAURANTS_PATH,
        ITEMS_PATH,
        SESSIONS_PATH,
        SESSION_ITEMS_PATH,
        SESSIONS_CHUNK_SIZE,
        SKIP_EXISTING,
    )
    from utils import get_logger  # type: ignore

    RAW_DATA_PATHS = {
        "users": USERS_PATH,
        "restaurants": RESTAURANTS_PATH,
        "items": ITEMS_PATH,
        "sessions": SESSIONS_PATH,
        "session_items": SESSION_ITEMS_PATH,
    }
    CHUNK_SIZES = {
        "users": 250_000,
        "restaurants": 50_000,
        "items": 100_000,
        "sessions": SESSIONS_CHUNK_SIZE,
        "session_items": max(100_000, SESSIONS_CHUNK_SIZE),
    }
    LOGGER = get_logger(__name__)

# ============================================================
# UTILS IMPORTS (Compatibility-safe)
# ============================================================

try:
    from utils import (
        set_random_seed,
        save_parquet,
        append_parquet,
        log_step,
        chunk_generator,
    )
except ImportError:
    from utils import set_random_seed, save_parquet, append_parquet, chunk_range  # type: ignore

    def log_step(message: str) -> None:
        LOGGER.info(message)

    def chunk_generator(total: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
        return chunk_range(total, chunk_size)


# ============================================================
# GLOBAL CONSTANTS
# ============================================================

USER_SEGMENTS = np.array(["budget", "premium", "frequent"], dtype=object)
USER_SEGMENT_PROBS = np.array([0.50, 0.20, 0.30], dtype=float)

CUISINES = np.array(
    [
        "north_indian",
        "south_indian",
        "chinese",
        "pizza",
        "burger",
        "dessert",
        "beverage",
        "biryani",
    ],
    dtype=object,
)
CUISINE_PROBS = np.array([0.18, 0.15, 0.12, 0.11, 0.10, 0.09, 0.12, 0.13], dtype=float)

CITIES = np.array(["city_a", "city_b", "city_c", "city_d"], dtype=object)
CITY_PROBS = np.array([0.38, 0.29, 0.19, 0.14], dtype=float)

ZONES = np.array([f"zone_{i}" for i in range(1, 9)], dtype=object)

MEAL_TIMES = np.array(["breakfast", "lunch", "snack", "dinner", "late_night"], dtype=object)
MEAL_PROBS = np.array([0.18, 0.30, 0.16, 0.27, 0.09], dtype=float)

CATEGORIES = np.array(["main", "snack", "dessert", "beverage", "side", "breakfast"], dtype=object)

CATEGORY_BY_MEAL: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "breakfast": (np.array(["breakfast", "beverage", "snack"], dtype=object), np.array([0.50, 0.30, 0.20], dtype=float)),
    "lunch": (np.array(["main", "side", "beverage", "dessert"], dtype=object), np.array([0.56, 0.16, 0.16, 0.12], dtype=float)),
    "snack": (np.array(["snack", "beverage", "dessert", "main"], dtype=object), np.array([0.54, 0.24, 0.14, 0.08], dtype=float)),
    "dinner": (np.array(["main", "side", "dessert", "beverage"], dtype=object), np.array([0.57, 0.16, 0.16, 0.11], dtype=float)),
    "late_night": (np.array(["snack", "beverage", "main", "dessert"], dtype=object), np.array([0.50, 0.22, 0.18, 0.10], dtype=float)),
}

SEGMENT_FREQ_MULT = {"budget": 1.0, "premium": 1.4, "frequent": 2.4}
SEGMENT_AOV_MULT = {"budget": 0.9, "premium": 1.5, "frequent": 1.15}
SEGMENT_ITEM_MULT = {"budget": 0.9, "premium": 1.2, "frequent": 1.3}

MEAL_ITEM_MULT = {
    "breakfast": 0.72,
    "lunch": 1.00,
    "snack": 0.86,
    "dinner": 1.14,
    "late_night": 0.80,
}

MEAL_CART_MULT = {
    "breakfast": 0.74,
    "lunch": 1.00,
    "snack": 0.84,
    "dinner": 1.18,
    "late_night": 0.86,
}

SEGMENT_PRICE_TEMP = {"budget": 160.0, "premium": 420.0, "frequent": 250.0}


# ============================================================
# INTERNAL HELPERS
# ============================================================


def _path(key: str) -> Path:
    return Path(RAW_DATA_PATHS[key])


def _chunk_size(key: str, default: int) -> int:
    value = CHUNK_SIZES.get(key, default)
    return int(value) if int(value) > 0 else default


def _skip_existing(path: Path) -> bool:
    return bool(SKIP_EXISTING and path.exists())


def _append(df: pd.DataFrame, path: Path, is_first_chunk: bool) -> None:
    sig = signature(append_parquet)
    if "is_first_chunk" in sig.parameters:
        append_parquet(df, path, is_first_chunk=is_first_chunk)
    else:
        if is_first_chunk and path.exists():
            path.unlink()
        append_parquet(df, path)


def _parquet_num_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _load_users() -> pd.DataFrame:
    return pd.read_parquet(_path("users"), columns=[
        "user_id", "order_frequency", "avg_order_value", "preferred_cuisine", "preferred_zone", "user_segment"
    ])


def _load_restaurants() -> pd.DataFrame:
    return pd.read_parquet(_path("restaurants"))


def _load_items() -> pd.DataFrame:
    return pd.read_parquet(_path("items"))


def _sample_hours(meal_times: np.ndarray) -> np.ndarray:
    hours = np.zeros(meal_times.shape[0], dtype=np.int16)

    idx = meal_times == "breakfast"
    hours[idx] = np.random.randint(7, 11, size=idx.sum())

    idx = meal_times == "lunch"
    hours[idx] = np.random.randint(12, 16, size=idx.sum())

    idx = meal_times == "snack"
    hours[idx] = np.random.randint(16, 19, size=idx.sum())

    idx = meal_times == "dinner"
    hours[idx] = np.random.randint(19, 23, size=idx.sum())

    idx = meal_times == "late_night"
    hours[idx] = np.random.choice(np.array([23, 0, 1], dtype=np.int16), size=idx.sum(), p=[0.5, 0.35, 0.15])

    return hours


def _choice_nonempty(values: np.ndarray, probs: np.ndarray | None, size: int) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=np.int64)
    if probs is None:
        return np.random.choice(values, size=size, replace=True)
    total = probs.sum()
    if total <= 0:
        return np.random.choice(values, size=size, replace=True)
    return np.random.choice(values, size=size, replace=True, p=probs / total)


# ============================================================
# DATA GENERATION
# ============================================================


def generate_users() -> None:
    """Generate users.parquet with segment-aware behavior and recency."""
    path = _path("users")
    if _skip_existing(path):
        log_step(f"Skipping users (exists): {path}")
        return

    log_step("Generating users...")
    n = int(DATA_SCALE.num_users)

    user_id = np.arange(1, n + 1, dtype=np.int64)
    user_segment = np.random.choice(USER_SEGMENTS, size=n, p=USER_SEGMENT_PROBS)

    base_freq = np.random.gamma(shape=2.2, scale=1.2, size=n)
    freq_mult = np.vectorize(SEGMENT_FREQ_MULT.get)(user_segment).astype(float)
    order_frequency = np.clip(base_freq * freq_mult, 0.2, None)

    base_aov = np.random.normal(loc=310.0, scale=85.0, size=n)
    aov_mult = np.vectorize(SEGMENT_AOV_MULT.get)(user_segment).astype(float)
    avg_order_value = np.clip(base_aov * aov_mult, 80.0, None)

    preferred_cuisine = np.random.choice(CUISINES, size=n, p=CUISINE_PROBS)
    preferred_zone = np.random.choice(ZONES, size=n)

    today = datetime.utcnow().date()
    signup_days_ago = np.random.randint(0, 730, size=n)
    signup_date = np.array([today - timedelta(days=int(d)) for d in signup_days_ago], dtype=object)

    activity = np.random.exponential(scale=22.0, size=n)
    recency_days = np.clip((activity / np.log1p(order_frequency + 1.0)).astype(np.int32), 0, 365)

    users = pd.DataFrame(
        {
            "user_id": user_id,
            "signup_date": pd.to_datetime(signup_date),
            "order_frequency": order_frequency.astype(np.float32),
            "avg_order_value": avg_order_value.astype(np.float32),
            "preferred_cuisine": preferred_cuisine,
            "preferred_zone": preferred_zone,
            "user_segment": user_segment,
            "recency_days": recency_days,
        }
    )

    save_parquet(users, path)
    log_step(f"users rows: {len(users):,}")


def generate_restaurants() -> None:
    """Generate restaurants.parquet."""
    path = _path("restaurants")
    if _skip_existing(path):
        log_step(f"Skipping restaurants (exists): {path}")
        return

    log_step("Generating restaurants...")
    n = int(DATA_SCALE.num_restaurants)

    restaurant_id = np.arange(1, n + 1, dtype=np.int64)
    cuisine = np.random.choice(CUISINES, size=n, p=CUISINE_PROBS)
    price_range = np.random.choice(np.array([1, 2, 3], dtype=np.int8), size=n, p=[0.42, 0.38, 0.20])
    rating = np.clip(np.random.normal(loc=4.1, scale=0.35, size=n), 2.5, 5.0)
    city = np.random.choice(CITIES, size=n, p=CITY_PROBS)
    zone = np.random.choice(ZONES, size=n)
    is_chain = np.random.binomial(1, 0.28, size=n).astype(bool)
    avg_order_volume = np.clip(np.random.gamma(shape=3.2, scale=18.0, size=n), 5.0, None)

    restaurants = pd.DataFrame(
        {
            "restaurant_id": restaurant_id,
            "cuisine": cuisine,
            "price_range": price_range,
            "rating": rating.astype(np.float32),
            "city": city,
            "zone": zone,
            "is_chain": is_chain,
            "avg_order_volume": avg_order_volume.astype(np.float32),
        }
    )

    save_parquet(restaurants, path)
    log_step(f"restaurants rows: {len(restaurants):,}")


def generate_items() -> None:
    """Generate items.parquet in chunks with popularity and price realism."""
    path = _path("items")
    if _skip_existing(path):
        log_step(f"Skipping items (exists): {path}")
        return

    log_step("Generating items...")

    restaurants = _load_restaurants()
    n = int(DATA_SCALE.num_items)
    chunk_size = _chunk_size("items", 100_000)

    rid = restaurants["restaurant_id"].to_numpy(dtype=np.int64)
    r_cuisine = restaurants["cuisine"].to_numpy()
    r_price_range = restaurants["price_range"].to_numpy(dtype=np.int8)
    r_vol = restaurants["avg_order_volume"].to_numpy(dtype=np.float64)

    rest_probs = r_vol / r_vol.sum()
    id_counter = 1
    first_chunk = True

    base_price_by_cat = {
        "main": 210.0,
        "snack": 110.0,
        "dessert": 130.0,
        "beverage": 90.0,
        "side": 80.0,
        "breakfast": 120.0,
    }

    cat_by_cuisine = {
        "north_indian": np.array(["main", "side", "dessert", "beverage"], dtype=object),
        "south_indian": np.array(["breakfast", "main", "beverage", "side"], dtype=object),
        "chinese": np.array(["main", "side", "beverage", "dessert"], dtype=object),
        "pizza": np.array(["main", "side", "beverage", "dessert"], dtype=object),
        "burger": np.array(["main", "side", "beverage", "dessert"], dtype=object),
        "dessert": np.array(["dessert", "beverage", "snack"], dtype=object),
        "beverage": np.array(["beverage", "snack", "dessert"], dtype=object),
        "biryani": np.array(["main", "side", "beverage", "dessert"], dtype=object),
    }

    cat_prob_by_cuisine = {
        "north_indian": np.array([0.50, 0.18, 0.18, 0.14]),
        "south_indian": np.array([0.30, 0.35, 0.20, 0.15]),
        "chinese": np.array([0.52, 0.20, 0.15, 0.13]),
        "pizza": np.array([0.55, 0.20, 0.15, 0.10]),
        "burger": np.array([0.50, 0.22, 0.18, 0.10]),
        "dessert": np.array([0.65, 0.25, 0.10]),
        "beverage": np.array([0.70, 0.20, 0.10]),
        "biryani": np.array([0.58, 0.18, 0.14, 0.10]),
    }

    for start, size in chunk_generator(n, chunk_size):
        chosen_idx = np.random.choice(np.arange(len(rid)), size=size, replace=True, p=rest_probs)
        chosen_restaurant = rid[chosen_idx]
        chosen_cuisine = r_cuisine[chosen_idx]
        chosen_price_range = r_price_range[chosen_idx]

        category = np.empty(size, dtype=object)
        for c in CUISINES:
            m = chosen_cuisine == c
            if not np.any(m):
                continue
            opts = cat_by_cuisine[c]
            probs = cat_prob_by_cuisine[c]
            category[m] = np.random.choice(opts, size=int(m.sum()), p=probs)

        price_base = np.vectorize(base_price_by_cat.get)(category).astype(float)
        price = np.random.lognormal(mean=np.log(price_base), sigma=0.25)
        price *= (0.80 + 0.22 * chosen_price_range)
        price = np.clip(price, 35.0, 1200.0)

        veg_prob = np.where(np.isin(category, ["dessert", "beverage", "breakfast", "side"]), 0.86, 0.58)
        veg_flag = np.random.binomial(1, veg_prob, size=size).astype(bool)

        pop_raw = np.random.lognormal(mean=0.1, sigma=0.8, size=size)
        popularity_score = (pop_raw - pop_raw.min()) / (pop_raw.max() - pop_raw.min() + 1e-9)

        df = pd.DataFrame(
            {
                "item_id": np.arange(id_counter, id_counter + size, dtype=np.int64),
                "restaurant_id": chosen_restaurant,
                "category": category,
                "price": price.astype(np.float32),
                "veg_flag": veg_flag,
                "popularity_score": popularity_score.astype(np.float32),
            }
        )

        _append(df, path, is_first_chunk=first_chunk)
        first_chunk = False
        id_counter += size

        log_step(f"items progress: {start + size:,}/{n:,}")

    log_step(f"items rows: {_parquet_num_rows(path):,}")


def generate_sessions() -> None:
    """Generate sessions.parquet in chunks with user-frequency and temporal realism."""
    path = _path("sessions")
    if _skip_existing(path):
        log_step(f"Skipping sessions (exists): {path}")
        return

    log_step("Generating sessions...")

    users = _load_users()
    restaurants = _load_restaurants()

    n = int(DATA_SCALE.num_sessions)
    chunk_size = _chunk_size("sessions", 100_000)

    user_ids = users["user_id"].to_numpy(dtype=np.int64)
    user_freq = users["order_frequency"].to_numpy(dtype=np.float64)
    user_aov = users["avg_order_value"].to_numpy(dtype=np.float64)
    user_pref_cuisine = users["preferred_cuisine"].to_numpy()
    user_pref_zone = users["preferred_zone"].to_numpy()
    user_segment = users["user_segment"].to_numpy()

    user_probs = user_freq / user_freq.sum()

    rest_ids = restaurants["restaurant_id"].to_numpy(dtype=np.int64)
    rest_city = restaurants["city"].to_numpy()
    rest_cuisine = restaurants["cuisine"].to_numpy()
    rest_zone = restaurants["zone"].to_numpy()
    rest_volume = restaurants["avg_order_volume"].to_numpy(dtype=np.float64)
    rest_probs = rest_volume / rest_volume.sum()

    idx_by_cuisine: Dict[str, np.ndarray] = {
        c: np.where(rest_cuisine == c)[0] for c in CUISINES
    }
    idx_by_zone: Dict[str, np.ndarray] = {
        z: np.where(rest_zone == z)[0] for z in ZONES
    }

    session_id = 1
    first_chunk = True
    now = datetime.utcnow()

    for start, size in chunk_generator(n, chunk_size):
        chosen_users_idx = np.random.choice(np.arange(len(user_ids)), size=size, replace=True, p=user_probs)

        sess_user_id = user_ids[chosen_users_idx]
        sess_user_seg = user_segment[chosen_users_idx]
        sess_user_pref_cuisine = user_pref_cuisine[chosen_users_idx]
        sess_user_pref_zone = user_pref_zone[chosen_users_idx]
        sess_user_aov = user_aov[chosen_users_idx]

        meal_time = np.random.choice(MEAL_TIMES, size=size, p=MEAL_PROBS)
        hour = _sample_hours(meal_time)
        day_of_week = np.random.randint(0, 7, size=size, dtype=np.int8)

        days_ago = np.random.randint(0, 90, size=size)
        minutes = np.random.randint(0, 60, size=size)
        base_time = np.array([now - timedelta(days=int(d)) for d in days_ago], dtype="datetime64[m]")
        timestamp = (base_time + hour.astype("timedelta64[h]") + minutes.astype("timedelta64[m]")).astype("datetime64[ns]")

        rest_idx = np.random.choice(np.arange(len(rest_ids)), size=size, replace=True, p=rest_probs)

        choose_pref_cuisine = np.random.rand(size) < 0.65
        choose_pref_zone = np.random.rand(size) < 0.45

        pref_cuisine_indices = np.where(choose_pref_cuisine)[0]
        for i in pref_cuisine_indices:
            pool = idx_by_cuisine.get(str(sess_user_pref_cuisine[i]))
            if pool is not None and pool.size > 0:
                rest_idx[i] = np.random.choice(pool)

        pref_zone_indices = np.where(choose_pref_zone)[0]
        for i in pref_zone_indices:
            pool = idx_by_zone.get(str(sess_user_pref_zone[i]))
            if pool is not None and pool.size > 0:
                rest_idx[i] = np.random.choice(pool)

        sess_restaurant_id = rest_ids[rest_idx]
        sess_city = rest_city[rest_idx]

        seg_item_mult = np.vectorize(SEGMENT_ITEM_MULT.get)(sess_user_seg).astype(float)
        meal_item_mult = np.vectorize(MEAL_ITEM_MULT.get)(meal_time).astype(float)

        lam = np.clip(DATA_SCALE.avg_items_per_session * seg_item_mult * meal_item_mult, 1.1, 16.0)
        item_count = np.maximum(1, np.random.poisson(lam=lam)).astype(np.int16)

        meal_cart_mult = np.vectorize(MEAL_CART_MULT.get)(meal_time).astype(float)
        noise = np.random.lognormal(mean=0.0, sigma=0.25, size=size)
        cart_value = np.clip(sess_user_aov * meal_cart_mult * noise * (0.76 + 0.10 * item_count), 80.0, 4000.0)

        df = pd.DataFrame(
            {
                "session_id": np.arange(session_id, session_id + size, dtype=np.int64),
                "user_id": sess_user_id,
                "restaurant_id": sess_restaurant_id,
                "timestamp": pd.to_datetime(timestamp),
                "hour": hour.astype(np.int8),
                "day_of_week": day_of_week,
                "meal_time": meal_time,
                "city": sess_city,
                "cart_value": cart_value.astype(np.float32),
                "item_count": item_count,
            }
        )

        _append(df, path, is_first_chunk=first_chunk)
        first_chunk = False
        session_id += size

        log_step(f"sessions progress: {start + size:,}/{n:,}")

    log_step(f"sessions rows: {_parquet_num_rows(path):,}")


def generate_session_items() -> None:
    """Generate session_items.parquet in chunks with co-occurrence and price sensitivity."""
    path = _path("session_items")
    if _skip_existing(path):
        log_step(f"Skipping session_items (exists): {path}")
        return

    log_step("Generating session_items...")

    items = _load_items()
    users = _load_users()

    item_ids = items["item_id"].to_numpy(dtype=np.int64)
    item_category = items["category"].to_numpy()
    item_price = items["price"].to_numpy(dtype=np.float64)
    item_restaurant = items["restaurant_id"].to_numpy(dtype=np.int64)
    item_pop = np.clip(items["popularity_score"].to_numpy(dtype=np.float64), 1e-6, None)

    user_seg_by_id = users.set_index("user_id")["user_segment"].to_dict()

    session_path = _path("sessions")
    pf = pq.ParquetFile(session_path)

    global_pool: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for seg in USER_SEGMENTS:
        for cat in CATEGORIES:
            m = item_category == cat
            ids = item_ids[m]
            if ids.size == 0:
                continue

            pop = item_pop[m]
            price = item_price[m]
            temp = SEGMENT_PRICE_TEMP[str(seg)]

            if seg == "premium":
                price_weight = np.sqrt(price)
            elif seg == "budget":
                price_weight = np.exp(-price / temp)
            else:
                price_weight = 1.0 / np.sqrt(price + 1.0)

            w = pop * price_weight
            w = np.clip(w, 1e-12, None)
            w = w / w.sum()
            global_pool[(str(seg), str(cat))] = (ids, w)

    rest_cat_pool: Dict[Tuple[int, str], np.ndarray] = {}
    grp = items.groupby(["restaurant_id", "category"], sort=False)["item_id"].apply(np.array)
    for key, val in grp.items():
        rest_cat_pool[(int(key[0]), str(key[1]))] = val

    first_chunk = True

    for batch in pf.iter_batches(batch_size=_chunk_size("sessions", 100_000)):
        s = batch.to_pandas()
        if s.empty:
            continue

        sess_id = s["session_id"].to_numpy(dtype=np.int64)
        sess_user = s["user_id"].to_numpy(dtype=np.int64)
        sess_rest = s["restaurant_id"].to_numpy(dtype=np.int64)
        sess_meal = s["meal_time"].to_numpy()
        sess_items = s["item_count"].to_numpy(dtype=np.int16)
        sess_cart = s["cart_value"].to_numpy(dtype=np.float64)

        counts = np.maximum(sess_items, 1).astype(np.int64)
        total_rows = int(counts.sum())
        if total_rows == 0:
            continue

        rep_idx = np.repeat(np.arange(len(sess_id), dtype=np.int64), counts)

        rep_session_id = sess_id[rep_idx]
        rep_restaurant = sess_rest[rep_idx]
        rep_item_count = counts[rep_idx]
        rep_meal = sess_meal[rep_idx]
        rep_cart = sess_cart[rep_idx]

        seg_arr = np.array([user_seg_by_id[int(uid)] for uid in sess_user], dtype=object)
        rep_segment = seg_arr[rep_idx]

        starts = np.repeat(np.cumsum(counts) - counts, counts)
        add_sequence = (np.arange(total_rows, dtype=np.int16) - starts + 1).astype(np.int16)

        category = np.empty(total_rows, dtype=object)
        for meal in MEAL_TIMES:
            m = rep_meal == meal
            if not np.any(m):
                continue
            opts, probs = CATEGORY_BY_MEAL[str(meal)]
            category[m] = np.random.choice(opts, size=int(m.sum()), p=probs)

        addon_prob = np.clip(0.12 + 0.06 * (rep_item_count - 1), 0.12, 0.62)
        addon_mask = (add_sequence > 1) & (np.random.rand(total_rows) < addon_prob)
        addon_cats = np.random.choice(
            np.array(["dessert", "snack", "beverage"], dtype=object),
            size=int(addon_mask.sum()),
            p=[0.45, 0.35, 0.20],
        )
        category[addon_mask] = addon_cats

        chosen_item_id = np.zeros(total_rows, dtype=np.int64)

        for seg in USER_SEGMENTS:
            for cat in CATEGORIES:
                m = (rep_segment == seg) & (category == cat)
                if not np.any(m):
                    continue
                pool = global_pool.get((str(seg), str(cat)))
                if pool is None:
                    continue
                ids, probs = pool
                chosen_item_id[m] = np.random.choice(ids, size=int(m.sum()), p=probs)

        unresolved = chosen_item_id == 0
        if np.any(unresolved):
            chosen_item_id[unresolved] = np.random.choice(item_ids, size=int(unresolved.sum()))

        same_rest_prob = np.clip(0.62 + 0.04 * (rep_item_count - 1), 0.62, 0.90)
        same_rest_mask = np.random.rand(total_rows) < same_rest_prob

        if np.any(same_rest_mask):
            tmp = pd.DataFrame(
                {
                    "idx": np.where(same_rest_mask)[0],
                    "restaurant_id": rep_restaurant[same_rest_mask],
                    "category": category[same_rest_mask],
                }
            )

            for (rid, cat), g in tmp.groupby(["restaurant_id", "category"], sort=False):
                pool = rest_cat_pool.get((int(rid), str(cat)))
                if pool is None or pool.size == 0:
                    continue
                idx = g["idx"].to_numpy(dtype=np.int64)
                chosen_item_id[idx] = np.random.choice(pool, size=len(idx), replace=True)

        chosen_price = item_price[chosen_item_id - 1]

        base_q = np.where(rep_segment == "premium", 0.18, np.where(rep_segment == "frequent", 0.14, 0.09))
        price_decay = np.exp(-chosen_price / 450.0)
        second_prob = np.clip(base_q * price_decay, 0.01, 0.35)

        quantity = (1 + (np.random.rand(total_rows) < second_prob).astype(np.int8)).astype(np.int8)
        third_prob = np.clip(0.15 * second_prob * (rep_cart / 600.0), 0.0, 0.08)
        quantity += (np.random.rand(total_rows) < third_prob).astype(np.int8)

        out = pd.DataFrame(
            {
                "session_id": rep_session_id,
                "item_id": chosen_item_id,
                "quantity": quantity,
                "add_sequence": add_sequence,
            }
        )

        _append(out, path, is_first_chunk=first_chunk)
        first_chunk = False

        log_step(f"session_items progress: +{len(out):,} rows")

    log_step(f"session_items rows: {_parquet_num_rows(path):,}")


# ============================================================
# VALIDATION
# ============================================================


def _log_validation_summary() -> None:
    users = _path("users")
    restaurants = _path("restaurants")
    items = _path("items")
    sessions = _path("sessions")
    session_items = _path("session_items")

    row_counts = {
        "users": _parquet_num_rows(users),
        "restaurants": _parquet_num_rows(restaurants),
        "items": _parquet_num_rows(items),
        "sessions": _parquet_num_rows(sessions),
        "session_items": _parquet_num_rows(session_items),
    }

    sess_df = pd.read_parquet(sessions, columns=["cart_value", "item_count"])
    avg_cart_value = float(sess_df["cart_value"].mean())
    avg_items_per_session = float(sess_df["item_count"].mean())

    user_df = pd.read_parquet(users, columns=["user_segment"])
    seg_dist = (user_df["user_segment"].value_counts(normalize=True) * 100.0).round(2)

    log_step("=== Validation Summary ===")
    for name, rows in row_counts.items():
        log_step(f"{name} rows: {rows:,}")
    log_step(f"avg_cart_value: {avg_cart_value:.2f}")
    log_step(f"avg_items_per_session: {avg_items_per_session:.2f}")
    log_step(f"segment_distribution_pct: {seg_dist.to_dict()}")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Run full synthetic raw data generation pipeline."""
    set_random_seed(int(RANDOM_SEED))
    log_step("Starting CSAO synthetic data generation")

    generate_users()
    generate_restaurants()
    generate_items()
    generate_sessions()
    generate_session_items()
    _log_validation_summary()

    log_step("Data generation completed")


if __name__ == "__main__":
    main()
