"""
config.py

Central configuration module for the data pipeline.
Handles paths, dataset scale, generation parameters,
and environment-level settings.

Import this file anywhere in the project:
    from config import settings
"""

from pathlib import Path
from dataclasses import dataclass
import os


# ============================================================
# ENVIRONMENT
# ============================================================

ENV = os.getenv("PIPELINE_ENV", "development")  # development | production
RANDOM_SEED = 42


# ============================================================
# PROJECT PATHS
# ============================================================

# data_pipeline/ directory
BASE_DIR = Path(__file__).resolve().parent

# CSAO_2.0 root directory
PROJECT_ROOT = BASE_DIR.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA FILE PATHS
# ============================================================

USERS_PATH = RAW_DATA_DIR / "users.parquet"
RESTAURANTS_PATH = RAW_DATA_DIR / "restaurants.parquet"
ITEMS_PATH = RAW_DATA_DIR / "items.parquet"
SESSIONS_PATH = RAW_DATA_DIR / "sessions.parquet"
SESSION_ITEMS_PATH = RAW_DATA_DIR / "session_items.parquet"

TRAIN_PATH = PROCESSED_DATA_DIR / "train.parquet"
VAL_PATH = PROCESSED_DATA_DIR / "val.parquet"
TEST_PATH = PROCESSED_DATA_DIR / "test.parquet"


# ============================================================
# DATA GENERATION SCALE
# ============================================================

@dataclass(frozen=True)
class DataScale:
    num_users: int
    num_restaurants: int
    num_items: int
    num_sessions: int
    avg_items_per_session: float


# Default scale (can be swapped per env)
DEFAULT_DATA_SCALE = DataScale(
    num_users=200_000,
    num_restaurants=3_000,
    num_items=30_000,
    num_sessions=2_000_000,
    avg_items_per_session=5.0,
)


# Smaller scale for quick debugging
DEV_DATA_SCALE = DataScale(
    num_users=10_000,
    num_restaurants=200,
    num_items=2_000,
    num_sessions=50_000,
    avg_items_per_session=4.0,
)

DATA_SCALE = DEV_DATA_SCALE if ENV == "development" else DEFAULT_DATA_SCALE


# ============================================================
# SESSION GENERATION SETTINGS
# ============================================================

SESSIONS_CHUNK_SIZE = 100_000
MAX_POSITIVES_PER_SESSION = 3


# ============================================================
# PIPELINE FLAGS
# ============================================================

SKIP_EXISTING = True
LOG_LEVEL = "INFO"


# ============================================================
# TRAINING SPLIT CONFIG
# ============================================================

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# ============================================================
# EXPORTABLE SETTINGS OBJECT
# ============================================================

@dataclass(frozen=True)
class Settings:
    env: str
    random_seed: int
    data_scale: DataScale
    sessions_chunk_size: int
    max_positives_per_session: int
    skip_existing: bool


settings = Settings(
    env=ENV,
    random_seed=RANDOM_SEED,
    data_scale=DATA_SCALE,
    sessions_chunk_size=SESSIONS_CHUNK_SIZE,
    max_positives_per_session=MAX_POSITIVES_PER_SESSION,
    skip_existing=SKIP_EXISTING,
)