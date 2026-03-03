"""
utils.py

Reusable utilities for CSAO data pipeline.
Designed for large-scale synthetic data generation with:
- Reproducibility
- Robust logging
- Memory-efficient parquet operations
- Dataset validation helpers
"""

from __future__ import annotations

import logging
import os
import sys
import random
from pathlib import Path
from typing import Iterable, Generator, Optional

import numpy as np
import pandas as pd


# ============================================================
# 🔹 REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Set global random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_random_seed(seed: int = 42) -> None:
    """
    Backward-compatible alias expected by generate_data.py.
    """
    set_seed(seed)


# ============================================================
# 🔹 LOGGING
# ============================================================

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger.

    Ensures no duplicate handlers if called multiple times.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

    return logger


# ============================================================
# 🔹 PARQUET HELPERS (Memory Efficient)
# ============================================================

def append_parquet(
    df: pd.DataFrame,
    path: Path,
    is_first_chunk: bool,
    compression: str = "snappy",
) -> None:
    """
    Append DataFrame to parquet using fastparquet.

    Handles string dtype compatibility and schema consistency.
    """

    from pandas.api.types import is_string_dtype

    # Convert Arrow string dtypes to object (fastparquet compatibility)
    if any(is_string_dtype(dtype) for dtype in df.dtypes):
        df = df.copy()
        for col in df.columns:
            if is_string_dtype(df[col].dtype):
                df[col] = df[col].astype("object")

    engine = "fastparquet"

    if is_first_chunk or not path.exists():
        df.to_parquet(path, engine=engine, compression=compression, index=False)
    else:
        df.to_parquet(
            path,
            engine=engine,
            compression=compression,
            index=False,
            append=True,
        )


def parquet_row_count(path: Path) -> int:
    """
    Efficiently count rows in a parquet file.
    """
    import fastparquet
    pf = fastparquet.ParquetFile(path)
    return pf.count()


def parquet_column_mean(path: Path, column: str) -> float:
    """
    Compute mean of a column without loading entire file.
    """
    import fastparquet

    pf = fastparquet.ParquetFile(path)
    total_sum = 0.0
    total_count = 0

    for rg in pf.iter_row_groups(columns=[column]):
        values = rg[column].to_numpy()
        total_sum += values.sum()
        total_count += len(values)

    return total_sum / total_count if total_count > 0 else float("nan")


# ============================================================
# 🔹 CHUNKING UTILITIES
# ============================================================

def chunk_range(
    total: int,
    chunk_size: int
) -> Generator[tuple[int, int], None, None]:
    """
    Yield (start, size) pairs for chunked processing.
    """
    start = 0
    while start < total:
        size = min(chunk_size, total - start)
        yield start, size
        start += size


def chunk_iterable(iterable: Iterable, chunk_size: int):
    """
    Yield chunks from any iterable.
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# ============================================================
# 🔹 MEMORY & DEBUGGING
# ============================================================

def memory_usage_mb(df: pd.DataFrame) -> float:
    """
    Return memory usage of DataFrame in MB.
    """
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def print_memory_usage(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print memory usage for debugging.
    """
    mem = memory_usage_mb(df)
    print(f"{name} memory usage: {mem:.2f} MB")


# ============================================================
# 🔹 VALIDATION HELPERS
# ============================================================

def dataset_summary(path: Path, label_col: Optional[str] = None) -> None:
    """
    Print summary statistics for a dataset.
    """
    rows = parquet_row_count(path)
    print(f"\n📊 Dataset Summary: {path.name}")
    print(f"Rows: {rows:,}")

    if label_col:
        mean_val = parquet_column_mean(path, label_col)
        print(f"{label_col} mean: {mean_val:.4f}")


def validate_no_nulls(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Raise error if nulls found in important columns.
    """
    null_counts = df[cols].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found:\n{null_counts}")


# ============================================================
# 🔹 FILE SYSTEM
# ============================================================

def ensure_dir(path: Path) -> None:
    """
    Create directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_parquet(
    df: pd.DataFrame,
    path: Path,
    compression: str = "snappy",
    index: bool = False,
) -> None:
    """
    Save a dataframe to parquet, creating parent directories when needed.
    """
    ensure_dir(path.parent)
    df.to_parquet(path, compression=compression, index=index)


def check_skip_existing(path: Path) -> bool:
    """
    Return True when pipeline is configured to skip existing outputs.
    """
    try:
        from config import settings  # Local import to keep utils standalone.
        skip_existing = settings.skip_existing
    except Exception:
        skip_existing = False

    return bool(skip_existing and path.exists())
