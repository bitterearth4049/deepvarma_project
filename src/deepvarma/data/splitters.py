from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class TimeSeriesSplitConfig:
    train_fraction: float
    val_fraction: float


def simple_train_val_test_split(
    T: int,
    cfg: TimeSeriesSplitConfig,
) -> Tuple[slice, slice, slice]:
    """
    Sequential split: [0, train) [train, val) [val, T)
    """
    train_end = int(T * cfg.train_fraction)
    val_end = train_end + int(T * cfg.val_fraction)

    if val_end >= T:
        raise ValueError("Train + val fractions leave no room for test set.")

    return slice(0, train_end), slice(train_end, val_end), slice(val_end, T)


def blocked_time_series_folds(
    T: int,
    n_splits: int,
    min_train_fraction: float = 0.5,
) -> List[Tuple[slice, slice]]:
    """
    Produce blocked folds (no shuffling) for time series CV.
    Folds: growing training window, fixed validation block.
    """
    if not (0 < min_train_fraction < 1.0):
        raise ValueError("min_train_fraction must be in (0, 1).")

    min_train_size = int(T * min_train_fraction)
    if min_train_size <= 0:
        raise ValueError("Not enough time steps for the requested min_train_fraction.")

    # approximate fold size for validation
    remaining = T - min_train_size
    if remaining <= n_splits:
        raise ValueError("Not enough remaining points to create validation folds.")

    fold_val_size = remaining // (n_splits + 1)
    folds: List[Tuple[slice, slice]] = []

    for i in range(n_splits):
        train_end = min_train_size + fold_val_size * i
        val_start = train_end
        val_end = val_start + fold_val_size
        if val_end > T:
            break
        folds.append((slice(0, train_end), slice(val_start, val_end)))

    if not folds:
        raise ValueError("Failed to create any folds, check T and n_splits.")

    return folds
