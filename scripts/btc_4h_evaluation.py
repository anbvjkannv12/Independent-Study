from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    index: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _validate_partition(name: str, frame: pd.DataFrame) -> None:
    required = {"open_time", "target_up"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
    timestamps = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"{name} open_time values must be strictly increasing")
    numeric = frame.drop(columns="open_time").apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains missing or non-finite values")
    if frame["target_up"].nunique(dropna=False) < 2:
        raise ValueError(f"{name} contains a single target class")


def build_expanding_folds(
    dataset: pd.DataFrame,
    initial_train_rows: int,
    validation_rows: int,
    test_rows: int,
    fold_count: int,
) -> list[WalkForwardFold]:
    """Return contiguous expanding-window folds without future-data overlap."""
    values = (initial_train_rows, validation_rows, test_rows, fold_count)
    if any(not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("fold sizes and fold_count must be positive integers")
    required_rows = initial_train_rows + fold_count * (validation_rows + test_rows)
    if len(dataset) < required_rows:
        raise ValueError("not enough rows for requested walk-forward folds")

    _validate_partition("dataset", dataset)
    folds: list[WalkForwardFold] = []
    for index in range(fold_count):
        train_end = initial_train_rows + index * (validation_rows + test_rows)
        validation_end = train_end + validation_rows
        test_end = validation_end + test_rows
        fold = WalkForwardFold(
            index=index,
            train=dataset.iloc[:train_end].copy(),
            validation=dataset.iloc[train_end:validation_end].copy(),
            test=dataset.iloc[validation_end:test_end].copy(),
        )
        _validate_partition(f"fold {index} train", fold.train)
        _validate_partition(f"fold {index} validation", fold.validation)
        _validate_partition(f"fold {index} test", fold.test)
        folds.append(fold)
    return folds
