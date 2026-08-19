from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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


def evaluate_directional_predictions(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    """Evaluate probabilities without conflating ranking with thresholded decisions."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = (probabilities >= threshold).astype(int)
    return {
        "sample_count": int(len(y_true)),
        "actual_up_rate": float(np.mean(y_true)),
        "predicted_up_rate": float(np.mean(labels)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, labels)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, labels, labels=[0, 1]).tolist(),
    }


def fit_platt_calibrator(
    validation_probabilities: np.ndarray, validation_target: pd.Series
) -> LogisticRegression:
    """Fit a probability-to-probability calibrator on validation data only."""
    calibrator = LogisticRegression(random_state=42, solver="lbfgs")
    calibrator.fit(np.asarray(validation_probabilities).reshape(-1, 1), validation_target)
    return calibrator


def select_balanced_accuracy_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose the smallest threshold with the best validation balanced accuracy."""
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.arange(0.05, 0.951, 0.01)
    scores = [balanced_accuracy_score(y_true, probabilities >= candidate) for candidate in candidates]
    return float(candidates[int(np.argmax(scores))])


def evaluate_zero_trade() -> dict[str, object]:
    return {"status": "no_directional_prediction", "coverage": 0.0}


def _constant_report(fold: WalkForwardFold, predicted_class: int) -> dict[str, object]:
    return {
        "validation": evaluate_directional_predictions(
            fold.validation["target_up"], np.full(len(fold.validation), predicted_class), 0.5
        ),
        "test": evaluate_directional_predictions(
            fold.test["target_up"], np.full(len(fold.test), predicted_class), 0.5
        ),
        "predicted_class": predicted_class,
    }


def evaluate_baselines(
    fold: WalkForwardFold, feature_names: tuple[str, ...]
) -> dict[str, object]:
    """Evaluate all comparison baselines with train-only fitting where required."""
    if "log_return_lag_1" not in fold.train.columns:
        raise ValueError("previous_direction baseline requires log_return_lag_1")
    training_majority = int(fold.train["target_up"].mode().iat[0])
    previous_direction = {
        "validation": evaluate_directional_predictions(
            fold.validation["target_up"],
            (fold.validation["log_return_lag_1"].to_numpy() > 0).astype(float),
            0.5,
        ),
        "test": evaluate_directional_predictions(
            fold.test["target_up"],
            (fold.test["log_return_lag_1"].to_numpy() > 0).astype(float),
            0.5,
        ),
    }
    logistic = make_pipeline(
        StandardScaler(), LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000)
    )
    logistic.fit(fold.train.loc[:, feature_names], fold.train["target_up"])
    validation_raw = logistic.predict_proba(fold.validation.loc[:, feature_names])[:, 1]
    test_raw = logistic.predict_proba(fold.test.loc[:, feature_names])[:, 1]
    calibrator = fit_platt_calibrator(validation_raw, fold.validation["target_up"])
    validation_probability = calibrator.predict_proba(validation_raw.reshape(-1, 1))[:, 1]
    test_probability = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
    threshold = select_balanced_accuracy_threshold(fold.validation["target_up"], validation_probability)
    return {
        "all_up": _constant_report(fold, 1),
        "training_majority": _constant_report(fold, training_majority),
        "previous_direction": previous_direction,
        "logistic_regression": {
            "validation": evaluate_directional_predictions(
                fold.validation["target_up"], validation_probability, threshold
            ),
            "test": evaluate_directional_predictions(fold.test["target_up"], test_probability, threshold),
            "threshold": threshold,
            "calibration": "platt",
        },
        "zero_trade": evaluate_zero_trade(),
    }
