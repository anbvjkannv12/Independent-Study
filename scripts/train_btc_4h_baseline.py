from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


EXCLUDED_MODEL_COLUMNS = {"open_time", "future_log_return", "target_up", "is_imputed"}
REQUIRED_COLUMNS = {"open_time", "future_log_return", "target_up", "is_imputed"}
MODEL_SETTINGS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": 1,
    "tree_method": "hist",
}


def _manifest_feature_names(manifest_path: Path) -> tuple[str, ...]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid manifest JSON: {manifest_path}") from exc

    names = manifest.get("feature_names")
    if not isinstance(names, list) or len(names) != 53 or not all(isinstance(name, str) for name in names):
        raise ValueError("Manifest must contain exactly 53 feature_names.")
    if len(set(names)) != len(names):
        raise ValueError("Manifest feature_names must be unique.")
    forbidden = EXCLUDED_MODEL_COLUMNS.intersection(names)
    if forbidden:
        raise ValueError(f"Manifest feature_names contain excluded columns: {', '.join(sorted(forbidden))}")
    return tuple(names)


def load_baseline_dataset(data_path: Path, manifest_path: Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Load the manifest-selected BTC 4h dataset with strict temporal validation."""
    feature_names = _manifest_feature_names(manifest_path)
    try:
        dataset = pd.read_csv(data_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found: {data_path}") from exc

    required = REQUIRED_COLUMNS.union(feature_names)
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    dataset = dataset[["open_time", *feature_names, "future_log_return", "target_up", "is_imputed"]].copy()
    dataset["open_time"] = pd.to_datetime(dataset["open_time"], utc=True, errors="coerce")
    if dataset["open_time"].isna().any():
        raise ValueError("open_time contains invalid timestamps.")
    if dataset["open_time"].duplicated().any():
        raise ValueError("open_time values must be unique.")
    if not dataset["open_time"].is_monotonic_increasing:
        raise ValueError("open_time values must be strictly increasing.")

    numeric = dataset.drop(columns="open_time")
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains missing or non-finite values.")
    dataset[numeric.columns] = numeric
    return dataset, feature_names


def split_chronologically(
    dataset: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Split an already validated dataset into contiguous train/validation/test frames."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train_fraction and validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction plus validation_fraction must be less than 1.")
    missing = sorted({"open_time", "target_up"}.difference(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    timestamps = pd.to_datetime(dataset["open_time"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("open_time contains invalid timestamps.")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("open_time values must be strictly increasing.")
    numeric = dataset.drop(columns="open_time").apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains missing or non-finite values.")
    n_rows = len(dataset)
    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= n_rows:
        raise ValueError("Fractions produce an empty chronological split.")
    splits = {
        "train": dataset.iloc[:train_end].copy(),
        "validation": dataset.iloc[train_end:validation_end].copy(),
        "test": dataset.iloc[validation_end:].copy(),
    }
    for name, frame in splits.items():
        if frame["target_up"].nunique(dropna=False) < 2:
            raise ValueError(f"{name} split contains a single target class.")
    return splits


def evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, object]:
    """Evaluate binary probabilities with a fixed 0.5 decision threshold."""
    labels = (np.asarray(probabilities) >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, labels)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, labels, labels=[0, 1]).tolist(),
    }


def _evaluate_constant_baseline(y_true: pd.Series, predicted_class: int) -> dict[str, object]:
    result = evaluate_predictions(y_true, np.full(len(y_true), predicted_class, dtype=float))
    result["predicted_class"] = predicted_class
    return result


def run_baseline(
    dataset: pd.DataFrame,
    feature_names: tuple[str, ...],
    random_state: int = 42,
) -> dict[str, object]:
    """Train and evaluate the fixed BTC 4h XGBoost baseline chronologically."""
    splits = split_chronologically(dataset)
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    model = XGBClassifier(
        **MODEL_SETTINGS,
        random_state=random_state,
    )
    model.fit(train.loc[:, feature_names], train["target_up"])

    def evaluate_split(frame: pd.DataFrame) -> dict[str, object]:
        probabilities = model.predict_proba(frame.loc[:, feature_names])[:, 1]
        return evaluate_predictions(frame["target_up"], probabilities)

    gain_importance = model.get_booster().get_score(importance_type="gain")
    sorted_gain_importance = dict(sorted(gain_importance.items(), key=lambda item: item[1], reverse=True))
    training_majority = int(train["target_up"].mode().iat[0])

    return {
        "model": {
            "validation": evaluate_split(validation),
            "test": evaluate_split(test),
            "gain_importance": sorted_gain_importance,
        },
        "benchmarks": {
            "all_up": {
                "validation": _evaluate_constant_baseline(validation["target_up"], 1),
                "test": _evaluate_constant_baseline(test["target_up"], 1),
            },
            "training_majority": {
                "validation": _evaluate_constant_baseline(validation["target_up"], training_majority),
                "test": _evaluate_constant_baseline(test["target_up"], training_majority),
            },
        },
    }


def _split_summary(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "count": len(frame),
            "start": frame["open_time"].iloc[0].isoformat(),
            "end": frame["open_time"].iloc[-1].isoformat(),
        }
        for name, frame in splits.items()
    }


def build_report(
    dataset: pd.DataFrame,
    feature_names: tuple[str, ...],
    data_path: Path,
    manifest_path: Path,
    random_state: int,
) -> dict[str, object]:
    """Create the serializable report only after a successful training run."""
    splits = split_chronologically(dataset)
    result = run_baseline(dataset, feature_names, random_state=random_state)
    return {
        "task": "BTCUSDT 4-hour direction classification",
        "paths": {
            "data": str(data_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
        "feature_names": list(feature_names),
        "splits": _split_summary(splits),
        "model_settings": {**MODEL_SETTINGS, "random_state": random_state},
        **result,
    }


def write_report_atomically(report: dict[str, object], output_path: Path) -> None:
    """Write a report via a sibling temporary file so failures preserve the old output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BTCUSDT 4-hour XGBoost baseline.")
    parser.add_argument("--data", type=Path, default=Path("data/features/btc_4h.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("logs/feature_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("logs/baseline_btc_4h.json"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dataset, feature_names = load_baseline_dataset(args.data, args.manifest)
        report = build_report(
            dataset,
            feature_names,
            args.data,
            args.manifest,
            args.random_state,
        )
        write_report_atomically(report, args.output)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Baseline training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
