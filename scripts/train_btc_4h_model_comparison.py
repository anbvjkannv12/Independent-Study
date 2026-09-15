from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from btc_4h_evaluation import (
    WalkForwardFold,
    build_expanding_folds,
    evaluate_baselines,
    evaluate_directional_predictions,
    fit_platt_calibrator,
    select_balanced_accuracy_threshold,
)
from btc_4h_model_adapters import (
    ModelSettings,
    fit_predict_lstm,
    fit_predict_transformer,
    fit_predict_xgboost,
)
from train_btc_4h_baseline import load_baseline_dataset


def _fold_summary(fold: WalkForwardFold) -> dict[str, object]:
    return {
        "index": fold.index,
        **{
            name: {
                "count": len(frame),
                "start": frame["open_time"].iloc[0].isoformat(),
                "end": frame["open_time"].iloc[-1].isoformat(),
            }
            for name, frame in (
                ("train", fold.train),
                ("validation", fold.validation),
                ("test", fold.test),
            )
        },
    }


def _probability_quintiles(y_true: pd.Series, probabilities: np.ndarray) -> list[dict[str, object]]:
    order = np.argsort(np.asarray(probabilities, dtype=float))
    groups = np.array_split(order, 5)
    return [
        {
            "quintile": index + 1,
            "sample_count": int(len(group)),
            "mean_probability": float(np.mean(np.asarray(probabilities)[group])),
            "actual_up_rate": float(np.mean(y_true.to_numpy()[group])),
        }
        for index, group in enumerate(groups)
        if len(group)
    ]


def _evaluate_model_fold(
    fold: WalkForwardFold,
    validation_raw: np.ndarray,
    test_raw: np.ndarray,
) -> dict[str, object]:
    calibrator = fit_platt_calibrator(validation_raw, fold.validation["target_up"])
    validation_probability = calibrator.predict_proba(np.asarray(validation_raw).reshape(-1, 1))[:, 1]
    test_probability = calibrator.predict_proba(np.asarray(test_raw).reshape(-1, 1))[:, 1]
    threshold = select_balanced_accuracy_threshold(fold.validation["target_up"], validation_probability)
    return {
        "threshold": threshold,
        "calibration": "platt",
        "validation": {
            **evaluate_directional_predictions(
                fold.validation["target_up"], validation_probability, threshold
            ),
            "uncalibrated_roc_auc": float(roc_auc_score(fold.validation["target_up"], validation_raw)),
        },
        "test": {
            **evaluate_directional_predictions(fold.test["target_up"], test_probability, threshold),
            "uncalibrated_roc_auc": float(roc_auc_score(fold.test["target_up"], test_raw)),
            "probability_quintiles": _probability_quintiles(fold.test["target_up"], test_probability),
        },
    }


def _aggregate_test_metrics(fold_reports: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    metric_names = ("roc_auc", "brier_score", "accuracy", "balanced_accuracy", "precision", "recall", "f1")
    return {
        name: {
            "mean": float(np.mean([report["test"][name] for report in fold_reports])),
            "sample_std": float(np.std([report["test"][name] for report in fold_reports], ddof=1)),
        }
        for name in metric_names
    }


def run_model_comparison(
    dataset: pd.DataFrame,
    feature_names: tuple[str, ...],
    initial_train_rows: int,
    validation_rows: int,
    test_rows: int,
    fold_count: int,
    model_settings: ModelSettings,
) -> dict[str, object]:
    folds = build_expanding_folds(
        dataset, initial_train_rows, validation_rows, test_rows, fold_count
    )
    adapters = {
        "xgboost": fit_predict_xgboost,
        "lstm": fit_predict_lstm,
        "transformer": fit_predict_transformer,
    }
    model_reports: dict[str, list[dict[str, object]]] = {name: [] for name in adapters}
    baseline_reports: dict[str, list[dict[str, object]]] = {}
    for fold in folds:
        fold_baselines = evaluate_baselines(fold, feature_names)
        for name, report in fold_baselines.items():
            baseline_reports.setdefault(name, []).append(report)
        for name, adapter in adapters.items():
            validation_raw, test_raw = adapter(
                fold.train, fold.validation, fold.test, feature_names, model_settings
            )
            model_reports[name].append(_evaluate_model_fold(fold, validation_raw, test_raw))
    return {
        "folds": [_fold_summary(fold) for fold in folds],
        "models": {
            name: {"folds": reports, "test_aggregate": _aggregate_test_metrics(reports)}
            for name, reports in model_reports.items()
        },
        "baselines": {name: {"folds": reports} for name, reports in baseline_reports.items()},
    }


def write_report_atomically(report: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 4-hour direction models for a supported USDT pair.")
    parser.add_argument("--symbol", choices=("BTC", "ETH", "SOL", "XRP"), default="BTC")
    parser.add_argument("--data", type=Path, default=Path("data/features/btc_4h.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("logs/feature_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("logs/btc_4h_model_comparison.json"))
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--initial-train-rows", type=int, default=19755)
    parser.add_argument("--validation-rows", type=int, default=4000)
    parser.add_argument("--test-rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dataset, feature_names = load_baseline_dataset(args.data, args.manifest)
        settings = ModelSettings(
            seed=args.seed,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
        )
        comparison = run_model_comparison(
            dataset,
            feature_names,
            args.initial_train_rows,
            args.validation_rows,
            args.test_rows,
            args.fold_count,
            settings,
        )
        report = {
            "task": f"{args.symbol}USDT 4-hour walk-forward model comparison",
            "paths": {"data": str(args.data.resolve()), "manifest": str(args.manifest.resolve())},
            "feature_names": list(feature_names),
            "fold_settings": {
                "initial_train_rows": args.initial_train_rows,
                "validation_rows": args.validation_rows,
                "test_rows": args.test_rows,
                "fold_count": args.fold_count,
            },
            "model_settings": vars(settings),
            **comparison,
        }
        write_report_atomically(report, args.output)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Model comparison failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
