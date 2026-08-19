import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from btc_4h_evaluation import (  # noqa: E402
    build_expanding_folds,
    evaluate_baselines,
    evaluate_zero_trade,
    select_balanced_accuracy_threshold,
)


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        timestamps = pd.date_range("2024-01-01", periods=72, freq="4h", tz="UTC")
        self.dataset = pd.DataFrame(
            {
                "open_time": timestamps,
                "feature": np.arange(72, dtype=float),
                "log_return_lag_1": np.where(np.arange(72) % 2, 0.01, -0.01),
                "target_up": [index % 2 for index in range(72)],
            }
        )

    def test_expanding_folds_are_contiguous_and_never_overlap(self):
        folds = build_expanding_folds(self.dataset, 24, 8, 8, 3)

        self.assertEqual([fold.index for fold in folds], [0, 1, 2])
        self.assertEqual([len(fold.train) for fold in folds], [24, 40, 56])
        for fold in folds:
            self.assertLess(fold.train.open_time.max(), fold.validation.open_time.min())
            self.assertLess(fold.validation.open_time.max(), fold.test.open_time.min())

    def test_expanding_folds_reject_insufficient_rows(self):
        with self.assertRaisesRegex(ValueError, "not enough rows"):
            build_expanding_folds(self.dataset.iloc[:71], 24, 8, 8, 3)


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        timestamps = pd.date_range("2024-01-01", periods=72, freq="4h", tz="UTC")
        self.dataset = pd.DataFrame(
            {
                "open_time": timestamps,
                "feature": np.arange(72, dtype=float),
                "log_return_lag_1": np.where(np.arange(72) % 2, 0.01, -0.01),
                "target_up": [index % 2 for index in range(72)],
            }
        )
        self.fold = build_expanding_folds(self.dataset, 24, 8, 8, 3)[0]

    def test_threshold_selection_maximizes_validation_balanced_accuracy(self):
        threshold = select_balanced_accuracy_threshold(
            pd.Series([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])
        )

        self.assertGreater(threshold, 0.4)
        self.assertLessEqual(threshold, 0.6)

    def test_zero_trade_is_not_reported_as_a_classifier(self):
        self.assertEqual(
            evaluate_zero_trade(),
            {"status": "no_directional_prediction", "coverage": 0.0},
        )

    def test_baselines_include_previous_direction_and_logistic_regression(self):
        report = evaluate_baselines(self.fold, ("feature", "log_return_lag_1"))

        self.assertEqual(
            set(report),
            {"all_up", "training_majority", "previous_direction", "logistic_regression", "zero_trade"},
        )
        self.assertIn("test", report["logistic_regression"])


if __name__ == "__main__":
    unittest.main()
