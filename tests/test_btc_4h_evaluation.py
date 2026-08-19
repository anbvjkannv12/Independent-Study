import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from btc_4h_evaluation import build_expanding_folds  # noqa: E402


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        timestamps = pd.date_range("2024-01-01", periods=72, freq="4h", tz="UTC")
        self.dataset = pd.DataFrame(
            {
                "open_time": timestamps,
                "feature": np.arange(72, dtype=float),
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


if __name__ == "__main__":
    unittest.main()
