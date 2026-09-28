from __future__ import annotations

import argparse
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from btc_4h_evaluation import build_expanding_folds
from run_three_class_target_comparison import (
    block_bootstrap_indices,
    fit_predict_xgboost_multiclass,
    non_overlapping_mean,
    persistence_direction,
    persistence_signal,
    primary_statistics,
    three_class_labels,
    trade_returns,
    validation_threshold,
)


class ThreeClassTargetComparisonTests(unittest.TestCase):
    def test_three_class_label_boundaries(self) -> None:
        r = np.array([-0.003, -0.002, 0.0, 0.002, 0.003])
        np.testing.assert_array_equal(three_class_labels(r, 0.002), np.array([0, 1, 1, 1, 2]))

    def test_threshold_uses_validation_only(self) -> None:
        validation = np.linspace(-1, 1, 101)
        test_a = np.array([100.0, 200.0])
        test_b = np.array([0.0, 0.0])
        t1 = validation_threshold(validation, 0.20)
        t2 = validation_threshold(validation, 0.20)
        self.assertEqual(t1, t2)
        self.assertEqual(validation_threshold(validation, 0.20), validation_threshold(validation, 0.20))
        self.assertEqual(t1, validation_threshold(validation, 0.20))
        self.assertNotEqual(test_a.mean(), test_b.mean())
        self.assertAlmostEqual((np.abs(validation) >= t1).mean(), 0.20, delta=0.02)

    def test_trade_returns(self) -> None:
        s = np.array([0.3, -0.2, 0.01])
        r = np.array([0.01, -0.01, 0.05])
        g = trade_returns(s, r, 0.1, 0.002)
        self.assertAlmostEqual(g[0], 0.008)
        self.assertAlmostEqual(g[1], 0.008)
        self.assertTrue(np.isnan(g[2]))

    def test_persistence_direction_uses_train_only(self) -> None:
        train_signal = np.arange(20, dtype=float)
        train_target = (train_signal < 10).astype(int)
        self.assertEqual(persistence_direction(train_signal, train_target), -1)

    def test_persistence_does_not_use_future_rows(self) -> None:
        frame = pd.DataFrame({"log_return": np.arange(20, dtype=float)})
        full = persistence_signal(frame)
        cut = persistence_signal(frame.iloc[:11].copy())
        self.assertEqual(full[10], cut[10])

    def test_multiclass_model_shape_sum_and_determinism(self) -> None:
        rng = np.random.default_rng(123)
        n = 300
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        r = x1 * 0.01 + rng.normal(scale=0.001, size=n)
        frame = pd.DataFrame({
            "open_time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "x1": x1,
            "x2": x2,
            "target_3c": three_class_labels(r, 0.002),
            "target_up": (r > 0).astype(int),
            "future_log_return": r,
        })
        train, validation, test = frame.iloc[:200], frame.iloc[200:250], frame.iloc[250:]
        val1, tst1 = fit_predict_xgboost_multiclass(train, validation, test, ("x1", "x2"), 42)
        val2, tst2 = fit_predict_xgboost_multiclass(train, validation, test, ("x1", "x2"), 42)
        self.assertEqual(val1.shape, (50, 3))
        self.assertEqual(tst1.shape, (50, 3))
        np.testing.assert_allclose(val1.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(tst1.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(val1, val2, atol=0, rtol=0)
        np.testing.assert_allclose(tst1, tst2, atol=0, rtol=0)

    def test_bootstrap_reproducible_and_fold_local(self) -> None:
        folds = [np.empty(100), np.empty(80)]
        a = block_bootstrap_indices(folds, 10, 3, 7)
        b = block_bootstrap_indices(folds, 10, 3, 7)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(len(left), 180)
            self.assertTrue(((left[:100] >= 0) & (left[:100] < 100)).all())
            self.assertTrue(((left[100:] >= 100) & (left[100:] < 180)).all())

    def test_purge_spacing(self) -> None:
        n = 240
        frame = pd.DataFrame({
            "open_time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "target_up": [0, 1] * 120,
            "future_log_return": np.tile([-0.01, 0.01], 120),
            "is_imputed": 0,
            "x": np.arange(n, dtype=float),
        })
        folds = build_expanding_folds(frame, 50, 20, 20, 2, purge_rows=24)
        for fold in folds:
            self.assertEqual(fold.validation.index[0] - fold.train.index[-1] - 1, 24)
            self.assertEqual(fold.test.index[0] - fold.validation.index[-1] - 1, 24)

    def test_m_is_mean_of_per_seed_means(self) -> None:
        fold = np.zeros(96, dtype=int)
        seed_a = np.full(96, np.nan); seed_a[0] = 0.010
        seed_b = np.full(96, np.nan); seed_b[:3] = 0.0
        stats = primary_statistics([seed_a, seed_b], [seed_b, seed_b], fold, replicates=5, seed=1)
        self.assertAlmostEqual(stats["m_T"], 0.005)

    def test_comparison_uses_each_arm_own_trades(self) -> None:
        # T and B never trade on the same row; B is better, so delta < 0 and p must exceed 0.5.
        fold = np.repeat([0, 1], 96)
        t_net = np.full(192, np.nan); t_net[::2] = -0.003
        b_net = np.full(192, np.nan); b_net[1::2] = -0.001
        stats = primary_statistics([t_net], [b_net], fold, replicates=200, seed=3)
        self.assertAlmostEqual(stats["delta"], -0.002)
        np.testing.assert_allclose(stats["fold_delta"], [-0.002, -0.002])
        self.assertGreater(stats["comparison_bootstrap"]["raw_one_sided_p_value"], 0.5)
        self.assertGreater(stats["signal_bootstrap"]["raw_one_sided_p_value"], 0.5)

    def test_non_overlapping_rows(self) -> None:
        frame = pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC"),
                              "net_return": [0.01, 9, 9, 9, 0.03, 9, 9, 9]})
        self.assertAlmostEqual(non_overlapping_mean([frame]), 0.02)

    def test_rejects_non_default_k(self) -> None:
        script = SCRIPTS / "run_three_class_target_comparison.py"
        proc = subprocess.run([sys.executable, str(script), "--k", "0.003", "--aggregate-only"], cwd=ROOT, text=True, capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("preregistered fixed defaults", proc.stderr)


if __name__ == "__main__":
    unittest.main()
