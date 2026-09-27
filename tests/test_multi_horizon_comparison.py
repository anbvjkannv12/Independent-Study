import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_multi_horizon_comparison import align_datasets, persistence_scores
from run_feature_ablation import moving_block_bootstrap_deltas


class HorizonTests(unittest.TestCase):
    def test_alignment_rejects_inconsistent_timestamps(self):
        a = pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")})
        b = a.iloc[[0, 2, 1, 3]].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            align_datasets({1: a, 4: b})

    def test_persistence_uses_only_current_and_past_returns(self):
        data = pd.DataFrame({"log_return": [-2., 1., 2., -3., 4.],
                             "future_log_return": [100.] * 5, "target_up": [1] * 5})
        original = persistence_scores(data, 3)
        np.testing.assert_array_equal(original, [0, 0, 1, 0, 1])
        changed = data.copy()
        changed.loc[4, "log_return"] = -100
        changed.loc[:, "future_log_return"] = -100
        changed.loc[:, "target_up"] = 0
        np.testing.assert_array_equal(persistence_scores(changed, 3)[:4], original[:4])

    def test_shared_seed_signal_differences_equal_paired_bootstrap(self):
        dates = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        a = pd.DataFrame({"fold": [0] * 12 + [1] * 12, "open_time": dates,
                          "target_up": [0, 1] * 12, "test_raw": [0.1, 0.9] * 12})
        b = a.copy()
        b["target_up"] = 1 - b.target_up
        null_a, null_b = a.copy(), b.copy()
        null_a["test_raw"] = 0.5
        null_b["test_raw"] = 0.5
        delta = moving_block_bootstrap_deltas(a, b, 2, 10, 17, allow_different_targets=True)
        separate = (moving_block_bootstrap_deltas(a, null_a, 2, 10, 17)
                    - moving_block_bootstrap_deltas(b, null_b, 2, 10, 17))
        np.testing.assert_allclose(delta, separate)

    def test_cross_horizon_bootstrap_uses_each_horizon_target(self):
        dates = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
        a = pd.DataFrame({"fold": 0, "open_time": dates, "target_up": [0, 1] * 6,
                          "test_raw": [0.1, 0.9] * 6})
        b = a.copy()
        b["target_up"] = 1 - b["target_up"]
        values = moving_block_bootstrap_deltas(a, b, 2, 3, 42, allow_different_targets=True)
        np.testing.assert_allclose(values, 1)


if __name__ == "__main__":
    unittest.main()
