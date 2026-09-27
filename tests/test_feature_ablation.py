import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_feature_ablation import (  # noqa: E402
    EXPECTED_REMAINDER,
    FEATURE_GROUPS,
    holm_adjust,
    moving_block_bootstrap_deltas,
    validate_feature_groups,
)


class FeatureGroupTests(unittest.TestCase):
    def test_preregistered_groups_are_disjoint_and_cover_42_features(self):
        grouped = [feature for group in FEATURE_GROUPS.values() for feature in group]

        self.assertEqual([len(FEATURE_GROUPS[key]) for key in FEATURE_GROUPS], [5, 12, 12, 4, 9])
        self.assertEqual(len(grouped), 42)
        self.assertEqual(len(set(grouped)), 42)

    def test_manifest_validation_accepts_only_expected_53_features(self):
        grouped = [feature for group in FEATURE_GROUPS.values() for feature in group]
        feature_names = tuple(grouped + sorted(EXPECTED_REMAINDER))

        validate_feature_groups(feature_names)
        with self.assertRaisesRegex(ValueError, "exactly 53"):
            validate_feature_groups(feature_names[:-1])


class StatisticalHelpersTests(unittest.TestCase):
    @staticmethod
    def _predictions(raw_values):
        rows = len(raw_values)
        return pd.DataFrame(
            {
                "fold": np.repeat([0, 1], rows // 2),
                "open_time": pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC"),
                "target_up": np.tile([0, 1], rows // 2),
                "test_raw": raw_values,
            }
        )

    def test_block_bootstrap_is_deterministic_and_preserves_pairing(self):
        targets = np.tile([0, 1], 24)
        strong = np.where(targets == 1, 0.9, 0.1)
        weak = np.linspace(0.2, 0.8, 48)
        control = self._predictions(strong)
        treatment = self._predictions(weak)

        first = moving_block_bootstrap_deltas(control, treatment, 4, 25, 7)
        second = moving_block_bootstrap_deltas(control, treatment, 4, 25, 7)

        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first > 0))

    def test_block_bootstrap_rejects_unpaired_rows(self):
        values = np.linspace(0.1, 0.9, 48)
        control = self._predictions(values)
        treatment = self._predictions(values)
        treatment.at[0, "open_time"] = treatment.at[1, "open_time"]

        with self.assertRaisesRegex(ValueError, "not paired"):
            moving_block_bootstrap_deltas(control, treatment, 4, 10, 1)

    def test_holm_adjustment_is_step_down_monotone(self):
        adjusted = holm_adjust({"G1": 0.001, "G2": 0.01, "G3": 0.03, "G4": 0.2, "G5": 0.8})

        self.assertAlmostEqual(adjusted["G1"]["adjusted_p_value"], 0.005)
        self.assertAlmostEqual(adjusted["G2"]["adjusted_p_value"], 0.04)
        self.assertAlmostEqual(adjusted["G3"]["adjusted_p_value"], 0.09)
        ordered = [adjusted[f"G{index}"]["adjusted_p_value"] for index in range(1, 6)]
        self.assertEqual(ordered, sorted(ordered))


if __name__ == "__main__":
    unittest.main()
