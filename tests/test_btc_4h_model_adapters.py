import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from btc_4h_model_adapters import (  # noqa: E402
    ModelSettings,
    fit_predict_lstm,
    fit_predict_transformer,
    fit_predict_xgboost,
    make_sequence_samples,
)


class ModelAdapterTests(unittest.TestCase):
    def setUp(self):
        self.feature_names = tuple(f"feature_{index}" for index in range(53))
        timestamps = pd.date_range("2024-01-01", periods=72, freq="4h", tz="UTC")
        self.frame = pd.DataFrame({"open_time": timestamps})
        for index, name in enumerate(self.feature_names):
            self.frame[name] = np.arange(72, dtype=float) + index
        self.frame["feature_0"] = np.arange(72, dtype=int)
        self.frame["target_up"] = [index % 2 for index in range(72)]
        self.train = self.frame.iloc[:48].copy()
        self.validation = self.frame.iloc[48:60].copy()
        self.test = self.frame.iloc[60:].copy()
        self.settings = ModelSettings(sequence_length=4, batch_size=8, max_epochs=1, patience=1)

    def test_sequence_samples_end_at_current_row_without_future_features(self):
        sequences, targets, times = make_sequence_samples(self.frame, self.feature_names, 4)

        self.assertEqual(sequences.shape, (69, 4, 53))
        self.assertEqual(len(targets), 69)
        self.assertEqual(times[0], self.frame.open_time.iloc[3])
        np.testing.assert_array_equal(
            sequences[0, -1], self.frame.loc[3, list(self.feature_names)].to_numpy()
        )

    def test_adapters_return_one_probability_per_validation_and_test_row(self):
        adapters = (fit_predict_xgboost, fit_predict_lstm, fit_predict_transformer)
        for adapter in adapters:
            validation_probability, test_probability = adapter(
                self.train, self.validation, self.test, self.feature_names, self.settings
            )
            self.assertEqual(len(validation_probability), len(self.validation))
            self.assertEqual(len(test_probability), len(self.test))
            self.assertTrue(np.isfinite(validation_probability).all())
            self.assertTrue(np.isfinite(test_probability).all())


if __name__ == "__main__":
    unittest.main()
