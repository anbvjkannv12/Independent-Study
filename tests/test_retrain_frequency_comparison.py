import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_retrain_frequency_comparison import retrain_schedule, predict_schedule


class RetrainFrequencyTests(unittest.TestCase):
    def test_schedules_cover_each_test_row_once(self):
        e, n = 31755, 43755
        for every, count in ((None, 1), (4000, 3), (1000, 12), (168, 72)):
            schedule = retrain_schedule(e, n, every)
            self.assertEqual(len(schedule), count)
            self.assertEqual(schedule[0][0], e)
            self.assertEqual(schedule[-1][1], n)
            for (a, b), (c, d) in zip(schedule, schedule[1:]):
                self.assertEqual(b, c)
                self.assertLess(a, b)
                self.assertLess(c, d)
            self.assertEqual(sum(b - a for a, b in schedule), 12000)
        self.assertEqual(retrain_schedule(e, n, 168)[-1][1] - retrain_schedule(e, n, 168)[-1][0], 72)

    def test_reject_invalid_schedules(self):
        with self.assertRaises(ValueError):
            retrain_schedule(180, 300, 0)

    def test_purge_and_initial_predictions_match_on_synthetic_data(self):
        n, e = 300, 180
        frame = pd.DataFrame({"open_time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
                              "feature": np.sin(np.arange(n) / 5), "target_up": np.arange(n) % 2})
        features = ("feature",)
        fixed, f_runs = predict_schedule(frame, features, 42, retrain_schedule(e, n, None), 24, e)
        updated, r_runs = predict_schedule(frame, features, 42, retrain_schedule(e, n, 40), 24, e)
        np.testing.assert_array_equal(fixed.test_raw.iloc[:40], updated.test_raw.iloc[:40])
        self.assertEqual(len(fixed), 120)
        self.assertEqual(len(updated), 120)
        for run in f_runs + r_runs:
            self.assertEqual(run["last_train_index"], run["prediction_start_index"] - 25)
            self.assertEqual(run["train_rows"], run["prediction_start_index"] - 24)
        self.assertEqual([len(updated[updated.train_rows == rows]) for rows in (156, 196, 236)], [40, 40, 40])


if __name__ == "__main__":
    unittest.main()
