from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_volatility_target_extension import (
    persistence_direction,
    volatility_labels,
    volatility_persistence_signal,
)


class VolatilityTargetExtensionTests(unittest.TestCase):
    def test_labels_use_train_threshold_strictly_greater(self) -> None:
        values = np.array([0.001, 0.002, 0.003])
        np.testing.assert_array_equal(volatility_labels(values, 0.002), np.array([0, 0, 1]))

    def test_persistence_signal_uses_past_absolute_returns_only(self) -> None:
        frame = pd.DataFrame({"log_return": [-1.0, 2.0, -3.0, 4.0, 100.0]})
        sig = volatility_persistence_signal(frame)
        self.assertTrue(np.isnan(sig[2]))
        self.assertEqual(sig[3], 10.0)
        cut = volatility_persistence_signal(frame.iloc[:4].copy())
        self.assertEqual(sig[3], cut[3])

    def test_persistence_direction_can_flip_on_train_only(self) -> None:
        signal = np.arange(10, dtype=float)
        target = (signal < 5).astype(int)
        self.assertEqual(persistence_direction(signal, target), -1)


if __name__ == "__main__":
    unittest.main()
