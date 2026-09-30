import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from btc_4h_model_adapters import TransformerClassifier
from build_4h_transformer_ptl import ProbabilityModel, load_lite, raw_windows, save_lite


class Transformer4hPtlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = TransformerClassifier(input_size=5, hidden_size=8, num_heads=2, sequence_length=4).eval()
        self.mean = np.arange(5, dtype=np.float32)
        self.scale = np.linspace(1, 2, 5, dtype=np.float32)
        self.x = torch.randn(3, 4, 5)

    def test_wrapper_scales_then_applies_sigmoid(self):
        wrapper = ProbabilityModel(self.model, self.mean, self.scale).eval()
        with torch.no_grad():
            expected = torch.sigmoid(self.model((self.x - torch.tensor(self.mean)) / torch.tensor(self.scale)))
            torch.testing.assert_close(wrapper(self.x), expected)

    def test_lite_round_trip_matches_eager_model(self):
        torch.backends.mha.set_fastpath_enabled(False)
        wrapper = ProbabilityModel(self.model, self.mean, self.scale).eval()
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, self.x[:1])
            expected = wrapper(self.x)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.ptl"
            save_lite(traced, path)
            lite = load_lite(path)
            with torch.no_grad():
                torch.testing.assert_close(lite(self.x), expected, atol=1e-5, rtol=1e-5)

    def test_raw_windows_end_at_each_row(self):
        frame = pd.DataFrame({"a": np.arange(6.0), "b": np.arange(6.0) * 10})
        windows = raw_windows(frame, ("a", "b"), 4)
        self.assertEqual(windows.shape, (3, 4, 2))
        np.testing.assert_array_equal(windows[-1, :, 0], [2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
