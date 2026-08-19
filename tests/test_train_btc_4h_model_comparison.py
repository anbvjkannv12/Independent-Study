import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_btc_4h_model_comparison.py"


class ModelComparisonCliTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_path = self.root / "btc_4h.csv"
        self.bad_data_path = self.root / "bad_btc_4h.csv"
        self.manifest_path = self.root / "feature_manifest.json"
        self.output_path = self.root / "comparison.json"
        self.feature_names = ("log_return_lag_1", *(f"feature_{index}" for index in range(52)))
        timestamps = pd.date_range("2024-01-01", periods=72, freq="4h", tz="UTC")
        dataset = pd.DataFrame({"open_time": timestamps})
        for index, name in enumerate(self.feature_names):
            dataset[name] = np.arange(72, dtype=float) + index
        dataset["future_log_return"] = np.where(np.arange(72) % 2, 0.01, -0.01)
        dataset["target_up"] = [index % 2 for index in range(72)]
        dataset["is_imputed"] = 0
        dataset.to_csv(self.data_path, index=False)
        dataset.assign(target_up=1).to_csv(self.bad_data_path, index=False)
        self.manifest_path.write_text(
            json.dumps({"feature_count": 53, "feature_names": list(self.feature_names)}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _command(self, data_path: Path) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT_PATH),
            "--data",
            str(data_path),
            "--manifest",
            str(self.manifest_path),
            "--output",
            str(self.output_path),
            "--fold-count",
            "3",
            "--initial-train-rows",
            "24",
            "--validation-rows",
            "8",
            "--test-rows",
            "8",
            "--sequence-length",
            "4",
            "--hidden-size",
            "8",
            "--num-heads",
            "2",
            "--batch-size",
            "8",
            "--max-epochs",
            "1",
            "--patience",
            "1",
        ]

    def test_cli_writes_three_model_folds_only_after_success(self):
        completed = subprocess.run(self._command(self.data_path), capture_output=True, text=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(set(report["models"]), {"xgboost", "lstm", "transformer"})
        self.assertEqual(len(report["folds"]), 3)
        self.assertEqual(set(report["baselines"]), {"all_up", "training_majority", "previous_direction", "logistic_regression", "zero_trade"})

    def test_cli_preserves_existing_output_if_one_model_fails(self):
        self.output_path.write_text('{"old": true}', encoding="utf-8")

        completed = subprocess.run(self._command(self.bad_data_path), capture_output=True, text=True)

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(json.loads(self.output_path.read_text(encoding="utf-8")), {"old": True})


class ComparisonDocumentationTests(unittest.TestCase):
    def test_comparison_documentation_and_notebook_are_read_only(self):
        documentation = (PROJECT_ROOT / "docs" / "btc_4h_model_comparison.md").read_text(
            encoding="utf-8"
        )
        notebook = json.loads(
            (PROJECT_ROOT / "run" / "03_btc_4h_model_comparison.ipynb").read_text(encoding="utf-8")
        )

        self.assertIn("train_btc_4h_model_comparison.py", documentation)
        source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertNotIn(".fit(", source)


if __name__ == "__main__":
    unittest.main()
