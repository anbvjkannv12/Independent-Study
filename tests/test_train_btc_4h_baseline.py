import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_btc_4h_baseline.py"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_btc_4h_baseline as baseline  # noqa: E402


load_baseline_dataset = baseline.load_baseline_dataset
split_chronologically = baseline.split_chronologically


class BaselineTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_path = self.root / "btc_4h.csv"
        self.manifest_path = self.root / "feature_manifest.json"
        self.feature_names = tuple(f"feature_{index}" for index in range(53))
        timestamps = pd.date_range("2024-01-01", periods=20, freq="4h", tz="UTC")
        dataset = pd.DataFrame({"open_time": timestamps})
        for index, name in enumerate(self.feature_names):
            dataset[name] = np.arange(20, dtype=float) + index
        dataset["future_log_return"] = np.linspace(-1.0, 1.0, 20)
        dataset["target_up"] = [index % 2 for index in range(20)]
        dataset["is_imputed"] = 0
        dataset.to_csv(self.data_path, index=False)
        self.manifest_path.write_text(
            json.dumps({"feature_count": 53, "feature_names": list(self.feature_names)}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_baseline_dependencies_are_importable(self):
        from sklearn.metrics import roc_auc_score
        from xgboost import XGBClassifier

        self.assertTrue(callable(roc_auc_score))
        self.assertTrue(callable(XGBClassifier))

    def test_loader_uses_manifest_features_and_orders_time(self):
        dataset, names = load_baseline_dataset(self.data_path, self.manifest_path)

        self.assertEqual(names, self.feature_names)
        self.assertTrue(dataset["open_time"].is_monotonic_increasing)
        self.assertNotIn("target_up", names)
        self.assertNotIn("open_time", names)
        self.assertNotIn("future_log_return", names)
        self.assertNotIn("is_imputed", names)
        self.assertEqual(list(dataset.columns), ["open_time", *self.feature_names, "future_log_return", "target_up", "is_imputed"])

    def test_loader_rejects_duplicate_timestamps(self):
        dataset = pd.read_csv(self.data_path)
        dataset.loc[1, "open_time"] = dataset.loc[0, "open_time"]
        dataset.to_csv(self.data_path, index=False)

        with self.assertRaisesRegex(ValueError, "unique"):
            load_baseline_dataset(self.data_path, self.manifest_path)

    def test_loader_rejects_unordered_timestamps(self):
        dataset = pd.read_csv(self.data_path)
        dataset.iloc[::-1].to_csv(self.data_path, index=False)

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            load_baseline_dataset(self.data_path, self.manifest_path)

    def test_loader_rejects_non_finite_values(self):
        dataset = pd.read_csv(self.data_path)
        dataset.loc[0, self.feature_names[0]] = np.inf
        dataset.to_csv(self.data_path, index=False)

        with self.assertRaisesRegex(ValueError, "finite"):
            load_baseline_dataset(self.data_path, self.manifest_path)

    def test_chronological_split_uses_requested_fractions(self):
        dataset, _ = load_baseline_dataset(self.data_path, self.manifest_path)
        splits = split_chronologically(dataset, train_fraction=0.70, validation_fraction=0.15)

        self.assertEqual({"train", "validation", "test"}, set(splits))
        self.assertEqual((14, 3, 3), tuple(len(splits[name]) for name in ("train", "validation", "test")))
        self.assertTrue(splits["train"]["open_time"].max() < splits["validation"]["open_time"].min())
        self.assertTrue(splits["validation"]["open_time"].max() < splits["test"]["open_time"].min())

    def test_split_rejects_duplicate_or_unordered_times(self):
        dataset, _ = load_baseline_dataset(self.data_path, self.manifest_path)
        duplicate = dataset.copy()
        duplicate.loc[1, "open_time"] = duplicate.loc[0, "open_time"]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            split_chronologically(duplicate)

        unordered = dataset.iloc[::-1].copy()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            split_chronologically(unordered)

    def test_split_rejects_missing_or_non_finite_values(self):
        dataset, _ = load_baseline_dataset(self.data_path, self.manifest_path)
        missing_time = dataset.drop(columns="open_time")
        with self.assertRaisesRegex(ValueError, "open_time"):
            split_chronologically(missing_time)

        non_finite = dataset.copy()
        non_finite.loc[0, self.feature_names[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            split_chronologically(non_finite)


class BaselineEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.feature_names = tuple(f"feature_{index}" for index in range(53))
        timestamps = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")
        self.dataset = pd.DataFrame({"open_time": timestamps})
        for index, name in enumerate(self.feature_names):
            self.dataset[name] = np.arange(40, dtype=float) + index
        self.dataset["future_log_return"] = np.linspace(-1.0, 1.0, 40)
        self.dataset["target_up"] = [index % 2 for index in range(40)]
        self.dataset["is_imputed"] = 0

    def test_metrics_use_probability_for_auc_and_threshold_for_labels(self):
        result = baseline.evaluate_predictions(pd.Series([0, 1, 1, 0]), np.array([0.1, 0.6, 0.4, 0.8]))

        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(result["roc_auc"], 0.5)

    def test_baseline_reports_all_up_and_training_majority(self):
        result = baseline.run_baseline(self.dataset, tuple(self.feature_names))

        self.assertEqual(result["benchmarks"]["all_up"]["validation"]["predicted_class"], 1)
        self.assertIn("training_majority", result["benchmarks"])

    def test_baseline_rejects_single_class_splits(self):
        single_class = self.dataset.copy()
        single_class["target_up"] = 1

        with self.assertRaisesRegex(ValueError, "single target class"):
            baseline.run_baseline(single_class, self.feature_names)


class BaselineCliTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_path = self.root / "btc_4h.csv"
        self.manifest_path = self.root / "feature_manifest.json"
        self.output_path = self.root / "baseline_btc_4h.json"
        self.feature_names = tuple(f"feature_{index}" for index in range(53))
        timestamps = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")
        dataset = pd.DataFrame({"open_time": timestamps})
        for index, name in enumerate(self.feature_names):
            dataset[name] = np.arange(40, dtype=float) + index
        dataset["future_log_return"] = np.linspace(-1.0, 1.0, 40)
        dataset["target_up"] = [index % 2 for index in range(40)]
        dataset["is_imputed"] = 0
        dataset.to_csv(self.data_path, index=False)
        self.manifest_path.write_text(
            json.dumps({"feature_count": 53, "feature_names": list(self.feature_names)}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cli_writes_complete_report_only_after_success(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--data",
                str(self.data_path),
                "--manifest",
                str(self.manifest_path),
                "--output",
                str(self.output_path),
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["task"], "BTCUSDT 4-hour direction classification")
        self.assertEqual(len(report["feature_names"]), 53)
        self.assertEqual(set(report["splits"]), {"train", "validation", "test"})
        self.assertIn("test", report["model"])
        self.assertIn("gain_importance", report["model"])
        self.assertEqual(report["model_settings"]["random_state"], 42)

    def test_cli_does_not_write_output_when_training_fails(self):
        bad_data_path = self.root / "bad_btc_4h.csv"
        pd.read_csv(self.data_path).assign(target_up=1).to_csv(bad_data_path, index=False)

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--data",
                str(bad_data_path),
                "--manifest",
                str(self.manifest_path),
                "--output",
                str(self.output_path),
            ],
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()
