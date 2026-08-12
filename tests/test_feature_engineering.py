import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from feature_engineering import (  # noqa: E402
    EXPECTED_FEATURE_NAMES,
    add_features,
    build_feature_dataset,
    generate_feature_datasets,
)


def make_raw_frame(rows: int = 96) -> pd.DataFrame:
    open_time = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    step = np.arange(rows, dtype=float)
    close = 100.0 + 0.2 * step + 2.0 * np.sin(step / 4.0)
    frame = pd.DataFrame(
        {
            "open_time": open_time,
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": 1000.0 + step * 3.0,
            "close_time": open_time + pd.Timedelta(minutes=59),
            "quote_asset_volume": close * (1000.0 + step * 3.0),
            "number_of_trades": 100.0 + step,
            "taker_buy_base": 450.0 + step,
            "taker_buy_quote": close * (450.0 + step),
            "is_imputed": (step == 40).astype(int),
        }
    )
    return pd.concat([frame.iloc[::-1], frame.iloc[[0]]], ignore_index=True)


class FeatureEngineeringTests(unittest.TestCase):
    def test_add_features_matches_original_53_feature_contract(self):
        dataset = add_features(make_raw_frame(), prediction_horizon=1)

        self.assertEqual(list(dataset.columns), [*EXPECTED_FEATURE_NAMES, "future_log_return", "target_up"])
        self.assertEqual(len(EXPECTED_FEATURE_NAMES), 53)
        self.assertNotIn("is_imputed", dataset.columns)
        self.assertTrue(dataset.index.is_monotonic_increasing)
        self.assertTrue(dataset.index.is_unique)
        self.assertFalse(dataset.isna().any().any())

    def test_labels_use_the_requested_future_horizon(self):
        raw = make_raw_frame()
        dataset = add_features(raw, prediction_horizon=4)
        timestamp = pd.Timestamp("2024-01-03", tz="UTC")
        close_by_time = raw.drop_duplicates("open_time").set_index("open_time")["close"]
        expected = np.log(close_by_time.loc[timestamp + pd.Timedelta(hours=4)] / close_by_time.loc[timestamp])

        self.assertAlmostEqual(dataset.loc[timestamp, "future_log_return"], expected)
        self.assertEqual(dataset.loc[timestamp, "target_up"], int(expected > 0))

    def test_output_dataset_preserves_audit_column_but_not_as_a_feature(self):
        output = build_feature_dataset(make_raw_frame(), prediction_horizon=1)

        self.assertEqual(output.columns[0], "open_time")
        self.assertIn("is_imputed", output.columns)
        self.assertEqual(
            list(output.columns[1:]),
            [*EXPECTED_FEATURE_NAMES, "future_log_return", "target_up", "is_imputed"],
        )
        self.assertEqual(int(output["is_imputed"].sum()), 1)

    def test_missing_required_column_is_rejected(self):
        raw = make_raw_frame().drop(columns="close")

        with self.assertRaisesRegex(ValueError, "close"):
            add_features(raw)

    def test_generate_feature_datasets_writes_csv_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "processed"
            output_dir = root / "features"
            manifest_path = root / "logs" / "feature_manifest.json"
            input_dir.mkdir()
            make_raw_frame().to_csv(input_dir / "btc_5y.csv", index=False)

            manifest = generate_feature_datasets(
                input_dir=input_dir,
                output_dir=output_dir,
                manifest_path=manifest_path,
                symbols=("BTCUSDT",),
                horizons=(4,),
            )

            output_path = output_dir / "btc_4h.csv"
            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(manifest["feature_count"], 53)
            self.assertEqual(manifest["horizons_hours"], [4])
            self.assertEqual(len(manifest["datasets"]), 1)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8"))["feature_count"], 53)


if __name__ == "__main__":
    unittest.main()
