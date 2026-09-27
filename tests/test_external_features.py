import io
import sys
import tempfile
import unittest
import zipfile
from urllib.error import HTTPError
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from external_features import build_external_features  # noqa: E402
from fetch_external_signals import _metric_is_unavailable, parse_metrics_zip  # noqa: E402
from feature_engineering import attach_external_features, build_feature_dataset  # noqa: E402
from test_feature_engineering import make_raw_frame  # noqa: E402


def write_external_fixtures(directory: Path, start="2024-01-01", days=10, hours=240) -> None:
    dates = pd.date_range(start, periods=days, freq="D", tz="UTC")
    pd.DataFrame(
        {
            "time": dates.strftime("%Y-%m-%d"),
            "AdrActCnt": np.linspace(100, 200, days),
            "CapMVRVCur": np.linspace(1.0, 2.0, days),
            "FlowInExNtv": np.linspace(10, 20, days),
            "FlowOutExNtv": np.linspace(20, 10, days),
        }
    ).to_csv(directory / "onchain_btcusdt.csv", index=False)
    pd.DataFrame(
        {
            "time": dates.strftime("%Y-%m-%d"),
            "fng_value": np.linspace(20, 80, days).round(),
            "fng_class": ["Fear"] * days,
        }
    ).to_csv(directory / "fear_greed.csv", index=False)

    settlements = pd.date_range(start, periods=days * 3, freq="8h", tz="UTC")
    pd.DataFrame(
        {
            "funding_time": settlements.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "funding_rate": np.linspace(-0.001, 0.001, len(settlements)),
        }
    ).to_csv(directory / "funding_btcusdt.csv", index=False)

    stamps = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    pd.DataFrame(
        {
            "time": stamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sum_open_interest": np.linspace(1000, 2000, hours),
            "sum_open_interest_value": np.linspace(1e6, 2e6, hours),
            "count_toptrader_long_short_ratio": np.linspace(0.8, 1.2, hours),
            "sum_toptrader_long_short_ratio": np.linspace(0.9, 1.1, hours),
            "count_long_short_ratio": np.linspace(1.0, 1.5, hours),
            "sum_taker_long_short_vol_ratio": np.linspace(0.95, 1.05, hours),
        }
    ).to_csv(directory / "open_interest_btcusdt.csv", index=False)


class ExternalAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        write_external_fixtures(self.directory)
        self.index = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
        self.features = build_external_features(self.index, self.directory, "BTCUSDT")

    def tearDown(self):
        self.tmp.cleanup()

    def test_daily_sources_are_published_with_one_day_lag(self):
        # The day-1 on-chain value must not be readable during day 1 itself.
        mvrv = self.features["onchain_mvrv"]
        self.assertTrue(mvrv.loc[: pd.Timestamp("2024-01-01 23:00", tz="UTC")].isna().all())
        self.assertAlmostEqual(mvrv.loc[pd.Timestamp("2024-01-02 00:00", tz="UTC")], 1.0)
        self.assertAlmostEqual(mvrv.loc[pd.Timestamp("2024-01-02 23:00", tz="UTC")], 1.0)
        self.assertAlmostEqual(
            self.features["sent_fng"].loc[pd.Timestamp("2024-01-02 06:00", tz="UTC")], 0.20
        )

    def test_funding_uses_the_last_settled_rate_only(self):
        funding = self.features["funding_rate"]
        first, second = funding.iloc[0], funding.loc[pd.Timestamp("2024-01-01 08:00", tz="UTC")]
        self.assertAlmostEqual(first, -0.001)
        self.assertGreater(second, first)
        # 07:00 still sees the 00:00 settlement, not the 08:00 one.
        self.assertAlmostEqual(funding.loc[pd.Timestamp("2024-01-01 07:00", tz="UTC")], first)

    def test_open_interest_changes_are_backward_looking(self):
        self.assertTrue(np.isnan(self.features["oi_log_change_1h"].iloc[0]))
        self.assertGreater(self.features["oi_log_change_1h"].iloc[5], 0)
        self.assertGreater(self.features["oi_account_ls_ratio"].iloc[10], 1.0)

    def test_values_beyond_the_staleness_tolerance_become_nan(self):
        index = pd.date_range("2024-02-01", periods=5, freq="h", tz="UTC")  # far past coverage
        stale = build_external_features(index, self.directory, "BTCUSDT")
        self.assertTrue(stale["funding_rate"].isna().all())
        self.assertTrue(stale["onchain_mvrv"].isna().all())

    def test_missing_sources_yield_fewer_columns_not_fabricated_values(self):
        features = build_external_features(self.index, self.directory, "SOLUSDT")
        self.assertNotIn("onchain_mvrv", features.columns)
        self.assertIn("sent_fng", features.columns)  # fear & greed is market-wide


class AttachToFeatureDatasetTests(unittest.TestCase):
    def test_attach_drops_uncovered_rows_and_keeps_column_order(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            write_external_fixtures(directory)
            raw = make_raw_frame(rows=240)
            raw["open_time"] = pd.date_range("2024-01-01", periods=len(raw), freq="h", tz="UTC").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            dataset = build_feature_dataset(raw, prediction_horizon=4)
            merged, names = attach_external_features(dataset, directory, "BTCUSDT")

            self.assertTrue(names)
            self.assertFalse(merged.isna().any().any())
            self.assertLess(len(merged), len(dataset))
            self.assertEqual(list(merged.columns)[0], "open_time")
            self.assertEqual(list(merged.columns)[-3:], ["future_log_return", "target_up", "is_imputed"])
            self.assertEqual(list(merged.columns)[54 : 54 + len(names)], names)


class MetricsArchiveTests(unittest.TestCase):
    def test_parse_metrics_zip_reads_the_archived_csv(self):
        csv_text = (
            "create_time,symbol,sum_open_interest,sum_open_interest_value,"
            "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
            "count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
            "2021-08-10 00:00:00,BTCUSDT,56591.028,2618651918.22,0.9005,1.0171,0.8633,1.0301\n"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("BTCUSDT-metrics-2021-08-10.csv", csv_text)
        rows = parse_metrics_zip(buffer.getvalue())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["time"], "2021-08-10T00:00:00Z")
        self.assertEqual(rows[0]["sum_open_interest"], "56591.028")


class CoinMetricsCoverageTests(unittest.TestCase):
    """Coin Metrics reports a missing metric two different ways; only those two may be skipped."""

    @staticmethod
    def _error(code: int, body: str) -> HTTPError:
        return HTTPError("https://example.invalid", code, "", {}, io.BytesIO(body.encode("utf-8")))

    def test_tier_restriction_and_missing_metric_are_both_treated_as_unavailable(self):
        # SOL's AdrActCnt is gated by the tier; SOL has no MVRV at all.
        self.assertTrue(_metric_is_unavailable(self._error(403, '{"error":{"type":"forbidden"}}')))
        self.assertTrue(_metric_is_unavailable(self._error(400, '{"error":{"type":"bad_parameter"}}')))

    def test_other_failures_still_surface(self):
        self.assertFalse(_metric_is_unavailable(self._error(400, '{"error":{"type":"unsupported_parameter"}}')))
        self.assertFalse(_metric_is_unavailable(self._error(500, "boom")))
        self.assertFalse(_metric_is_unavailable(self._error(400, "not json")))


if __name__ == "__main__":
    unittest.main()
