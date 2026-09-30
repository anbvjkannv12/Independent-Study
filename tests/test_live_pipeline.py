import json
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from feature_engineering import add_features
from fetch_binance_5y import KLINE_COLUMNS, write_csv_atomic
import live_pipeline as lp


def synthetic_raw(start: str, n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.uniform(10, 20, n)
    times = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "open_time": times.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "open": open_, "high": np.maximum(open_, close) * 1.001, "low": np.minimum(open_, close) * 0.999,
        "close": close, "volume": volume,
        "close_time": (times + pd.Timedelta(minutes=59, seconds=59)).strftime("%Y-%m-%dT%H:%M:%S.999Z"),
        "quote_asset_volume": volume * close, "number_of_trades": rng.integers(100, 200, n),
        "taker_buy_base": volume / 2, "taker_buy_quote": volume * close / 2,
    })


def binance_row(open_time: pd.Timestamp) -> list[object]:
    ms = int(open_time.value // 1_000_000)
    return [ms, "1", "2", "0.5", "1.5", "10", ms + 3_599_999, "15", 5, "5", "7.5", "0"]


class LivePipelineTests(unittest.TestCase):
    def test_only_closed_candles(self):
        now = datetime(2026, 9, 30, 10, 30, tzinfo=UTC)
        self.assertEqual(lp.last_closed_open_ms(now), int(datetime(2026, 9, 30, 9, tzinfo=UTC).timestamp() * 1000))

    def test_incremental_fetch_requests_after_last_row_and_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = lp.Paths(Path(tmp))
            raw = synthetic_raw("2026-09-30 00:00", 5)
            rows = [{c: row[c] for c in KLINE_COLUMNS} for row in raw.to_dict("records")]
            write_csv_atomic(paths.raw_csv("BTC"), rows)
            urls = []

            def fake_request(url, timeout, retries):
                urls.append(url)
                start = int(parse_qs(urlparse(url).query)["startTime"][0])
                # Re-send the last known row too: the merge must not duplicate it.
                return [binance_row(pd.Timestamp(start - lp.HOUR_MS, unit="ms", tz="UTC"))] + [
                    binance_row(pd.Timestamp(start + i * lp.HOUR_MS, unit="ms", tz="UTC")) for i in range(3)]

            fetched = lp.update_raw("BTC", paths, datetime(2026, 9, 30, 8, 10, tzinfo=UTC), "http://x", fake_request)
            self.assertEqual(int(parse_qs(urlparse(urls[0]).query)["startTime"][0]),
                             int(pd.Timestamp("2026-09-30 05:00", tz="UTC").value // 1_000_000))
            merged = pd.read_csv(paths.raw_csv("BTC"))
            self.assertEqual(fetched, 4)
            self.assertEqual(len(merged), 8)  # 00:00..07:00, the 07:00 candle closed at 08:00
            self.assertTrue(merged.open_time.is_unique and merged.open_time.is_monotonic_increasing)
            self.assertEqual(merged.open_time.iloc[-1], "2026-09-30T07:00:00Z")

    def test_keep_unlabeled_rows(self):
        raw = synthetic_raw("2026-01-01 00:00", 200)
        default = add_features(raw, 4)
        live = add_features(raw, 4, keep_unlabeled=True)
        self.assertEqual(len(live), len(default) + 4)
        self.assertTrue(live.future_log_return.iloc[-4:].isna().all())
        self.assertFalse(live[lp.FEATURES].isna().any().any())
        pd.testing.assert_frame_equal(live.loc[default.index], default)

    def test_labels_use_timestamps_and_wait_for_close(self):
        processed = pd.DataFrame({"open_time": pd.date_range("2026-09-01", periods=6, freq="h", tz="UTC"),
                                  "close": [1.0, 2.0, 3.0, 4.0, 8.0, 9.0]}).drop(index=2)
        times = pd.Series(pd.date_range("2026-09-01", periods=2, freq="h", tz="UTC").astype(str))
        labels = lp.realized_labels(processed, times)
        self.assertAlmostEqual(labels.iloc[0], np.log(8.0 / 1.0))
        self.assertAlmostEqual(labels.iloc[1], np.log(9.0 / 2.0))
        later = lp.realized_labels(processed, pd.Series(["2026-09-01 03:00:00+00:00"]))
        self.assertTrue(np.isnan(later.iloc[0]))

    def test_retrain_purges_last_24_labeled_rows(self):
        labeled = pd.DataFrame({"x": range(100)})
        train = lp.rolling_training_rows(labeled)
        self.assertEqual(train.index[-1], labeled.index[-1] - 24)

    def test_rolling_model_valid_at_prediction_time(self):
        metas = [{"model_id": "rolling_20261001", "effective_from": "2026-10-01T03:00:00+00:00"},
                 {"model_id": "rolling_20261101", "effective_from": "2026-11-01T03:00:00+00:00"}]
        pick = lambda t: (lp.rolling_model_for(metas, pd.Timestamp(t, tz="UTC")) or {}).get("model_id")
        self.assertIsNone(pick("2026-10-01 01:00"))
        self.assertEqual(pick("2026-10-01 02:00"), "rolling_20261001")
        self.assertEqual(pick("2026-11-01 01:00"), "rolling_20261001")
        self.assertEqual(pick("2026-11-01 02:00"), "rolling_20261101")

    def test_backfilled_flag(self):
        t = pd.Timestamp("2026-10-01 00:00", tz="UTC")
        self.assertFalse(lp.is_backfilled(t, t + pd.Timedelta(hours=1, minutes=5)))
        self.assertTrue(lp.is_backfilled(t, t + pd.Timedelta(hours=3)))

    def test_predictions_are_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = lp.Paths(Path(tmp))
            paths.models.mkdir(parents=True)
            features = add_features(synthetic_raw("2026-08-01 00:00", 400), 4, keep_unlabeled=True)
            train = features[features.index <= lp.FROZEN_CUTOFF]
            lp.train_model(train, train.future_log_return > 0).save_model(paths.frozen_model("BTC"))
            now = pd.Timestamp("2026-08-20 00:00", tz="UTC")  # after the last synthetic row
            first = lp.new_prediction_rows("BTC", features[lp.FEATURES], lp.read_predictions(paths), paths, now)
            after_cutoff = int((features.index > lp.FROZEN_CUTOFF).sum())
            self.assertEqual(len(first), after_cutoff)
            self.assertTrue((first.model_id == "frozen").all() and (first.backfilled == 1).all())
            lp.write_predictions(paths, first)
            second = lp.new_prediction_rows("BTC", features[lp.FEATURES], lp.read_predictions(paths), paths, now)
            self.assertEqual(len(second), 0)

    def test_feature_consistency_check_uses_real_files_when_present(self):
        root = Path(__file__).resolve().parents[1]
        processed = root / "data/processed/btc_5y.csv"
        if not processed.exists() or not (root / "data/features/btc_4h.csv").exists():
            self.skipTest("local data files not available")
        features = lp.live_features(pd.read_csv(processed))
        lp.check_feature_consistency("BTC", features, root)


if __name__ == "__main__":
    unittest.main()
