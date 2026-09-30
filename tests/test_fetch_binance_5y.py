import json
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fetch_binance_5y import (  # noqa: E402
    KLINE_COLUMNS,
    build_manifest_entry,
    fetch_klines,
    find_missing_open_times,
    load_config,
    map_kline_row,
    request_json,
    resolve_time_window,
    run_symbol,
    subtract_years,
    validate_rows,
    write_csv_atomic,
)


def valid_row(open_time: str = "2024-01-01T00:00:00Z") -> dict[str, object]:
    return {
        "open_time": open_time,
        "open": 1.0,
        "high": 2.0,
        "low": 1.0,
        "close": 1.5,
        "volume": 10.0,
        "close_time": "2024-01-01T00:59:59.999Z",
        "quote_asset_volume": 15.0,
        "number_of_trades": 1,
        "taker_buy_base": 5.0,
        "taker_buy_quote": 7.0,
    }


class FetchContractTests(unittest.TestCase):
    def test_map_kline_row_uses_expected_columns_and_types(self):
        row = [
            1704067200000,
            "42000.1",
            "42100.2",
            "41900.0",
            "42050.3",
            "12.5",
            1704070799999,
            "525000.4",
            1234,
            "6.2",
            "260000.7",
            "0",
        ]
        mapped = map_kline_row(row)
        self.assertEqual(tuple(mapped), KLINE_COLUMNS)
        self.assertEqual(mapped["open_time"], "2024-01-01T00:00:00Z")
        self.assertEqual(mapped["close"], 42050.3)
        self.assertEqual(mapped["number_of_trades"], 1234)

    def test_find_missing_open_times_reports_one_hour_gap(self):
        missing = find_missing_open_times([0, 3_600_000, 10_800_000], 3_600_000)
        self.assertEqual(missing, [{"from": 7_200_000, "to": 7_200_000, "count": 1}])

    def test_validate_rows_rejects_duplicate_open_times(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows([valid_row(), valid_row()], 3_600_000)


class PaginationTests(unittest.TestCase):
    def test_fetch_klines_advances_by_last_open_time_plus_interval(self):
        calls = []

        def fake_request(url, timeout_seconds, max_retries):
            calls.append(url)
            if len(calls) == 1:
                return [[0, "1", "2", "1", "1", "1", 0, "1", 1, "1", "1", "0"]]
            return []

        rows = fetch_klines("BTCUSDT", "1h", 0, 3_600_000, fake_request)
        self.assertEqual(len(rows), 1)
        self.assertIn("startTime=3600000", calls[1])

    def test_request_json_retries_http_429_then_succeeds(self):
        attempts = []

        class FakeResponse:
            def read(self):
                return json.dumps({"ok": True}).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_opener(request, timeout):
            attempts.append(request.full_url)
            if len(attempts) == 1:
                raise HTTPError(request.full_url, 429, "rate limit", {}, None)
            return FakeResponse()

        result = request_json(
            "https://api.binance.com/api/v3/time",
            timeout_seconds=1,
            max_retries=2,
            opener=fake_opener,
            sleep=lambda _: None,
        )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(attempts), 2)


class OutputTests(unittest.TestCase):
    def test_write_csv_atomic_does_not_replace_target_when_validation_fails(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "btc_5y.csv"
            target.write_text("old", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_csv_atomic(target, [valid_row(), valid_row()])
            self.assertEqual(target.read_text(encoding="utf-8"), "old")

    def test_manifest_entry_contains_quality_summary(self):
        entry = build_manifest_entry("BTCUSDT", [valid_row()], "success", None, 3_600_000)
        self.assertEqual(entry["status"], "success")
        self.assertEqual(entry["row_count"], 1)
        self.assertIn("missing_intervals", entry)


class ConfigurationTests(unittest.TestCase):
    def test_subtract_years_clamps_leap_day_to_february_28(self):
        value = datetime(2024, 2, 29, tzinfo=UTC)
        self.assertEqual(subtract_years(value, 5), datetime(2019, 2, 28, tzinfo=UTC))

    def test_run_symbol_writes_csv_and_returns_manifest_entry(self):
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "btc_5y.csv"

            def fake_request(url, timeout_seconds, max_retries):
                if "startTime=0" in url:
                    return [[0, "1", "2", "1", "1.5", "1", 3599999, "1.5", 1, "0.5", "0.75", "0"]]
                return []

            config = {
                "base_url": "https://api.binance.com",
                "interval": "1h",
                "interval_ms": 3_600_000,
                "page_limit": 1000,
                "timeout_seconds": 1,
                "max_retries": 1,
                "request_delay_seconds": 0,
                "output_path": str(output_path),
                "start_ms": 0,
                "end_ms": 0,
            }
            entry = run_symbol("BTCUSDT", config, fake_request)
            self.assertEqual(entry["status"], "success")
            self.assertTrue(output_path.exists())

    def test_resolve_time_window_ends_at_last_complete_hour(self):
        now = datetime(2026, 8, 9, 12, 34, tzinfo=UTC)
        start_ms, end_ms = resolve_time_window({"interval_ms": 3_600_000, "years": 5}, now)
        self.assertEqual(datetime.fromtimestamp(end_ms / 1000, tz=UTC).isoformat(), "2026-08-09T11:00:00+00:00")
        self.assertEqual(datetime.fromtimestamp(start_ms / 1000, tz=UTC).isoformat(), "2021-08-09T11:00:00+00:00")

    def test_load_config_applies_safe_defaults(self):
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps({"symbols": ["BTCUSDT"]}), encoding="utf-8")
            config = load_config(config_path)
            self.assertEqual(config["interval"], "1h")
            self.assertEqual(config["interval_ms"], 3_600_000)
            self.assertEqual(config["years"], 5)


class DocumentationTests(unittest.TestCase):
    def test_documentation_has_required_paths_and_security_note(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        agement = (PROJECT_ROOT / "agement.md").read_text(encoding="utf-8")
        self.assertIn("BTCUSDT", readme)
        self.assertIn("XRPUSDT", readme)
        self.assertIn("不使用", readme)
        self.assertIn("API key", readme)
        self.assertIn("交接使用說明.md", agement)

    def test_run_notebook_is_valid_json(self):
        notebook = json.loads((PROJECT_ROOT / "run" / "01_fetch_and_validate.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertTrue(any("fetch_binance_5y.py" in "".join(cell["source"]) for cell in notebook["cells"]))


if __name__ == "__main__":
    unittest.main()
