import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fetch_binance_5y import KLINE_COLUMNS  # noqa: E402
from impute_missing_klines import impute_rows, process_project  # noqa: E402


def valid_row(open_time: str = "2024-01-01T00:00:00Z", close: str = "10.0") -> dict[str, object]:
    return {
        "open_time": open_time,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": "5.0",
        "close_time": open_time.replace(":00:00Z", ":59:59.999Z"),
        "quote_asset_volume": "50.0",
        "number_of_trades": "7",
        "taker_buy_base": "2.0",
        "taker_buy_quote": "20.0",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KLINE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class ImputeRowsTests(unittest.TestCase):
    def test_impute_rows_inserts_missing_hour_with_flag(self):
        rows = [
            valid_row("2024-01-01T00:00:00Z", close="10.0"),
            valid_row("2024-01-01T02:00:00Z", close="12.0"),
        ]

        processed, gaps = impute_rows(rows, 3_600_000)

        self.assertEqual([row["open_time"] for row in processed], [
            "2024-01-01T00:00:00Z",
            "2024-01-01T01:00:00Z",
            "2024-01-01T02:00:00Z",
        ])
        self.assertEqual(processed[1]["open"], "10.0")
        self.assertEqual(processed[1]["high"], "10.0")
        self.assertEqual(processed[1]["low"], "10.0")
        self.assertEqual(processed[1]["close"], "10.0")
        self.assertEqual(processed[1]["volume"], "0")
        self.assertEqual(processed[1]["quote_asset_volume"], "0")
        self.assertEqual(processed[1]["number_of_trades"], "0")
        self.assertEqual(processed[1]["is_imputed"], "1")
        self.assertEqual(processed[0]["is_imputed"], "0")
        self.assertEqual(
            gaps,
            [{"from": "2024-01-01T01:00:00Z", "to": "2024-01-01T01:00:00Z", "count": 1}],
        )

    def test_impute_rows_adds_only_flags_when_no_gap_exists(self):
        rows = [
            valid_row("2024-01-01T00:00:00Z", close="10.0"),
            valid_row("2024-01-01T01:00:00Z", close="11.0"),
        ]

        processed, gaps = impute_rows(rows, 3_600_000)

        self.assertEqual(gaps, [])
        self.assertEqual(len(processed), 2)
        self.assertEqual([row["is_imputed"] for row in processed], ["0", "0"])


class ProjectProcessingTests(unittest.TestCase):
    def test_process_project_writes_processed_csv_and_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "data" / "raw" / "btc_5y.csv"
            rows = [
                valid_row("2024-01-01T00:00:00Z", close="10.0"),
                valid_row("2024-01-01T02:00:00Z", close="12.0"),
            ]
            write_csv(raw_path, rows)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "symbols": ["BTCUSDT"],
                        "interval": "1h",
                        "interval_ms": 3_600_000,
                        "output_dir": "data/raw",
                    }
                ),
                encoding="utf-8",
            )

            manifest = process_project(config_path)

            processed_path = root / "data" / "processed" / "btc_5y.csv"
            processed_rows = read_csv(processed_path)
            self.assertEqual(len(read_csv(raw_path)), 2)
            self.assertEqual(len(processed_rows), 3)
            self.assertEqual(processed_rows[1]["is_imputed"], "1")
            self.assertEqual(manifest["symbols"][0]["imputed_count"], 1)
            self.assertTrue((root / "logs" / "imputation_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
