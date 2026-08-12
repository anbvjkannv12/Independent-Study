import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fetch_binance_5y import KLINE_COLUMNS  # noqa: E402
from check_data_quality import check_csv, check_project, write_report  # noqa: E402


def valid_row(open_time: str = "2024-01-01T00:00:00Z") -> dict[str, object]:
    return {
        "open_time": open_time,
        "open": "1.0",
        "high": "2.0",
        "low": "1.0",
        "close": "1.5",
        "volume": "10.0",
        "close_time": "2024-01-01T00:59:59.999Z",
        "quote_asset_volume": "15.0",
        "number_of_trades": "1",
        "taker_buy_base": "5.0",
        "taker_buy_quote": "7.0",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KLINE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class CsvCheckTests(unittest.TestCase):
    def test_check_csv_returns_warning_for_missing_intervals(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "btc_5y.csv"
            write_csv(path, [valid_row(), valid_row("2024-01-01T02:00:00Z")])

            result = check_csv(path, 3_600_000)

            self.assertEqual(result["status"], "warning")
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["warnings"][0]["type"], "missing_intervals")

    def test_check_csv_reports_invalid_rows(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "btc_5y.csv"
            row = valid_row()
            row["high"] = "0.5"
            write_csv(path, [row])

            result = check_csv(path, 3_600_000)

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["errors"][0]["type"], "validation_error")


class ProjectCheckTests(unittest.TestCase):
    def make_config(self, directory: Path) -> dict[str, object]:
        return {
            "symbols": ["BTCUSDT"],
            "interval_ms": 3_600_000,
            "output_dir": str(directory / "data"),
        }

    def test_check_project_reports_missing_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"symbols": []}), encoding="utf-8")

            result = check_project(self.make_config(root), manifest_path)

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["errors"][0]["type"], "missing_file")

    def test_check_project_reports_manifest_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "data" / "btc_5y.csv"
            write_csv(csv_path, [valid_row()])
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "symbols": [
                            {
                                "symbol": "BTCUSDT",
                                "row_count": 99,
                                "first_open_time": "2024-01-01T00:00:00Z",
                                "last_open_time": "2024-01-01T00:00:00Z",
                                "missing_intervals": [],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = check_project(self.make_config(root), manifest_path)

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["errors"][0]["type"], "manifest_mismatch")

    def test_fail_on_warnings_changes_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(
                root / "data" / "btc_5y.csv",
                [valid_row(), valid_row("2024-01-01T02:00:00Z")],
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "symbols": [
                            {
                                "symbol": "BTCUSDT",
                                "row_count": 2,
                                "first_open_time": "2024-01-01T00:00:00Z",
                                "last_open_time": "2024-01-01T02:00:00Z",
                                "missing_intervals": [
                                    {
                                        "from": "2024-01-01T01:00:00Z",
                                        "to": "2024-01-01T01:00:00Z",
                                        "count": 1,
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = check_project(self.make_config(root), manifest_path, fail_on_warnings=True)

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["warnings"][0]["type"], "missing_intervals")

    def test_write_report_creates_json(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.json"
            report = {"status": "passed", "symbols": [], "errors": [], "warnings": []}

            write_report(report, path)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "passed")


if __name__ == "__main__":
    unittest.main()
