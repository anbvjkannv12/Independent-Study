from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fetch_binance_5y import KLINE_COLUMNS, load_config, validate_rows


SUMMARY_FIELDS = ("row_count", "first_open_time", "last_open_time", "missing_intervals")


def _issue(issue_type: str, message: str, **details: Any) -> dict[str, object]:
    result: dict[str, object] = {"type": issue_type, "message": message}
    if details:
        result["details"] = details
    return result


def _symbol_filename(symbol: str) -> str:
    return f"{symbol.lower().replace('usdt', '')}_5y.csv"


def _parse_timestamp(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_csv(path: Path, interval_ms: int) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path),
        "status": "failed",
        "row_count": 0,
        "summary": None,
        "errors": [],
        "warnings": [],
    }
    errors = result["errors"]
    warnings = result["warnings"]
    assert isinstance(errors, list)
    assert isinstance(warnings, list)

    if not path.exists():
        errors.append(_issue("missing_file", f"CSV file does not exist: {path}"))
        return result

    try:
        rows = _load_rows(path)
    except (OSError, UnicodeError, csv.Error) as error:
        errors.append(_issue("read_error", f"Could not read CSV file: {error}"))
        return result

    result["row_count"] = len(rows)
    try:
        for row in rows:
            _parse_timestamp(row["open_time"])
            _parse_timestamp(row["close_time"])
        summary = validate_rows(rows, interval_ms)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        errors.append(_issue("validation_error", str(error)))
        return result

    result["summary"] = summary
    missing_intervals = summary["missing_intervals"]
    if missing_intervals:
        warnings.append(
            _issue(
                "missing_intervals",
                "One or more expected interval timestamps are missing",
                intervals=missing_intervals,
            )
        )
    result["status"] = "warning" if warnings else "passed"
    return result


def _read_manifest(path: Path) -> dict[str, dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("symbols", [])
    return {str(entry["symbol"]): entry for entry in entries}


def _compare_manifest(symbol: str, summary: dict[str, object], entry: dict[str, object]) -> dict[str, object] | None:
    mismatches: dict[str, dict[str, object]] = {}
    for field in SUMMARY_FIELDS:
        actual = summary.get(field)
        expected = entry.get(field)
        if actual != expected:
            mismatches[field] = {"csv": actual, "manifest": expected}
    if not mismatches:
        return None
    return _issue(
        "manifest_mismatch",
        f"CSV summary does not match manifest for {symbol}",
        fields=mismatches,
    )


def check_project(
    config: dict[str, object],
    manifest_path: Path,
    *,
    fail_on_warnings: bool = False,
) -> dict[str, object]:
    base_dir = manifest_path.parent
    output_dir = Path(str(config.get("output_dir", "data/raw")))
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "passed",
        "symbols": [],
        "errors": [],
        "warnings": [],
    }
    symbol_results = report["symbols"]
    errors = report["errors"]
    warnings = report["warnings"]
    assert isinstance(symbol_results, list)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)

    manifest_entries: dict[str, dict[str, object]] = {}
    if not manifest_path.exists():
        errors.append(_issue("missing_manifest", f"Manifest file does not exist: {manifest_path}"))
    else:
        try:
            manifest_entries = _read_manifest(manifest_path)
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            errors.append(_issue("manifest_error", f"Could not read manifest: {error}"))

    interval_ms = int(config["interval_ms"])
    for configured_symbol in config.get("symbols", []):
        symbol = str(configured_symbol)
        csv_path = output_dir / _symbol_filename(symbol)
        symbol_result = check_csv(csv_path, interval_ms)
        symbol_result["symbol"] = symbol
        symbol_results.append(symbol_result)

        for error in symbol_result["errors"]:
            errors.append({"symbol": symbol, **error})
        for warning in symbol_result["warnings"]:
            warnings.append({"symbol": symbol, **warning})

        summary = symbol_result["summary"]
        if isinstance(summary, dict):
            manifest_entry = manifest_entries.get(symbol)
            if manifest_entry is None:
                errors.append(_issue("manifest_missing_symbol", f"Manifest has no entry for {symbol}", symbol=symbol))
            else:
                mismatch = _compare_manifest(symbol, summary, manifest_entry)
                if mismatch:
                    errors.append({"symbol": symbol, **mismatch})

    if errors or (fail_on_warnings and warnings):
        report["status"] = "failed"
    elif warnings:
        report["status"] = "warning"
    return report


def write_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Binance OHLCV CSV data quality.")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    config = load_config(config_path)
    project_root = config_path.parent
    manifest_path = args.manifest or project_root / str(config.get("manifest_path", "logs/fetch_manifest.json"))
    report_path = args.report or project_root / "logs/data_quality_report.json"
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    if not report_path.is_absolute():
        report_path = project_root / report_path

    check_config = dict(config)
    output_dir = Path(str(check_config.get("output_dir", "data/raw")))
    if not output_dir.is_absolute():
        check_config["output_dir"] = str(project_root / output_dir)
    report = check_project(check_config, manifest_path, fail_on_warnings=args.fail_on_warnings)
    write_report(report, report_path)
    print(f"Data quality: {report['status']}")
    print(f"Symbols checked: {len(report['symbols'])}")
    print(f"Errors: {len(report['errors'])}; warnings: {len(report['warnings'])}")
    print(f"Report: {report_path}")
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
