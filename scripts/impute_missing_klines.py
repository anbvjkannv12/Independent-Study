from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fetch_binance_5y import KLINE_COLUMNS, load_config, validate_rows


PROCESSED_COLUMNS = (*KLINE_COLUMNS, "is_imputed")
PRICE_COLUMNS = ("open", "high", "low", "close")
ACTIVITY_COLUMNS = ("volume", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote")


def _symbol_filename(symbol: str) -> str:
    return f"{symbol.lower().replace('usdt', '')}_5y.csv"


def _parse_utc(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _format_utc(value: datetime) -> str:
    formatted = value.astimezone(UTC).isoformat(timespec="milliseconds")
    return formatted.replace("+00:00", "Z").replace(".000Z", "Z")


def _load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strip_imputation_flag(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{column: row[column] for column in KLINE_COLUMNS} for row in rows]


def _with_flag(row: dict[str, object], value: str) -> dict[str, object]:
    flagged = {column: row[column] for column in KLINE_COLUMNS}
    flagged["is_imputed"] = value
    return flagged


def _imputed_row(previous_row: dict[str, object], open_time: datetime, interval_ms: int) -> dict[str, object]:
    close_value = str(previous_row["close"])
    row: dict[str, object] = {
        "open_time": _format_utc(open_time),
        "close_time": _format_utc(open_time + timedelta(milliseconds=interval_ms - 1)),
        "is_imputed": "1",
    }
    for column in PRICE_COLUMNS:
        row[column] = close_value
    for column in ACTIVITY_COLUMNS:
        row[column] = "0"
    return {column: row[column] for column in PROCESSED_COLUMNS}


def impute_rows(rows: list[dict[str, object]], interval_ms: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    validate_rows(rows, interval_ms)

    processed: list[dict[str, object]] = [_with_flag(rows[0], "0")]
    gaps: list[dict[str, object]] = []
    interval = timedelta(milliseconds=interval_ms)

    for previous, current in zip(rows, rows[1:]):
        previous_open_time = _parse_utc(previous["open_time"])
        current_open_time = _parse_utc(current["open_time"])
        missing_count = int((current_open_time - previous_open_time) / interval) - 1
        if missing_count > 0:
            first_missing = previous_open_time + interval
            last_missing = previous_open_time + interval * missing_count
            gaps.append(
                {
                    "from": _format_utc(first_missing),
                    "to": _format_utc(last_missing),
                    "count": missing_count,
                }
            )
            for index in range(1, missing_count + 1):
                processed.append(_imputed_row(previous, previous_open_time + interval * index, interval_ms))
        processed.append(_with_flag(current, "0"))

    validate_rows(_strip_imputation_flag(processed), interval_ms)
    return processed, gaps


def write_processed_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROCESSED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def process_symbol(symbol: str, raw_path: Path, processed_path: Path, interval_ms: int) -> dict[str, object]:
    rows = _load_rows(raw_path)
    processed_rows, gaps = impute_rows(rows, interval_ms)
    write_processed_csv(processed_path, processed_rows)
    return {
        "symbol": symbol,
        "input_path": str(raw_path),
        "output_path": str(processed_path),
        "original_count": len(rows),
        "processed_count": len(processed_rows),
        "imputed_count": len(processed_rows) - len(rows),
        "gaps": gaps,
    }


def process_project(
    config_path: Path,
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    project_root = config_path.parent
    raw_dir = Path(str(config.get("output_dir", "data/raw")))
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir

    resolved_output_dir = output_dir or project_root / "data" / "processed"
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = project_root / resolved_output_dir
    resolved_manifest_path = manifest_path or project_root / "logs" / "imputation_manifest.json"
    if not resolved_manifest_path.is_absolute():
        resolved_manifest_path = project_root / resolved_manifest_path

    entries = []
    for configured_symbol in config.get("symbols", []):
        symbol = str(configured_symbol)
        entries.append(
            process_symbol(
                symbol,
                raw_dir / _symbol_filename(symbol),
                resolved_output_dir / _symbol_filename(symbol),
                int(config["interval_ms"]),
            )
        )

    manifest: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "input_dir": str(raw_dir),
        "output_dir": str(resolved_output_dir),
        "symbols": entries,
    }
    write_manifest(resolved_manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create processed kline CSV files with missing rows imputed.")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest = process_project(args.config, args.output_dir, args.manifest)
    print("Imputation complete")
    print(f"Symbols processed: {len(manifest['symbols'])}")
    print(f"Rows imputed: {sum(int(entry['imputed_count']) for entry in manifest['symbols'])}")
    print(f"Output dir: {manifest['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
