from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
)

PRICE_COLUMNS = {"open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"}
INTERVAL_MS = {"1h": 3_600_000}
DEFAULT_BASE_URL = "https://api.binance.com"
DEFAULT_INTERVAL = "1h"
DEFAULT_PAGE_LIMIT = 1000
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 5


def _format_utc(milliseconds: int) -> str:
    value = datetime.fromtimestamp(milliseconds / 1000, tz=UTC).isoformat(timespec="milliseconds")
    return value.replace("+00:00", "Z").replace(".000Z", "Z")


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _to_milliseconds(value: str) -> int:
    return int(_parse_utc(value).timestamp() * 1000)


def subtract_years(value: datetime, years: int) -> datetime:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def map_kline_row(row: list[object]) -> dict[str, object]:
    if len(row) < 12:
        raise ValueError("Binance kline row must contain at least 12 values")
    return {
        "open_time": _format_utc(int(row[0])),
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
        "close_time": _format_utc(int(row[6])),
        "quote_asset_volume": float(row[7]),
        "number_of_trades": int(row[8]),
        "taker_buy_base": float(row[9]),
        "taker_buy_quote": float(row[10]),
    }


def find_missing_open_times(open_times: list[int], interval_ms: int) -> list[dict[str, int]]:
    missing: list[dict[str, int]] = []
    for previous, current in zip(open_times, open_times[1:]):
        gap_count = (current - previous) // interval_ms - 1
        if gap_count <= 0:
            continue
        first_missing = previous + interval_ms
        last_missing = previous + interval_ms * gap_count
        missing.append({"from": first_missing, "to": last_missing, "count": gap_count})
    return missing


def validate_rows(rows: list[dict[str, object]], interval_ms: int) -> dict[str, object]:
    if not rows:
        raise ValueError("no kline rows returned")
    if any(tuple(row.keys()) != KLINE_COLUMNS for row in rows):
        raise ValueError("unexpected CSV columns")

    open_times = [_to_milliseconds(str(row["open_time"])) for row in rows]
    if len(open_times) != len(set(open_times)):
        raise ValueError("duplicate open_time values")
    if open_times != sorted(open_times):
        raise ValueError("open_time values are not sorted")

    for row in rows:
        numeric_values = [float(row[column]) for column in PRICE_COLUMNS]
        if any(value < 0 for value in numeric_values):
            raise ValueError("OHLCV values cannot be negative")
        if float(row["high"]) < max(float(row["open"]), float(row["close"])):
            raise ValueError("high is below open or close")
        if float(row["low"]) > min(float(row["open"]), float(row["close"])):
            raise ValueError("low is above open or close")
        if int(row["number_of_trades"]) < 0:
            raise ValueError("number_of_trades cannot be negative")

    missing = find_missing_open_times(open_times, interval_ms)
    return {
        "row_count": len(rows),
        "first_open_time": rows[0]["open_time"],
        "last_open_time": rows[-1]["open_time"],
        "duplicate_count": 0,
        "missing_intervals": [
            {"from": _format_utc(item["from"]), "to": _format_utc(item["to"]), "count": item["count"]}
            for item in missing
        ],
    }


def request_json(
    url: str,
    timeout_seconds: int,
    max_retries: int,
    opener: Callable[..., Any] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        request = Request(url, headers={"User-Agent": "five-year-binance-data-project/1.0"})
        try:
            with opener(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            last_error = error
            retryable = error.code == 429 or 500 <= error.code <= 599
        except URLError as error:
            last_error = error
            retryable = True
        if not retryable or attempt >= max_retries:
            raise last_error
        sleep(min(60.0, 2**attempt))
    raise RuntimeError("request retry loop exited unexpectedly")


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    request_json_fn: Callable[[str, int, int], Any],
    *,
    base_url: str = DEFAULT_BASE_URL,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    request_delay_seconds: float = 0.0,
) -> list[list[object]]:
    interval_ms = INTERVAL_MS[interval]
    cursor = start_ms
    rows: list[list[object]] = []
    while cursor <= end_ms:
        query = urlencode(
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": page_limit,
            }
        )
        url = f"{base_url.rstrip('/')}/api/v3/klines?{query}"
        page = request_json_fn(url, timeout_seconds, max_retries)
        if not page:
            break
        rows.extend(page)
        last_open_time = int(page[-1][0])
        next_cursor = last_open_time + interval_ms
        if next_cursor <= cursor:
            raise RuntimeError("Binance pagination cursor did not advance")
        cursor = next_cursor
        if last_open_time >= end_ms:
            break
        if request_delay_seconds:
            time.sleep(request_delay_seconds)
    return rows


def write_csv_atomic(path: Path, rows: list[dict[str, object]], interval_ms: int = 3_600_000) -> None:
    validate_rows(rows, interval_ms)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=KLINE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with temporary_path.open("r", encoding="utf-8", newline="") as handle:
            written_rows = list(csv.DictReader(handle))
        if len(written_rows) != len(rows):
            raise ValueError("temporary CSV row count changed during write")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def build_manifest_entry(
    symbol: str,
    rows: list[dict[str, object]],
    status: str,
    error: str | None,
    interval_ms: int = 3_600_000,
) -> dict[str, object]:
    entry: dict[str, object] = {"symbol": symbol, "status": status, "error": error}
    if rows:
        entry.update(validate_rows(rows, interval_ms))
    else:
        entry.update(
            {
                "row_count": 0,
                "first_open_time": None,
                "last_open_time": None,
                "duplicate_count": 0,
                "missing_intervals": [],
            }
        )
    return entry


def run_symbol(
    symbol: str,
    config: dict[str, object],
    request_json_fn: Callable[[str, int, int], Any] = request_json,
) -> dict[str, object]:
    interval = str(config["interval"])
    interval_ms = int(config["interval_ms"])
    rows: list[dict[str, object]] = []
    try:
        raw_rows = fetch_klines(
            symbol,
            interval,
            int(config["start_ms"]),
            int(config["end_ms"]),
            request_json_fn,
            base_url=str(config["base_url"]),
            page_limit=int(config["page_limit"]),
            timeout_seconds=int(config["timeout_seconds"]),
            max_retries=int(config["max_retries"]),
            request_delay_seconds=float(config["request_delay_seconds"]),
        )
        rows = [map_kline_row(row) for row in raw_rows]
        rows.sort(key=lambda row: _to_milliseconds(str(row["open_time"])))
        output_path = Path(str(config["output_path"]))
        write_csv_atomic(output_path, rows, interval_ms)
        return build_manifest_entry(symbol, rows, "success", None, interval_ms)
    except Exception as error:
        return build_manifest_entry(symbol, rows, "error", str(error), interval_ms)


def load_config(config_path: Path) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    interval = str(config.get("interval", DEFAULT_INTERVAL))
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported interval: {interval}")
    config["interval"] = interval
    config["interval_ms"] = int(config.get("interval_ms", INTERVAL_MS[interval]))
    config["page_limit"] = int(config.get("page_limit", DEFAULT_PAGE_LIMIT))
    config["timeout_seconds"] = int(config.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    config["max_retries"] = int(config.get("max_retries", DEFAULT_MAX_RETRIES))
    config["request_delay_seconds"] = float(config.get("request_delay_seconds", 0.2))
    config["base_url"] = str(config.get("base_url", DEFAULT_BASE_URL))
    config["years"] = int(config.get("years", 5))
    return config


def resolve_time_window(config: dict[str, object], now: datetime | None = None) -> tuple[int, int]:
    current = (now or datetime.now(UTC)).astimezone(UTC)
    interval_ms = int(config["interval_ms"])
    current_ms = int(current.timestamp() * 1000)
    end_ms = (current_ms // interval_ms) * interval_ms - interval_ms
    end_datetime = datetime.fromtimestamp(end_ms / 1000, tz=UTC)
    start_datetime = subtract_years(end_datetime, int(config["years"]))
    return int(start_datetime.timestamp() * 1000), end_ms


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download five years of Binance Spot 1h klines.")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    config = load_config(config_path)
    project_root = config_path.parent
    start_ms, end_ms = resolve_time_window(config)
    config["start_ms"] = start_ms
    config["end_ms"] = end_ms
    symbols = [str(symbol) for symbol in config["symbols"]]
    output_dir = project_root / str(config.get("output_dir", "data/raw"))
    manifest_path = project_root / str(config.get("manifest_path", "logs/fetch_manifest.json"))

    if args.dry_run:
        print(json.dumps({"symbols": symbols, "interval": config["interval"], "start_ms": start_ms, "end_ms": end_ms, "endpoint": f"{config['base_url']}/api/v3/klines"}, indent=2))
        return 0

    generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    entries: list[dict[str, object]] = []
    for symbol in symbols:
        symbol_config = dict(config)
        symbol_config["output_path"] = str(output_dir / f"{symbol.lower().replace('usdt', '')}_5y.csv")
        print(f"Downloading {symbol} ...", flush=True)
        entry = run_symbol(symbol, symbol_config)
        entries.append(entry)
        print(f"{symbol}: {entry['status']} ({entry['row_count']} rows)", flush=True)

    manifest = {
        "generated_at": generated_at,
        "config": {key: value for key, value in config.items() if key not in {"output_path"}},
        "symbols": entries,
    }
    write_manifest(manifest_path, manifest)
    return 0 if all(entry["status"] == "success" for entry in entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
