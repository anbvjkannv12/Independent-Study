"""Fetch on-chain, sentiment and derivatives signals from key-free public sources.

Raw downloads land in ``data/external/``; alignment and leakage control happen in
``scripts/external_features.py``. No API key is read or stored.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
COINMETRICS_METRICS = ("AdrActCnt", "CapMVRVCur", "FlowInExNtv", "FlowOutExNtv")
COINMETRICS_ASSETS = dict(zip(("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"), ("btc", "eth", "sol", "xrp")))
FAPI_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
VISION_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
FNG_URL = "https://api.alternative.me/fng/"

OI_COLUMNS = (
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)

SOURCES = ("onchain", "funding", "open_interest", "fear_greed")
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 5
DEFAULT_DELAY = 0.2


def _request(url: str, timeout: int = DEFAULT_TIMEOUT, max_retries: int = DEFAULT_MAX_RETRIES, delay: float = 0.0) -> bytes:
    """GET with retry on 429/5xx, returning the raw body. 404 is raised for the caller to interpret."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        request = Request(url, headers={"User-Agent": "five-year-external-signals/1.0"})
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if delay:
                time.sleep(delay)
            return payload
        except HTTPError as error:
            last_error = error
            retryable = error.code == 429 or 500 <= error.code <= 599
        except URLError as error:
            last_error = error
            retryable = True
        if not retryable or attempt >= max_retries:
            raise last_error
        time.sleep(min(60.0, 2**attempt))
    raise RuntimeError("request retry loop exited unexpectedly")


def _json(url: str, timeout: int = DEFAULT_TIMEOUT, max_retries: int = DEFAULT_MAX_RETRIES, delay: float = 0.0) -> Any:
    return json.loads(_request(url, timeout, max_retries, delay).decode("utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str]) -> None:
    columns = list(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _utc_ms(value: date) -> int:
    return int(datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp() * 1000)


def _iso(milliseconds: int) -> str:
    stamp = datetime.fromtimestamp(milliseconds / 1000, tz=UTC)
    return stamp.isoformat(timespec="seconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- on-chain


def _coinmetrics_page(asset: str, metrics: Iterable[str], start: date, end: date, **kwargs: Any) -> Any:
    query = urlencode(
        {
            "assets": asset,
            "metrics": ",".join(metrics),
            "frequency": "1d",
            "page_size": 10000,
            "start_time": str(start),
            "end_time": str(end),
        }
    )
    return _json(f"{COINMETRICS_URL}?{query}", **kwargs)


def _metric_is_unavailable(error: HTTPError) -> bool:
    """Coin Metrics answers 403 when the tier forbids a metric and 400/bad_parameter when the
    asset simply has no such metric (SOL has no MVRV, XRP has no exchange flows). Both mean
    "not available here"; any other 400 is a real query bug and must surface."""
    if error.code in (403, 404):
        return True
    if error.code != 400:
        return False
    try:
        return json.loads(error.read().decode("utf-8"))["error"]["type"] == "bad_parameter"
    except (ValueError, KeyError):
        return False


def fetch_onchain(asset: str, start: date, end: date, **kwargs: Any) -> list[dict[str, Any]]:
    """Community tier coverage differs per asset, so unsupported metrics are skipped, not fatal."""
    available: list[str] = []
    for metric in COINMETRICS_METRICS:
        try:
            _coinmetrics_page(asset, (metric,), start, start, **kwargs)
        except HTTPError as error:
            if _metric_is_unavailable(error):
                continue
            raise
        available.append(metric)
    if not available:
        return []

    collected: dict[str, dict[str, Any]] = {}
    payload = _coinmetrics_page(asset, available, start, end, **kwargs)
    while True:
        for item in payload.get("data", []):
            day = item["time"][:10]
            entry = collected.setdefault(day, {"time": day})
            for metric in available:
                value = item.get(metric)
                entry[metric] = float(value) if value not in (None, "") else ""
        next_url = payload.get("next_page_url")
        if not next_url:
            break
        payload = _json(next_url, **kwargs)
    return [collected[key] for key in sorted(collected)]


# ---------------------------------------------------------------------------- funding


def fetch_funding(symbol: str, start: date, end: date, **kwargs: Any) -> list[dict[str, Any]]:
    """Funding settles every eight hours; a rate only exists once its settlement has passed."""
    cursor = _utc_ms(start)
    end_ms = _utc_ms(end + timedelta(days=1))
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    while cursor <= end_ms:
        query = urlencode({"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000})
        page = _json(f"{FAPI_URL}?{query}", **kwargs)
        if not page:
            break
        for item in page:
            funding_time = int(item["fundingTime"])
            if funding_time in seen:
                continue
            seen.add(funding_time)
            rows.append({"funding_time": _iso(funding_time), "funding_rate": float(item["fundingRate"])})
        last_time = int(page[-1]["fundingTime"])
        if last_time + 1 <= cursor:
            raise RuntimeError("funding rate pagination cursor did not advance")
        cursor = last_time + 1
        if len(page) < 1000:
            break
    rows.sort(key=lambda row: row["funding_time"])
    return rows


# ---------------------------------------------------------------------- open interest


def parse_metrics_zip(payload: bytes) -> list[dict[str, str]]:
    """Read one data.binance.vision daily metrics archive into 5-minute rows."""
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            for record in csv.DictReader(io.TextIOWrapper(handle, "utf-8")):
                row = {"time": record.get("create_time", "").replace(" ", "T") + "Z"}
                row.update({column: record.get(column, "") for column in OI_COLUMNS})
                rows.append(row)
    return rows


def _archived_day(symbol: str, day: date, cache_dir: Path, **kwargs: Any) -> bytes | None:
    """Return one archived metrics file, using the local cache when present."""
    name = f"{symbol}-metrics-{day.isoformat()}.zip"
    cached = cache_dir / name
    if cached.exists():
        return cached.read_bytes()
    try:
        payload = _request(f"{VISION_URL}/{symbol}/{name}", **kwargs)
    except HTTPError as error:
        if error.code == 404:
            return None
        raise
    cached.write_bytes(payload)
    return payload


def fetch_open_interest(
    symbol: str,
    start: date,
    end: date,
    cache_dir: Path,
    workers: int = 8,
    **kwargs: Any,
) -> list[dict[str, str]]:
    """Daily 5-minute archives, sampled down to the on-the-hour observation."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    days = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for payload in pool.map(lambda day: _archived_day(symbol, day, cache_dir, **kwargs), days):
            if payload is None:
                continue
            rows.extend(row for row in parse_metrics_zip(payload) if row["time"].endswith(":00:00Z"))
    return sorted(rows, key=lambda row: row["time"])


# -------------------------------------------------------------------------- sentiment


def fetch_fear_greed(**kwargs: Any) -> list[dict[str, Any]]:
    payload = _json(f"{FNG_URL}?limit=0&format=json", **kwargs)
    rows = [
        {
            "time": datetime.fromtimestamp(int(item["timestamp"]), tz=UTC).strftime("%Y-%m-%d"),
            "fng_value": int(item["value"]),
            "fng_class": item["value_classification"],
        }
        for item in payload.get("data", [])
    ]
    rows.sort(key=lambda row: row["time"])
    return rows


# ----------------------------------------------------------------------------- driver


def collect(
    symbols: Iterable[str],
    output_dir: Path,
    start: date,
    end: date,
    sources: Iterable[str] = SOURCES,
    workers: int = 8,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fetch every requested source and return the manifest describing what landed."""
    symbols = tuple(symbols)
    sources = tuple(sources)
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "window": {"start": str(start), "end": str(end)},
        "symbols": list(symbols),
        "sources": {},
    }

    if "fear_greed" in sources:
        rows = fetch_fear_greed(**kwargs)
        path = output_dir / "fear_greed.csv"
        _write_csv(path, rows, ("time", "fng_value", "fng_class"))
        manifest["sources"]["fear_greed"] = {
            "path": str(path),
            "provider": "alternative.me",
            "frequency": "1d",
            "row_count": len(rows),
            "start": rows[0]["time"] if rows else None,
            "end": rows[-1]["time"] if rows else None,
            "scope": "whole-market index, identical for every symbol",
        }

    for symbol in symbols:
        if "onchain" in sources:
            asset = COINMETRICS_ASSETS.get(symbol, symbol.lower())
            rows = fetch_onchain(asset, start, end, **kwargs)
            path = output_dir / f"onchain_{symbol.lower()}.csv"
            columns = ("time", *COINMETRICS_METRICS)
            if rows:
                columns = ("time", *[m for m in COINMETRICS_METRICS if m in rows[0]])
                _write_csv(path, rows, columns)
            manifest["sources"].setdefault("onchain", {})[symbol] = {
                "path": str(path) if rows else None,
                "provider": "coinmetrics-community",
                "frequency": "1d",
                "row_count": len(rows),
                "start": rows[0]["time"] if rows else None,
                "end": rows[-1]["time"] if rows else None,
                "metrics": [column for column in columns if column != "time"] if rows else [],
                "unavailable_metrics": [m for m in COINMETRICS_METRICS if not rows or m not in rows[0]],
            }

        if "funding" in sources:
            rows = fetch_funding(symbol, start, end, **kwargs)
            path = output_dir / f"funding_{symbol.lower()}.csv"
            _write_csv(path, rows, ("funding_time", "funding_rate"))
            manifest["sources"].setdefault("funding", {})[symbol] = {
                "path": str(path),
                "provider": "binance-fapi",
                "frequency": "8h",
                "row_count": len(rows),
                "start": rows[0]["funding_time"] if rows else None,
                "end": rows[-1]["funding_time"] if rows else None,
            }

        if "open_interest" in sources:
            cache_dir = output_dir / "cache" / "open_interest"
            rows = fetch_open_interest(symbol, start, end, cache_dir, workers=workers, **kwargs)
            path = output_dir / f"open_interest_{symbol.lower()}.csv"
            _write_csv(path, rows, ("time", *OI_COLUMNS))
            manifest["sources"].setdefault("open_interest", {})[symbol] = {
                "path": str(path),
                "provider": "data.binance.vision",
                "frequency": "1h sampled from 5m",
                "row_count": len(rows),
                "start": rows[0]["time"] if rows else None,
                "end": rows[-1]["time"] if rows else None,
            }

    return manifest


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch on-chain, sentiment and derivatives signals.")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external"))
    parser.add_argument("--manifest", type=Path, default=Path("logs/external_manifest.json"))
    parser.add_argument("--start", type=_parse_date, default=_parse_date("2021-08-01"))
    parser.add_argument("--end", type=_parse_date, default=datetime.now(UTC).date())
    parser.add_argument("--sources", default=",".join(SOURCES), help=f"comma separated subset of {','.join(SOURCES)}")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads for the open interest archive")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    symbols = [str(symbol) for symbol in config["symbols"]]
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    unknown = [item for item in sources if item not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown sources: {', '.join(unknown)}")

    manifest = collect(
        symbols,
        args.output_dir,
        args.start,
        args.end,
        sources=sources,
        workers=args.workers,
        timeout=args.timeout,
        max_retries=args.max_retries,
        delay=args.delay,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for source, entry in manifest["sources"].items():
        if "row_count" in entry:
            print(f"{source}: {entry['row_count']} rows", flush=True)
        else:
            for symbol, detail in entry.items():
                print(f"{source} {symbol}: {detail['row_count']} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
