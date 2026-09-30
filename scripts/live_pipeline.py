"""Live data update and forward-test pipeline (docs/next_phase_work_plan.md, work B).

Subcommands:
  init     copy raw history, train the frozen model (once) and the first rolling model
  update   hourly: fetch closed candles, impute, check, predict, backfill labels
  retrain  monthly: train a new rolling model on all known labels (24-row purge)
  status   print data/prediction freshness and the frozen model's recent AUC

Never writes to data/raw, data/processed or data/features (inputs of earlier experiments).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from check_data_quality import check_csv
from feature_engineering import EXPECTED_FEATURE_NAMES, add_features
from fetch_binance_5y import KLINE_COLUMNS, fetch_klines, map_kline_row, request_json, write_csv_atomic
from impute_missing_klines import process_symbol
from train_btc_4h_baseline import MODEL_SETTINGS, load_baseline_dataset

SYMBOLS = ("BTC", "ETH", "SOL", "XRP")
HOUR_MS = 3_600_000
HORIZON = 4
PURGE_ROWS = 24
SEED = 42
FROZEN_CUTOFF = pd.Timestamp("2026-08-09 06:00", tz="UTC")
CONSISTENCY_WINDOW = (pd.Timestamp("2026-08-02 00:00", tz="UTC"), FROZEN_CUTOFF)
PREDICTION_COLUMNS = ("symbol", "open_time", "model_id", "p_up", "predicted_at", "backfilled",
                      "label_due_at", "future_log_return", "target_up")
FEATURES = list(EXPECTED_FEATURE_NAMES)


class Paths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.raw = root / "data/live/raw"
        self.processed = root / "data/live/processed"
        self.models = root / "models/live"
        self.logs = root / "logs/live"
        self.predictions = self.logs / "predictions.csv"

    def raw_csv(self, symbol: str) -> Path:
        return self.raw / f"{symbol.lower()}_1h.csv"

    def processed_csv(self, symbol: str) -> Path:
        return self.processed / f"{symbol.lower()}_1h.csv"

    def frozen_model(self, symbol: str) -> Path:
        return self.models / f"{symbol.lower()}_4h_frozen.json"


def _utc_iso(ts: pd.Timestamp | datetime) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


# ---------- data ----------

def last_closed_open_ms(now: datetime) -> int:
    """open_time (ms) of the newest fully closed 1h candle at `now`."""
    return (int(now.timestamp() * 1000) // HOUR_MS) * HOUR_MS - HOUR_MS


def merge_klines(existing: list[dict[str, object]], new_raw: list[list[object]]) -> list[dict[str, object]]:
    """Append freshly fetched Binance rows; on duplicate open_time the new row wins."""
    by_time = {str(row["open_time"]): {c: row[c] for c in KLINE_COLUMNS} for row in existing}
    for row in map(map_kline_row, new_raw):
        by_time[str(row["open_time"])] = row
    return [by_time[key] for key in sorted(by_time)]


def update_raw(symbol: str, paths: Paths, now: datetime, base_url: str, request_fn=request_json) -> int:
    path = paths.raw_csv(symbol)
    with path.open("r", encoding="utf-8", newline="") as handle:
        existing = list(csv.DictReader(handle))
    start_ms = int(pd.Timestamp(existing[-1]["open_time"]).value // 1_000_000) + HOUR_MS
    end_ms = last_closed_open_ms(now)
    if start_ms > end_ms:
        return 0
    new_raw = fetch_klines(f"{symbol}USDT", "1h", start_ms, end_ms, request_fn, base_url=base_url)
    new_raw = [row for row in new_raw if int(row[0]) <= end_ms]  # never keep an unclosed candle
    if new_raw:
        write_csv_atomic(path, merge_klines(existing, new_raw), HOUR_MS)
    return len(new_raw)


def refresh_processed(symbol: str, paths: Paths) -> pd.DataFrame:
    # Missing hours in raw are only warnings; imputation fills them. Anything else stops the run.
    quality = check_csv(paths.raw_csv(symbol), HOUR_MS)
    if quality["status"] == "failed":
        raise RuntimeError(f"{symbol} raw data failed quality check: {quality['errors']}")
    process_symbol(symbol, paths.raw_csv(symbol), paths.processed_csv(symbol), HOUR_MS)
    processed = pd.read_csv(paths.processed_csv(symbol))
    if (pd.to_datetime(processed.open_time, utc=True).diff().dropna() != pd.Timedelta(hours=1)).any():
        raise RuntimeError(f"{symbol} processed data is not a continuous hourly series")
    return processed


def live_features(processed: pd.DataFrame) -> pd.DataFrame:
    """53 features for every row with complete features, including rows whose label is not known yet."""
    features = add_features(processed, prediction_horizon=HORIZON, keep_unlabeled=True)
    return features[FEATURES]


def realized_labels(processed: pd.DataFrame, open_times: pd.Series) -> pd.Series:
    """log(close[t+4h] / close[t]) looked up by timestamp; NaN until the t+4h candle has closed."""
    close = pd.Series(processed["close"].to_numpy(dtype=float), index=pd.to_datetime(processed["open_time"], utc=True))
    times = pd.to_datetime(open_times, utc=True)
    now_close = close.reindex(times).to_numpy()
    later_close = close.reindex(times + pd.Timedelta(hours=HORIZON)).to_numpy()
    return pd.Series(np.log(later_close / now_close), index=open_times.index)


def check_feature_consistency(symbol: str, features: pd.DataFrame, root: Path) -> None:
    """Live features must equal the experiment feature file on a shared historical window."""
    reference = pd.read_csv(root / f"data/features/{symbol.lower()}_4h.csv")
    reference["open_time"] = pd.to_datetime(reference["open_time"], utc=True)
    reference = reference.set_index("open_time")
    start, end = CONSISTENCY_WINDOW
    shared = reference.index[(reference.index >= start) & (reference.index <= end)].intersection(features.index)
    if len(shared) < 100:
        raise RuntimeError(f"{symbol} consistency window has only {len(shared)} shared rows")
    diff = np.abs(features.loc[shared, FEATURES].to_numpy(dtype=float) - reference.loc[shared, FEATURES].to_numpy(dtype=float))
    if np.nanmax(diff) > 1e-9:
        raise RuntimeError(f"{symbol} live features differ from data/features by {np.nanmax(diff):.3g}")


# ---------- models ----------

def train_model(features: pd.DataFrame, target: pd.Series) -> XGBClassifier:
    model = XGBClassifier(**MODEL_SETTINGS, random_state=SEED)
    model.fit(features[FEATURES], target.astype(int))
    return model


def rolling_training_rows(labeled: pd.DataFrame) -> pd.DataFrame:
    """All rows with a known label except the newest PURGE_ROWS."""
    if len(labeled) <= PURGE_ROWS:
        raise ValueError("not enough labeled rows to retrain")
    return labeled.iloc[: len(labeled) - PURGE_ROWS]


def rolling_models(paths: Paths, symbol: str) -> list[dict[str, object]]:
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in paths.models.glob(f"{symbol.lower()}_4h_rolling_*.meta.json")]
    return sorted(metas, key=lambda m: m["effective_from"])


def rolling_model_for(metas: list[dict[str, object]], open_time: pd.Timestamp) -> dict[str, object] | None:
    """The rolling model in force when row t could first be predicted (its candle closes at t+1h)."""
    usable = [m for m in metas if pd.Timestamp(m["effective_from"]) <= open_time + pd.Timedelta(hours=1)]
    return usable[-1] if usable else None


def load_model(path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(path)
    return model


# ---------- predictions ----------

def read_predictions(paths: Paths) -> pd.DataFrame:
    if not paths.predictions.exists():
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.read_csv(paths.predictions, dtype={"symbol": str, "model_id": str})


def write_predictions(paths: Paths, frame: pd.DataFrame) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    tmp = paths.predictions.with_suffix(".csv.tmp")
    frame.loc[:, PREDICTION_COLUMNS].to_csv(tmp, index=False)
    tmp.replace(paths.predictions)


def is_backfilled(open_time: pd.Timestamp, predicted_at: pd.Timestamp) -> bool:
    """A prediction made more than 1h after the candle closed was not live."""
    return predicted_at - (open_time + pd.Timedelta(hours=1)) > pd.Timedelta(hours=1)


def new_prediction_rows(symbol: str, features: pd.DataFrame, existing: pd.DataFrame, paths: Paths,
                        now: pd.Timestamp) -> pd.DataFrame:
    """Frozen predictions for every row after the cutoff; rolling predictions with the model valid at that time."""
    done = set(zip(existing.loc[existing.symbol == symbol, "open_time"], existing.loc[existing.symbol == symbol, "model_id"]))
    candidates = features[features.index > FROZEN_CUTOFF]
    assignments: dict[str, tuple[Path, list[pd.Timestamp]]] = {}
    metas = rolling_models(paths, symbol)
    for t in candidates.index:
        key = _utc_iso(t)
        if (key, "frozen") not in done:
            assignments.setdefault("frozen", (paths.frozen_model(symbol), []))[1].append(t)
        meta = rolling_model_for(metas, t)
        if meta and (key, meta["model_id"]) not in done:
            assignments.setdefault(meta["model_id"], (paths.models / meta["model_file"], []))[1].append(t)
    rows = []
    for model_id, (model_path, times) in assignments.items():
        p_up = load_model(model_path).predict_proba(candidates.loc[times, FEATURES])[:, 1]
        for t, p in zip(times, p_up):
            rows.append({"symbol": symbol, "open_time": _utc_iso(t), "model_id": model_id, "p_up": float(p),
                         "predicted_at": _utc_iso(now), "backfilled": int(is_backfilled(t, now)),
                         "label_due_at": _utc_iso(t + pd.Timedelta(hours=HORIZON + 1)),
                         "future_log_return": np.nan, "target_up": np.nan})
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def backfill_labels(predictions: pd.DataFrame, symbol: str, processed: pd.DataFrame) -> pd.DataFrame:
    mask = (predictions.symbol == symbol) & predictions.future_log_return.isna()
    if mask.any():
        r = realized_labels(processed, predictions.loc[mask, "open_time"])
        predictions.loc[mask, "future_log_return"] = r
        known = mask & predictions.future_log_return.notna()
        predictions.loc[known, "target_up"] = (predictions.loc[known, "future_log_return"] > 0).astype(int)
    return predictions


# ---------- subcommands ----------

def cmd_init(paths: Paths, symbols: tuple[str, ...], now: pd.Timestamp) -> None:
    paths.models.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.models / "frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "cutoff": _utc_iso(FROZEN_CUTOFF), "created_at": _utc_iso(now), "git_revision": _git_revision(paths.root),
        "xgboost_version": xgboost.__version__, "models": {}}
    for symbol in symbols:
        if not paths.raw_csv(symbol).exists():
            paths.raw.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(paths.root / f"data/raw/{symbol.lower()}_5y.csv", paths.raw_csv(symbol))
        if paths.frozen_model(symbol).exists():
            print(f"{symbol}: frozen model already exists, not retrained")
            continue
        feature_path = paths.root / f"data/features/{symbol.lower()}_4h.csv"
        dataset, names = load_baseline_dataset(feature_path, paths.root / "logs/feature_manifest.json")
        if list(names) != FEATURES or dataset.open_time.iloc[-1] != FROZEN_CUTOFF:
            raise ValueError(f"{symbol} feature file does not match the frozen specification")
        train_model(dataset, dataset.target_up).save_model(paths.frozen_model(symbol))
        manifest["models"][symbol] = {"model_file": paths.frozen_model(symbol).name, "model_sha256": _sha256(paths.frozen_model(symbol)),
                                      "feature_file_sha256": _sha256(feature_path), "train_rows": len(dataset),
                                      "train_start": _utc_iso(dataset.open_time.iloc[0]), "train_end": _utc_iso(dataset.open_time.iloc[-1])}
        print(f"{symbol}: frozen model trained on {len(dataset)} rows")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for symbol in symbols:
        if not rolling_models(paths, symbol):
            cmd_retrain(paths, (symbol,), now)


def verify_frozen(paths: Paths, symbol: str) -> None:
    manifest = json.loads((paths.models / "frozen_manifest.json").read_text(encoding="utf-8"))
    if _sha256(paths.frozen_model(symbol)) != manifest["models"][symbol]["model_sha256"]:
        raise RuntimeError(f"{symbol} frozen model changed since it was created")


def cmd_update(paths: Paths, symbols: tuple[str, ...], now: pd.Timestamp, base_url: str) -> dict[str, object]:
    predictions = read_predictions(paths)
    log: dict[str, object] = {"run_at": _utc_iso(now), "symbols": {}}
    for symbol in symbols:
        verify_frozen(paths, symbol)
        fetched = update_raw(symbol, paths, now.to_pydatetime(), base_url)
        processed = refresh_processed(symbol, paths)
        features = live_features(processed)
        check_feature_consistency(symbol, features, paths.root)
        new = new_prediction_rows(symbol, features, predictions, paths, now)
        predictions = pd.concat([predictions, new], ignore_index=True) if len(new) else predictions
        predictions = backfill_labels(predictions, symbol, processed)
        log["symbols"][symbol] = {"fetched_rows": fetched, "latest_open_time": str(processed.open_time.iloc[-1]),
                                  "new_predictions": len(new)}
    write_predictions(paths, predictions)
    return log


def cmd_retrain(paths: Paths, symbols: tuple[str, ...], now: pd.Timestamp) -> None:
    paths.models.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        processed = pd.read_csv(paths.processed_csv(symbol)) if paths.processed_csv(symbol).exists() else refresh_processed(symbol, paths)
        features = live_features(processed)
        labels = realized_labels(processed, pd.Series(features.index, index=features.index))
        labeled = features[labels.notna()]
        train = rolling_training_rows(labeled)
        model_id = f"rolling_{now:%Y%m%d}"
        model_file = f"{symbol.lower()}_4h_{model_id}.json"
        if (paths.models / model_file).exists():
            raise FileExistsError(f"{model_file} already exists")
        train_model(train, labels[train.index] > 0).save_model(paths.models / model_file)
        meta = {"model_id": model_id, "model_file": model_file, "effective_from": _utc_iso(now),
                "train_rows": len(train), "train_start": _utc_iso(train.index[0]), "train_end": _utc_iso(train.index[-1]),
                "purge_rows": PURGE_ROWS, "processed_sha256": _sha256(paths.processed_csv(symbol)),
                "git_revision": _git_revision(paths.root), "xgboost_version": xgboost.__version__}
        (paths.models / f"{symbol.lower()}_4h_{model_id}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{symbol}: {model_id} trained on {len(train)} rows up to {meta['train_end']}")


def cmd_status(paths: Paths, symbols: tuple[str, ...], now: pd.Timestamp) -> None:
    predictions = read_predictions(paths)
    for symbol in symbols:
        processed = pd.read_csv(paths.processed_csv(symbol), usecols=["open_time"])
        latest = pd.Timestamp(processed.open_time.iloc[-1])
        age = (now - latest - pd.Timedelta(hours=1)) / pd.Timedelta(hours=1)
        rows = predictions[(predictions.symbol == symbol) & (predictions.model_id == "frozen")]
        recent = rows[pd.to_datetime(rows.open_time) > now - pd.Timedelta(days=30)].dropna(subset=["target_up"])
        auc = roc_auc_score(recent.target_up, recent.p_up) if recent.target_up.nunique() == 2 else float("nan")
        warn = "  WARNING: no new data for >3h" if age > 3 else ""
        print(f"{symbol}: latest candle {latest:%Y-%m-%d %H:%M} UTC ({age:.1f}h ago), frozen predictions {len(rows)}, "
              f"labeled last 30d {len(recent)}, frozen AUC last 30d {auc:.4f}{warn}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("init", "update", "retrain", "status"))
    parser.add_argument("--symbols", nargs="+", choices=SYMBOLS, default=SYMBOLS)
    parser.add_argument("--base-url", default="https://api.binance.com",
                        help="use https://data-api.binance.vision where api.binance.com is blocked")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    paths, symbols, now = Paths(args.root), tuple(args.symbols), pd.Timestamp.now(tz="UTC")
    run_log: dict[str, object] = {"command": args.command, "run_at": _utc_iso(now)}
    try:
        if args.command == "init":
            cmd_init(paths, symbols, now)
        elif args.command == "update":
            run_log.update(cmd_update(paths, symbols, now, args.base_url))
        elif args.command == "retrain":
            cmd_retrain(paths, symbols, now)
        else:
            cmd_status(paths, symbols, now)
            return 0
        run_log["status"] = "ok"
        return 0
    except Exception as exc:  # logged, then reported through the exit code for the scheduler
        run_log.update(status="error", error=f"{type(exc).__name__}: {exc}")
        print(run_log["error"], file=sys.stderr)
        return 1
    finally:
        if args.command != "status":
            paths.logs.mkdir(parents=True, exist_ok=True)
            with (paths.logs / "run_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(run_log, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
