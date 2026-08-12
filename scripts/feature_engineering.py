from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


LAGS = (1, 2, 3, 6, 12, 24)
ROLLING_WINDOWS = (6, 12, 24)
DEFAULT_HORIZONS = (1, 4, 12, 24)
SYMBOL_FILES = {
    "BTCUSDT": "btc_5y.csv",
    "ETHUSDT": "eth_5y.csv",
    "SOLUSDT": "sol_5y.csv",
    "XRPUSDT": "xrp_5y.csv",
}
RAW_NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
)
EXPECTED_FEATURE_NAMES = (
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "log_return",
    "range_pct",
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "atr_14_pct",
    "log_volume",
    "volume_change",
    "trade_intensity",
    "taker_buy_ratio",
    "quote_volume_change",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "log_return_lag_1",
    "volume_change_lag_1",
    "taker_buy_ratio_lag_1",
    "log_return_lag_2",
    "volume_change_lag_2",
    "taker_buy_ratio_lag_2",
    "log_return_lag_3",
    "volume_change_lag_3",
    "taker_buy_ratio_lag_3",
    "log_return_lag_6",
    "volume_change_lag_6",
    "taker_buy_ratio_lag_6",
    "log_return_lag_12",
    "volume_change_lag_12",
    "taker_buy_ratio_lag_12",
    "log_return_lag_24",
    "volume_change_lag_24",
    "taker_buy_ratio_lag_24",
    "return_mean_6",
    "return_std_6",
    "volume_mean_6",
    "range_mean_6",
    "return_mean_12",
    "return_std_12",
    "volume_mean_12",
    "range_mean_12",
    "return_mean_24",
    "return_std_24",
    "volume_mean_24",
    "range_mean_24",
    "rsi_14",
    "macd_pct",
    "macd_signal_pct",
    "bollinger_z_20",
)


def _prepare_raw(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"open_time", *RAW_NUMERIC_COLUMNS}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = raw.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    if df["open_time"].isna().any():
        raise ValueError("open_time contains invalid timestamps.")
    df = df.sort_values("open_time").drop_duplicates("open_time")
    df = df.set_index("open_time")
    for col in RAW_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "is_imputed" not in df:
        df["is_imputed"] = 0
    df["is_imputed"] = pd.to_numeric(df["is_imputed"], errors="coerce").fillna(0).astype(int)
    return df


def add_features(raw: pd.DataFrame, prediction_horizon: int = 1) -> pd.DataFrame:
    """Build the original project's 53 features and two target columns."""
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon must be a positive integer.")

    df = _prepare_raw(raw)
    close = df["close"].clip(lower=1e-12)
    volume = df["volume"].clip(lower=0)

    df["log_return"] = np.log(close / close.shift(1))
    df["future_log_return"] = np.log(close.shift(-prediction_horizon) / close)
    df["target_up"] = (df["future_log_return"] > 0).astype(int)
    df["range_pct"] = (df["high"] - df["low"]) / close
    df["body_pct"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["upper_shadow_pct"] = (df["high"] - df[["open", "close"]].max(axis=1)) / close
    df["lower_shadow_pct"] = (df[["open", "close"]].min(axis=1) - df["low"]) / close

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14_pct"] = true_range.rolling(14).mean() / close

    df["log_volume"] = np.log1p(volume)
    df["volume_change"] = np.log1p(volume) - np.log1p(volume.shift(1))
    df["trade_intensity"] = df["number_of_trades"] / volume.replace(0, np.nan)
    df["taker_buy_ratio"] = df["taker_buy_base"] / volume.replace(0, np.nan)
    quote_volume = df["quote_asset_volume"].clip(lower=0)
    df["quote_volume_change"] = np.log1p(quote_volume) - np.log1p(quote_volume.shift(1))

    hour = df.index.hour
    day = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * day / 7)
    df["dow_cos"] = np.cos(2 * np.pi * day / 7)

    for lag in LAGS:
        df[f"log_return_lag_{lag}"] = df["log_return"].shift(lag)
        df[f"volume_change_lag_{lag}"] = df["volume_change"].shift(lag)
        df[f"taker_buy_ratio_lag_{lag}"] = df["taker_buy_ratio"].shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"return_mean_{window}"] = df["log_return"].rolling(window).mean()
        df[f"return_std_{window}"] = df["log_return"].rolling(window).std()
        df[f"volume_mean_{window}"] = df["log_volume"].rolling(window).mean()
        df[f"range_mean_{window}"] = df["range_pct"].rolling(window).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    df["macd_pct"] = macd / close
    df["macd_signal_pct"] = macd.ewm(span=9, adjust=False).mean() / close

    rolling_close_mean = close.rolling(20).mean()
    rolling_close_std = close.rolling(20).std()
    df["bollinger_z_20"] = (close - rolling_close_mean) / rolling_close_std.replace(0, np.nan)

    dataset = df[[*EXPECTED_FEATURE_NAMES, "future_log_return", "target_up"]].replace(
        [np.inf, -np.inf], np.nan
    )
    return dataset.dropna()


def build_feature_dataset(raw: pd.DataFrame, prediction_horizon: int = 1) -> pd.DataFrame:
    """Return output-ready features with timestamp and imputation audit data."""
    prepared = _prepare_raw(raw)
    dataset = add_features(prepared.reset_index(), prediction_horizon=prediction_horizon)
    audit = prepared["is_imputed"].rename("is_imputed")
    output = dataset.join(audit, how="left").reset_index(names="open_time")
    output["is_imputed"] = output["is_imputed"].fillna(0).astype(int)
    return output[["open_time", *EXPECTED_FEATURE_NAMES, "future_log_return", "target_up", "is_imputed"]]


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise ValueError("horizons must be comma-separated positive integers.") from exc
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must be comma-separated positive integers.")
    return horizons


def _validate_output(dataset: pd.DataFrame) -> None:
    expected = ["open_time", *EXPECTED_FEATURE_NAMES, "future_log_return", "target_up", "is_imputed"]
    if list(dataset.columns) != expected:
        raise ValueError("Generated dataset columns do not match the original feature contract.")
    timestamps = pd.to_datetime(dataset["open_time"], utc=True)
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("Generated open_time values must be sorted and unique.")
    numeric = dataset.drop(columns="open_time")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Generated dataset contains NaN or infinite values.")


def generate_feature_datasets(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    symbols: Iterable[str],
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> dict[str, object]:
    """Generate all requested datasets and write a manifest after validation."""
    outputs: list[dict[str, object]] = []
    symbols = tuple(symbols)
    horizons = tuple(horizons)
    prepared_outputs: list[tuple[Path, pd.DataFrame]] = []
    for symbol in symbols:
        if symbol not in SYMBOL_FILES:
            raise ValueError(f"Unsupported symbol: {symbol}")
        input_path = input_dir / SYMBOL_FILES[symbol]
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        raw = pd.read_csv(input_path)
        for horizon in horizons:
            dataset = build_feature_dataset(raw, prediction_horizon=horizon)
            _validate_output(dataset)
            output_path = output_dir / f"{symbol[:-4].lower()}_{horizon}h.csv"
            prepared_outputs.append((output_path, dataset))
            outputs.append(
                {
                    "symbol": symbol[:-4],
                    "horizon_hours": horizon,
                    "input_file": str(input_path),
                    "output_file": str(output_path),
                    "row_count": len(dataset),
                    "imputed_row_count": int(dataset["is_imputed"].sum()),
                    "start_utc": dataset["open_time"].min().isoformat(),
                    "end_utc": dataset["open_time"].max().isoformat(),
                }
            )

    for output_path, dataset in prepared_outputs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False, encoding="utf-8-sig")

    manifest = {
        "feature_count": len(EXPECTED_FEATURE_NAMES),
        "feature_names": list(EXPECTED_FEATURE_NAMES),
        "label_definition": {
            "classification": "target_up = future_log_return > 0",
            "regression": "future_log_return = log(close[t+h] / close[t])",
        },
        "horizons_hours": list(horizons),
        "leakage_controls": [
            "Rows are sorted by UTC open_time before feature engineering.",
            "Lag and rolling features use past observations only.",
            "The target columns are excluded from feature_names.",
            "is_imputed is retained for audit and excluded from feature_names.",
        ],
        "datasets": outputs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build original-project features and labels.")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--manifest", type=Path, default=Path("logs/feature_manifest.json"))
    parser.add_argument("--horizons", default=','.join(map(str, DEFAULT_HORIZONS)))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    symbols = config.get("symbols", list(SYMBOL_FILES))
    generate_feature_datasets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        symbols=symbols,
        horizons=_parse_horizons(args.horizons),
    )


if __name__ == "__main__":
    main()
