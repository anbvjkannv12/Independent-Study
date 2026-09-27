"""Align on-chain, sentiment and derivatives signals onto the hourly kline index.

Every series is merged backward-in-time only (``merge_asof`` with an explicit
tolerance), and daily series carry a one-day publication lag, so a bar at time
``t`` never sees a value that was published after ``t``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# How stale a value may be before the bar is treated as uncovered.
ONCHAIN_TOLERANCE = pd.Timedelta("3D")
SENTIMENT_TOLERANCE = pd.Timedelta("3D")
FUNDING_TOLERANCE = pd.Timedelta("1D")
OPEN_INTEREST_TOLERANCE = pd.Timedelta("6h")
PUBLICATION_LAG = pd.Timedelta("1D")


def _z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def _read(path: Path, time_column: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if time_column not in frame.columns or frame.empty:
        return None
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    frame = frame.dropna(subset=[time_column]).sort_values(time_column).drop_duplicates(time_column)
    for column in frame.columns.drop(time_column):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.set_index(time_column)


def _has(frame: pd.DataFrame, *columns: str) -> bool:
    """A metric an asset does not track arrives as a present but empty column."""
    return all(column in frame.columns and frame[column].notna().any() for column in columns)


def onchain_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily exchange flow, active address and MVRV features (available subset only)."""
    out = pd.DataFrame(index=frame.index)
    if _has(frame, "AdrActCnt"):
        active = frame["AdrActCnt"].clip(lower=1)
        out["onchain_active_addr_chg"] = np.log(active) - np.log(active.shift(1))
        out["onchain_active_addr_z30"] = _z(np.log(active), 30)
    if _has(frame, "CapMVRVCur"):
        out["onchain_mvrv"] = frame["CapMVRVCur"]
        out["onchain_mvrv_z90"] = _z(frame["CapMVRVCur"], 90)
    if _has(frame, "FlowInExNtv", "FlowOutExNtv"):
        inflow = frame["FlowInExNtv"].clip(lower=0)
        outflow = frame["FlowOutExNtv"].clip(lower=0)
        netflow = inflow - outflow
        out["onchain_exch_netflow_z30"] = _z(netflow, 30)
        out["onchain_exch_flow_ratio"] = np.log1p(inflow) - np.log1p(outflow)
        out["onchain_exch_inflow_chg"] = np.log1p(inflow) - np.log1p(inflow.shift(1))
    return out.dropna(how="all")


def funding_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Funding rate level, short-window average and standardised crowding."""
    rate = frame["funding_rate"]
    out = pd.DataFrame(index=frame.index)
    out["funding_rate"] = rate
    out["funding_rate_mean_3"] = rate.rolling(3).mean()
    out["funding_rate_mean_9"] = rate.rolling(9).mean()
    out["funding_rate_z90"] = _z(rate, 90)
    return out.dropna(how="all")


def open_interest_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Open interest momentum plus the long/short crowding ratios shipped alongside it."""
    out = pd.DataFrame(index=frame.index)
    open_interest = frame["sum_open_interest"].clip(lower=1e-12)
    log_oi = np.log(open_interest)
    out["oi_log_change_1h"] = log_oi.diff()
    out["oi_log_change_24h"] = log_oi.diff(24)
    out["oi_notional_z168"] = _z(np.log(frame["sum_open_interest_value"].clip(lower=1e-12)), 168)
    out["oi_toptrader_ls_ratio"] = frame["sum_toptrader_long_short_ratio"]
    out["oi_account_ls_ratio"] = frame["count_long_short_ratio"]
    out["oi_taker_ls_vol_ratio"] = frame["sum_taker_long_short_vol_ratio"]
    return out.dropna(how="all")


def sentiment_features(fng: pd.DataFrame | None, news: pd.DataFrame | None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fear & Greed index features, plus transformer-scored news/social features if present."""
    fng_out = None
    if fng is not None and "fng_value" in fng:
        value = fng["fng_value"] / 100.0
        fng_out = pd.DataFrame(index=fng.index)
        fng_out["sent_fng"] = value
        fng_out["sent_fng_chg_1d"] = value.diff()
        fng_out["sent_fng_z30"] = _z(value, 30)
        fng_out = fng_out.dropna(how="all")

    news_out = None
    if news is not None and "sent_mean" in news:
        news_out = pd.DataFrame(index=news.index)
        news_out["sent_news_mean"] = news["sent_mean"]
        news_out["sent_news_mean_3"] = news["sent_mean"].rolling(3).mean()
        if "sent_pos_ratio" in news:
            news_out["sent_news_pos_ratio"] = news["sent_pos_ratio"]
        if "sent_count" in news:
            news_out["sent_news_volume_z30"] = _z(np.log1p(news["sent_count"]), 30)
        news_out = news_out.dropna(how="all")
    return fng_out, news_out


def _merge(index: pd.DatetimeIndex, source: pd.DataFrame | None, tolerance: pd.Timedelta, lag: pd.Timedelta | None = None) -> pd.DataFrame:
    if source is None or source.empty:
        return pd.DataFrame(index=index)
    right = source.copy()
    if lag is not None:
        right.index = right.index + lag
        right = right[~right.index.duplicated(keep="last")]
    right = right.sort_index().reset_index(names="__ts")
    left = pd.DataFrame({"__ts": index}).sort_values("__ts")
    merged = pd.merge_asof(left, right, on="__ts", direction="backward", tolerance=tolerance)
    return merged.set_index("__ts").reindex(index)


def build_external_features(index: pd.DatetimeIndex, external_dir: Path, symbol: str) -> pd.DataFrame:
    """Return every external feature that exists for ``symbol``, aligned to ``index``.

    Columns are only produced for sources that were actually fetched, so SOL (no
    community on-chain coverage) and XRP (no exchange flows) simply get fewer
    columns instead of a column full of fabricated values.
    """
    index = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    key = symbol.lower()
    external_dir = Path(external_dir)

    onchain = _read(external_dir / f"onchain_{key}.csv", "time")
    funding = _read(external_dir / f"funding_{key}.csv", "funding_time")
    open_interest = _read(external_dir / f"open_interest_{key}.csv", "time")
    fng = _read(external_dir / "fear_greed.csv", "time")
    news = _read(external_dir / f"news_sentiment_{key}.csv", "time")
    if news is None:
        news = _read(external_dir / "news_sentiment.csv", "time")
    fng_features, news_features = sentiment_features(fng, news)

    blocks = [
        _merge(index, onchain_features(onchain) if onchain is not None else None, ONCHAIN_TOLERANCE, PUBLICATION_LAG),
        _merge(index, funding_features(funding) if funding is not None else None, FUNDING_TOLERANCE),
        _merge(index, open_interest_features(open_interest) if open_interest is not None else None, OPEN_INTEREST_TOLERANCE),
        _merge(index, fng_features, SENTIMENT_TOLERANCE, PUBLICATION_LAG),
        _merge(index, news_features, SENTIMENT_TOLERANCE, PUBLICATION_LAG),
    ]
    features = pd.concat([block for block in blocks if not block.empty], axis=1) if any(not b.empty for b in blocks) else pd.DataFrame(index=index)
    return features.replace([np.inf, -np.inf], np.nan)
