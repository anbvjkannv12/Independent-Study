# Missing Kline Imputation Design

## Goal

Create processed CSV files for model training by filling small 1-hour OHLCV gaps while keeping the raw Binance CSV files unchanged.

## Decision

Use forward-fill imputation for price fields and explicit zeroes for activity fields.

- Price fields: `open`, `high`, `low`, and `close` are filled with the previous row's `close`.
- Activity fields: `volume`, `quote_asset_volume`, `number_of_trades`, `taker_buy_base`, and `taker_buy_quote` are filled with `0`.
- Timestamp fields: `open_time` is the missing interval timestamp, and `close_time` is `open_time + interval - 1 ms`.
- Traceability field: add `is_imputed`, using `0` for original rows and `1` for generated rows.

## Rationale

The current raw data has 7 missing 1-hour candles per symbol out of 43,818 rows, so the missing rate is about 0.016%. The effect on overall model training should be very small. Still, copying previous volume would create fake trading activity, so activity fields use zeroes. The `is_imputed` flag keeps the model and project report honest about which rows were generated.

## Scope

- Read raw files from `data/raw`.
- Write processed files to `data/processed`.
- Do not mutate raw CSV files.
- Produce a processing manifest in `logs/imputation_manifest.json`.
- Keep compatibility with the existing fetch column schema plus one new `is_imputed` column.

## Testing

Tests should prove that missing hourly rows are inserted, imputed rows use the intended price and activity values, original rows are marked as non-imputed, and a no-gap file remains unchanged except for the added flag column.
