# Data Quality Check Design

## Goal

Provide a repeatable command-line quality check for the four Binance 1-hour OHLCV CSV files produced by this project.

## Scope

The checker reads `config.json`, inspects each configured symbol CSV, compares the measured summary with `logs/fetch_manifest.json`, and writes a machine-readable report to `logs/data_quality_report.json`. It does not modify raw data.

## Checks

- Required symbol files exist and are non-empty.
- CSV columns exactly match `KLINE_COLUMNS`.
- Timestamps are valid UTC values, sorted by `open_time`, and unique.
- OHLCV and trade-count values are numeric, non-negative, and obey high/low consistency rules.
- Missing 1-hour intervals are reported as warnings.
- Row count and first/last open times agree with the fetch manifest.

Structural failures return a non-zero exit code. Missing intervals return warnings and a zero exit code by default because the current manifest documents known gaps. `--fail-on-warnings` makes warnings fail the command for stricter runs.

## Output

The report contains one entry per symbol, a top-level status, counts of errors and warnings, and the exact evidence needed for follow-up. Terminal output is concise and suitable for notebook or CI use.

