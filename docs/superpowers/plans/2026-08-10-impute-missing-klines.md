# Missing Kline Imputation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible preprocessing step that fills small missing 1-hour kline gaps and records the decision for the project.

**Architecture:** Add a focused script in `scripts/impute_missing_klines.py` that reuses `fetch_binance_5y.KLINE_COLUMNS`, reads raw CSVs, inserts missing rows, writes processed CSVs, and emits a manifest. Keep existing raw-data validation separate from processed-data creation.

**Tech Stack:** Python standard library, `unittest`, existing project scripts.

## Global Constraints

- Raw CSV files in `data/raw` must not be modified.
- Processed CSV files must include original kline columns plus `is_imputed`.
- Price fields in imputed rows use the previous row's `close`.
- Activity fields in imputed rows use `0`.
- CLI paths resolve relative to the provided `config.json`.

---

### Task 1: Imputation Behavior

**Files:**
- Create: `tests/test_impute_missing_klines.py`
- Create: `scripts/impute_missing_klines.py`

**Interfaces:**
- Consumes: `KLINE_COLUMNS` from `scripts/fetch_binance_5y.py`
- Produces: `impute_rows(rows: list[dict[str, object]], interval_ms: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]`

- [x] **Step 1: Write the failing test**

```python
def test_impute_rows_inserts_missing_hour_with_flag():
    rows = [
        valid_row("2024-01-01T00:00:00Z", close="10.0"),
        valid_row("2024-01-01T02:00:00Z", close="12.0"),
    ]

    processed, gaps = impute_rows(rows, 3_600_000)

    assert processed[1]["open_time"] == "2024-01-01T01:00:00Z"
    assert processed[1]["open"] == "10.0"
    assert processed[1]["volume"] == "0"
    assert processed[1]["is_imputed"] == "1"
    assert processed[0]["is_imputed"] == "0"
    assert gaps == [{"from": "2024-01-01T01:00:00Z", "to": "2024-01-01T01:00:00Z", "count": 1}]
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_impute_missing_klines -v`

Expected: import failure because `impute_missing_klines` does not exist.

- [x] **Step 3: Write minimal implementation**

Implement timestamp parsing, missing-row construction, row copying with `is_imputed`, and gap summary generation.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_impute_missing_klines -v`

Expected: PASS

### Task 2: Project CLI and Documentation

**Files:**
- Modify: `scripts/impute_missing_klines.py`
- Modify: `agement.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `load_config(config_path: Path) -> dict[str, object]`
- Produces: `process_project(config_path: Path, output_dir: Path | None, manifest_path: Path | None) -> dict[str, object]`

- [x] **Step 1: Add CLI tests**

Test that project processing writes a processed CSV and manifest without changing raw input.

- [x] **Step 2: Run tests to verify failure**

Run: `python -m unittest tests.test_impute_missing_klines -v`

Expected: failure because CLI project processing is not implemented yet.

- [x] **Step 3: Implement project processing**

Resolve paths from config, process each configured symbol, write `data/processed/*_5y.csv`, and write `logs/imputation_manifest.json`.

- [x] **Step 4: Update documentation**

Append the project decision to `agement.md` and add README usage commands.

- [x] **Step 5: Verify full project**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.
