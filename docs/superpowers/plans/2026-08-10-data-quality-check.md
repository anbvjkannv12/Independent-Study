# Data Quality Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repeatable command-line quality checker for the project's Binance 1-hour OHLCV CSV files.

**Architecture:** Keep ingestion in `scripts/fetch_binance_5y.py` and add a focused `scripts/check_data_quality.py` module. The checker reuses the existing row validator, adds file/manifest reconciliation, emits a JSON report, and returns a non-zero status only for structural errors unless strict warning mode is enabled.

**Tech Stack:** Python standard library, `unittest`, CSV, JSON.

## Global Constraints

- Read symbols, paths, interval, and manifest location from `config.json`.
- Do not modify files under `data/raw`.
- Use the existing `KLINE_COLUMNS` and `validate_rows` contracts.
- Missing intervals are warnings by default and failures only with `--fail-on-warnings`.

---

### Task 1: Add the checker contract tests

**Files:**
- Modify: `tests/test_fetch_binance_5y.py`
- Create: `tests/test_check_data_quality.py`

**Interfaces:**
- Tests will import `check_csv`, `check_project`, and `write_report` from `scripts/check_data_quality.py`.

- [ ] **Step 1: Write tests for a valid CSV and documented missing intervals**

```python
def test_check_csv_returns_warning_for_missing_intervals(self):
    result = check_csv(csv_path, interval_ms=3_600_000)
    self.assertEqual(result["errors"], [])
    self.assertEqual(result["warnings"][0]["type"], "missing_intervals")
```

- [ ] **Step 2: Write tests for missing files, invalid rows, and manifest mismatch**

```python
def test_check_project_reports_missing_file(self):
    result = check_project(config, manifest_path)
    self.assertEqual(result["status"], "failed")
    self.assertEqual(result["errors"][0]["type"], "missing_file")
```

- [ ] **Step 3: Write tests for strict warning mode and JSON report output**

```python
def test_fail_on_warnings_changes_status(self):
    result = check_project(config, manifest_path, fail_on_warnings=True)
    self.assertEqual(result["status"], "failed")

def test_write_report_creates_json(self):
    write_report(report, output_path)
    self.assertEqual(json.loads(output_path.read_text())["status"], report["status"])
```

- [ ] **Step 4: Run the new tests and confirm they fail before implementation**

Run: `C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.test_check_data_quality -v`

Expected: FAIL because `scripts/check_data_quality.py` does not exist yet.

### Task 2: Implement CSV and project checks

**Files:**
- Create: `scripts/check_data_quality.py`

**Interfaces:**
- `check_csv(path: Path, interval_ms: int) -> dict[str, object]` returns `path`, `status`, `row_count`, `summary`, `errors`, and `warnings`.
- `check_project(config: dict[str, object], manifest_path: Path, *, fail_on_warnings: bool = False) -> dict[str, object]` returns `generated_at`, `status`, `symbols`, `errors`, and `warnings`.
- `write_report(report: dict[str, object], path: Path) -> None` writes UTF-8 JSON with indentation.

- [ ] **Step 1: Load CSV rows with `csv.DictReader` and preserve validation errors as structured records**

```python
with path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
summary = validate_rows(rows, interval_ms)
```

- [ ] **Step 2: Convert validator results into warnings for missing intervals and errors for structural failures**

```python
warnings = []
if summary["missing_intervals"]:
    warnings.append({"type": "missing_intervals", "details": summary["missing_intervals"]})
```

- [ ] **Step 3: Reconcile each CSV summary with its manifest entry**

Compare `row_count`, `first_open_time`, `last_open_time`, and `missing_intervals`; emit a `manifest_mismatch` error when any value differs.

- [ ] **Step 4: Add an argparse CLI for config, manifest, report, and strict warning mode**

Run shape: `python scripts/check_data_quality.py --config config.json --manifest logs/fetch_manifest.json --report logs/data_quality_report.json --fail-on-warnings`.

### Task 3: Verify the project result and documentation

**Files:**
- Modify: `README.md`
- Create: `logs/data_quality_report.json`

- [ ] **Step 1: Run all unit tests**

Run: `C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 2: Run the checker against the current project data**

Run: `C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts/check_data_quality.py --config config.json --manifest logs/fetch_manifest.json --report logs/data_quality_report.json`

Expected: exit code 0, four symbols checked, and missing-interval warnings recorded.

- [ ] **Step 3: Document the normal and strict commands in `README.md`**

Add the two commands above and explain that normal mode reports known gaps as warnings while strict mode returns failure for any warning.

- [ ] **Step 4: Inspect the generated JSON report and confirm it contains four symbol entries**

Check: `status`, `symbols`, `errors`, and `warnings` are present and each configured symbol appears once.

