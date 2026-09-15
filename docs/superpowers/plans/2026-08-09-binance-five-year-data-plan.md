# Binance 五年資料專案 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立獨立的 `C:\Users\user\專題 new (五年)` 資料專案，下載並驗證 Binance BTC、ETH、SOL、XRP 最近五年 1 小時現貨 K 線資料。

**Architecture:** 新專案使用 Python 標準函式庫呼叫 Binance 公開 `/api/v3/klines` 端點，逐交易對分頁下載、暫存、驗證後原子替換正式 CSV。設定、腳本、執行 notebook、品質 manifest 與研究文件全部位於新專案，不依賴原專題的相對路徑。

**Tech Stack:** Python 3.10+、`urllib.request`、`csv`、`json`、`datetime`、Jupyter Notebook JSON；測試使用 Python 內建 `unittest` 與假的 HTTP transport，不需要 API key。

## Global Constraints

- 新專案根目錄固定為 `C:\Users\user\專題 new (五年)`。
- 只抓 `BTCUSDT`、`ETHUSDT`、`SOLUSDT`、`XRPUSDT` 的 Binance Spot `1h` K 線。
- 使用公開行情端點，不讀取、保存或傳送使用者提供的 API key。
- 輸出 CSV 使用 UTF-8，時間使用 UTC ISO 8601 字串。
- 正式 CSV 只能由通過驗證的暫存檔取代。
- 原專題的 `data`、`src`、`models`、`notebooks` 不得修改。
- 下載錯誤、缺漏區段與品質摘要必須寫入 `logs\fetch_manifest.json`。
- `run` 內的 notebook 必須使用新專案內相對路徑，不能依賴原專題。

---

### Task 1: 建立抓取器的測試骨架與資料契約

**Files:**
- Create: `C:\Users\user\專題 new (五年)\tests\test_fetch_binance_5y.py`
- Create: `C:\Users\user\專題 new (五年)\scripts\fetch_binance_5y.py`

**Interfaces:**
- `KLINE_COLUMNS: tuple[str, ...]` 定義 11 個 CSV 欄位。
- `map_kline_row(row: list[object]) -> dict[str, object]` 將 Binance 12 欄回應映射成 CSV 欄位。
- `find_missing_open_times(open_times: list[int], interval_ms: int) -> list[dict[str, int]]` 回傳缺漏區段。
- `validate_rows(rows: list[dict[str, object]], interval_ms: int) -> dict[str, object]` 回傳驗證摘要，驗證失敗時拋出 `ValueError`。

- [ ] **Step 1: Write the failing contract tests**

```python
import unittest

from fetch_binance_5y import KLINE_COLUMNS, find_missing_open_times, map_kline_row, validate_rows


class FetchContractTests(unittest.TestCase):
    def test_map_kline_row_uses_expected_columns_and_types(self):
        row = [1704067200000, "42000.1", "42100.2", "41900.0", "42050.3", "12.5",
               1704070799999, "525000.4", 1234, "6.2", "260000.7", "0"]
        mapped = map_kline_row(row)
        self.assertEqual(tuple(mapped), KLINE_COLUMNS)
        self.assertEqual(mapped["open_time"], "2024-01-01T00:00:00Z")
        self.assertEqual(mapped["close"], 42050.3)
        self.assertEqual(mapped["number_of_trades"], 1234)

    def test_find_missing_open_times_reports_one_hour_gap(self):
        missing = find_missing_open_times([0, 3_600_000, 10_800_000], 3_600_000)
        self.assertEqual(missing, [{"from": 7_200_000, "to": 7_200_000, "count": 1}])

    def test_validate_rows_rejects_duplicate_open_times(self):
        rows = [{"open_time": "2024-01-01T00:00:00Z", "open": 1, "high": 2,
                 "low": 1, "close": 1.5, "volume": 10, "close_time": "2024-01-01T00:59:59.999Z",
                 "quote_asset_volume": 15, "number_of_trades": 1, "taker_buy_base": 5,
                 "taker_buy_quote": 7}]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows(rows + rows, 3_600_000)
```

- [ ] **Step 2: Run the contract tests and verify they fail**

Run: `python -m unittest discover -s tests -v`
Expected: FAIL because `fetch_binance_5y.py` and the public functions do not exist yet.

- [ ] **Step 3: Implement the data contract functions**

Implement `KLINE_COLUMNS`, numeric conversion, UTC formatting, duplicate detection, chronological ordering, non-negative OHLCV checks, and missing-interval reporting. Do not add network calls in this task.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `python -m unittest discover -s tests -v`
Expected: PASS for mapping, gap detection, duplicate rejection, and all existing contract tests.

- [ ] **Step 5: Commit the task**

Run: `git add scripts/fetch_binance_5y.py tests/test_fetch_binance_5y.py; git commit -m "feat: define five-year kline data contract"`
Expected: one commit containing only the new project contract files.

### Task 2: Add paginated Binance download and retry behavior

**Files:**
- Modify: `C:\Users\user\專題 new (五年)\scripts\fetch_binance_5y.py`
- Modify: `C:\Users\user\專題 new (五年)\tests\test_fetch_binance_5y.py`
- Create: `C:\Users\user\專題 new (五年)\config.json`

**Interfaces:**
- `fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, request_json: Callable) -> list[list[object]]` returns all pages without duplicate boundary rows.
- `request_json(url: str, timeout_seconds: int, max_retries: int) -> object` performs public HTTP GET with exponential backoff for 429, 5xx, and transient URL errors.
- `subtract_years(value: datetime, years: int) -> datetime` handles leap-day input by clamping to February 28.

- [ ] **Step 1: Write failing pagination and retry tests**

```python
import unittest


class PaginationTests(unittest.TestCase):
    def test_fetch_klines_advances_by_last_open_time_plus_interval(self):
        calls = []

        def fake_request(url, timeout_seconds, max_retries):
            calls.append(url)
            if len(calls) == 1:
                return [[0, "1", "2", "1", "1", "1", 0, "1", 1, "1", "1", "0"]]
            return []

        rows = fetch_klines("BTCUSDT", "1h", 0, 3_600_000, fake_request)
        self.assertEqual(len(rows), 1)
        self.assertIn("startTime=3600000", calls[1])

    def test_request_json_retries_http_429_then_succeeds(self):
        responses = [HTTPError(url="x", code=429, msg="rate limit", hdrs=None, fp=None), {"ok": True}]
        result = request_json("x", timeout_seconds=1, max_retries=2, opener=fake_opener(responses))
        self.assertEqual(result, {"ok": True})
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m unittest discover -s tests -v`
Expected: FAIL because pagination and retry functions are not implemented.

- [ ] **Step 3: Implement configuration, pagination, and retry logic**

Create `config.json` with `base_url`, `interval`, `interval_ms`, `symbols`, `years`, `page_limit`, `timeout_seconds`, `max_retries`, and `request_delay_seconds`. Build query strings with `urllib.parse.urlencode`; advance `startTime` by one interval after the last returned candle; stop on an empty page or when the end time is reached. Never add authentication headers.

- [ ] **Step 4: Run focused tests and a no-network dry run**

Run: `python -m unittest discover -s tests -v`
Expected: PASS, including pagination and retry tests.

Run: `python scripts/fetch_binance_5y.py --dry-run --config config.json`
Expected: print four symbol plans and no CSV or network request.

- [ ] **Step 5: Commit the task**

Run: `git add scripts/fetch_binance_5y.py tests/test_fetch_binance_5y.py config.json; git commit -m "feat: add paginated public Binance downloader"`
Expected: one commit containing downloader, config, and tests.

### Task 3: Add atomic CSV output and manifest generation

**Files:**
- Modify: `C:\Users\user\專題 new (五年)\scripts\fetch_binance_5y.py`
- Modify: `C:\Users\user\專題 new (五年)\tests\test_fetch_binance_5y.py`
- Create: `C:\Users\user\專題 new (五年)\data\raw\.gitkeep`
- Create: `C:\Users\user\專題 new (五年)\logs\.gitkeep`

**Interfaces:**
- `write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None` writes and validates a temporary file before replacing the target.
- `build_manifest_entry(symbol: str, rows: list[dict[str, object]], status: str, error: str | None) -> dict[str, object]` returns a JSON-safe quality summary.
- `run_symbol(symbol: str, config: dict[str, object], request_json: Callable) -> dict[str, object]` downloads, validates, writes one CSV, and returns its manifest entry.

- [ ] **Step 1: Write failing atomic-output and manifest tests**

```python
import tempfile
import unittest


class OutputTests(unittest.TestCase):
    def test_write_csv_atomic_does_not_replace_target_when_validation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "btc_5y.csv"
            target.write_text("old", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_csv_atomic(target, invalid_rows())
            self.assertEqual(target.read_text(encoding="utf-8"), "old")

    def test_manifest_entry_contains_quality_summary(self):
        entry = build_manifest_entry("BTCUSDT", valid_rows(), "success", None)
        self.assertEqual(entry["status"], "success")
        self.assertEqual(entry["row_count"], len(valid_rows()))
        self.assertIn("missing_intervals", entry)
```

- [ ] **Step 2: Run tests and verify the new tests fail**

Run: `python -m unittest discover -s tests -v`
Expected: FAIL because atomic output and manifest functions are not implemented.

- [ ] **Step 3: Implement output and manifest handling**

Create `data\raw` and `logs` if missing. Write UTF-8 CSV to `.<name>.tmp`, reopen it for validation, then replace the target with `os.replace`. Write a manifest containing `generated_at`, config snapshot without secrets, per-symbol status, row count, first/last UTC timestamps, duplicate count, missing intervals, and error details. A failed symbol must not produce a formal CSV.

- [ ] **Step 4: Run tests and verify output safety**

Run: `python -m unittest discover -s tests -v`
Expected: PASS, including target preservation and manifest content.

- [ ] **Step 5: Commit the task**

Run: `git add scripts/fetch_binance_5y.py tests/test_fetch_binance_5y.py data/raw/.gitkeep logs/.gitkeep; git commit -m "feat: add validated atomic outputs and manifest"`
Expected: one commit containing output and manifest behavior.

### Task 4: Add project documentation and the `run` notebook area

**Files:**
- Create: `C:\Users\user\專題 new (五年)\README.md`
- Create: `C:\Users\user\專題 new (五年)\agement.md`
- Create: `C:\Users\user\專題 new (五年)\時程.md`
- Create: `C:\Users\user\專題 new (五年)\run\README.md`
- Create: `C:\Users\user\專題 new (五年)\run\01_fetch_and_validate.ipynb`

**Interfaces:**
- The notebook invokes `scripts\fetch_binance_5y.py` using `Path(__file__).resolve().parents[1]`-equivalent project-root discovery, or a notebook-safe `Path.cwd()` check that refuses to run from another project.
- `README.md` documents the exact command and output files.
- `agement.md` documents the five-year research scope and links to `時程.md`.

- [ ] **Step 1: Write documentation acceptance checks**

```python
import unittest


class DocumentationTests(unittest.TestCase):
    def test_documentation_has_required_paths_and_security_note(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        agement = Path("agement.md").read_text(encoding="utf-8")
        self.assertIn("BTCUSDT", readme)
        self.assertIn("XRPUSDT", readme)
        self.assertIn("不使用", readme)
        self.assertIn("API key", readme)
        self.assertIn("時程.md", agement)

    def test_run_notebook_is_valid_json(self):
        notebook = json.loads(Path("run/01_fetch_and_validate.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertTrue(any("fetch_binance_5y.py" in "".join(cell["source"]) for cell in notebook["cells"]))
```

- [ ] **Step 2: Run the documentation checks and verify they fail**

Run: `python -m unittest discover -s tests -v`
Expected: FAIL because the documentation and notebook do not exist yet.

- [ ] **Step 3: Create the documentation and notebook**

Copy the relevant research context from the existing `docs\agent.md` into `agement.md`, update the data scope to five years, and preserve the project’s model handoff context. Write `時程.md` with phases for download, validation, feature engineering, retraining, walk-forward evaluation, and final comparison. Create the notebook with cells for project-root verification, dry run, full fetch command, manifest loading, and CSV summary. Do not embed API credentials.

- [ ] **Step 4: Run documentation and notebook validation**

Run: `python -m unittest discover -s tests -v`
Expected: PASS, including valid notebook JSON and required security/path text.

Run: `jupyter nbconvert --to notebook --execute run/01_fetch_and_validate.ipynb --output run/01_fetch_and_validate.executed.ipynb --ExecutePreprocessor.timeout=120`
Expected: notebook executes its dry-run and validation cells without importing files from the original project.

- [ ] **Step 5: Commit the task**

Run: `git add README.md agement.md 時程.md run/README.md run/01_fetch_and_validate.ipynb; git commit -m "docs: add five-year project run notebook and schedule"`
Expected: one commit containing documentation and the run notebook.

### Task 5: Create the external project and perform a verified download

**Files:**
- Create or modify only inside: `C:\Users\user\專題 new (五年)`

**Interfaces:**
- The completed project contains all files from Tasks 1 through 4.
- The downloader is run from the new project root and writes four verified CSVs plus `logs\fetch_manifest.json`.

- [ ] **Step 1: Verify the destination path before writing**

Run PowerShell with an explicit resolved destination check for `C:\Users\user\專題 new (五年)`. Confirm it is not equal to `C:\Users\user\專題` and that all created paths remain under the destination.

- [ ] **Step 2: Create the destination directory and copy only approved project files**

Create `data\raw`, `scripts`, `tests`, `logs`, and `run` under the destination. Do not copy `.venv`, model outputs, notebooks, credentials, or unrelated files.

- [ ] **Step 3: Run the dry-run and test suite from the destination**

Run: `python scripts\fetch_binance_5y.py --dry-run --config config.json`
Expected: four planned symbols, five-year window, `1h`, public endpoint, and no output files changed.

Run: `python -m unittest discover -s tests -v`
Expected: PASS.

- [ ] **Step 4: Run the full public download**

Run: `python scripts\fetch_binance_5y.py --config config.json`
Expected: each symbol either completes with a success manifest entry or stops with a recorded error; no API key is requested.

- [ ] **Step 5: Verify the downloaded artifacts**

Run: `python -c "import json; from pathlib import Path; m=json.loads(Path('logs/fetch_manifest.json').read_text(encoding='utf-8')); print(json.dumps(m, ensure_ascii=False, indent=2))"`
Expected: four symbol entries with row counts, UTC boundaries, and missing-interval results.

Run: `python -c "import csv; from pathlib import Path; required={'btc_5y.csv','eth_5y.csv','sol_5y.csv','xrp_5y.csv'}; actual={p.name for p in Path('data/raw').glob('*.csv')}; assert required <= actual; [next(csv.DictReader(p.open(encoding='utf-8', newline=''))) for p in [Path('data/raw')/name for name in required]]; print('CSV artifact verification: PASS')"`
Expected: all four CSVs exist with the required header and at least one row.

- [ ] **Step 6: Commit the completed external project if Git is available**

Run: `git add .; git commit -m "feat: add verified five-year Binance dataset project"`
Expected: commit succeeds only if the destination is a Git repository; otherwise leave the files in place and report that no commit was made.

## Plan Self-Review

- Spec coverage: destination, symbols, interval, public API, CSV schema, atomic writes, quality manifest, run notebook, schedule, agement documentation, and security constraints each have an implementation task.
- Completion scan: no unfinished instructions or unspecified error-handling steps are used.
- Interface consistency: Tasks 1 through 3 define the functions used by later downloader and test steps; Task 4 validates the documented CLI path.
- Environment caveat: Git commands may fail if the desktop sandbox cannot resolve the Chinese workspace path or write `.git`; implementation must continue without destructive Git operations and report the limitation.
