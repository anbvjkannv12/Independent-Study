# 原專題特徵與標籤流程整合 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 從目前 `data/processed` 產生與原專題完全相同的 53 個特徵及 1/4/12/24 小時標籤資料集。

**Architecture:** 新增單一、可匯入的 `scripts/feature_engineering.py`，集中保留原專題的特徵公式、欄位順序與資料驗證；CLI 負責讀取四幣種檔案、逐 horizon 產生輸出，並寫出可追溯的 manifest。`is_imputed` 只隨輸出保存，不進入特徵清單。

**Tech Stack:** Python 3, pandas, NumPy, unittest，沿用目前專題既有測試與腳本風格。

## Global Constraints

- 必須保留原專題 53 個特徵名稱與順序。
- 分類標籤為 `future_log_return > 0`；回歸標籤為 `log(close[t+h] / close[t])`。
- 僅使用 UTC 排序後的目前與過去觀測計算特徵。
- 不修改 `data/processed` 或 `C:\Users\user\專題`。
- 不新增第三方依賴。

---

### Task 1: Feature contract tests

**Files:**
- Create: `tests/test_feature_engineering.py`
- Read: `models/popular_coins/feature_manifest.json`

**Interfaces:**
- Tests import `EXPECTED_FEATURE_NAMES`, `add_features`, and `build_feature_dataset` from `scripts.feature_engineering`.

- [ ] **Step 1: Write the failing tests**

  Add tests for exact 53-column contract, formula-based labels, no leakage columns, sorted/deduplicated UTC index, `is_imputed` exclusion, and CLI dataset output helper.

- [ ] **Step 2: Run tests to verify they fail**

  Run `python -m unittest tests.test_feature_engineering -v`.
  Expected: import failure because `scripts.feature_engineering` does not exist yet.

### Task 2: Feature engineering implementation

**Files:**
- Create: `scripts/feature_engineering.py`
- Modify: `tests/test_feature_engineering.py` only if a test assertion needs clarification from the actual contract

**Interfaces:**
- `add_features(raw: pd.DataFrame, prediction_horizon: int = 1) -> pd.DataFrame` returns the feature dataset with a UTC datetime index.
- `build_feature_dataset(raw: pd.DataFrame, prediction_horizon: int = 1) -> pd.DataFrame` returns output-ready data including `open_time` and `is_imputed`.
- `EXPECTED_FEATURE_NAMES: tuple[str, ...]` exposes the exact 53-column contract.

- [ ] **Step 1: Implement the minimal feature and label pipeline**

  Port the original formulas for OHLCV-derived features, log return, ATR, volume features, time features, lag features, rolling features, RSI, MACD, and Bollinger z-score. Exclude raw OHLCV fields, `is_imputed`, `future_log_return`, and `target_up` from the feature list; preserve the manifest order.

- [ ] **Step 2: Run focused tests**

  Run `python -m unittest tests.test_feature_engineering -v`.
  Expected: PASS.

- [ ] **Step 3: Run the existing test suite**

  Run `python -m unittest discover -s tests -v`.
  Expected: all existing and new tests pass.

### Task 3: CLI and manifest integration

**Files:**
- Modify: `scripts/feature_engineering.py`
- Create: `tests/test_feature_engineering_cli.py` if CLI behavior needs isolated coverage

**Interfaces:**
- CLI command: `python scripts/feature_engineering.py --config config.json`.
- Optional arguments: `--input-dir`, `--output-dir`, `--manifest`, and comma-separated `--horizons`.

- [ ] **Step 1: Add input discovery and output writing**

  Map `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, and `XRPUSDT` to the processed file names, write one CSV per symbol/horizon, and create parent directories as needed.

- [ ] **Step 2: Add manifest generation**

  Record exact feature names, labels, horizons, input/output paths, row counts, imputed row counts, and UTC ranges. Write the manifest only after all requested datasets validate successfully.

- [ ] **Step 3: Run a small real-data smoke test**

  Run the CLI against the four existing processed files with `--horizons 1` and a temporary output directory, then verify generated rows and 53 features.

- [ ] **Step 4: Run the full test suite and CLI validation**

  Run `python -m unittest discover -s tests -v` and then run the default CLI to produce the project feature datasets.

### Task 4: Documentation and handoff

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the feature command and outputs**

  Add the exact command, output directory, horizon behavior, and note that `is_imputed` is retained for audit but excluded from model features.

- [ ] **Step 2: Verify final artifacts**

  Confirm `data/features/` and `logs/feature_manifest.json` contain the requested outputs and rerun the test suite before reporting completion.
