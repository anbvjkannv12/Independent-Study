# BTC 4h XGBoost Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可由命令列重跑的 BTCUSDT 4 小時漲跌 XGBoost baseline，產出可追溯評估結果。

**Architecture:** `scripts/train_btc_4h_baseline.py` 以純函式完成載入、驗證、時間切分、訓練與評估，CLI 只呼叫這些函式並以原子方式寫出 JSON。測試使用合成資料；notebook 只讀取結果 JSON。

**Tech Stack:** Python 3、pandas、numpy、scikit-learn、xgboost、unittest、Jupyter。

## Global Constraints

- 特徵必須完全讀取 `logs/feature_manifest.json` 的 53 個 `feature_names`。
- `open_time`、`future_log_return`、`target_up`、`is_imputed` 不得進入模型特徵。
- 資料依時間順序切為 70% 訓練、15% 驗證、15% 測試；不得洗牌或重新取樣。
- 固定亂數種子 42 與 0.5 類別門檻；ROC-AUC 使用正類機率。
- 報告 Accuracy、Precision、Recall、F1、ROC-AUC、2×2 混淆矩陣，並在同一資料上比較全上漲與訓練集多數類別。
- 缺欄、非有限數、重複或未排序時間、任一切分單一類別時，程式非零失敗且不產生結果檔。

---

## File Structure

- Create `requirements-baseline.txt`: `scikit-learn>=1.4,<2.0` 與 `xgboost>=2.0,<3.0`。
- Create `scripts/train_btc_4h_baseline.py`: 純函式與 CLI。
- Create `tests/test_train_btc_4h_baseline.py`: 合成資料單元／整合測試。
- Create `docs/btc_4h_baseline.md`: 安裝、重跑與指標解讀。
- Create `run/02_xgboost_baseline_btc_4h.ipynb`: 結果報告檢視。

### Task 1: 安裝依賴並建立可驗證的訓練骨架

**Files:**
- Create: `requirements-baseline.txt`
- Create: `tests/test_train_btc_4h_baseline.py`
- Create: `scripts/train_btc_4h_baseline.py`

**Interfaces:**
- Produces: `load_baseline_dataset(data_path: Path, manifest_path: Path) -> tuple[pd.DataFrame, tuple[str, ...]]`
- Produces: `split_chronologically(dataset: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15) -> dict[str, pd.DataFrame]`

- [ ] **Step 1: 寫入失敗測試**

```python
def test_baseline_dependencies_are_importable(self):
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    self.assertTrue(callable(roc_auc_score))
    self.assertTrue(callable(XGBClassifier))

def test_loader_uses_manifest_features_and_orders_time(self):
    dataset, names = load_baseline_dataset(self.data_path, self.manifest_path)
    self.assertEqual(names, tuple(self.feature_names))
    self.assertTrue(dataset["open_time"].is_monotonic_increasing)
    self.assertNotIn("target_up", names)
```

- [ ] **Step 2: 驗證 RED 狀態**

Run: `& $py -m unittest tests.test_train_btc_4h_baseline -v`

Expected: FAIL，因目前缺少 sklearn、xgboost 與函式實作。

- [ ] **Step 3: 安裝依賴並建立最小載入器**

Create `requirements-baseline.txt`:

```text
scikit-learn>=1.4,<2.0
xgboost>=2.0,<3.0
```

Run: `& $py -m pip install -r requirements-baseline.txt`。

Implement loader to parse `open_time` as UTC, reject missing manifest features / `target_up` / non-finite data, require unique strictly increasing time, and return only the manifest feature tuple.

- [ ] **Step 4: 驗證 GREEN 狀態**

Run: `& $py -m unittest tests.test_train_btc_4h_baseline -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add requirements-baseline.txt scripts/train_btc_4h_baseline.py tests/test_train_btc_4h_baseline.py; git commit -m "build: add baseline training foundation"`。

### Task 2: 時間切分與契約錯誤處理

**Files:**
- Modify: `scripts/train_btc_4h_baseline.py`
- Modify: `tests/test_train_btc_4h_baseline.py`

**Interfaces:**
- Consumes: `load_baseline_dataset`。
- Produces: contiguous `train`, `validation`, and `test` frames.

- [ ] **Step 1: 寫入失敗測試**

```python
def test_chronological_split_is_contiguous_and_non_overlapping(self):
    splits = split_chronologically(self.dataset)
    self.assertEqual([len(splits[x]) for x in ("train", "validation", "test")], [14, 3, 3])
    self.assertLess(splits["train"].open_time.max(), splits["validation"].open_time.min())
    self.assertLess(splits["validation"].open_time.max(), splits["test"].open_time.min())

def test_loader_rejects_duplicate_time(self):
    with self.assertRaisesRegex(ValueError, "strictly increasing"):
        load_baseline_dataset(self.duplicate_time_path, self.manifest_path)
```

- [ ] **Step 2: 驗證 RED 狀態**

Run: `& $py -m unittest tests.test_train_btc_4h_baseline.BaselineDataTests -v`

Expected: FAIL，因 splitter 尚未實作。

- [ ] **Step 3: 實作最小時間切分與失敗條件**

```python
def split_chronologically(dataset, train_fraction=0.70, validation_fraction=0.15):
    train_end = int(len(dataset) * train_fraction)
    validation_end = train_end + int(len(dataset) * validation_fraction)
    return {"train": dataset.iloc[:train_end].copy(), "validation": dataset.iloc[train_end:validation_end].copy(), "test": dataset.iloc[validation_end:].copy()}
```

Require all three frames non-empty, have non-overlapping ranges, and report explicit errors for missing columns, unordered / duplicate time, and NaN or infinity.

- [ ] **Step 4: 驗證 GREEN 狀態並提交**

Run: `& $py -m unittest tests.test_train_btc_4h_baseline.BaselineDataTests -v`

Expected: PASS.

Run: `git add scripts/train_btc_4h_baseline.py tests/test_train_btc_4h_baseline.py; git commit -m "feat: enforce chronological baseline splits"`。

### Task 3: 指標、對照組與 XGBoost

**Files:**
- Modify: `scripts/train_btc_4h_baseline.py`
- Modify: `tests/test_train_btc_4h_baseline.py`

**Interfaces:**
- Produces: `evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, object]`
- Produces: `run_baseline(dataset: pd.DataFrame, feature_names: tuple[str, ...], random_state: int = 42) -> dict[str, object]`

- [ ] **Step 1: 寫入失敗測試**

```python
def test_metrics_use_probability_for_auc_and_threshold_for_labels(self):
    result = evaluate_predictions(pd.Series([0, 1, 1, 0]), np.array([0.1, 0.6, 0.4, 0.8]))
    self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
    self.assertAlmostEqual(result["roc_auc"], 0.5)

def test_baseline_reports_all_up_and_training_majority(self):
    result = run_baseline(self.dataset, tuple(self.feature_names))
    self.assertEqual(result["benchmarks"]["all_up"]["validation"]["predicted_class"], 1)
    self.assertIn("training_majority", result["benchmarks"])
```

- [ ] **Step 2: 驗證 RED 狀態**

Run: `& $py -m unittest tests.test_train_btc_4h_baseline.BaselineEvaluationTests -v`

Expected: FAIL，因兩個函式尚未實作。

- [ ] **Step 3: 實作最小評估與固定模型**

`evaluate_predictions` uses sklearn Accuracy / Precision / Recall / F1 with `zero_division=0`, `roc_auc_score(y_true, probabilities)`, and `confusion_matrix(..., labels=[0, 1]).tolist()`; labels are `(probabilities >= 0.5).astype(int)`.

`run_baseline` must fail when a split has one target class; fit `XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=1, tree_method="hist")`; evaluate model and both constant baselines on validation and test; sort `get_score(importance_type="gain")` descending.

- [ ] **Step 4: 驗證 GREEN 狀態並提交**

Run: `& $py -m unittest tests.test_train_btc_4h_baseline.BaselineEvaluationTests -v`

Expected: PASS.

Run: `git add scripts/train_btc_4h_baseline.py tests/test_train_btc_4h_baseline.py; git commit -m "feat: evaluate XGBoost baseline"`。

### Task 4: CLI、JSON 成果、Notebook 與實際執行

**Files:**
- Modify: `scripts/train_btc_4h_baseline.py`
- Modify: `tests/test_train_btc_4h_baseline.py`
- Create: `docs/btc_4h_baseline.md`
- Create: `run/02_xgboost_baseline_btc_4h.ipynb`

**Interfaces:**
- Consumes: `run_baseline` result.
- Produces: `logs/baseline_btc_4h.json`.

- [ ] **Step 1: 寫入失敗 CLI 測試**

```python
def test_cli_writes_complete_report_only_after_success(self):
    completed = subprocess.run([sys.executable, str(SCRIPT_PATH), "--data", str(self.data_path), "--manifest", str(self.manifest_path), "--output", str(self.output_path)], capture_output=True, text=True)
    self.assertEqual(completed.returncode, 0, completed.stderr)
    report = json.loads(self.output_path.read_text(encoding="utf-8"))
    self.assertEqual(report["task"], "BTCUSDT 4-hour direction classification")
    self.assertEqual(len(report["feature_names"]), 53)
    self.assertIn("test", report["model"])
```

- [ ] **Step 2: 驗證 RED 狀態**

Run: `& $py -m unittest tests.test_train_btc_4h_baseline.BaselineCliTests -v`

Expected: FAIL，因 CLI 與 JSON 產出尚未實作。

- [ ] **Step 3: 實作 CLI 與文件**

CLI defaults: `--data data/features/btc_4h.csv`, `--manifest logs/feature_manifest.json`, `--output logs/baseline_btc_4h.json`, `--random-state 42`. Write JSON to a same-directory temporary file and use `Path.replace()` after successful training only. Include split counts/ranges, paths, 53 feature names, fixed model settings, model and benchmark metrics, and gain importances.

Document commands:

```powershell
$py = "C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py -m pip install -r requirements-baseline.txt
& $py scripts\train_btc_4h_baseline.py
```

Notebook reads JSON, displays split ranges, model-versus-benchmark table and feature importance; it must not fit a model.

- [ ] **Step 4: 驗證、執行五年資料並提交**

Run:

```powershell
& $py -m unittest discover -s tests -v
& $py scripts\train_btc_4h_baseline.py
```

Expected: all tests PASS; JSON has 53 names, three split summaries, model/benchmark validation and test reports, and importance values.

Run: `git add scripts/train_btc_4h_baseline.py tests/test_train_btc_4h_baseline.py docs/btc_4h_baseline.md run/02_xgboost_baseline_btc_4h.ipynb logs/baseline_btc_4h.json; git commit -m "feat: publish reproducible BTC baseline"`。

## Plan Self-Review

- Spec coverage: Tasks 1–2 implement environment, data contract, and strict chronological slicing; Task 3 implements the model, all required metrics, benchmarks and importance; Task 4 implements atomic reporting, reproducibility assets and end-to-end verification.
- Placeholder scan: no deferred or ambiguous implementation elements remain.
- Interface consistency: loader returns the dataset/features consumed by training; splitter is used by training; evaluation reports are used uniformly for model and baselines; CLI serializes the training result.
