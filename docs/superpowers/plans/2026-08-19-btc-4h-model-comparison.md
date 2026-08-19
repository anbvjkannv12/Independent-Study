# BTC 4h Three-model Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 BTCUSDT 四小時漲跌的可重跑 walk-forward 評估，公平比較 XGBoost、LSTM 與 Transformer。

**Architecture:** 將時間 folds、校準、門檻、指標與基準組集中在 `scripts/btc_4h_evaluation.py`；各模型只實作訓練與機率預測 adapter。CLI runner 依相同 folds 呼叫所有 adapter，全部成功後原子寫入一個 JSON；文件與 notebook 只讀取 JSON。

**Tech Stack:** Python 3、pandas、numpy、scikit-learn、xgboost、PyTorch、unittest、Jupyter。

**Spec:** `docs/superpowers/specs/2026-08-19-btc-4h-model-comparison-design.md`

## Global Constraints

- 只使用 `data/features/btc_4h.csv` 和 manifest 指定的 53 個特徵；禁止使用 `open_time`、`future_log_return`、`target_up`、`is_imputed` 作特徵。
- 每個 fold 是嚴格遞增且無重疊的 train／validation／test；至少三個完整 expanding-window folds。
- 模型只以 fold train 擬合；Platt calibration 與門檻只以 fold validation 選擇；test 永不參與擬合、校準或選擇。
- 未校準機率計算 ROC-AUC；校準後機率與 validation 選定門檻計算決策指標。
- 每個模型與方向基準必須輸出 ROC-AUC、Brier、accuracy、balanced accuracy、precision、recall、F1、2×2 混淆矩陣、實際／預測上漲率與樣本數。
- 零交易訊號只輸出 `coverage: 0.0` 與 `status: "no_directional_prediction"`，不得偽造分類指標。
- 每個深度模型序列只能包含當列及以前的 53 個特徵；固定 seed，輸出序列長度、batch size、epoch 上限與 early-stopping 設定。
- 任何 fold 或模型失敗時 CLI 非零結束，且不得取代既有結果 JSON。

---

## File Structure

- Create: `requirements-model-comparison.txt` — 深度模型與共用評估的明確相依性。
- Create: `scripts/btc_4h_evaluation.py` — folds、校準、門檻、指標、基準組與 JSON-safe 摘要。
- Create: `scripts/btc_4h_model_adapters.py` — XGBoost、Logistic Regression、LSTM、Transformer 的統一機率預測 adapter。
- Create: `scripts/train_btc_4h_model_comparison.py` — CLI 編排、資料讀取、原子輸出。
- Create: `tests/test_btc_4h_evaluation.py` — 評估契約與 leakage 防護。
- Create: `tests/test_btc_4h_model_adapters.py` — 序列資料與三模型 adapter。
- Create: `tests/test_train_btc_4h_model_comparison.py` — 端到端 CLI 與原子輸出。
- Create: `docs/btc_4h_model_comparison.md` — 安裝、重跑與結果解讀。
- Create: `run/03_btc_4h_model_comparison.ipynb` — 只讀結果比較。

### Task 1: 建立相依性與 expanding-window fold 契約

**Files:**
- Create: `requirements-model-comparison.txt`
- Create: `scripts/btc_4h_evaluation.py`
- Create: `tests/test_btc_4h_evaluation.py`

**Interfaces:**
- Produces: `WalkForwardFold(index: int, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame)`。
- Produces: `build_expanding_folds(dataset: pd.DataFrame, initial_train_rows: int, validation_rows: int, test_rows: int, fold_count: int) -> list[WalkForwardFold]`。

- [ ] **Step 1: 寫入失敗測試**

```python
def test_expanding_folds_are_contiguous_and_never_overlap(self):
    folds = build_expanding_folds(self.dataset, 24, 8, 8, 3)
    self.assertEqual([fold.index for fold in folds], [0, 1, 2])
    self.assertEqual([len(fold.train) for fold in folds], [24, 32, 40])
    for fold in folds:
        self.assertLess(fold.train.open_time.max(), fold.validation.open_time.min())
        self.assertLess(fold.validation.open_time.max(), fold.test.open_time.min())

def test_expanding_folds_reject_empty_or_single_class_partitions(self):
    with self.assertRaisesRegex(ValueError, "not enough rows"):
        build_expanding_folds(self.dataset.iloc[:30], 24, 8, 8, 3)
```

- [ ] **Step 2: 驗證 RED 狀態**

Run: `& $py -m unittest tests.test_btc_4h_evaluation.WalkForwardTests -v`

Expected: FAIL，因 `btc_4h_evaluation` 與 `build_expanding_folds` 尚不存在。

- [ ] **Step 3: 新增最小相依性與 fold 實作**

Create `requirements-model-comparison.txt`:

```text
-r requirements-baseline.txt
torch>=2.3,<3.0
```

Implement immutable fold container and loop:

```python
@dataclass(frozen=True)
class WalkForwardFold:
    index: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

def build_expanding_folds(dataset, initial_train_rows, validation_rows, test_rows, fold_count):
    required = initial_train_rows + fold_count * (validation_rows + test_rows)
    if len(dataset) < required:
        raise ValueError("not enough rows for requested walk-forward folds")
    folds = []
    for index in range(fold_count):
        train_end = initial_train_rows + index * (validation_rows + test_rows)
        validation_end = train_end + validation_rows
        test_end = validation_end + test_rows
        folds.append(WalkForwardFold(index, dataset.iloc[:train_end].copy(), dataset.iloc[train_end:validation_end].copy(), dataset.iloc[validation_end:test_end].copy()))
    return folds
```

Validate each partition's ordered unique timestamps, finite values and at least two target classes before returning it.

- [ ] **Step 4: 驗證 GREEN 狀態**

Run: `& $py -m unittest tests.test_btc_4h_evaluation.WalkForwardTests -v`

Expected: PASS。

- [ ] **Step 5: Commit**

Run: `git add requirements-model-comparison.txt scripts/btc_4h_evaluation.py tests/test_btc_4h_evaluation.py; git commit -m "feat: add BTC walk-forward fold contract"`

### Task 2: 實作共同指標、基準組、校準與門檻選擇

**Files:**
- Modify: `scripts/btc_4h_evaluation.py`
- Modify: `tests/test_btc_4h_evaluation.py`

**Interfaces:**
- Produces: `evaluate_directional_predictions(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, object]`。
- Produces: `fit_platt_calibrator(validation_probabilities: np.ndarray, validation_target: pd.Series) -> LogisticRegression`。
- Produces: `select_balanced_accuracy_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float`。
- Produces: `evaluate_baselines(fold: WalkForwardFold, feature_names: tuple[str, ...]) -> dict[str, object]`。

- [ ] **Step 1: 寫入失敗測試**

```python
def test_threshold_selection_uses_validation_only(self):
    threshold = select_balanced_accuracy_threshold(pd.Series([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    self.assertGreaterEqual(threshold, 0.4)
    self.assertLessEqual(threshold, 0.6)

def test_zero_trade_is_not_reported_as_classifier(self):
    result = evaluate_zero_trade()
    self.assertEqual(result, {"status": "no_directional_prediction", "coverage": 0.0})

def test_baselines_include_previous_direction_and_logistic(self):
    report = evaluate_baselines(self.fold, self.feature_names)
    self.assertEqual(set(report), {"all_up", "training_majority", "previous_direction", "logistic_regression", "zero_trade"})
```

- [ ] **Step 2: 驗證 RED 狀態**

Run: `& $py -m unittest tests.test_btc_4h_evaluation.EvaluationContractTests -v`

Expected: FAIL，因評估、校準與基準函式不存在。

- [ ] **Step 3: 實作最小共同評估邏輯**

Implement metrics with `accuracy_score`、`balanced_accuracy_score`、`precision_score(..., zero_division=0)`、`recall_score`、`f1_score`、`roc_auc_score`、`brier_score_loss` and `confusion_matrix(..., labels=[0, 1]).tolist()`.

Fit Platt scaling with `LogisticRegression(random_state=42, solver="lbfgs")` on `validation_probabilities.reshape(-1, 1)` and validation target only. Search thresholds `np.arange(0.05, 0.951, 0.01)`; choose greatest balanced accuracy and break ties by smallest threshold. The previous-direction baseline must use `log_return_lag_1 > 0` only and fail if that column is absent. Fit Logistic Regression only on train, then calibrate it on validation under the same rule.

- [ ] **Step 4: 驗證 GREEN 狀態**

Run: `& $py -m unittest tests.test_btc_4h_evaluation.EvaluationContractTests -v`

Expected: PASS。

- [ ] **Step 5: Commit**

Run: `git add scripts/btc_4h_evaluation.py tests/test_btc_4h_evaluation.py; git commit -m "feat: add calibrated BTC evaluation baselines"`

### Task 3: 實作 XGBoost、LSTM 與 Transformer 統一 adapter

**Files:**
- Create: `scripts/btc_4h_model_adapters.py`
- Create: `tests/test_btc_4h_model_adapters.py`

**Interfaces:**
- Produces: `make_sequence_samples(frame: pd.DataFrame, feature_names: tuple[str, ...], sequence_length: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]`。
- Produces: `fit_predict_xgboost(train, validation, test, feature_names, seed) -> tuple[np.ndarray, np.ndarray]`。
- Produces: `fit_predict_lstm(train, validation, test, feature_names, settings) -> tuple[np.ndarray, np.ndarray]`。
- Produces: `fit_predict_transformer(train, validation, test, feature_names, settings) -> tuple[np.ndarray, np.ndarray]`。

- [ ] **Step 1: 寫入失敗測試**

```python
def test_sequence_samples_end_at_current_row_without_future_features(self):
    sequences, targets, times = make_sequence_samples(self.frame, self.feature_names, 4)
    self.assertEqual(sequences.shape, (len(self.frame) - 3, 4, 53))
    self.assertEqual(times[0], self.frame.open_time.iloc[3])
    np.testing.assert_array_equal(sequences[0, -1], self.frame.loc[3, list(self.feature_names)].to_numpy())

def test_all_adapters_return_one_probability_per_validation_and_test_row(self):
    validation_probability, test_probability = fit_predict_xgboost(self.train, self.validation, self.test, self.feature_names, 42)
    self.assertEqual(len(validation_probability), len(self.validation))
    self.assertEqual(len(test_probability), len(self.test))
```

- [ ] **Step 2: 驗證 RED 狀態**

Run: `& $py -m unittest tests.test_btc_4h_model_adapters -v`

Expected: FAIL，因 adapters 不存在。

- [ ] **Step 3: 實作最小 adapter 與序列模型**

Define a `ModelSettings` dataclass with `seed=42`, `sequence_length=24`, `batch_size=128`, `max_epochs=30`, `patience=5`, `learning_rate=0.001`, `hidden_size=32`, and `num_heads=4`. Fit feature scaling from train only; transform validation/test with that fitted scaler. For sequences, prepend the necessary preceding rows from the earlier partition but generate labels only for the requested partition, so every validation/test row has a prediction without future context.

Implement `LSTMClassifier(nn.Module)` with one `nn.LSTM(input_size=53, hidden_size=settings.hidden_size, batch_first=True)` followed by `nn.Linear(hidden_size, 1)`. Implement `TransformerClassifier(nn.Module)` with input projection to `hidden_size`, learned positional embedding of `sequence_length`, `nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)`, one encoder layer, final-token pooling and `nn.Linear(hidden_size, 1)`. Use `BCEWithLogitsLoss`, Adam, validation loss early stopping, and restore the best validation-loss state. Seeds must set `random`, `numpy`, and `torch`.

- [ ] **Step 4: 驗證 GREEN 狀態**

Run: `& $py -m unittest tests.test_btc_4h_model_adapters -v`

Expected: PASS. Then run `& $py -m unittest discover -s tests -v`.

- [ ] **Step 5: Commit**

Run: `git add scripts/btc_4h_model_adapters.py tests/test_btc_4h_model_adapters.py; git commit -m "feat: add BTC LSTM and Transformer adapters"`

### Task 4: 編排共同 walk-forward 執行與原子 JSON 結果

**Files:**
- Create: `scripts/train_btc_4h_model_comparison.py`
- Create: `tests/test_train_btc_4h_model_comparison.py`

**Interfaces:**
- Consumes: `load_baseline_dataset`, `build_expanding_folds`, common evaluators and all three adapters。
- Produces: `run_model_comparison(dataset, feature_names, settings) -> dict[str, object]`。
- Produces: `logs/btc_4h_model_comparison.json`。

- [ ] **Step 1: 寫入失敗 CLI 測試**

```python
def test_cli_writes_three_model_folds_only_after_success(self):
    completed = subprocess.run([sys.executable, str(SCRIPT_PATH), "--data", str(self.data_path), "--manifest", str(self.manifest_path), "--output", str(self.output_path), "--fold-count", "3", "--initial-train-rows", "24", "--validation-rows", "8", "--test-rows", "8", "--max-epochs", "1"], capture_output=True, text=True)
    self.assertEqual(completed.returncode, 0, completed.stderr)
    report = json.loads(self.output_path.read_text(encoding="utf-8"))
    self.assertEqual(set(report["models"]), {"xgboost", "lstm", "transformer"})
    self.assertEqual(len(report["folds"]), 3)

def test_cli_preserves_existing_output_if_one_model_fails(self):
    self.output_path.write_text('{"old": true}', encoding="utf-8")
    completed = subprocess.run([sys.executable, str(SCRIPT_PATH), "--data", str(self.bad_data_path), "--output", str(self.output_path)], capture_output=True, text=True)
    self.assertNotEqual(completed.returncode, 0)
    self.assertEqual(json.loads(self.output_path.read_text(encoding="utf-8")), {"old": True})
```

- [ ] **Step 2: 驗證 RED 狀態**

Run: `& $py -m unittest tests.test_train_btc_4h_model_comparison -v`

Expected: FAIL，因 CLI 與 JSON schema 不存在。

- [ ] **Step 3: 實作 runner 與結果 schema**

Add CLI defaults: data `data/features/btc_4h.csv`, manifest `logs/feature_manifest.json`, output `logs/btc_4h_model_comparison.json`, `--fold-count 3`, `--initial-train-rows 24000`, `--validation-rows 4000`, `--test-rows 4000`, and the Task 3 model settings. For each fold, call every model adapter, calibrate and select threshold from validation only, then report test metrics and probability quintiles. Calculate per-model mean and sample standard deviation across fold test metrics. Use a sibling temporary JSON file and `Path.replace()` only after all folds and all models pass.

- [ ] **Step 4: 驗證 GREEN 狀態**

Run: `& $py -m unittest tests.test_train_btc_4h_model_comparison -v`

Expected: PASS. Then run `& $py -m unittest discover -s tests -v`.

- [ ] **Step 5: Commit**

Run: `git add scripts/train_btc_4h_model_comparison.py tests/test_train_btc_4h_model_comparison.py; git commit -m "feat: run BTC walk-forward model comparison"`

### Task 5: 產出可閱讀的比較文件與實際五年資料結果

**Files:**
- Create: `docs/btc_4h_model_comparison.md`
- Create: `run/03_btc_4h_model_comparison.ipynb`
- Create: `logs/btc_4h_model_comparison.json`

**Interfaces:**
- Consumes: Task 4 JSON。
- Produces: 可重跑命令與只讀結果檢視。

- [ ] **Step 1: 寫入失敗文件／notebook 測試**

```python
def test_comparison_documentation_and_notebook_are_read_only(self):
    documentation = (PROJECT_ROOT / "docs" / "btc_4h_model_comparison.md").read_text(encoding="utf-8")
    notebook = json.loads((PROJECT_ROOT / "run" / "03_btc_4h_model_comparison.ipynb").read_text(encoding="utf-8"))
    self.assertIn("train_btc_4h_model_comparison.py", documentation)
    self.assertNotIn(".fit(", "".join("".join(cell.get("source", [])) for cell in notebook["cells"]))
```

- [ ] **Step 2: 驗證 RED 狀態**

Run: `& $py -m unittest tests.test_train_btc_4h_model_comparison.ComparisonDocumentationTests -v`

Expected: FAIL，因文件與 notebook 不存在。

- [ ] **Step 3: 建立結果閱讀介面並執行完整資料**

Document the dependency install and run commands:

```powershell
$py = "C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py -m pip install -r requirements-model-comparison.txt
& $py scripts\train_btc_4h_model_comparison.py
```

Notebook must load only `logs/btc_4h_model_comparison.json`, display fold ranges, per-model mean/std test table, fold table, calibration quintiles and a warning that metrics do not prove profitability. Run the full test suite, run the CLI on real data, then confirm JSON has three models, at least three folds, all five baselines, calibration metadata and aggregate test metrics.

- [ ] **Step 4: 驗證 GREEN 狀態**

Run: `& $py -m unittest discover -s tests -v`

Expected: PASS. Run: `& $py scripts\train_btc_4h_model_comparison.py`.

- [ ] **Step 5: Commit**

Run: `git add docs/btc_4h_model_comparison.md run/03_btc_4h_model_comparison.ipynb logs/btc_4h_model_comparison.json tests/test_train_btc_4h_model_comparison.py; git commit -m "docs: publish BTC four-hour model comparison"`

## Plan Self-Review

- Spec coverage: Task 1 implements continuous expanding folds; Task 2 implements shared metrics, five baselines, calibration and validation-only threshold choice; Task 3 supplies the three requested model families and future-safe sequences; Task 4 enforces one schema and atomic output; Task 5 publishes only read-only results and actual-data verification.
- Placeholder scan: no TBD, TODO, deferred implementation, or unnamed interfaces remain.
- Type consistency: Task 1 fold containers flow to Task 2 baseline evaluation and Task 4 runner; Task 3 adapters all return validation/test probability arrays consumed by Task 4.
