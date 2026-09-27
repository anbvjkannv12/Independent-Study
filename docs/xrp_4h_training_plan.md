# XRPUSDT 4 小時方向分類 訓練模型計劃書

> 建立日期：2026-09-20　｜　研究範圍權威文件：[agement.md](../agement.md)　｜　排程：[時程.md](../時程.md)
>
> 本計劃書只規劃 **4 小時漲跌方向分類**。不規劃回歸任務（舊專題已確認為負結果，見 [log_return_regression_comparison.md](log_return_regression_comparison.md)），也不規劃 1h／12h／24h 建模。

## 1. 目標與範圍

| 項目 | 內容 |
| --- | --- |
| 標的 | `XRPUSDT`（Binance Spot） |
| 任務 | 二元分類：`target_up = future_log_return > 0`，預測視窗 **4 小時** |
| 模型 | XGBoost、LSTM、Transformer 三者比較 |
| 定案標準 | 三個 test folds 的平均與逐 fold test ROC-AUC 一致性；**允許的結論包含「無可靠優勝模型」** |
| 排定期間 | 2026-09-24 ～ 09-27（[時程.md](../時程.md) 階段 7） |
| 實際狀態 | **尚未開始**（今日 2026-09-20）。校正後特徵檔已備妥，但**全歷史比較尚未以校正後資料重跑**（見第 9 節） |

**本輪不做的事**：不重新設計特徵、不做超參數搜尋、不做 Stacking 融合（`agement.md` 已說明「表現穩定」的前提條件尚未達成）、不啟動三段式目標與外部特徵的新實驗。

**預期結果**：ROC-AUC 落在 **0.50–0.53** 區間。這是依既有結果寫下的**預期值，不是要達成的目標**；本計劃書不設定「把 AUC 拉高到某個數字」的績效承諾。時程.md 特別註記 XRP 事件驅動特性明顯，需留意跨期穩定性——這是本輪的重點。

## 2. 資料來源與期間

| 項目 | 內容 |
| --- | --- |
| 原始資料 | Binance Spot `XRPUSDT` 1 小時 K 線，`data/raw/xrp_5y.csv`，43,818 列 |
| 補值後資料 | `data/processed/xrp_5y.csv`，其中 `is_imputed=1` 共 **7 列**（佔 0.016%） |
| 特徵資料（原始版） | `data/features/xrp_4h.csv`，**43,755 列** |
| 特徵資料（校正後） | `data/features/revisions/2026-09-16/xrp_4h.csv`，**43,795 列**、7 個補值列 |
| 特徵 manifest（校正後） | `logs/sol_xrp_4h_feature_manifest_2026-09-16.json` |
| 時間範圍 | 2021-08-10 11:00 UTC ～ 2026-08-09 06:00 UTC（全部以 UTC 表示） |
| 來源記錄 | `logs/fetch_manifest.json`、`logs/data_quality_report.json`、`logs/imputation_manifest.json`、`logs/feature_manifest.json` |

**缺口與補值**：XRP 原始資料有 7 個缺失小時。補值規則見 `agement.md`：OHLC 全部取前一根的 `close`（視為平盤），成交量類欄位補 `0`，並以 `is_imputed` 標記。`is_imputed` **不是模型特徵**，只留作稽核欄。

**已修正的資料缺陷**：與 ETH、SOL 相同的零成交量補值缺陷。校正前的 `data/features/xrp_4h.csv` 因補值列成交量為 0，使 `trade_intensity` 與 `taker_buy_ratio` 變成非有限值而在 `dropna()` 被移除，少了 **40 小時**（43,755 → 43,795）。`scripts/feature_engineering.py` 已修正並加上回歸測試。**XRP 的校正後特徵檔已產出，但全歷史比較尚未重跑**，這是本輪的第一項工作。

## 3. 特徵清單摘要（引用 `logs/feature_manifest.json`）

固定使用 manifest 記錄的 **53 個特徵**，順序不變。這 53 欄沿用舊專題 `src/lstm_experiment_utils.py` 的 `add_features()` 定義，**本計劃書不新增、不移除、不重新發明任何特徵**。

| 群組 | 欄位 | 數量 |
| --- | --- | ---: |
| 成交量與筆數原始欄 | `quote_asset_volume`、`number_of_trades`、`taker_buy_base`、`taker_buy_quote` | 4 |
| K 棒形狀與波動 | `log_return`、`range_pct`、`body_pct`、`upper_shadow_pct`、`lower_shadow_pct`、`atr_14_pct` | 6 |
| 量能衍生 | `log_volume`、`volume_change`、`trade_intensity`、`taker_buy_ratio`、`quote_volume_change` | 5 |
| 時間週期 | `hour_sin`、`hour_cos`、`dow_sin`、`dow_cos` | 4 |
| Lag（1／2／3／6／12／24） | `log_return_lag_*`、`volume_change_lag_*`、`taker_buy_ratio_lag_*` | 18 |
| Rolling（6／12／24） | `return_mean_*`、`return_std_*`、`volume_mean_*`、`range_mean_*` | 12 |
| 技術指標 | `rsi_14`、`macd_pct`、`macd_signal_pct`、`bollinger_z_20` | 4 |
| **合計** | | **53** |

**標籤定義**（manifest `label_definition`）：`future_log_return = log(close[t+4] / close[t])`、`target_up = future_log_return > 0`。

**manifest 記錄的防洩漏控制**：依 UTC `open_time` 排序後才做特徵；lag 與 rolling 只用過去觀測；標籤欄位排除在 `feature_names` 之外；`is_imputed` 保留稽核但排除在特徵之外。

**特徵與 XRP 事件驅動特性的落差（須寫進報告）**：53 欄全部來自 K 線本身（價格、量能、時間、技術指標），沒有任何欄位能表達訴訟、上架、清算等事件資訊。XRP 的大幅波動常由這類外生事件觸發，這是本特徵集的結構性限制，而不是模型調參可以彌補的。

**外部訊號（鏈上／情緒／衍生品）不納入本輪。** [external_features_ab_comparison.md](external_features_ab_comparison.md)（2026-09-16）已在相同資料列上做過受控 A/B。XRP 有兩個必須寫進報告的限制：

- **XRP 沒有交易所流量資料**，外部欄位只有 17 個（BTC／ETH 為 20）。
- 加入外部特徵後，**XRP LSTM 降 −0.0164，是 12 組比較中最大的退步**；Transformer 升 +0.0078、XGBoost 升 +0.0007，方向不一致，整體不構成效果證據。

## 4. 模型架構與超參數規劃

三個模型的實作在 `scripts/btc_4h_model_adapters.py`（檔名雖為 btc，實際為四幣種共用）。**本輪沿用既有設定，不做超參數搜尋**——在訊號僅 0.50–0.53 的情況下，調參的改善很難與雜訊區分，且會消耗 9/30 前的時程。

共用執行設定（`ModelSettings`）：`seed=42`、`sequence_length=24`、`batch_size=128`、`max_epochs=30`、`patience=5`、`learning_rate=0.001`、`hidden_size=32`、`num_heads=4`。

### 4.1 XGBoost

| 參數 | 舊專題 4h 設定 | 本輪 5 年資料設定 | 是否調整・理由 |
| --- | --- | --- | --- |
| `n_estimators` | 300 | 200 | **已調整**。5 年資料的 walk-forward train 段最大到 35,795 列，早期驗證顯示 200 棵已收斂 |
| `max_depth` | 3 | 4 | **已調整**。特徵數固定 53，稍深一層換取交互作用表達力 |
| `learning_rate` | 0.03 | 0.05 | **已調整**，配合較少的樹數 |
| `subsample` | 未指定 | 0.8 | 新增，降低過擬合 |
| `colsample_bytree` | 未指定 | 0.8 | 新增 |
| `objective` ／ `eval_metric` | — | `binary:logistic` ／ `logloss` | — |
| `tree_method` | — | `hist` | 為 CPU 執行速度 |

> 舊專題出處：`src/multi_horizon_experiment_utils.py` 的 `run_xgboost()` 內建參數，屬「調整後（預測 4、12、24 小時）」實驗設定。

### 4.2 LSTM

| 參數 | 舊專題 4h 設定 | 本輪 5 年資料設定 | 是否調整・理由 |
| --- | --- | --- | --- |
| 框架 | TensorFlow／Keras | **PyTorch** | **已調整**。新專題統一改用 PyTorch（torch 2.14.0+cu126） |
| `sequence_length` | 24（default） | 24 | 未調整。註：舊專題 `xgboost_vs_lstm_comparison.md` 的延伸實驗曾指出 XRP 的 48 小時序列較有潛力，但那是 1h horizon 的結論，**本輪不採納、不另開實驗**，列為未來工作 |
| 隱藏單元 | `lstm_units=64`（default） | `hidden_size=32` | **已調整**。53 特徵 × 24 步的樣本下 64 單元容易過擬合 |
| 網路結構 | LSTM(64) → Dropout(0.20) → Dense(32, relu) → Dropout(0.10) → Dense(1, sigmoid) | `nn.LSTM(53→32)` → `Linear(32,1)` | **已調整**。簡化為單層，避免在弱訊號上疊加不可解釋的容量 |
| Loss／Optimizer | `binary_crossentropy`／Adam(1e-3, clipnorm=1.0) | `BCEWithLogitsLoss`／Adam(1e-3) | 等價 |
| Epochs | 15（固定） | `max_epochs=30` + `patience=5` early stopping | **已調整**，改由 validation loss 決定停點 |
| `batch_size` | 64 | 128 | **已調整** |

> 舊專題出處：`DEFAULT_MODEL_CONFIG`（`src/multi_horizon_experiment_utils.py:41`）與 `build_lstm_model()`。**XRP 在舊專題 4h 沒有 per-symbol override**，走 default（`sequence_length=24`、`lstm_units=64`）；舊專題僅對 BTC 4h 設過 override（`sequence_length=12`、`lstm_units=32`）。本輪採用的 32 單元即沿用該 override 的容量設定，但序列長度維持 24。

### 4.3 Transformer

| 參數 | 舊專題 4h 設定 | 本輪 5 年資料設定 | 是否調整・理由 |
| --- | --- | --- | --- |
| 框架 | TensorFlow／Keras（`src/train_week5_transformer.py`） | **PyTorch** | **已調整** |
| `sequence_length` | 24（default） | 24 | 未調整 |
| `d_model` | 64（default） | 32 | **已調整**，與 LSTM 同步縮小容量 |
| `ff_dim` | 128（default） | 64（`dim_feedforward = hidden_size * 2`） | **已調整**，維持 2× 比例 |
| `num_heads` | `NUM_HEADS`（`d_model` 須整除） | 4（key dim = 32 / 4 = 8） | — |
| Encoder 層數 | `NUM_ENCODER_BLOCKS` | **1** | 簡化 |
| 位置編碼 | `layers.Embedding` 學習式 | 學習式 `nn.Parameter(1, 24, 32)` | 等價 |
| `dropout` | 0.20 | `TransformerEncoderLayer` 預設 | 未特別調整 |
| 輸出 | 序列聚合後 Dense(1) | 取**最後一個 timestep** → `Linear(32,1)` | 與 LSTM 取最後隱藏狀態一致 |

> 舊專題出處：`src/train_week5_transformer.py` 的 `build_transformer_model()` 與 `DEFAULT_MODEL_CONFIG`。XRP 在舊專題 4h 同樣沒有 override，走 default（`d_model=64`、`ff_dim=128`）。

## 5. 訓練與驗證方法（walk-forward 與防洩漏）

**絕不使用隨機切分。** 切分邏輯在 `scripts/btc_4h_evaluation.py` 的 `build_expanding_folds()`，採 expanding-window walk-forward：

| 參數 | 值 |
| --- | ---: |
| `initial_train_rows` | 19,755（校正後資料用 **19,795**，補回被移除的 40 小時，使日曆時間對齊） |
| `validation_rows` | 4,000 |
| `test_rows` | 4,000 |
| `fold_count` | 3 |

| Fold | train | validation | test |
| --- | --- | --- | --- |
| 1 | 2021-08-10 → 2023-11-13 | 2023-11-13 → 2024-04-27 | **2024-04-27 → 2024-10-11** |
| 2 | 2021-08-10 → 2024-10-11 | 2024-10-11 → 2025-03-27 | **2025-03-27 → 2025-09-09** |
| 3 | 2021-08-10 → 2025-09-09 | 2025-09-09 → 2026-02-23 | **2026-02-23 → 2026-08-09** |

四個幣種使用**完全相同的 fold 日期**，這是跨幣種可比性的前提。

### 防洩漏作法（逐項對應程式碼）

1. **時間順序**：`_validate_partition()` 檢查每個 partition 的 `open_time` 嚴格遞增、無重複、無 NaN 與非有限值；train < validation < test 連續銜接且不重疊。
2. **序列樣本不跨界**：`make_sequence_samples()` 只使用「結束於預測列」的視窗；validation／test 的前 23 列歷史由 `_scaled_partition_sequences()` 從**前一段**取得，不使用未來列。
3. **標準化只 fit 在 train**：`StandardScaler().fit(train)`，validation／test 只做 `transform`。
4. **機率校準只用 validation**：`fit_platt_calibrator()` 以該 fold 的 validation 擬合 Platt scaling，test 完全不參與。
5. **門檻只用 validation**：`select_balanced_accuracy_threshold()` 在 0.05–0.95（每 0.01）中挑 validation balanced accuracy 最佳者，再套用到 test。
6. **標籤未來值不入特徵**：`future_log_return`、`target_up`、`is_imputed` 皆不在 `feature_names` 內。
7. **測試守住流程**：`tests/test_btc_4h_evaluation.py`、`tests/test_btc_4h_model_comparison.py` 驗證 folds 不重疊、門檻僅依 validation 選擇、`zero_trade` 不被誤報為分類器、基準組完整性。

### 執行環境與指令

五年專題隨附的 Codex runtime **沒有 `torch`**。`docs/xrp_4h_model_comparison.md` 目前記錄的重跑指令仍指向該 runtime，**本輪必須改用 `.venv-transformer`**（Python 3.12.14、Torch 2.14.0+cu126、Pandas 2.3.3、NumPy 2.5.0、XGBoost 3.2.0、scikit-learn 1.9.0），並在定案時更新該文件的指令。

```powershell
& ".venv-transformer\Scripts\python.exe" scripts\train_btc_4h_model_comparison.py --symbol XRP --data data\features\revisions\2026-09-16\xrp_4h.csv --manifest logs\sol_xrp_4h_feature_manifest_2026-09-16.json --output logs\xrp_4h_model_comparison_2026-09-16.json --initial-train-rows 19795 --validation-rows 4000 --test-rows 4000 --fold-count 3
```

每次執行需記錄：資料與 manifest 的 SHA-256、Git revision、套件版本——這是 ETH 定案時補齊的項目，XRP 尚未補（見第 8 節風險 #7）。

## 6. 評估指標

### 6.1 主要指標（決定定案的依據）

| 指標 | 用途 | 判讀原則 |
| --- | --- | --- |
| **ROC-AUC**（test） | 主要選模指標，衡量排序能力，不受門檻影響 | 0.50 = 隨機。逐 fold 都要看，不能只看平均 |
| **樣本標準差**（3 folds） | 穩定性 | 平均差距小於 fold 標準差時，不得宣稱某模型較優 |
| **Brier score** | 機率校準品質 | 二元目標下約 0.25 即為弱辨識。XRP 三模型皆約 0.2505–0.2507，為四幣中最差 |
| Accuracy／Balanced accuracy | 輔助 | 類別比例接近 50／50，Accuracy 單獨無意義 |
| **Precision／Recall／F1** | **XRP 的重點診斷** | XRP 的 F1 與 Accuracy 落差最大，必須與 `predicted_up_rate` 併列 |
| 機率五分位實際上漲率 | 診斷 | **XRP 必做**：檢查是否單調，並確認 Fold 2 的異常是否由單一事件期間主導 |

### 6.2 次要指標（Sharpe、最大回撤）— 需補程式，列為待辦

舊專題 [xgboost_vs_lstm_comparison.md](xgboost_vs_lstm_comparison.md) 的呈現方式（幣種 × 模型 × Accuracy／F1／Log loss／策略報酬／Sharpe／最大回撤）**新專題尚未實作**。

- 可重用的實作：`src/lstm_experiment_utils.py:267` 的 `strategy_metrics(y_proba, future_log_returns, threshold)`，已回傳 `strategy_total_return`、`strategy_sharpe`（年化係數 `sqrt(24*365)`）、`max_drawdown`。
- 資料已具備：`data/features/xrp_4h.csv` 已含 `future_log_return` 欄。
- 待辦：把 `strategy_metrics()` 接進 `scripts/btc_4h_evaluation.py` 的 fold 報告，輸出到既有的比較 JSON。

**使用限制（必須寫進報告）**：

1. **未納入手續費與滑價**，不得解讀為可交易績效。
2. 目前訊號在 ROC-AUC 0.50–0.53，**交易面指標不得作為選模依據**——在近乎隨機的排序上，Sharpe 主要反映 test 期的市場走向與雜訊，而非模型能力。XRP 又特別容易被少數事件行情主導：舊專題 3 年資料的 XRP 兩個模型策略報酬皆為負（XGBoost −0.1292、LSTM −0.0198），正是這個風險的實例。
3. 僅作為「若強行交易會發生什麼」的敘述性補充。

### 6.3 共同基準組

`evaluate_baselines()` 對每個 fold 產出五種基準，三個模型必須與之並列呈現：

| 基準 | 說明 |
| --- | --- |
| `all_up` | 永遠預測上漲 |
| `training_majority` | 訓練集多數類別 |
| `previous_direction` | 以 `log_return_lag_1 > 0` 當預測 |
| `logistic_regression` | 53 特徵 + StandardScaler，同樣只用 validation 做校準與門檻 |
| `zero_trade` | 不交易，`coverage = 0`，**不是分類器**，不參與 ROC-AUC 排名 |

> **XRP 特別注意**：在外部特徵執行中，Logistic Regression 基準達 0.5103，與 XGBoost（0.5216）、LSTM（0.5102）同級。XRP 的定案敘述必須說明三個模型相對線性基準沒有明顯優勢。

## 7. 比較基準

### 7.1 對照 BTC（已定案）

BTC 已於 2026-08-19 選定 **Transformer** 為定案模型，**不再重新訓練**。XRP 使用完全相同的 fold 日期、特徵契約與評估流程，因此可直接並列。

| 幣種 | XGBoost | LSTM | Transformer |
| --- | ---: | ---: | ---: |
| BTCUSDT（定案基準） | 0.5289 | 0.5302 | **0.5318** |
| XRPUSDT（原始版，待重跑） | 0.5164 | **0.5231** | 0.5157 |

> 來源：`logs/btc_4h_model_comparison.json`、`logs/xrp_4h_model_comparison.json`

XRP 的模型排序與 BTC 不同（BTC 由 Transformer 最高，XRP 由 LSTM 最高），這本身就是「沒有跨幣種穩定優勝模型」的證據之一。

### 7.2 對照舊專題 3 年資料（XRP 4h 分類）

| 模型 | Accuracy | F1 | Log loss |
| --- | ---: | ---: | ---: |
| XGBoost | 0.5050 | 0.4740 | 0.7033 |
| LSTM | 0.5206 | 0.4823 | 0.6988 |
| Transformer | 0.5033 | 0.4719 | 0.6961 |

> 來源：`models/調整後(預測4,12,24小時)/{xgboost,lstm,transformer}/classification/metrics_summary.csv`（3 年資料、5 folds、`TEST_SIZE=720`、`GAP=24`）

**比較的正確寫法**：兩階段的切分設定不同（3 年用 5 folds × 720 列、24 列 gap；5 年用 3 folds × 4,000 列），**數字不可直接相減**。可以說的是「兩個資料期間、兩套獨立實作下，XRP 4h 的方向訊號都停在接近隨機的量級，且兩階段都是 LSTM 略高於另外兩者」——這個一致性值得在報告中指出，但幅度仍不足以支撐任何交易宣稱。舊專題 `final_model_comparison_and_conclusion.md` 對 XRP 4h 的判讀也是「Accuracy 與 F1 不一致，需保守解讀」，與本階段的風險 #3 相同。

### 7.3 XRP 兩個版本的狀態（引用時必須指明版本）

| 版本 | 資料 | XGBoost | LSTM | Transformer |
| --- | --- | ---: | ---: | ---: |
| 2026-08-27 原始 | `data/features/xrp_4h.csv`（43,755 列） | 0.5164 | 0.5231 | 0.5157 |
| 2026-09-16 校正後 | `revisions/2026-09-16/xrp_4h.csv`（43,795 列） | **尚未重跑** | **尚未重跑** | **尚未重跑** |

ETH 與 SOL 的校正前後差異分別在 ±0.006 與 +0.004 以內，預期 XRP 也是同量級、不改變結論，但**在重跑完成前不得假設**。

## 8. 風險與待決事項

| # | 項目 | 說明 | 處置 |
| --- | --- | --- | --- |
| 1 | **校正後全歷史比較尚未執行** | 特徵檔已備妥，但 `logs/` 下只有原始版與外部特徵 A/B，沒有 `xrp_4h_model_comparison_2026-09-16.json` | **本輪第一優先**，見第 9 節 |
| 2 | **訊號強度** | 預期 ROC-AUC 0.50–0.53；Brier 約 0.2505–0.2507，為四幣中最差 | 照實寫。不得宣稱可交易、可獲利或可風險預警 |
| 3 | **F1 極低而 Accuracy 正常** | 原始版平均 F1：LSTM **0.2052**、XGBoost 0.3683、Transformer 0.4226，但 Accuracy 為 0.5172／0.5090／0.5025 | LSTM 的 0.2052 代表門檻幾乎把所有樣本判為下跌。**必須併列 `predicted_up_rate`**，且不得只以 Accuracy 宣稱模型有效 |
| 4 | **LSTM 的「穩定」可能是假象** | LSTM 三 fold 為 0.5204／0.5227／0.5263，樣本標準差僅 0.0030（四幣最小），但同時 F1 最低 | 需確認這是穩定的弱排序訊號，還是模型幾乎輸出常數機率 → 查機率五分位的分佈寬度 |
| 5 | **Transformer 跨 fold 崩壞** | 0.5366／**0.4845**／0.5261，樣本標準差 0.0276；Fold 2（2025-03-27 ～ 2025-09-09）低於隨機 | 逐 fold 檢查該期間是否由單一事件行情主導（XRP 事件驅動特性） |
| 6 | **特徵無法表達事件** | 53 欄全來自 K 線，沒有訴訟／上架／清算等外生事件資訊 | 列為研究限制，不是調參可解 |
| 7 | **缺可追溯性檢視** | XRP 尚未做 ETH 等級的檢視（資料／manifest 雜湊、Git revision、套件版本、逐 fold 與基準組核對） | **本輪必辦**，見第 9 節 |
| 8 | **文件記錄的執行環境錯誤** | `docs/xrp_4h_model_comparison.md` 的重跑指令指向無 torch 的 Codex runtime | 定案時一併更正為 `.venv-transformer` |
| 9 | **資料是否充足** | 43,795 列（5 年 × 1 小時）。[external_features_ab_comparison.md](external_features_ab_comparison.md) 的附帶發現顯示：歷史截短到 33k 列，XRP 三模型變化僅 +0.0045／+0.0035／−0.0011 | **訊號薄弱不是資料量造成的**，不要提出「再抓更多年資料」當作解方 |
| 10 | **XRP 缺交易所流量資料** | 外部欄位只有 17 個；且加入外部特徵後 LSTM 退步 −0.0164，是 12 組中最大 | 跨幣種比較表必須註明；不得把 XRP 與 BTC／ETH 的外部實驗當作同等條件 |
| 11 | **交易面指標未實作** | 見 6.2 | 列為本輪待辦；若 9/30 前來不及則移入「未來工作」，不得為此延後報告 |

**明確排除在本輪之外（寫成「未來工作」）**：三段式目標（`cost_aware_target`，`src/lstm_experiment_utils.py:140`）、XRP 的 48 小時序列延伸實驗、特徵／標籤漂移深入檢驗、Stacking 融合、1h／12h／24h 建模、外部訊號再實驗。

## 9. 時程規劃

| 日期 | 工作 | 狀態 |
| --- | --- | --- |
| 已完成 | 產出校正後特徵檔 `revisions/2026-09-16/xrp_4h.csv` 與 manifest | ✅ |
| 已完成 | 外部特徵 A/B（`_base_`／`_ext_` 兩組） | ✅ |
| 2026-09-24 | **以校正後資料重跑 XRP 4h 三模型比較**（風險 #1），輸出 `logs/xrp_4h_model_comparison_2026-09-16.json` | 待辦 |
| 2026-09-25 | **可追溯性檢視**：核對三個 folds、五種基準、validation-only 校準與門檻；記錄資料與 manifest 雜湊、Git revision、套件版本 | 待辦 |
| 2026-09-25 ～ 09-26 | 逐 fold 診斷：`predicted_up_rate`、機率五分位單調性、LSTM 是否近似常數輸出（風險 #3、#4）、Transformer Fold 2 是否由單一事件主導（風險 #5） | 待辦 |
| 2026-09-26 ～ 09-27 | 更新 [xrp_4h_model_comparison.md](xrp_4h_model_comparison.md)：補入校正後數字、正確執行環境、定案結論 | 待辦 |
| 2026-09-27 | 交易面指標接線（6.2） | 待辦（可延後） |
| 2026-09-28 ～ 09-30 | 併入四幣種結果整理與報告撰寫 | 待辦 |

> **時程風險**：XRP 是最後一個幣種，排定視窗只有 4 天，且 ETH 與 SOL 都有補追工作可能佔用同一段時間。重跑一次三模型比較在 CPU 上約需數分鐘，不是瓶頸；瓶頸在逐 fold 診斷與文件撰寫。若 6.2 的交易面指標無法在 09-27 前完成，**優先保住 9/30 的報告交付**，把該項移入「未來工作」，而不是壓縮第 9 節的可追溯性檢視與逐 fold 診斷。

## 10. 產出物清單

| 產出 | 路徑 | 狀態 |
| --- | --- | --- |
| 特徵資料（校正後） | `data/features/revisions/2026-09-16/xrp_4h.csv` | ✅ |
| 特徵 manifest | `logs/sol_xrp_4h_feature_manifest_2026-09-16.json` | ✅ |
| 比較結果（原始版，保留） | `logs/xrp_4h_model_comparison.json` | ✅ |
| 外部特徵 A/B | `logs/xrp_4h_model_comparison_{base,ext}_2026-09-16.json` | ✅ |
| 比較結果（校正後） | `logs/xrp_4h_model_comparison_2026-09-16.json` | **待產出**（風險 #1） |
| 定案文件 | [xrp_4h_model_comparison.md](xrp_4h_model_comparison.md) | 待補（校正後數字、可追溯性檢視、定案結論） |
| 跨幣種彙整 | [multi_asset_4h_model_comparison.md](multi_asset_4h_model_comparison.md) | 待更新為校正後數字 |
| 交易面指標 | 併入上述比較 JSON | 待辦（6.2） |

## 參考文件

- [agement.md](../agement.md)：研究範圍權威文件
- [時程.md](../時程.md)：階段排程
- [整合說明_舊專題參考.md](整合說明_舊專題參考.md)：舊專題（3 年）與新專題（5 年）的關係
- [feature_engineering.md](feature_engineering.md)：特徵與標籤定義
- [multi_horizon_experiment_guide.md](multi_horizon_experiment_guide.md)：舊專題逐幣種逐 horizon 參數設定
- [xgboost_vs_lstm_comparison.md](xgboost_vs_lstm_comparison.md)：交易面指標的呈現格式
- [final_model_comparison_and_conclusion.md](final_model_comparison_and_conclusion.md)：舊專題 3 年資料總結論（分類優於回歸）
- [external_features_ab_comparison.md](external_features_ab_comparison.md)：外部訊號的否定結果
- [eth_4h_training_plan.md](eth_4h_training_plan.md)・[sol_4h_training_plan.md](sol_4h_training_plan.md)：前兩階段同格式計劃書
