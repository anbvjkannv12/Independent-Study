# ETHUSDT 4 小時方向分類 訓練模型計劃書

> 建立日期：2026-09-20　｜　研究範圍權威文件：[agement.md](../agement.md)　｜　排程：[時程.md](../時程.md)
>
> 本計劃書只規劃 **4 小時漲跌方向分類**。不規劃回歸任務，也不規劃 1h／12h／24h 建模。

## 1. 目標與範圍

| 項目 | 內容 |
| --- | --- |
| 標的 | `ETHUSDT`（Binance Spot） |
| 任務 | 二元分類：`target_up = future_log_return > 0`，預測視窗 **4 小時** |
| 模型 | XGBoost、LSTM、Transformer 三者比較 |
| 定案標準 | 三個 test folds 的平均與逐 fold test ROC-AUC 一致性；**允許的結論包含「無可靠優勝模型」** |
| 排定期間 | 2026-09-11 ～ 09-17（[時程.md](../時程.md) 階段 5） |
| 實際狀態 | **已逾期**（今日 2026-09-20）。模型比較與可追溯性檢視均已完成，剩餘工作為第 6.2 節的交易面指標補件 |

**本輪不做的事**：不重新設計特徵、不做超參數搜尋、不做 Stacking 融合（`agement.md` 已說明「表現穩定」的前提條件尚未達成）、不啟動三段式目標與外部特徵的新實驗。

**預期結果**：ROC-AUC 落在 **0.50–0.53** 區間。這是依 BTC 與既有 ETH 結果寫下的**預期值，不是要達成的目標**；本計劃書不設定「把 AUC 拉高到某個數字」的績效承諾。

## 2. 資料來源與期間

| 項目 | 內容 |
| --- | --- |
| 原始資料 | Binance Spot `ETHUSDT` 1 小時 K 線，`data/raw/eth_5y.csv`，43,818 列 |
| 補值後資料 | `data/processed/eth_5y.csv`，43,825 列，其中 `is_imputed=1` 共 **7 列**（佔 0.016%） |
| 特徵資料（原始版） | `data/features/eth_4h.csv`，**43,756 列** |
| 特徵資料（校正後） | `data/features/revisions/2026-09-10/eth_4h.csv`，**43,796 列**、7 個補值列、無時間跳點 |
| 時間範圍 | 2021-08-10 11:00 UTC ～ 2026-08-09 06:00 UTC（全部以 UTC 表示） |
| 來源記錄 | `logs/fetch_manifest.json`、`logs/data_quality_report.json`、`logs/imputation_manifest.json`、`logs/feature_manifest.json` |

**缺口與補值**：ETH 原始資料有 7 個缺失小時（2021-08-13、2021-09-29、2023-03-24 三段）。補值規則見 `agement.md`：OHLC 全部取前一根的 `close`（視為平盤），成交量類欄位補 `0`，並以 `is_imputed` 標記。`is_imputed` **不是模型特徵**，只留作稽核欄。

**已修正的資料缺陷（ETH 專屬，必須寫進報告）**：校正前的 `data/features/eth_4h.csv` 因補值列成交量為 0，使 `trade_intensity` 與 `taker_buy_ratio` 變成非有限值而在 `dropna()` 被移除，造成 **11 個時間跳點、合計 40 小時**。`scripts/feature_engineering.py` 已修正為「零成交量補值列的這兩個比率固定為 0」，並加上回歸測試。因此引用 ETH 數字時**必須指明版本**：校正前（43,756 列）或校正後（43,796 列）。

## 3. 特徵清單摘要（引用 `logs/feature_manifest.json`）

固定使用 manifest 記錄的 **53 個特徵**，順序不變。這 53 欄的定義見 `scripts/feature_engineering.py` 的 `add_features()`，**本計劃書不新增、不移除、不重新發明任何特徵**。

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

**外部訊號（鏈上／情緒／衍生品）不納入本輪。** [external_features_ab_comparison.md](external_features_ab_comparison.md)（2026-09-16）已在相同資料列上做過受控 A/B：ETH 加入 20 個外部欄位後，12 組比較的差異平均為 −0.0017；ETH XGBoost 的 +0.0113 是 12 次比較中的兩個離群值之一（另一個方向相反），不構成效果證據。

## 4. 模型架構與超參數規劃

三個模型的實作在 `scripts/btc_4h_model_adapters.py`（檔名雖為 btc，實際為四幣種共用）。**本輪沿用既有設定，不做超參數搜尋**——在訊號僅 0.50–0.53 的情況下，調參的改善很難與雜訊區分，且會消耗 9/30 前的時程。

共用執行設定（`ModelSettings`）：`seed=42`、`sequence_length=24`、`batch_size=128`、`max_epochs=30`、`patience=5`、`learning_rate=0.001`、`hidden_size=32`、`num_heads=4`。

### 4.1 XGBoost

| 參數 | 本輪 5 年資料設定 | 說明 |
| --- | --- | --- |
| `n_estimators` | 200 | 5 年資料的 walk-forward train 段最大到 35,795 列，早期驗證顯示 200 棵已收斂 |
| `max_depth` | 4 | 特徵數固定 53，稍深一層換取交互作用表達力 |
| `learning_rate` | 0.05 | 配合較少的樹數 |
| `subsample` | 0.8 | 降低過擬合 |
| `colsample_bytree` | 0.8 | — |
| `objective` ／ `eval_metric` | `binary:logistic` ／ `logloss` | — |
| `tree_method` | `hist` | 為 CPU 執行速度 |

### 4.2 LSTM

| 參數 | 本輪 5 年資料設定 | 說明 |
| --- | --- | --- |
| 框架 | **PyTorch** | 新專題統一改用 PyTorch（torch 2.14.0+cu126） |
| `sequence_length` | 24 | — |
| 隱藏單元 | `hidden_size=32` | 53 特徵 × 24 步的樣本下 64 單元容易過擬合 |
| 網路結構 | `nn.LSTM(53→32)` → `Linear(32,1)` | 簡化為單層，避免在弱訊號上疊加不可解釋的容量 |
| Loss／Optimizer | `BCEWithLogitsLoss`／Adam(1e-3) | — |
| Epochs | `max_epochs=30` + `patience=5` early stopping | 改由 validation loss 決定停點 |
| `batch_size` | 128 | — |

### 4.3 Transformer

| 參數 | 本輪 5 年資料設定 | 說明 |
| --- | --- | --- |
| 框架 | **PyTorch** | — |
| `sequence_length` | 24 | — |
| `d_model` | 32 | 與 LSTM 同步縮小容量 |
| `ff_dim` | 64（`dim_feedforward = hidden_size * 2`） | 維持 2× 比例 |
| `num_heads` | 4（key dim = 32 / 4 = 8） | — |
| Encoder 層數 | **1** | 簡化 |
| 位置編碼 | 學習式 `nn.Parameter(1, 24, 32)` | — |
| `dropout` | `TransformerEncoderLayer` 預設 | 未特別調整 |
| 輸出 | 取**最後一個 timestep** → `Linear(32,1)` | 與 LSTM 取最後隱藏狀態一致 |

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

五年專題隨附的 Codex runtime **沒有 `torch`**，必須使用 `.venv-transformer`（Python 3.12.14、Torch 2.14.0+cu126、Pandas 2.3.3、NumPy 2.5.0、XGBoost 3.2.0、scikit-learn 1.9.0）。

```powershell
& ".venv-transformer\Scripts\python.exe" scripts\train_btc_4h_model_comparison.py --symbol ETH --data data\features\revisions\2026-09-10\eth_4h.csv --manifest logs\eth_4h_feature_manifest_2026-09-10.json --output logs\eth_4h_model_comparison_2026-09-10.json --initial-train-rows 19795 --validation-rows 4000 --test-rows 4000 --fold-count 3
```

每次執行需記錄：資料與 manifest 的 SHA-256、Git revision、套件版本。ETH 校正後資料的 SHA-256 為 `D9FE91EB2ADD309F7EBE49E26B6460FDA429F1CEADA85E450BF192C01B730CBE`，manifest 為 `2FF2BA910EF87403A47652EEDA20D5625424B8FA0848C290D3566CF528E054BF`。

## 6. 評估指標

### 6.1 主要指標（決定定案的依據）

| 指標 | 用途 | 判讀原則 |
| --- | --- | --- |
| **ROC-AUC**（test） | 主要選模指標，衡量排序能力，不受門檻影響 | 0.50 = 隨機。逐 fold 都要看，不能只看平均 |
| **樣本標準差**（3 folds） | 穩定性 | 平均差距小於 fold 標準差時，不得宣稱某模型較優 |
| **Brier score** | 機率校準品質 | 二元目標下約 0.25 即為弱辨識 |
| Accuracy／Balanced accuracy | 輔助 | 類別比例接近 50／50，Accuracy 單獨無意義 |
| Precision／Recall／F1 | 輔助 | F1 隨門檻大幅變動，須與 `predicted_up_rate` 一起看 |
| 機率五分位實際上漲率 | 診斷 | 檢查是否單調；不單調代表排序訊號不可靠 |

### 6.2 次要指標（Sharpe、最大回撤）— 需補程式，列為待辦

交易面指標（幣種 × 模型 × Accuracy／F1／Log loss／策略報酬／Sharpe／最大回撤）**尚未實作**。

- 資料已具備：`data/features/eth_4h.csv` 已含 `future_log_return` 欄。
- 待辦：把 `strategy_metrics()` 接進 `scripts/btc_4h_evaluation.py` 的 fold 報告，輸出到既有的比較 JSON。

**使用限制（必須寫進報告）**：

1. **未納入手續費與滑價**，不得解讀為可交易績效。
2. 目前訊號在 ROC-AUC 0.50–0.53，**交易面指標不得作為選模依據**——在近乎隨機的排序上，Sharpe 主要反映 test 期的市場走向與雜訊，而非模型能力。
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

## 7. 比較基準

### 7.1 對照 BTC（已定案）

BTC 已於 2026-08-19 選定 **Transformer** 為定案模型，**不再重新訓練**。ETH 使用完全相同的 fold 日期、特徵契約與評估流程，因此可直接並列。

| 幣種 | XGBoost | LSTM | Transformer |
| --- | ---: | ---: | ---: |
| BTCUSDT（定案基準） | 0.5289 | 0.5302 | **0.5318** |
| ETHUSDT（校正後） | 0.5221 | 0.5232 | 0.5257 |

> 來源：`logs/btc_4h_model_comparison.json`、`logs/eth_4h_model_comparison_2026-09-10.json`

### 7.3 ETH 三個版本的數字對照（引用時必須指明版本）

| 版本 | 資料 | XGBoost | LSTM | Transformer |
| --- | --- | ---: | ---: | ---: |
| 2026-08-27 原始 | `data/features/eth_4h.csv`（43,756 列） | 0.5275 | 0.5226 | 0.5261 |
| 2026-09-10 校正後 | `revisions/2026-09-10/eth_4h.csv`（43,796 列） | 0.5221 | 0.5232 | 0.5257 |
| 2026-09-16 重跑 | 同上 | 0.5221 | 0.5232 | 0.5257 |

2026-09-16 重跑與 09-10 **逐 fold 完全一致**，確認流程可重現（`seed=42` 生效）。README 目前引用的是 2026-08-27 原始版本，報告定稿時需統一為校正後版本。

## 8. 風險與待決事項

| # | 項目 | 說明 | 處置 |
| --- | --- | --- | --- |
| 1 | **訊號強度** | 預期 ROC-AUC 0.50–0.53，Brier 約 0.2495–0.2499 | 照實寫。不得宣稱可交易、可獲利或可風險預警 |
| 2 | **Transformer 跨 fold 不穩** | 平均 0.5257 最高，但 Fold 3 掉到 **0.5063**，樣本標準差 0.0173 為三者中最大 | ETH 的定案結論寫為「**無可靠優勝模型**」，不選一個可供交易的模型 |
| 3 | **F1 與 Accuracy 不一致** | 校正後 Transformer 平均 F1 僅 0.3882、LSTM 0.4157，遠低於 Accuracy 約 0.513，代表門檻把大量樣本判為下跌 | 報表必須同時列出 `predicted_up_rate`，避免單看 Accuracy 誤判 |
| 4 | **資料版本分歧** | 原始版與校正版兩套數字並存，README 尚未更新 | 報告定稿前統一改用校正後版本，並保留原始版供可追溯 |
| 5 | **資料是否充足** | 43,796 列（5 年 × 1 小時）看似充足，但 [external_features_ab_comparison.md](external_features_ab_comparison.md) 的附帶發現顯示：把歷史截短到 33k 列，ETH 三個模型的 AUC 反而各升 +0.0031／+0.0049／+0.0065 | **訊號薄弱不是資料量造成的**，不要提出「再抓更多年資料」當作解方 |
| 6 | **Logistic Regression 咬得很緊** | 在外部特徵執行中，LR 基準達 0.5305，與三個模型相當 | 報告須說明：三個模型的額外複雜度沒有換到穩定回報 |
| 7 | **交易面指標未實作** | 見 6.2 | 列為本輪待辦；若 9/30 前來不及則移入「未來工作」，不得為此延後報告 |
| 8 | **執行環境仍借用外部 venv** | 需 `.venv-transformer` 才有 torch | 已固定；環境快照（套件版本）隨結果記錄 |

**明確排除在本輪之外（寫成「未來工作」）**：三段式目標、特徵／標籤漂移深入檢驗、Stacking 融合、1h／12h／24h 建模、外部訊號再實驗。

## 9. 時程規劃

| 日期 | 工作 | 狀態 |
| --- | --- | --- |
| 2026-09-11 ～ 09-17 | ETH 三模型結果確認與定案（排定視窗） | **已逾期**，實質工作已完成 |
| 已完成 | 校正零成交量補值列、重跑 ETH 4h、保留舊版產物 | ✅ |
| 已完成 | 記錄資料／manifest 雜湊、Git revision、套件版本 | ✅ |
| 已完成 | 核對三個 folds、五種基準、validation-only 校準與門檻 | ✅ |
| 已完成 | 依一致性、平均 ROC-AUC 與 Brier 確認無穩定優勢，定案為「無可靠優勝模型」 | ✅ |
| 2026-09-20 ～ 09-23 | **補追**：交易面指標接線（6.2），與 SOL 階段並行 | 待辦 |
| 2026-09-24 ～ 09-27 | 統一 README 與各文件的引用版本為校正後數字 | 待辦 |
| 2026-09-28 ～ 09-30 | 併入四幣種結果整理與報告撰寫 | 待辦 |

> **時程風險**：ETH 已逾期，須與 SOL（09-18 ～ 09-23）並行補追。若 6.2 的交易面指標無法在 09-23 前完成，**優先保住 9/30 的報告交付**，把該項移入「未來工作」，而不是壓縮驗證步驟。

## 10. 產出物清單

| 產出 | 路徑 | 狀態 |
| --- | --- | --- |
| 特徵資料（校正後） | `data/features/revisions/2026-09-10/eth_4h.csv` | ✅ |
| 特徵 manifest | `logs/eth_4h_feature_manifest_2026-09-10.json` | ✅ |
| 比較結果（校正後） | `logs/eth_4h_model_comparison_2026-09-10.json` | ✅ |
| 比較結果（重跑驗證） | `logs/eth_4h_model_comparison_2026-09-16_rerun.json` | ✅ |
| 比較結果（原始版，保留） | `logs/eth_4h_model_comparison.json` | ✅ |
| 外部特徵 A/B | `logs/eth_4h_model_comparison_{base,ext}_2026-09-16.json` | ✅ |
| 定案文件 | [eth_4h_model_comparison.md](eth_4h_model_comparison.md) | ✅ |
| 跨幣種彙整 | [multi_asset_4h_model_comparison.md](multi_asset_4h_model_comparison.md) | 待更新為校正後數字 |
| 交易面指標 | 併入上述比較 JSON | 待辦（6.2） |

## 參考文件

- [agement.md](../agement.md)：研究範圍權威文件
- [時程.md](../時程.md)：階段排程
- [feature_engineering.md](feature_engineering.md)：特徵與標籤定義
- [external_features_ab_comparison.md](external_features_ab_comparison.md)：外部訊號的否定結果
