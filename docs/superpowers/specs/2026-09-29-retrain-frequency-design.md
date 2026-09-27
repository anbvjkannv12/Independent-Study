# 重新訓練頻率 vs 固定模型：預先登記設計

> 狀態：**定案**。在任何重新訓練結果產生之前 commit。若須更改判準，另開 v2 文件，勿修改本文件。
> 來源： 工作 3。研究問題：在相同 4h 特徵、評估期和預設 XGBoost 下，頻繁重訓能否穩定改善固定模型的 test AUC？不回答交易獲利或其他視窗問題。

### 3.3 固定不變的條件

| 項目 | 設定 |
|---|---|
| 資料 | `data/features/<幣種>_4h.csv`，四幣種各自獨立，不跨幣種對齊 |
| 特徵 | `logs/feature_manifest.json` 的 53 個特徵，用 `load_baseline_dataset` 載入 |
| 標籤 | `target_up`（4 小時後收盤價是否高於現在） |
| 模型 | `fit_predict_xgboost` 與預設 `ModelSettings`（200 棵樹、深度 4、學習率 0.05、subsample 0.8、colsample 0.8），只改 `seed` |
| Seeds | 42、43、44 |
| 訓練窗 | expanding：每次重新訓練都用資料開頭到截止點的全部列 |
| Purge | 24 列：在第 s 列開始預測時，訓練資料為第 0 列到第 s − 25 列（`dataset.iloc[:s-24]`）。4h 標籤只需要往後 4 小時，24 列足夠 |
| 校準與門檻 | **不做**。主要指標是未校準 AUC，校準與門檻不影響 AUC，也就不需要 validation 區段 |
| 主要指標 | 評估期間 12,000 列上的未校準 ROC-AUC，三 seed 各算一次後取平均 |
| 運算 | 全部 CPU，XGBoost `n_jobs=1`、`tree_method="hist"`；可用多個 process 平行跑不同幣種 |

### 3.4 評估期間

每個幣種取 4h 特徵檔**最後 12,000 列**作為評估期間，切成 3 段、每段 4,000 列（段 0、1、2）。段只用於逐段檢查和 bootstrap 分層，不代表重新訓練的時間點。

| 幣種 | 總列數 | 評估期第一列索引 | 段 0 開始 | 段 1 開始 | 段 2 開始 | 最後一列 | 固定模型訓練列數 |
|---|---:|---:|---|---|---|---|---:|
| BTC | 43,755 | 31,755 | 2025-03-27 07:00 | 2025-09-09 23:00 | 2026-02-23 15:00 | 2026-08-09 06:00 | 31,731 |
| ETH | 43,756 | 31,756 | 2025-03-27 07:00 | 2025-09-09 23:00 | 2026-02-23 15:00 | 2026-08-09 06:00 | 31,732 |
| SOL | 43,755 | 31,755 | 2025-03-27 06:00 | 2025-09-09 22:00 | 2026-02-23 14:00 | 2026-08-09 06:00 | 31,731 |
| XRP | 43,755 | 31,755 | 2025-03-27 07:00 | 2025-09-09 23:00 | 2026-02-23 15:00 | 2026-08-09 06:00 | 31,731 |

時間皆為 UTC。SOL 早 1 小時，是因為 SOL 在 `2026-02-25` 有一列被刪除。

### 3.5 實驗組別

令 E = 評估期第一列索引、N = 總列數。每一組都要對 E 到 N − 1 的 12,000 列各產生一個預測，每列恰好一次。

| 組別 | 重新訓練時間點（列索引） | 每個 seed 的訓練次數 | 代表的實務做法 |
|---|---|---:|---|
| F：固定 | 只在 E 訓練一次，預測全部 12,000 列 | 1 | 訓練一次後不再更新 |
| R4000 | E、E+4,000、E+8,000 | 3 | 目前 walk-forward 的做法，約每 167 天 |
| R1000 | E、E+1,000、…、E+11,000 | 12 | 約每 42 天 |
| R168 | E、E+168、…、E+71×168（最後一次只預測剩下的 72 列） | 72 | 每 7 天 |

每次重新訓練時，模型預測從該時間點到下一個時間點前一列（最後一次到 N − 1）。

**總訓練次數**：每幣種每 seed 1 + 3 + 12 + 72 = 88 次；× 3 seeds × 4 幣種 = 1,056 次。以每次 4～6 秒估計，單一 process 約 70～105 分鐘，四個幣種各開一個 process 約 20～30 分鐘。實際時間寫進結果 JSON。

**必須成立的一致性**：F 和 R4000 在段 0 使用完全相同的訓練資料與 seed，預測必須逐列完全相同；R1000、R168 的前 1,000 列、前 168 列也分別和 F 相同。程式要檢查這些一致性，不一致就報錯停止。這項檢查可以確認訓練是可重現的。

### 3.6 判定規則（事先定案）

**比較**：每個幣種內，R4000、R1000、R168 各和 F 比較，共 3 個比較。

- ΔAUC = 該組三 seed 平均 AUC − F 三 seed 平均 AUC（12,000 列合併計算）。
- 逐段 ΔAUC：同樣定義，只算單一段的 4,000 列。
- ε（seed 雜訊門檻）：同一組內三個 seed 兩兩相減的 |AUC 差| 取最大值；比較時取 max(ε_該組, ε_F)。
- Bootstrap：moving block，block 長度 48 列，在每一段內各自重抽，2,000 次，seed 20260929。直接呼叫 `run_feature_ablation.moving_block_bootstrap_deltas(a, b, 48, 2000, 20260929, score_columns=("test_raw_seed42","test_raw_seed43","test_raw_seed44"))`，其中 `fold` 欄位填段編號 0、1、2。單尾 p = (1 + #{重抽 ΔAUC ≤ 0}) / 2,001。
- 每個幣種內 3 個比較做 Holm 校正（`run_feature_ablation.holm_adjust`），α = 0.05。

**判定為「該組比固定模型穩定地好」須同時符合：**

1. **逐段都為正**：預測和 F 不同的每一段，逐段 ΔAUC 都 > 0。R4000 看段 1、2（段 0 和 F 相同）；R1000、R168 看段 0、1、2。
2. **顯著**：Holm 校正後 p ≤ 0.05。
3. **大於 seed 雜訊**：合併 ΔAUC > ε。

不符合時寫「沒有證據顯示該重新訓練頻率比固定模型好」，不可寫成「重新訓練沒用」。ΔAUC < 0 且未校正 95% CI 上界 < 0 時，另外標記「可能弱於固定模型」，只作描述。

**停止條件與下一步：**

| 結果 | 下一步 |
|---|---|
| 某一組在 ≥ 2 個幣種通過 | 採用該重新訓練頻率作為之後所有實驗的標準。若有多組都在 ≥ 2 個幣種通過，選**訓練次數最少**的一組（順序 R4000 < R1000 < R168），因為成本較低、效果已足夠 |
| 某一組只在 1 個幣種通過 | 列為候選假說，不改變標準；和 1h 候選一起等新資料獨立驗證 |
| 沒有任何組通過 | 結論為「沒有證據顯示重新訓練頻率能改善 4h 方向訊號」。不為了救結果去改訓練窗、調參或換特徵。進入總報告第七節第 4 項（三段式目標）的預先登記 |

### 3.7 描述性檢查（不參與判定）

1. **衰退曲線**：F 組依「距離訓練結束的列數」把 12,000 列切成 12 個 1,000 列的區間，算每個區間的三 seed 平均 AUC，列成表格。用來直接觀察衰退：如果 AUC 隨區間明顯下降，表示衰退存在；如果沒有趨勢，表示 validation 與 test 的差距可能來自其他原因（例如 validation 期間剛好比較好預測）。
2. 各組逐段 AUC、各段 test 上漲率。
3. 各組每次重新訓練的訓練列數（第一次與最後一次）。

### 3.8 程式修改

1. **新增 `scripts/run_retrain_frequency_comparison.py`**：
   - 直接 import 使用：`load_baseline_dataset`、`fit_predict_xgboost`、`ModelSettings`、`moving_block_bootstrap_deltas`、`holm_adjust`、`write_report_atomically`、`_write_csv_atomically`，不重寫。
   - `fit_predict_xgboost` 需要 validation 參數；因為不做校準，把預測區段同時傳給 validation 與 test，取第二個回傳值。不修改 `btc_4h_model_adapters.py`。
   - 函式 `retrain_schedule(eval_start: int, eval_end: int, every: int | None) -> list[tuple[int, int]]`：回傳每次重新訓練的（預測起點、預測終點）列索引。`every=None` 代表 F。
   - CLI 參數：`--symbols`（預設四幣種）、`--purge-rows`（預設 24，小於 24 就報錯）、`--eval-rows`（預設 12,000）、`--output-dir`（預設 `logs/retrain_frequency`）。
   - 輸出 JSON 記錄：git revision、輸入 CSV 的 SHA-256、Python 與 xgboost 版本、每組的重新訓練時間點與訓練列數、實際執行時間。
2. **新增 `tests/test_retrain_frequency_comparison.py`**：
   - `retrain_schedule`：四組的預測區間首尾相接、沒有重疊、完全涵蓋評估期間；R168 最後一段是 72 列。
   - Purge：每次訓練資料的最後一列索引 = 預測起點 − 25。
   - 用 300 列合成資料、評估最後 120 列，比較 `every=None`（F）與 `every=40`，確認前 40 列預測逐列相同。

**輸出：**

| 內容 | 位置 |
|---|---|
| 逐列預測 | `logs/retrain_frequency/<幣種>/<組別>_seed<42、43、44>_predictions.csv`（欄位：`segment`、`open_time`、`target_up`、`test_raw`、`train_rows`） |
| 各幣種結果 | `logs/retrain_frequency/<幣種>/retrain_frequency_comparison.json` |
| 四幣種彙整 | `logs/retrain_frequency/multi_asset_retrain_frequency.json` |
| 報告 | `docs/retrain_frequency_results.md` |

### 3.9 限制（寫進預先登記與報告）

- 評估期間（2025-03-27 至 2026-08-09）和多視窗實驗 fold 1、fold 2 的 test 期間、以及先前 4h 實驗的 test 期間重疊，不是獨立確認。
- 只測 expanding 訓練窗。sliding 訓練窗（只用最近 N 列）是另一個問題，不在本次範圍。
- 只測預設 XGBoost；不做 Logistic、LSTM、Transformer。
- AUC 不代表可交易，未扣交易成本。
