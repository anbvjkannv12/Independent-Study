# 下一階段工作計畫：三段式目標與前向測試

> 撰寫日期：2026-09-28  
> 前置文件：[`retrain_frequency_conclusions.md`](retrain_frequency_conclusions.md)、[`multi_horizon_conclusions.md`](multi_horizon_conclusions.md)、[`five_year_4h_final_report.md`](five_year_4h_final_report.md) 第七節  
> 本文件列出下一階段每項工作的目的、輸入、步驟、通過條件與失敗時的處理。工作 A 是研究主線；工作 B 是工程支線，可和 A 同時進行；工作 C 要等 2027-01-25 之後才能開始。

## 總覽

| 編號 | 工作 | 性質 | 前置條件 | 預估時間 | 主要產出 |
|---|---|---|---|---|---|
| A | 三段式目標（上漲／下跌／不交易）vs 二分類加信心門檻 | 研究主線 | 無（重訓頻率實驗已依停止條件指向此項） | 預先登記半天、實作 1～1.5 天、執行約 20 分鐘、報告半天 | 預先登記文件、`scripts/run_three_class_target_comparison.py`、結果與結論文件 |
| B | 持續更新資料與前向測試流程 | 工程支線 | 無 | 2～3 天 | `scripts/live_*.py`、`data/live/`、`models/live/`、`logs/live/predictions.csv`、Windows 排程 |
| C | 新資料獨立驗證（1h 候選、BTC G4、工作 A 的候選） | 研究確認 | 2027-01-25 之後 | 另開文件 | 獨立驗證預先登記與報告 |

**執行原則（沿用前幾個實驗）**

1. 先寫預先登記文件並 commit，commit 之前不可執行任何會產生結果的程式。
2. 定案後若要修改，另開 v2 文件並寫明原因；不可為了救結果去改規則。
3. 所有會跑出數字的程式都要有單元測試，全部通過後才執行。
4. 結果寫成「沒有證據顯示……」或「通過預定判準」，不寫成「沒用」或「有效」。
5. 2027-01-25 之前，不用 2026-08-09 之後的新資料做任何調參、選特徵或改規則。

---

## 已確認的資料與程式事實（2026-09-28 查核）

以下事實會影響工作 A 與 B 的設計，先記錄下來：

1. **4h 特徵檔列數**：BTC 43,755、ETH 43,756、SOL 43,755、XRP 43,755；每列是一根 1 小時 K 線，最後一列都是 `2026-08-09 06:00 UTC`。
2. **切分沿用多視窗實驗的 4h 版面**：`initial = 列數 − 3 × (4000 + 4000 + 2 × 24)`，也就是 BTC／SOL／XRP 為 19,611、ETH 為 19,612。本實驗只用 4h 一個視窗，**不做跨視窗對齊**，所以列索引和多視窗實驗（有對齊）可能差幾列，但時段幾乎相同：

   | 幣種 | fold | train 最後一列 | validation | test |
   |---|---|---|---|---|
   | BTC | 0 | 2023-11-07 05:00 | 2023-11-08 06:00 ～ 2024-04-22 21:00 | 2024-04-23 22:00 ～ 2024-10-07 13:00 |
   | BTC | 1 | 2024-10-07 13:00 | 2024-10-08 14:00 ～ 2025-03-24 06:00 | 2025-03-25 07:00 ～ 2025-09-07 22:00 |
   | BTC | 2 | 2025-09-07 22:00 | 2025-09-08 23:00 ～ 2026-02-22 14:00 | 2026-02-23 15:00 ～ 2026-08-09 06:00 |

   ETH、XRP 的時間晚 1 小時或相同；SOL 在 fold 1、2 早 1 小時（SOL 在 2026-02-25 有一列被刪除）。實際值由程式寫進 JSON。

3. **不交易區間的比例**：用 fold 0 的 train 區段（只看訓練資料，不看 test）計算 `|future_log_return| < k` 的比例：

   | 幣種 | 4h 報酬絕對值中位數 | k = 0.001 | **k = 0.002** | k = 0.003 | k = 0.004 |
   |---|---:|---:|---:|---:|---:|
   | BTC | 0.0043 | 14.7% | **27.9%** | 38.7% | 47.8% |
   | ETH | 0.0056 | 11.7% | **22.5%** | 31.7% | 39.8% |
   | SOL | 0.0103 | 5.8% | **11.5%** | 17.2% | 22.4% |
   | XRP | 0.0070 | 8.7% | **16.6%** | 24.7% | 31.7% |

4. **Binance 現貨手續費**：一般帳戶 maker／taker 皆為 0.1%，一買一賣來回約 0.2%，換成對數報酬約 0.002。
5. **現有 `fit_predict_xgboost`**（`scripts/btc_4h_model_adapters.py`）寫死 `objective="binary:logistic"`，只能做二分類；三分類要另寫，**不修改**這個檔案。
6. **現有 `moving_block_bootstrap_deltas`**（`scripts/run_feature_ablation.py`）只能算 AUC 差，不能直接用在「每筆交易平均報酬」這種指標；要照同一套區塊抽樣方式另寫一個小函式。`holm_adjust` 可以直接沿用。
7. **現有 `persistence_scores`**（`scripts/run_multi_horizon_comparison.py`）回傳的是 0／1，沒有幅度，不能拿來依信心排序；工作 A 的 persistence 基準要直接用 `log_return.rolling(4).sum()` 的連續值。
8. **`feature_engineering.py` 的 `build_feature_dataset` 最後會 `dropna()`**，會把最後 h 列（標籤還不知道的列）一起刪掉。前向測試需要預測「最新、還沒有答案」的那一列，所以工作 B 要改用 `add_features` 的輸出，只刪除**特徵**缺值的列，保留標籤缺值的列。
9. **`fetch_binance_5y.py` 每次都重新下載 5 年**（`resolve_time_window` 以現在往回推 5 年），沒有增量模式；但 `fetch_klines(symbol, interval, start_ms, end_ms, request_json_fn, ...)` 可以直接指定起訖時間，工作 B 可以重用。
10. **`.gitignore`** 已排除 `data/` 與 `logs/`，但**沒有排除 `models/`**；工作 B 新增的 `models/live/` 要加進 `.gitignore`。

以下所有指令都使用 main 的 Python 環境：

```powershell
$py = "C:\Users\user\專題\.venv-transformer\Scripts\python.exe"
```

---

## 工作 A：三段式目標 vs 二分類加信心門檻

### A.1 研究問題

多視窗、重訓頻率兩個實驗都顯示：4h 方向訊號存在但很弱（AUC 約 0.51～0.53），而且小幅波動根本不夠付手續費。與其每一列都硬猜漲跌，不如讓模型「只在有把握、而且預期波動夠大時才出手」。

> 在相同資料、特徵、切分與 XGBoost 設定下，**直接用三段式標籤（上漲／下跌／不交易）訓練的模型**，挑出來的交易是否比**二分類模型加信心門檻**挑出來的交易，扣除手續費後平均報酬更高？

**為什麼要和「二分類加信心門檻」比，而不是和「每列都交易」比**：二分類模型本來就可以只在 `p` 離 0.5 很遠時才交易。如果三段式只是贏過「每列都交易」，那只證明「少交易可以省手續費」，沒有證明三段式標籤有用。公平的比較是：兩者都只挑同樣比例的交易時點，看誰挑得比較好。

**本實驗不回答**：
- 實際交易能不能賺錢：沒有算滑價、資金管理、部位重疊。
- 哪一種模型最好：只測 XGBoost。
- 1h 等其他視窗是否適用。
- 最佳的 k 或最佳的交易比例：這兩個都事先固定，不做搜尋。

### A.2 流程

1. **從 main 建 worktree**：

   ```powershell
   git -C C:\Users\user\專題 worktree add -b three-class-target "C:\Users\user\orca\workspaces\專題\three-class-target" main
   ```

2. **預先登記**：把本節 A.3～A.10、A.15 抄寫成 `docs/superpowers/specs/<定案日期>-three-class-target-design.md`，標註「定案」後 commit。**commit 之前不可執行任何會產生結果的程式**（包括「先跑一個幣種看看」）。
3. **實作與測試**（A.11、A.12）：測試全部通過後 commit。
4. **執行實驗**（A.13），產生逐列預測、JSON 與 `docs/three_class_target_results.md`，commit。
5. **寫結論文件** `docs/three_class_target_conclusions.md`，格式同 `retrain_frequency_conclusions.md`，commit。
6. **更新總報告**：`five_year_4h_final_report.md` 第七節第 4 項標上「已完成」並附結論連結；第八節加入本實驗的文件與 logs 路徑。
7. 完整測試通過後，fast-forward 合併回 main，並把 `logs/three_class_target/` 複製到 main 的 `logs/`（不進 git）。

### A.3 固定不變的條件

| 項目 | 設定 |
|---|---|
| 資料 | `data/features/<幣種>_4h.csv`，四幣種各自獨立，不跨幣種、不跨視窗對齊 |
| 特徵 | `logs/feature_manifest.json` 的 53 個特徵，用 `load_baseline_dataset` 載入 |
| 報酬 | `future_log_return`（4 小時後的對數報酬，特徵檔內建） |
| 切分 | `build_expanding_folds(dataset, initial, 4000, 4000, 3, purge_rows=24)`，`initial = 列數 − 3 × 8048` |
| 模型超參數 | 與 `fit_predict_xgboost` 相同：200 棵樹、深度 4、學習率 0.05、subsample 0.8、colsample 0.8、`n_jobs=1`、`tree_method="hist"` |
| Seeds | 42、43、44 |
| 手續費 c | 0.002（對數報酬；來回 0.1% × 2） |
| 不交易門檻 k | 0.002（和手續費相同，意思是「漲跌幅不夠付手續費就算不交易」） |
| 主要交易比例 q | 20%（由 validation 決定門檻，見 A.6） |
| 類別權重 | **不加**。三類比例不平均是資料的真實狀態，不另外調整 |
| 運算 | 全部 CPU，四個幣種可各開一個 process 平行跑 |

**為什麼 k 用固定的 0.002，而不是依波動度縮放**：手續費是固定百分比，和波動度無關；「不夠付手續費」這個定義有明確的經濟意義，也不需要從資料估計任何參數，不會有偷看資料的疑慮。代價是各幣種的不交易比例差很多（SOL 11.5%、BTC 27.9%），這一點寫進限制。

### A.4 標籤定義

令 r = `future_log_return`：

| 類別 | 條件 | 編碼 |
|---|---|---|
| 下跌 | r < −k | 0 |
| 不交易 | −k ≤ r ≤ k | 1 |
| 上漲 | r > k | 2 |

- 欄位名稱 `target_3c`，由 `future_log_return` 在載入後計算，不寫回特徵檔。
- 邊界 r 剛好等於 ±k 時歸入「不交易」。
- 每個 fold 的 train 區段必須三類都存在，否則報錯停止。
- `target_up` 維持原定義（r > 0），給二分類組使用。

### A.5 實驗組別

| 組別 | 模型 | 分數 s（越大越偏向上漲，越小越偏向下跌） | 說明 |
|---|---|---|---|
| **T**：三段式 | XGBoost `objective="multi:softprob"`、`num_class=3`，用 `target_3c` 訓練 | s = P(上漲) − P(下跌) | 本實驗要測的做法 |
| **B**：二分類加門檻 | 現有 `fit_predict_xgboost`，用 `target_up` 訓練 | s = p − 0.5 | 比較基準 |
| **P**：persistence | 不訓練 | s = d × 過去 4 小時累積報酬（`log_return.rolling(4).sum()`） | 描述性基準，不參與判定 |

**P 的方向 d**：在每個 fold 的 **train 區段**計算 `rolling(4).sum()` 對 `target_up` 的 AUC，AUC ≥ 0.5 則 d = +1，否則 d = −1。方向只用訓練資料決定，所以不是看過 test 答案後才選的方向，和多視窗報告中的「best-direction persistence」不同。

**訓練次數**：每幣種 × 每 fold × 每 seed，T、B 各訓練 1 次 → 2 × 3 × 3 × 4 = **72 次**。P 不用訓練。

### A.6 交易規則（三組共用）

對每一組、每個 fold、每個 seed：

1. 在 validation 區段算出每列的 |s|，取第 (1 − q) 分位數作為門檻 t。q = 20% 時，就是 validation 中信心最高的 20% 才交易。
2. test 區段中 |s| ≥ t 的列視為**交易**：s > 0 做多、s < 0 做空（s = 0 不交易）。
3. 每筆交易的淨報酬 g = sign(s) × r − c。
4. 門檻 t 只由 validation 決定，test 完全不參與。因此 test 實際的交易比例可能不是剛好 20%，實際比例寫進 JSON 與報告。

### A.7 主要指標

**每筆交易的平均淨報酬 m**：該組在 12,000 列 test（三個 fold 合併）中所有交易的 g 取平均。T、B 各 seed 先算 m，再取三 seed 平均；P 沒有 seed。

- 若某個 seed 在 test 中交易次數 < 200，該 seed 的 m 記為無效，該幣種的判定直接記為「交易次數不足，無法判定」。
- m 為對數報酬，報告同時列出換算的百分比。

### A.8 判定規則（事先定案）

**關卡一：訊號關卡（T 本身扣手續費後是否 > 0）**

- 逐 fold：T 在每個 fold 的 m 都 > 0。
- 顯著：moving block bootstrap（區塊長度 48 列、在每個 fold 內各自重抽、2,000 次、seed 20261001），單尾 p = (1 + #{重抽 m ≤ 0}) / 2001；四個幣種的 p 做 Holm 校正（`holm_adjust`），校正後 p ≤ 0.05。

**關卡二：比較關卡（T 是否比 B 好）**

- Δ = m_T − m_B（三 seed 平均）。
- 逐 fold：每個 fold 的 Δ 都 > 0。
- 顯著：同上 bootstrap（同一組重抽索引同時套用在 T 與 B，保持配對），單尾 p = (1 + #{重抽 Δ ≤ 0}) / 2001；四個幣種 Holm 校正後 p ≤ 0.05。
- 大於 seed 雜訊：Δ > ε，ε = max(ε_T, ε_B)，ε_組 = 該組三個 seed 兩兩相減 |m 差| 的最大值。

**判定結果**

| 關卡一 | 關卡二 | 該幣種的判定文字 |
|---|---|---|
| 通過 | 通過 | 三段式目標比二分類加門檻穩定地好，且扣手續費後仍有正報酬 |
| 未通過 | 通過 | 三段式挑的交易比二分類好，但扣手續費後沒有證據顯示有正報酬 |
| 通過 | 未通過 | 扣手續費後有正報酬，但沒有證據顯示比二分類加門檻好 |
| 未通過 | 未通過 | 沒有證據顯示三段式目標能改善可用性 |

m_T < 0 且未校正 95% CI 上界 < 0 時，另外標記「扣手續費後可能虧損」，只作描述。

**停止條件與下一步**

| 結果 | 下一步 |
|---|---|
| ≥ 2 個幣種兩關都通過 | 三段式目標列為之後實驗的標準做法；加入工作 C 的新資料驗證清單；之後可以預先登記交易成本回測 |
| 只有 1 個幣種兩關都通過 | 列為候選假說，不改變標準；加入工作 C 的新資料驗證清單 |
| 沒有幣種兩關都通過 | 結論為「沒有證據顯示三段式目標能改善 4h 方向訊號的可用性」。**不為了救結果去改 k、q、特徵或超參數**。五年 4h 研究主線至此收尾，之後只剩工作 C 的新資料驗證與報告撰寫 |

### A.9 描述性檢查（不參與判定）

1. **交易比例曲線**：q = 5%、10%、20%、30%、50%、100% 時，T、B、P 的 m、交易次數與命中率。q = 100% 就是每列都交易。
2. **命中率**：交易中 sign(s) = sign(r) 的比例。
3. **大行情捕捉率**：交易中 |r| > c 的比例（有沒有挑到夠付手續費的波動）。
4. **T 的方向排序能力**：用 s_T 對 `target_up` 算 AUC，和 B 的 AUC 比較，看改成三分類後方向排序有沒有變差。
5. **T 的「不交易」類別**：P(不交易) 最高的列，實際 |r| ≤ k 的比例（precision）與 recall。
6. **各 fold 的三類比例**：train、validation、test 各自的下跌／不交易／上漲比例。
7. **不重疊版本**：4h 標籤在相鄰列之間會重疊，連續幾小時的交易其實押在同一段行情上。額外只取 `open_time` 小時數為 0、4、8、12、16、20 的列重算 m，看結論是否一致。
8. **validation 與 test 的實際交易比例**：確認門檻沒有在 test 期間嚴重偏移。

### A.10 未納入本實驗的變化（事先說明，避免事後加碼）

以下變化**刻意不做**，因為每多試一種，就多一次「碰巧撞到好結果」的機會：

- k 的敏感度分析（0.001、0.003）。
- 依波動度縮放的 k。
- 類別權重或 focal loss。
- 其他模型（Logistic、LSTM、Transformer）。
- 1h、12h、24h 視窗。

若本實驗通過，上述變化可以在工作 C 之後另開預先登記。

### A.11 程式修改

**新增 `scripts/run_three_class_target_comparison.py`**

直接 import 使用，不重寫：

| 函式 | 來源 |
|---|---|
| `load_baseline_dataset` | `scripts/train_btc_4h_baseline.py` |
| `build_expanding_folds` | `scripts/btc_4h_evaluation.py` |
| `fit_predict_xgboost`、`ModelSettings` | `scripts/btc_4h_model_adapters.py` |
| `holm_adjust`、`_write_csv_atomically` | `scripts/run_feature_ablation.py` |

需要新寫的函式：

| 函式 | 內容 |
|---|---|
| `three_class_labels(r: np.ndarray, k: float) -> np.ndarray` | 依 A.4 回傳 0／1／2 |
| `fit_predict_xgboost_multiclass(train, validation, test, features, seed) -> tuple[np.ndarray, np.ndarray]` | 超參數與 `fit_predict_xgboost` 相同，改成 `objective="multi:softprob"`、`num_class=3`、`eval_metric="mlogloss"`；回傳 validation、test 的 (n, 3) 機率矩陣 |
| `persistence_signal(dataset) -> np.ndarray` | `log_return.rolling(4, min_periods=4).sum()`，在切 fold 之前對整份資料計算，只用過去資料 |
| `persistence_direction(train_signal, train_target) -> int` | 依 A.5 回傳 +1 或 −1 |
| `validation_threshold(val_scores, q) -> float` | `np.quantile(np.abs(val_scores), 1 − q)` |
| `trade_returns(scores, r, t, c) -> np.ndarray` | 交易列回傳 sign(s) × r − c，不交易列回傳 NaN |
| `mean_trade_return(g) -> float` | 忽略 NaN 取平均 |
| `block_bootstrap_indices(folds, block_length, replicates, seed)` | 產生重抽的列索引，抽樣方式與 `moving_block_bootstrap_deltas` 相同（每個 fold 內抽 48 列的區塊，截到該 fold 長度）。T 與 B 共用同一組索引 |

CLI 參數：

| 參數 | 預設 | 限制 |
|---|---|---|
| `--symbols` | BTC ETH SOL XRP | |
| `--purge-rows` | 24 | < 24 報錯 |
| `--k` | 0.002 | 只允許預設值；要改必須先開 v2 預先登記（程式直接拒絕其他值） |
| `--cost` | 0.002 | 同上 |
| `--coverage` | 0.20 | 同上；A.9 的其他 q 由程式內固定清單產生 |
| `--output-dir` | `logs/three_class_target` | |

**輸出**

| 內容 | 位置 |
|---|---|
| 逐列預測（每組每 seed） | `logs/three_class_target/<幣種>/<T 或 B>_seed<42、43、44>_predictions.csv`；欄位：`fold`、`open_time`、`future_log_return`、`target_up`、`target_3c`、`score`、`p_down`、`p_neutral`、`p_up`（B 組只有 `p_up`）、`threshold`、`traded`、`net_return` |
| P 組逐列 | `logs/three_class_target/<幣種>/P_predictions.csv` |
| 各幣種結果 | `logs/three_class_target/<幣種>/three_class_target_comparison.json` |
| 四幣種彙整 | `logs/three_class_target/multi_asset_three_class_target.json` |
| 報告 | `docs/three_class_target_results.md` |

JSON 必須記錄：git revision、輸入 CSV 的 SHA-256、Python 與 xgboost 版本、每個 fold 的時間範圍與列數、三類比例、每組每 seed 的門檻 t、validation 與 test 實際交易比例、交易次數、m、逐 fold m、Δ、ε、bootstrap 原始與校正後 p 值、95% CI、判定結果、實際執行秒數。

### A.12 單元測試：新增 `tests/test_three_class_target_comparison.py`

| 測試 | 驗證內容 |
|---|---|
| 標籤邊界 | r = −0.003、−0.002、0、0.002、0.003 分別得到 0、1、1、1、2 |
| 門檻只看 validation | 改動 test 分數不影響 t；validation 上 \|s\| ≥ t 的比例 ≈ q |
| 交易報酬 | 手算的小例子：s = [0.3, −0.2, 0.01]、r = [0.01, −0.01, 0.05]、t = 0.1、c = 0.002 → g = [0.008, 0.008, NaN] |
| persistence 方向只用 train | 造一組 train 上負相關、test 上正相關的資料，d 必須是 −1 |
| persistence 不用未來資料 | 截斷資料到 t 後算出的訊號，和用完整資料算出的位置 t 相同 |
| 三分類模型 | 300 列合成資料可以訓練；回傳形狀為 (n, 3)；每列機率和為 1（誤差 ≤ 1e-6）；相同 seed 兩次結果完全相同 |
| bootstrap 可重現 | 相同 seed 產生相同索引；每個 fold 的抽樣只落在該 fold 內 |
| Purge | 每個 fold 中，validation 第一列索引 − train 最後一列索引 − 1 = 24，test 同理 |
| 拒絕非預設參數 | `--k 0.003` 必須報錯 |

執行：

```powershell
& $py -m unittest discover -s tests -v
```

目前完整測試是 83 項，新增後全部必須通過。

### A.13 執行

```powershell
& $py scripts\run_three_class_target_comparison.py --symbols BTC --output-dir logs\three_class_target
& $py scripts\run_three_class_target_comparison.py --symbols ETH --output-dir logs\three_class_target
& $py scripts\run_three_class_target_comparison.py --symbols SOL --output-dir logs\three_class_target
& $py scripts\run_three_class_target_comparison.py --symbols XRP --output-dir logs\three_class_target
```

四個指令可以在四個終端機同時執行。全部完成後，再執行一次彙整（程式提供 `--aggregate-only`，只讀四個幣種的 JSON，做 Holm 校正並產生報告）：

```powershell
& $py scripts\run_three_class_target_comparison.py --aggregate-only --output-dir logs\three_class_target
```

**執行期間的一致性檢查**（程式內建，任何一項不成立就報錯停止）：

1. 每個 fold 的 test 恰好 4,000 列，三個 fold 合計 12,000 列。
2. T、B、P 三組的 `(fold, open_time)` 完全相同。
3. 每列三類機率和為 1。
4. B 組的 test AUC 三 seed 平均，要和多視窗實驗 4h 結果差距在 0.01 以內（切分沒有對齊，不會完全相同；差太多代表程式寫錯）。

**預估時間**：三分類 XGBoost 每棵樹要算 3 類，單次訓練約為二分類的 3 倍，約 12～20 秒。每幣種 18 次訓練，單一幣種約 3～6 分鐘；四幣種平行約 5～10 分鐘。實際時間寫進 JSON。

### A.14 報告格式：`docs/three_class_target_results.md`

1. 主表：幣種 × 組別（T、B、P），欄位為交易次數、實際交易比例、m（對數與百分比）、逐 fold m、命中率、關卡一 p、關卡二 Δ、ε、Holm p、95% CI、判定。
2. 交易比例曲線表（A.9 第 1 項）。
3. 三類比例表（A.9 第 6 項）。
4. T 與 B 的方向 AUC 對照（A.9 第 4 項）。
5. 不重疊版本的 m（A.9 第 7 項）。
6. 結論：依 A.8 的判定文字，一字不改。
7. 限制（A.15）。

### A.15 限制（寫進預先登記與報告）

- test 期間（2024-04 至 2026-08）和多視窗、重訓頻率、先前 4h 實驗的 test 期間重疊，這是第四次使用同一段資料，不是獨立確認。
- 固定 k = 0.002 讓各幣種的不交易比例差很多（SOL 約 11%、BTC 約 28%）。
- 只扣固定手續費，沒有滑價、資金費率、部位上限。
- 4h 標籤逐小時重疊，主要指標中的交易不是互相獨立的交易，不能直接加總成實際報酬。
- 只測 XGBoost。

### A.16 失敗與異常的處理

| 狀況 | 處理 |
|---|---|
| 單元測試失敗 | 修程式，不修測試的預期值（除非測試本身寫錯，需在 commit 訊息說明） |
| 某 fold 的 train 缺少某一類 | 報錯停止；寫進結論文件，不改 k |
| 某 seed 交易次數 < 200 | 該幣種判定為「交易次數不足，無法判定」，不改 q |
| B 組 AUC 和多視窗結果差 > 0.01 | 停下來查原因（通常是切分或特徵載入錯誤），修正後全部重跑 |
| 發現實驗程式有錯且已產生結果 | 依規定開 v2 預先登記，寫明錯誤與修正方式，修正後重跑全部 |

---

## 工作 B：持續更新資料與前向測試流程

### B.1 目的

1. 讓資料與模型自動跟上新行情，之後網頁可以直接讀取預測結果。
2. 累積 2026-08-09 之後、模型從沒看過的預測紀錄，給工作 C 使用。

**兩個模型要分開**：

| 模型 | 訓練資料 | 何時更新 | 用途 |
|---|---|---|---|
| **凍結模型（frozen）** | 只用到 2026-08-09 06:00 為止的列 | **永遠不更新** | 工作 C 的獨立驗證。因為模型固定，之後的預測隨時都可以補算，結果完全相同 |
| **滾動模型（rolling）** | 到重新訓練當下為止的全部已知標籤列（扣 24 列 purge） | 每月 1 日 | 網頁顯示與實際運作 |

工作 C 只能用凍結模型的預測做判定；滾動模型的預測只作描述。

### B.2 資料夾與檔案（全部不進 git）

```
data/live/raw/<幣種>_1h.csv            增量更新的原始 K 線
data/live/processed/<幣種>_1h.csv      補值後的連續資料
models/live/<幣種>_4h_frozen.json      凍結模型
models/live/<幣種>_4h_rolling_<YYYYMMDD>.json  每月的滾動模型
models/live/<幣種>_4h_rolling_<YYYYMMDD>.meta.json  訓練範圍、資料 SHA-256、git revision
models/live/current.json               指向目前使用中的滾動模型
logs/live/predictions.csv              所有預測（凍結與滾動）
logs/live/run_log.jsonl                每次執行的紀錄
```

- **不寫入** `data/raw/`、`data/processed/`、`data/features/`：這些是前幾個實驗的輸入，JSON 裡記錄了它們的 SHA-256，覆寫之後就無法重現。
- 在 `.gitignore` 加上 `models/live/`（`data/` 與 `logs/` 已經排除）。

### B.3 步驟

#### B-1：初始化（只做一次）

1. 把 `data/raw/<幣種>_5y.csv` **複製**到 `data/live/raw/<幣種>_1h.csv`，作為增量更新的起點。
2. 訓練凍結模型：用 `data/features/<幣種>_4h.csv` 全部 43,755 列（最後一列 2026-08-09 06:00）、預設 XGBoost、seed 42，存成 `models/live/<幣種>_4h_frozen.json`，同時記錄特徵檔的 SHA-256。
3. 訓練第一個滾動模型（此時資料範圍和凍結模型相同）。

#### B-2：增量抓取（每小時）

新增 `scripts/live_update_data.py`：

1. 讀 `data/live/raw/<幣種>_1h.csv` 最後一列的 `open_time`。
2. `end_ms` = 目前時間所在小時的前一小時（只抓**已收盤**的 K 線，同 `resolve_time_window` 的做法）。
3. 呼叫 `fetch_klines(symbol, "1h", 最後一列 + 1 小時, end_ms, request_json)`，重用 `fetch_binance_5y.py` 的 `request_json`、`map_kline_row`、`validate_rows`。
4. 新資料附加到檔案後面；`open_time` 重複的列以新資料為準；寫檔用暫存檔再改名（同 `write_csv_atomic`）。
5. 用 `impute_missing_klines.process_symbol` 產生 `data/live/processed/<幣種>_1h.csv`，用 `check_data_quality.check_csv` 檢查；品質檢查有錯誤就停止，不做預測。
6. 設定 `base_url` 可以切換成 `https://data-api.binance.vision`（Binance 公開行情端點，將來搬到雲端時用得到）。

#### B-3：產生最新特徵（每小時）

在 `scripts/live_predict.py` 內：

1. 讀 `data/live/processed/<幣種>_1h.csv`。
2. 呼叫 `feature_engineering.add_features(raw, prediction_horizon=4)`。
3. **只刪除 53 個特徵中有缺值的列**，保留 `future_log_return` 缺值的最新幾列（這幾列就是要預測的列）。
4. **特徵一致性檢查**：取 2026-08-02 至 2026-08-09 06:00 這段歷史列，和 `data/features/<幣種>_4h.csv` 同時間的列比對，53 個特徵差距都要 ≤ 1e-9。不一致代表線上特徵和實驗用的特徵算法不同，必須停止。這項檢查每天做一次即可。

#### B-4：預測與記錄（每小時）

1. 對每個幣種、每個尚未預測過的列（`open_time` 大於 `predictions.csv` 中該幣種 × 該模型的最後一列），分別用凍結模型與滾動模型預測 `p_up`。
2. 附加寫入 `logs/live/predictions.csv`，欄位：

   | 欄位 | 內容 |
   |---|---|
   | `symbol` | BTC／ETH／SOL／XRP |
   | `open_time` | 被預測的那一列（UTC） |
   | `model_id` | `frozen` 或 `rolling_<YYYYMMDD>` |
   | `p_up` | 模型輸出的上漲機率 |
   | `predicted_at` | 實際執行預測的時間（UTC） |
   | `backfilled` | 1 表示是事後補算（例如電腦關機錯過的時段），0 表示即時預測 |
   | `label_due_at` | 答案可以取得的時間 = `open_time` + 5 小時（第 t+4 根 K 線收盤） |
   | `future_log_return` | 先留空，到期後回填 |
   | `target_up` | 先留空，到期後回填 |

3. 以 `(symbol, open_time, model_id)` 為唯一鍵，重複執行不會產生重複列。
4. **補算規則**：凍結模型可以補算任何錯過的時段（模型固定，結果和即時預測相同）。滾動模型補算時，必須使用**當時**有效的滾動模型版本（依 `models/live/*.meta.json` 的生效日期），並標記 `backfilled=1`。

#### B-5：回填答案（每小時）

對 `label_due_at` 已過、`target_up` 還是空的列，從 `data/live/processed/` 取 `close[t]` 與 `close[t+4 小時]`（**用時間戳查，不用列數**），計算 `future_log_return` 與 `target_up`。

#### B-6：每月重新訓練

新增 `scripts/live_retrain.py`，每月 1 日執行：

1. 用 B-3 的方法產生特徵，只保留標籤已知的列。
2. 訓練資料：到最後一個已知標籤列之前 24 列為止（purge 24 列），也就是 `dataset.iloc[:最後已知列索引 − 24]`。
3. 預設 XGBoost、seed 42。
4. 存成 `models/live/<幣種>_4h_rolling_<YYYYMMDD>.json`，`meta.json` 記錄：訓練起訖時間、列數、特徵 SHA-256、git revision、xgboost 版本、生效時間。
5. 更新 `models/live/current.json`。舊模型不刪除，補算時要用。

**為什麼每月一次**：重訓頻率實驗顯示，每 7 天或每 42 天重新訓練，都沒有穩定比每 167 天好；每月一次已經足夠，成本也低。

#### B-7：Windows 工作排程器

因為專案路徑含中文，排程器直接呼叫 Python 容易出錯，先寫兩個 PowerShell 包裝檔：

`scripts/live_hourly.ps1`：

```powershell
Set-Location "C:\Users\user\專題"
$py = "C:\Users\user\專題\.venv-transformer\Scripts\python.exe"
& $py scripts\live_update_data.py
if ($LASTEXITCODE -eq 0) { & $py scripts\live_predict.py }
```

`scripts/live_monthly.ps1`：

```powershell
Set-Location "C:\Users\user\專題"
& "C:\Users\user\專題\.venv-transformer\Scripts\python.exe" scripts\live_retrain.py
```

建立排程（在 PowerShell 執行）：

```powershell
schtasks /Create /TN "crypto-live-hourly" /SC HOURLY /ST 00:05 /TR "powershell -NoProfile -ExecutionPolicy Bypass -File \"C:\Users\user\專題\scripts\live_hourly.ps1\""
schtasks /Create /TN "crypto-live-monthly" /SC MONTHLY /D 1 /ST 03:00 /TR "powershell -NoProfile -ExecutionPolicy Bypass -File \"C:\Users\user\專題\scripts\live_monthly.ps1\""
```

- 每小時第 5 分鐘執行，讓 Binance 有時間完成上一根 K 線。
- 在工作排程器的「條件」頁勾選「喚醒電腦以執行此工作」，在「設定」頁勾選「錯過排定時間後盡快執行」。
- 電腦關機期間錯過的時段，下次執行時會由 B-2 的增量抓取補上資料，B-4 會補算預測並標記 `backfilled=1`。

**為什麼每小時預測，而不是每 4 小時**：離線實驗是每小時一列、每列預測 4 小時後的方向。前向測試要和離線評估的方式相同，工作 C 才能直接比較；而且工作 C 需要 4,000 列，每小時一列才能在 2027-01-25 前後湊滿。

#### B-8：監控

1. 每次執行在 `logs/live/run_log.jsonl` 寫一行：執行時間、各幣種新抓的列數、最新 `open_time`、預測列數、回填列數、錯誤訊息。
2. `scripts/live_status.py`：印出各幣種最新資料時間、最新預測時間、距離現在幾小時、凍結模型最近 30 天的滾動 AUC（描述性）。超過 3 小時沒有新資料就印出警告。

### B.4 單元測試：新增 `tests/test_live_pipeline.py`

| 測試 | 驗證內容 |
|---|---|
| 增量抓取 | 用假的 `request_json_fn`，確認只要求最後一列之後的資料、合併後沒有重複、時間連續 |
| 只抓已收盤 K 線 | 目前時間 10:30 時，`end_ms` 對應 09:00 開盤的 K 線 |
| 保留未知標籤列 | 產生的特徵中最後 4 列的 `future_log_return` 為 NaN，但特徵完整 |
| 特徵一致性 | 用 `data/features/btc_4h.csv` 的前 500 列對應的原始資料重算，差距 ≤ 1e-9 |
| 預測不重複 | 同一批資料執行兩次，`predictions.csv` 列數不變 |
| 回填時間 | `label_due_at` 未到時不回填；到期後的值等於用時間戳查到的 `log(close[t+4h]/close[t])` |
| 重訓 purge | 訓練資料最後一列索引 = 最後已知標籤列索引 − 25 |
| 補算使用當時的模型 | 造兩個不同生效日期的滾動模型，補算時各時段用到正確的版本 |

### B.5 將來搬到雲端（網頁上線時）

1. 網頁只讀 `predictions.csv`（或改存資料庫），**不在網頁伺服器上跑模型**。
2. 排程搬到雲端主機的 cron 或平台排程，執行同樣的 `live_*.py`。
3. 主機選**非美國區域**（例如新加坡、東京、法蘭克福）；或把 `base_url` 改成 `https://data-api.binance.vision`。Binance 會拒絕美國 IP 的請求。
4. 模型檔與 `predictions.csv` 放在持久化儲存（雲端磁碟或物件儲存），避免主機重啟後遺失。
5. 運算量很小（每次訓練約 5 秒、預測不到 1 秒），最便宜的方案就夠用，不需要 GPU。

### B.6 獨立性規則

- 2027-01-25 之前，`logs/live/predictions.csv` 可以看，但**不可**根據它調整模型、特徵、門檻或任何實驗規則。
- 凍結模型一旦建立就不可重新訓練或覆寫；檔案的 SHA-256 在建立當天寫進 `models/live/frozen_manifest.json`，工作 C 開始前先核對。

### B.7 完成條件

- [ ] B.4 的測試全部通過，完整測試全部通過。
- [ ] 排程連續執行 72 小時，`run_log.jsonl` 沒有錯誤。
- [ ] `predictions.csv` 中四幣種 × 兩個模型都有連續的每小時預測，到期的列都已回填答案。
- [ ] 特徵一致性檢查通過。
- [ ] 手動把電腦關機 3 小時再開機，確認錯過的時段有補上且標記 `backfilled=1`。
- [ ] README「下一步」一節的描述和實際流程一致。

---

## 工作 C：新資料獨立驗證（2027-01-25 之後）

**對象**：
1. 1h 候選（XRP 1h 通過，BTC、ETH 接近）。
2. BTC G4 時間特徵候選。
3. 工作 A 若有候選（只有 1 個幣種通過），也一併驗證。
4. 凍結 4h 模型在新資料上的 AUC（確認 4h 的弱訊號本身是否延續）。

**屆時步驟**（細節另開預先登記文件）：

1. 核對凍結模型的 SHA-256。
2. 取 2026-08-09 06:00 之後、扣掉 24 列 purge 的 4,000 列作為 test。
3. 判定規則沿用各原始實驗，不重新調整。
4. 1h 要改成主要視窗，必須在新資料上**至少 2 個幣種**通過「比 4h 穩定地強」；1h 還要額外報告 XGBoost 是否勝過用訓練資料決定方向的 persistence，排除「只是短期反轉」的可能。

在此之前，不對任何候選假說做額外調參或特徵修改。

---

## 建議時程

| 日期（約） | 工作 |
|---|---|
| 第 1 天 | 工作 A 預先登記文件定案並 commit |
| 第 2～3 天 | 工作 A 實作與測試；工作 B 的 B-1、B-2 |
| 第 3 天 | 工作 A 執行（約 10 分鐘）與報告 |
| 第 4 天 | 工作 A 結論文件、更新總報告、合併；工作 B 的 B-3～B-5 |
| 第 5 天 | 工作 B 的 B-6～B-8 與測試，建立排程 |
| 第 6～8 天 | 工作 B 連續執行 72 小時觀察 |
| 2027-01-25 之後 | 工作 C |

## 目前不做的事與原因

| 不做 | 原因 | 什麼條件下才做 |
|---|---|---|
| 交易成本回測（完整版） | 工作 A 已經扣了固定手續費；完整回測（滑價、部位、資金）需要先有確認過的訊號 | 工作 A ≥ 2 個幣種通過，或工作 C 通過 |
| LSTM／Transformer 複驗 | 多數組合 XGBoost 與 Logistic 差距在 ±0.01 以內，訊號接近線性；4h 階段深度模型也沒有穩定勝過 XGBoost | 某個設定在新資料上 ≥ 2 個幣種通過 |
| 新增特徵、調參 | 外部特徵 A/B、消融 v1、v2 都沒有找到穩定改善；再試會增加偶然撞到的機會 | 無 |
| k、q 的搜尋 | 見 A.10 | 工作 A 通過後，另開預先登記 |
| sliding 訓練窗 | 重訓頻率實驗沒有找到改善，優先度低 | 無 |

## 總檢查清單

- [ ] A：預先登記 commit 早於任何產生結果的程式
- [ ] A：新增測試與完整測試全部通過
- [ ] A：執行期一致性檢查全部通過
- [ ] A：結果、結論文件完成，總報告第七、八節更新
- [ ] A：合併回 main，logs 複製到 main
- [ ] B：`.gitignore` 加入 `models/live/`
- [ ] B：凍結模型建立並記錄 SHA-256
- [ ] B：排程連續 72 小時無錯誤
- [ ] B：README「下一步」一節與實際流程一致
- [ ] C：2027-01-25 後另開預先登記
