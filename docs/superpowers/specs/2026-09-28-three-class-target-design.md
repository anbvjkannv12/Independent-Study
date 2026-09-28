# 三段式目標實驗預先登記（定案）

> 定案日期：2026-09-28  
> 狀態：定案  
> 規則：本文件 commit 之前不可執行任何會產生結果的程式；定案後若要修改，另開 v2 文件並寫明原因。

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
