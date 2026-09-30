# 多視窗實驗之後的工作計畫

> 歷史文件：本計畫中的工作 0～4 已由後續 `next_phase_work_plan.md` 與 `final_stage_work_plan.md` 接續或取代，保留作為研究流程紀錄。  
> 撰寫日期：2026-09-28  
> 前置文件：[`multi_horizon_conclusions.md`](multi_horizon_conclusions.md)、[`multi_horizon_comparison_results.md`](multi_horizon_comparison_results.md)、[`five_year_4h_final_report.md`](five_year_4h_final_report.md) 第七節  
> 本文件列出接下來每項工作的目的、輸入、步驟、通過條件與失敗時的處理。依編號順序執行，前一項沒有通過就不做下一項。

## 總覽

| 編號 | 工作 | 前置條件 | 預估時間 | 產出 |
|---|---|---|---|---|
| 0 | 把多視窗結果檔複製到 main | 無 | 1 分鐘 | `logs/multi_horizon/`（不進 git） |
| 1 | 驗證多視窗實驗的輸出 | 工作 0 | 2～3 小時（寫程式 + 執行） | `scripts/verify_multi_horizon_outputs.py`、`logs/multi_horizon/verification.json` |
| 2 | 更新總報告與多視窗報告 | 工作 1 全部通過 | 1 小時 | 修改 `five_year_4h_final_report.md`、`multi_horizon_comparison_results.md` |
| 3 | 重新訓練頻率 vs 固定模型（預先登記 → 實作 → 執行 → 報告） | 工作 1 全部通過 | 預先登記半天、實作 1 天、執行約 30 分鐘 | 見 3.8 |
| 4 | 等新資料的獨立驗證（1h 候選、BTC G4） | 2027-01-25 之後 | — | 另開文件 |

目前**不做**的事：深度學習（LSTM／Transformer）複驗、1h／12h／24h 調參、新增特徵、三段式目標、交易成本回測。原因見文末「目前不做的事與原因」。

## 已確認的資料事實（2026-09-28 查核）

以下事實會影響工作 1 與工作 3 的設計，先記錄：

1. **特徵檔的每一列是一根 1 小時 K 線**。4h 特徵檔列數：BTC 43,755、ETH 43,756、SOL 43,755、XRP 43,755，最後一列都是 `2026-08-09 06:00 UTC`。
2. **特徵檔有 11～12 處時間缺口**，不是每列都間隔 1 小時。例如四幣種都有 `2021-08-13 01:00 → 12:00`（缺 10 列）；BTC 另有 `2025-01-10 09:00 → 11:00`；SOL 另有 `2026-02-25 18:00 → 20:00`；XRP 另有 `2023-01-31 18:00 → 20:00`。
3. **缺口不影響標籤正確性**：`data/processed/<幣種>_5y.csv` 沒有任何缺口。標籤 `future_log_return = log(close[t+h]/close[t])` 是在連續序列上用 `shift(-h)` 算的，之後才由 `dropna()` 刪掉特徵無效的列（例如成交量為 0 的 `2021-08-13 05:00`，`volume_change` 會變成無限大）。所以每個標籤都是剛好 h 小時。
4. **缺口會影響三件用「列數」計算的事**：
   - purge 24 列在跨越缺口時會超過 24 小時，這是比較保守的方向，沒有洩漏問題。
   - persistence 基準用 `rolling(h)` 加總最近 h 列報酬，跨越缺口時會漏掉被刪除那幾小時的報酬。受影響的列數最多是 h × 缺口數。
   - block bootstrap 的區塊以列數計算，影響可忽略。
5. **XGBoost 單次訓練時間**（CPU、`n_jobs=1`、BTC 4h）：31,731 列約 3.9 秒；43,563 列約 5.8 秒。
6. **多視窗結果檔在 `.gitignore` 範圍內**（`.gitignore` 第 11 行排除 `logs/`），只存在 worktree `C:\Users\user\orca\workspaces\專題\multi-horizon-comparison\logs\multi_horizon\`，main 的 `logs\` 沒有這些檔案。

以下所有指令都使用 main 的 Python 環境：

```powershell
$py = "C:\Users\user\專題\.venv-transformer\Scripts\python.exe"
```

---

## 工作 0：把多視窗結果檔複製到 main

**目的**：main 上的報告引用 `logs/multi_horizon/`，但 main 沒有這些檔案。

```powershell
Copy-Item -Recurse "C:\Users\user\orca\workspaces\專題\multi-horizon-comparison\logs\multi_horizon" "C:\Users\user\專題\logs\multi_horizon"
```

**通過條件**：`C:\Users\user\專題\logs\multi_horizon\` 下有 `multi_asset_multi_horizon.json`，以及 `btc`、`eth`、`sol`、`xrp` 四個資料夾；每個資料夾有 12 個 `*_predictions.csv` 和 1 個 `multi_horizon_comparison.json`，共 48 + 5 = 53 個檔案。

這些檔案不進 git，維持 `.gitignore` 現狀。

---

## 工作 1：驗證多視窗實驗的輸出

**目的**：工作 3 會沿用同一套切分、bootstrap 與 Holm 程式。先確認多視窗實驗的數字可以從原始資料獨立重算，沒有洩漏，再往下做。

### 1.1 做法

在 main 開 branch `verify-multi-horizon`，新增 `scripts/verify_multi_horizon_outputs.py`。這支程式：

- 只讀檔案，不重新訓練模型。
- 讀取 `logs/multi_horizon/`、`data/features/<幣種>_<h>h.csv`、`data/processed/<幣種>_5y.csv`。
- 逐項執行下面的 V1～V8，每項印出 `PASS` 或 `FAIL`，並附上不一致的具體數值。
- 把結果寫進 `logs/multi_horizon/verification.json`。
- 任何一項 `FAIL` 就以 exit code 1 結束。

執行：

```powershell
& $py scripts\verify_multi_horizon_outputs.py
& $py -m unittest discover -s tests -v
```

### 1.2 檢查項目與通過條件

| 編號 | 檢查 | 做法 | 通過條件 |
|---|---|---|---|
| V1 | 單元測試 | 執行 `unittest discover -s tests` | 全部通過，0 failures、0 errors |
| V2 | 列數與配對 | 讀 48 個預測檔 | 每個檔案 12,000 列；`fold` 0、1、2 各 4,000 列。同一幣種內，4 個視窗 × 3 個 seed 的 `(fold, open_time)` 欄位完全相同 |
| V3 | 切分與 JSON 一致 | 比對預測檔每個 fold 的第一列與最後一列 `open_time`，和 JSON `horizons.<h>.folds[i].test.start/end` | 完全相同 |
| V4 | Purge | 用對齊後的資料（四個視窗 `open_time` 的交集）找出每個 fold 的 train 最後一列、validation 第一列與最後一列、test 第一列的**列索引** | `validation 第一列索引 − train 最後一列索引 − 1 = 24`，且 `test 第一列索引 − validation 最後一列索引 − 1 = 24`，四幣種 × 四視窗 × 三 fold 共 48 組全部成立 |
| V5 | 標籤正確 | 每個預測檔的每一列：從 `data/processed/<幣種>_5y.csv` 取 `close[t]` 與 `close[t+h 小時]`（用時間戳查，不用列數），算 `log(close[t+h]/close[t]) > 0` | 和預測檔的 `target_up` 100% 相同。另外，特徵檔 `future_log_return` 和重算值的差距 ≤ 1e-9 |
| V6 | AUC 可重算 | 對每個預測檔算 `roc_auc_score(target_up, test_raw)`，三 seed 平均；逐 fold 同樣計算 | 單一 seed 和 JSON `pooled_auc_by_seed` 差距 ≤ 1e-12；三 seed 平均和 `mean_pooled_auc` 差距 ≤ 1e-12；逐 fold 平均四捨五入到小數 6 位後，和 `multi_horizon_comparison_results.md` 表格一致 |
| V7 | Persistence 沒有用到未來資料 | ① 對每個幣種 × 視窗，用 `numpy.random.default_rng(20260928)` 抽 200 個 test 列 t，把資料截斷到 t（`dataset.iloc[:t+1]`）後呼叫 `persistence_scores`，取最後一個值；② 用完整資料算出位置 t 的值 | ① 和 ② 完全相同，共 16 × 200 = 3,200 次比對。另外，重算的 persistence test AUC 和 JSON `baselines[i].persistence.test.roc_auc` 差距 ≤ 1e-12 |
| V8 | Holm 校正可重算 | 從 JSON 讀 16 個訊號關卡原始 p 值，以及每個幣種 3 個比較的原始 p 值，自己實作 Holm | 校正後 p 值和 JSON `adjusted_p_value` 差距 ≤ 1e-12，`signal_passes`、`decision` 相同 |

另外記錄一項**描述性數字，不判定通過與否**：各幣種 × 視窗 × fold 的 test 列中，persistence 的 `rolling(h)` 視窗跨過特徵檔缺口的列數（見「已確認的資料事實」第 2、4 點）。

- 目前已知 SOL 的 `2026-02-25 18:00 → 20:00` 缺口落在 SOL fold 2 的 test 期間，最多影響 24 列（24h 視窗），占該 fold 4,000 列的 0.6%。
- 若任何 fold 受影響的比例 > 1%，在 `multi_horizon_conclusions.md` 第七節註明。persistence 只是描述性基準，不影響判定，不需重跑。

### 1.3 已知的統計特性（不是錯誤，寫進驗證報告）

- Bootstrap 是 2,000 次，單尾 p 的最小值是 1/2001 ≈ 0.0005。16 格 Holm 校正後，最小值是 16/2001 ≈ 0.007996。訊號關卡表中有 5 格（BTC 1h、BTC 4h、ETH 1h、ETH 12h、XRP 1h）正好是 0.007996，代表 2,000 次重抽都大於 0，p 值被重抽次數限制住，不是計算錯誤。

### 1.4 通過與失敗的處理

- **全部 PASS**：把驗證程式和結果摘要 commit 到 `verify-multi-horizon`，fast-forward 合併回 main。在 `multi_horizon_conclusions.md` 第七節改寫為「已完成驗證（日期、commit）」，然後進行工作 2 與工作 3。
- **任何一項 FAIL**：
  1. 不做工作 2、3。
  2. 找出原因，寫進 `multi_horizon_conclusions.md` 第七節：哪一項、哪個幣種 × 視窗 × fold、預期值與實際值。
  3. 如果是實驗程式的錯誤，依預先登記文件「定案後修改必須另開 v2 文件」的規定，新增 `docs/superpowers/specs/<日期>-multi-horizon-comparison-v2-design.md`，寫明錯誤與修正方式，修正後重跑全部 16 格，再重做工作 1。
  4. 如果是驗證程式本身的錯誤，修正驗證程式後重跑，不需要重跑實驗。

---

## 工作 2：更新總報告與多視窗報告

**前置條件**：工作 1 全部通過。在工作 1 的 branch 或另一個 branch 上修改，完成後合併回 main。

### 2.1 `docs/five_year_4h_final_report.md`

| 位置 | 修改 |
|---|---|
| 第五節標題「1h／12h／24h 是否需要訓練」 | 改為「1h／12h／24h 多視窗比較（已完成）」 |
| 第五節內文 | 保留原本三段作為背景，最後加一段：完成日期 2026-09-28；1h 四幣種都通過訊號關卡，只有 XRP 1h 通過「比 4h 穩定地強」的判定；12h、24h 沒有任何幣種比 4h 好；依預先登記，4h 維持主要視窗，1h 列為候選假說；連結 `multi_horizon_conclusions.md` |
| 第七節第 1 項 | 前面加上「（已完成，2026-09-28）」，後面加一句結論與連結 |
| 第七節第 3 項 | 後面加「（下一項執行，見 `next_work_plan.md` 工作 3）」 |
| 第八節 | 新增五行：預先登記 `docs/superpowers/specs/2026-09-27-multi-horizon-comparison-design.md`、結果 `docs/multi_horizon_comparison_results.md`、結論 `docs/multi_horizon_conclusions.md`、logs `logs/multi_horizon/`（不進 git，來源 worktree 路徑）、驗證 `logs/multi_horizon/verification.json` |

### 2.2 `docs/multi_horizon_comparison_results.md`：補上反向 persistence

**問題**：報告中的「XGB−persistence」是拿 XGB 和 persistence 比。但 1h 的 persistence AUC 大多在 0.46～0.49，低於 0.5，代表短期反轉；把 persistence 的預測反過來用，AUC 會高於 0.5。所以直接相減會高估 XGB 的優勢。

**修改**：

1. 在 `scripts/run_multi_horizon_comparison.py` 的 `render_report` 描述性檢查那一行，加入 `best-direction persistence = max(p, 1 − p)`（p 為 persistence AUC），以及 `XGB − best-direction persistence`。
2. 在「長視窗 XGBoost 不勝 persistence」段落後加一句：「persistence AUC < 0.5 表示該視窗短期反轉；比較 XGB 的額外資訊時應使用 best-direction persistence。」
3. **不重新訓練**。`render_report` 只讀取 `reports` 字典，而這個字典就是 `logs/multi_horizon/multi_asset_multi_horizon.json` 的 `symbols` 欄位。寫一段程式載入 JSON 後呼叫 `render_report(data["symbols"], None)`，重新產生 markdown 即可。
4. 判定表（訊號關卡與相對 4h 比較）的所有數字必須和修改前完全相同；只允許描述性檢查段落改變。用 `git diff` 確認。

**注意**：`max(p, 1 − p)` 是看過 test 結果後才選方向，會讓 persistence 看起來偏好，只能當描述性的上限參考，不可用於判定。這句話要一起寫進報告。

---

## 工作 3：重新訓練頻率 vs 固定模型

### 3.1 研究問題

在同一段評估期間、同一組特徵與同一個 XGBoost 設定下，只改變「多久用最新資料重新訓練一次」：

> 較頻繁地重新訓練，4h 方向預測的 test AUC 是否穩定高於只訓練一次的固定模型？

**為什麼做**：v2 與多視窗實驗都觀察到 validation AUC 普遍高於 test AUC（例如 BTC 4h fold 1：validation 0.547、test 0.514），代表訊號會隨時間衰退。如果衰退來自市場結構改變，較頻繁地重新訓練應該能減緩。這也是多視窗預先登記文件停止條件指定的下一步。

**本實驗不回答**：交易是否獲利、哪個模型最好、train 視窗該用 expanding 還是 sliding、1h 等其他視窗是否適用。

### 3.2 流程

1. 從 main 建 worktree：

   ```powershell
   git -C C:\Users\user\專題 worktree add -b retrain-frequency "C:\Users\user\orca\workspaces\專題\retrain-frequency" main
   ```

2. 把本節 3.3～3.9 抄寫成預先登記文件 `docs/superpowers/specs/2026-09-29-retrain-frequency-design.md`，標註「定案」後 commit。**commit 之前不可執行任何會產生結果的程式。** 定案後若要修改，另開 v2 文件並寫明原因。
3. 實作程式與測試（3.8），測試全部通過後 commit。
4. 執行實驗，產生結果與報告，commit。
5. 另寫結論文件 `docs/retrain_frequency_conclusions.md`，格式同 `multi_horizon_conclusions.md`，commit 後 fast-forward 合併回 main。

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

---

## 工作 4：等新資料的獨立驗證

**適用對象**：

1. 1h 候選假說（XRP 1h 通過；BTC、ETH 1h 接近通過）。
2. BTC G4 時間特徵候選假說（總報告第七節第 2 項）。
3. 工作 3 若有「只在 1 個幣種通過」的候選，也一併驗證。

**可以開始的時間**：現有資料最後一列是 2026-08-09 06:00 UTC。獨立驗證需要至少一個完整 test 段，也就是 4,000 列，再加上 24 列 purge 與最長 24 小時的標籤，約 4,048 小時，因此最早是 **2027-01-25**。

**屆時的步驟**：

1. 重新抓資料並跑完整資料流程（fetch → 品質檢查 → 補值 → 特徵工程），產生新的特徵檔。舊的特徵檔不覆寫，另存。
2. 另開預先登記文件：訓練資料到 2026-08-09 06:00 為止，test 只使用 2026-08-09 之後的列；判定規則沿用原實驗，不重新調整。
3. 1h 要切換成主要視窗，必須在新資料上**至少 2 個幣種**通過「比 4h 穩定地強」的判定。

**在此之前**：不對候選假說做任何額外調參或特徵修改，以免新資料的驗證失去獨立性。

---

## 目前不做的事與原因

| 不做 | 原因 | 什麼條件下才做 |
|---|---|---|
| LSTM／Transformer 複驗 | 多視窗實驗沒有任何視窗在 ≥ 2 個幣種通過（預先登記的門檻）；多數組合 XGBoost 與 Logistic 的 AUC 差距在 ±0.01 以內，訊號接近線性，深度模型不太可能學到更多；4h 階段的 LSTM、Transformer 也沒有穩定勝過 XGBoost（README 表格） | 某個視窗或設定在新資料上於 ≥ 2 個幣種通過 |
| 1h／12h／24h 調參 | 預先登記規定不為了救結果去調參；12h、24h 沒有訊號 | 無 |
| 新增特徵 | 外部特徵 A/B、消融 v1、v2 都沒有找到能穩定改善的特徵變動 | 無 |
| 三段式目標（上漲／下跌／不交易） | 排在重新訓練頻率之後（總報告第七節第 4 項） | 工作 3 結果為「沒有任何組通過」或完成採用後 |
| 交易成本回測 | 需要先有確認過的訊號，否則只是在弱訊號上算報酬 | 某個設定通過新資料獨立驗證 |
