# 多視窗 1h／4h／12h／24h 方向分類比較：預先登記與操作文件

> 狀態：**定案**（2026-09-27）。文末「已確認的操作細節」六項都採用建議。  
> 前置文件：[`../../five_year_4h_final_report.md`](../../five_year_4h_final_report.md) 第五、七節；[`2026-09-26-feature-ab-test-v2-design.md`](2026-09-26-feature-ab-test-v2-design.md)。  
> 本文件在看到任何多視窗結果之前定案。定案後若需要修改，必須另開 v2 文件並寫明原因，不可直接改本文件。

## 一、研究問題

在**同一套資料列、同一套 walk-forward 切分、同一個模型設定**下，只改變預測視窗 h ∈ {1, 4, 12, 24} 小時：

1. 哪些「幣種 × 視窗」的方向訊號**顯著高於隨機**（ROC-AUC > 0.5）？
2. 相對於目前的 4h，有沒有哪個視窗的方向訊號**穩定較強**？

本實驗**不回答**：交易是否獲利、哪個模型最好、哪些特徵有用。不調超參數，不新增特徵。

## 二、為什麼要做

4h 階段的三輪受控實驗（外部特徵 A/B、特徵消融 v1、v2）都沒有找到能穩定改善的特徵變動，四幣種 test AUC 維持在 0.50～0.53。這表示 4h 的瓶頸比較可能在訊號本身，而不是在特徵選擇。因此下一步是改變問題本身，也就是預測視窗，而不是繼續微調 4h 的特徵。

## 三、固定不變的條件

| 項目 | 設定 | 來源 |
| --- | --- | --- |
| 資料 | `data/features/<幣種>_{1,4,12,24}h.csv`（已產出，不重做特徵工程） | `scripts/feature_engineering.py` |
| 特徵 | `logs/feature_manifest.json` 的 53 個特徵，四個視窗完全相同 | — |
| 標籤 | `target_up = log(close[t+h] / close[t]) > 0` | 特徵工程 manifest |
| 模型 | `fit_predict_xgboost` 與預設 `ModelSettings`，只改 `seed` | `scripts/btc_4h_model_adapters.py` |
| 校準 | 只用 validation 做 Platt 校準 | `fit_platt_calibrator` |
| 門檻 | 只用 validation，依 balanced accuracy 選定 | `select_balanced_accuracy_threshold` |
| 主要指標 | 合併 test 列上的**未校準** ROC-AUC（`test_raw`），同 v2 | — |
| 多重比較 | Holm 校正，α = 0.05 | 同 v2 |
| Bootstrap | moving block，block 長度 48 列（四個視窗相同），在每個 fold 內各自重抽，2,000 次，seed 20260927 | — |
| 運算裝置 | **全部使用 CPU**，不使用 GPU；XGBoost 維持 `n_jobs=1`、`tree_method="hist"` | GPU 與 CPU 的浮點運算順序不同，同 seed 結果也可能有小差異，會影響 ε 等級（約 0.005）的比較 |

## 四、資料對齊與 purge（本實驗的關鍵修改）

### 4.1 目前的問題

檢查 `data/features/` 的實際列數（BTC）：

| 視窗 | 列數 | 最後一列 `open_time` |
| --- | ---: | --- |
| 1h | 43,758 | 2026-08-09 09:00 |
| 4h | 43,755 | 2026-08-09 06:00 |
| 12h | 43,747 | 2026-08-08 22:00 |
| 24h | 43,735 | 2026-08-08 10:00 |

ETH 在每個視窗都多 1 列。`build_expanding_folds` 是從資料開頭**依列數**切分，所以如果直接套用，四個視窗的 test 期間會錯開幾個到二十幾個小時，比較的就不是同一段市場。

另外，`build_expanding_folds` 目前沒有 purge。train 最後 h−1 列的標籤會用到 validation 期間的收盤價（validation 與 test 的交界也一樣）。4h 的影響很小，但 24h 會有 23 列重疊，所以一定要處理。

### 4.2 做法

1. **時間對齊**：每個幣種取四個視窗 `open_time` 的**交集**，四份資料都只保留交集內的列。對齊後四個視窗的列數與時間戳必須完全相同，不同則直接報錯停止。
2. **Purge**：在 train→validation、validation→test 兩個交界各丟棄 **P = 24 列**（四個視窗都用同樣的 P，見 Q1）。這 24 列不參與該 fold 的訓練與評估；到下一個 fold 時，這些列會成為 expanding train 的一部分。
3. **切分**：每個 fold 的排列為 `train | purge 24 | validation 4,000 | purge 24 | test 4,000`，共 3 個 folds，最後一個 test 對齊到資料尾端。

以 BTC 為例：交集 43,735 列；每個 fold 占 4,000 + 4,000 + 2×24 = 8,048 列；初始訓練列數 = 43,735 − 3 × 8,048 = **19,591**。其他幣種的初始訓練列數由程式依交集列數自動計算，並寫進輸出 JSON。

> 因為對齊和 purge 讓 test 期間與先前 4h 實驗略有不同，**4h 也要在新切分下重跑**，不沿用 `logs/*_4h_model_comparison*.json` 的數字。報告中可以把舊 4h 數字並列作為參考，但判定只用新切分的結果。

## 五、實驗組別

每個幣種、每個視窗：

| 組別 | 內容 | seeds | 次數 |
| --- | --- | --- | ---: |
| XGBoost（主要） | 53 特徵 | 42、43、44 | 3 seeds × 3 folds = 9 |
| Logistic Regression（基準） | 53 特徵，`evaluate_baselines` 內建 | — | 3 folds |
| Persistence（基準，新增） | 預測方向 = 過去 h 小時報酬的正負號，也就是 `sign(Σ log_return[t−h+1..t])`；只用 t 當下已知的資料 | — | 3 folds |
| all-up、training-majority、zero-trade | 沿用既有基準 | — | 3 folds |

四幣種 × 四視窗 × 9 = **144 次 XGBoost 訓練**。依 v1、v2 實測（每次約 3 秒），訓練約 7～8 分鐘，加上 bootstrap，全部預估 **40～60 分鐘**（CPU）。

**加速方式**：用多個 process 平行跑不同的幣種或 seed，每個 process 內的 XGBoost 仍維持 `n_jobs=1`。這只改變執行順序，不影響結果，所以可以在不動判準的情況下縮短時間。

**本輪不跑 LSTM／Transformer**（見 Q3）。有視窗通過步驟 2 之後，另開文件複驗深度模型；GPU 支援（`ModelSettings` 新增 `device` 欄位，預設 `"cpu"`）也在那時一起加，並先用 BTC 一個 fold 比較 CPU／GPU 的時間與 AUC 差異。

> 為什麼另外加 persistence 基準：既有的 `previous_direction` 用的是 `log_return_lag_1`（上一根 1h K 線），對 12h／24h 不是合理的比較對象。如果長視窗的 AUC 只是反映趨勢延續，persistence 基準就能直接看出來。

## 六、流程與判定規則（事先定案）

### 步驟 1：訊號關卡（每個幣種 × 視窗，共 16 格）

- 統計量：XGBoost 三顆 seed 各自在合併 12,000 列 test 上的 AUC，取平均。
- 檢定：對「平均 AUC − 0.5」做 block bootstrap，單尾 p = (1 + #{重抽值 ≤ 0}) / 2,001。
- **16 格一起做 Holm 校正**。和 v2 的前置關卡不同，這一步本身就是研究結論，所以要校正。
- 通過條件：Holm 校正後 p ≤ 0.05。

### 步驟 2：與 4h 比較（每個幣種內，1h／12h／24h 各對 4h）

- ΔAUC(h) = 平均 AUC(h) − 平均 AUC(4h)。四個視窗對齊到同樣的列，所以可以**配對 bootstrap**：同一組重抽索引同時套用到 h 與 4h 的預測。
- 逐 fold ΔAUC：同樣的定義，只算單一 fold 的 4,000 列。
- ε（seed 雜訊門檻）：同一視窗三顆 seed 兩兩配對的 |ΔAUC| 取最大值；比較時取 max(ε_h, ε_4h)，作法同 v2，偏向保守。
- 每個幣種內 3 個比較做 Holm 校正。

**判定為「h 的方向訊號較 4h 穩定地強」須同時符合：**

1. 3 個 fold 的 ΔAUC 都 > 0。
2. Holm 校正後 p ≤ 0.05。
3. 合併 ΔAUC > ε。
4. 視窗 h 在步驟 1 通過。

其他結果的寫法：

- 不符合 → 「沒有證據顯示 h 比 4h 好」，不能寫成「h 比較差」或「h 沒用」。
- ΔAUC < 0 且未校正 95% CI 上界 < 0 → 標記「h 可能弱於 4h」，只作描述。

### 步驟 3：描述性檢查（不參與判定）

- XGBoost 平均 AUC 對 Logistic Regression、persistence 的差值。如果 XGBoost 贏不過 persistence，報告要寫明長視窗的訊號可能主要來自趨勢延續。
- 校準後 Brier，對「各 fold 訓練集上漲率常數」的 Brier。
- 各視窗 test 期的上漲率，因為長視窗的類別比例可能較不平衡。
- validation AUC 與 test AUC 的落差（v2 觀察到的衰退現象是否隨視窗改變）。

### 停止條件

| 結果 | 下一步 |
| --- | --- |
| 沒有任何幣種有 h 通過步驟 2 | 結論為「改變視窗沒有證據能改善方向訊號」。**不為了救結果去調參數、換特徵或換 fold。** 進入報告第七節下一項：重新訓練頻率 vs 固定模型。 |
| 有 h 在 1 個幣種通過 | 列為候選假說，不宣稱發現；等 2026-08-09 之後的新資料累積後獨立驗證。 |
| 同一個 h 在 ≥ 2 個幣種通過 | 該視窗成為下一階段主要視窗：另開文件，做深度模型複驗與新資料的獨立驗證。 |

## 七、需要的程式修改

1. **`scripts/btc_4h_evaluation.py`**：`build_expanding_folds` 新增 `purge_rows: int = 0` 參數。預設 0 時行為與現在完全相同，既有測試不需改。
2. **新增 `scripts/run_multi_horizon_comparison.py`**：
   - 讀四個視窗的 CSV → 取 `open_time` 交集 → 驗證四份的時間戳一致。
   - 依交集列數計算 `initial_train_rows`，呼叫 `build_expanding_folds(..., purge_rows=24)`。
   - XGBoost 三 seed 直接重用 `run_feature_ablation.run_xgboost_arm`；`pooled_auc`、`moving_block_bootstrap_deltas`、`holm_adjust` 也直接 import，不重寫。
   - 基準用 `evaluate_baselines`，另外加上 persistence 基準。
   - CLI 參數：`--symbols`、`--horizons`、`--purge-rows`、`--block-length`、`--output-dir`，預設值照本文件。
3. **`tests/`**：
   - `test_btc_4h_evaluation.py`：新增 purge 測試，驗證 train 尾端、validation 頭尾、test 開頭之間相隔正好 `purge_rows` 列。
   - 新增 `tests/test_multi_horizon_comparison.py`：時間戳不一致時報錯；persistence 基準在 t 只用到 t 以前（含 t）的資料。

## 八、輸出

| 內容 | 位置 |
| --- | --- |
| 逐列預測 | `logs/multi_horizon/<幣種>/<h>h_seed{42,43,44}_predictions.csv` |
| 各幣種結果 | `logs/multi_horizon/<幣種>/multi_horizon_comparison.json`（含切分、對齊列數、ε、bootstrap、判定） |
| 四幣種彙整 | `logs/multi_horizon/multi_asset_multi_horizon.json` |
| 報告 | `docs/multi_horizon_comparison_results.md` |

輸出 JSON 另外記錄：git revision、四份輸入 CSV 的 SHA-256、Python／xgboost 版本。既有的 `logs/*_4h_*` 檔案一律不覆寫。

## 九、執行步驟

### 步驟 0：建立 worktree 與準備環境

所有工作都在新的 worktree 裡進行，不在 `C:\Users\user\專題`（main）或 `soapfish` 裡做。原因：

- `soapfish` 停在舊的 commit（`f475505`），缺少 main 上的 `run_feature_ablation.py`、v2 設計文件與最終報告。
- 直接在 main 上做，會和主資料夾其他未 commit 的工作混在一起。

固定路徑與名稱：

| 項目 | 值 |
| --- | --- |
| 主資料夾（main） | `C:\Users\user\專題` |
| 新 worktree 路徑 | `C:\Users\user\orca\workspaces\專題\multi-horizon-comparison` |
| 新 branch | `multi-horizon-comparison`，從 main 的最新 commit 分出 |

#### 0-1. 確認 main 是最新的

```powershell
git -C C:\Users\user\專題 status --short
git -C C:\Users\user\專題 log --oneline -3
git -C C:\Users\user\專題 worktree list
```

檢查：

- `log` 最上面應包含 `docs: update 4h final stage report` 之後的 commit（撰寫本文件時 main 為 `a025399`）。
- 新 worktree 只會帶進 **已 commit** 的內容。如果 `status` 顯示 main 有本實驗需要、但尚未 commit 的檔案（例如 `scripts/`、`tests/` 的修改），先處理再往下做。
- `worktree list` 中不應已經有 `multi-horizon-comparison`；如果有，表示之前建過，直接跳到 0-3。

#### 0-2. 建立 worktree 與 branch

```powershell
git -C C:\Users\user\專題 worktree add -b multi-horizon-comparison "C:\Users\user\orca\workspaces\專題\multi-horizon-comparison" main
```

如果出現 `a branch named 'multi-horizon-comparison' already exists`，表示 branch 已存在但沒有 worktree，改用（不加 `-b`）：

```powershell
git -C C:\Users\user\專題 worktree add "C:\Users\user\orca\workspaces\專題\multi-horizon-comparison" multi-horizon-comparison
```

確認：

```powershell
cd "C:\Users\user\orca\workspaces\專題\multi-horizon-comparison"
git branch --show-current                              # 應為 multi-horizon-comparison
git log --oneline -1                                   # 應與 main 最新 commit 相同
Test-Path scripts\run_feature_ablation.py              # 應為 True
Test-Path docs\five_year_4h_final_report.md            # 應為 True
```

#### 0-3. 複製被 `.gitignore` 排除的資料

`data/`、`logs/` 不在 git 裡，新 worktree 沒有這些檔案，需要從主資料夾複製。只複製本實驗需要的部分：

```powershell
$src = "C:\Users\user\專題"
$dst = "C:\Users\user\orca\workspaces\專題\multi-horizon-comparison"

New-Item -ItemType Directory -Force "$dst\data" | Out-Null
Copy-Item -Recurse "$src\data\features" "$dst\data\features"
Copy-Item -Recurse "$src\logs" "$dst\logs"
```

- `data\features`：四幣種 × 四視窗共 16 份特徵 CSV。
- `logs`：`feature_manifest.json`（53 個特徵的清單），以及舊 4h 結果（BTC 試跑檢查清單要拿來比對）。
- 如果目的地已經有 `data\features` 或 `logs`，`Copy-Item` 會把來源放進子資料夾而不是覆蓋。**重複執行前先確認目的地不存在**，不要重複複製。

#### 0-4. Python 環境

**直接使用主資料夾的 venv，不複製。** venv 內含 PyTorch，檔案很大，而且本實驗不修改套件。程式碼從目前所在的 worktree 讀取，所以用主資料夾的 Python 執行，跑的仍然是新 worktree 的程式。

```powershell
$py = "C:\Users\user\專題\.venv-transformer\Scripts\python.exe"
& $py -c "import sys, xgboost, sklearn, pandas; print(sys.version); print('xgboost', xgboost.__version__); print('sklearn', sklearn.__version__); print('pandas', pandas.__version__)"
```

撰寫本文件時確認的版本：xgboost 3.2.0、torch 2.14.0+cu126。xgboost 版本若不同，4h 結果可能和舊結果有差異，要記錄在報告裡。

#### 0-5. 確認資料列數與第四節一致

```powershell
& $py -c "import pandas as pd; [print(f'{s}_{h}h', len(d), d['open_time'].iloc[-1]) for s in ('btc','eth','sol','xrp') for h in (1,4,12,24) for d in [pd.read_csv(f'data/features/{s}_{h}h.csv', usecols=['open_time'])]]"
```

BTC 應為 1h 43,758 列、4h 43,755 列、12h 43,747 列、24h 43,735 列，最後時間與第四節表格相同；ETH 每個視窗多 1 列。數字不同表示複製到的是別版本的特徵資料，**先查清楚再往下做**。

#### 0-6. 把本文件搬進新 worktree 並 commit

本文件目前在 `soapfish`（`fix-usage-typo` branch），還沒 commit。搬到新 worktree 後 commit，作為這個 branch 的第一個 commit，證明設計早於任何結果。

```powershell
Move-Item "C:\Users\user\orca\workspaces\專題\soapfish\docs\superpowers\specs\2026-09-27-multi-horizon-comparison-design.md" "C:\Users\user\orca\workspaces\專題\multi-horizon-comparison\docs\superpowers\specs\"

cd "C:\Users\user\orca\workspaces\專題\multi-horizon-comparison"
git add docs\superpowers\specs\2026-09-27-multi-horizon-comparison-design.md
git commit -m "docs: pre-register multi-horizon comparison design"
```

**這個 commit 必須在任何實驗結果產生之前完成。** 之後若要修改設計，另開 v2 文件，不直接改這份。

#### 步驟 0 完成的檢查清單

- [ ] 位於 `C:\Users\user\orca\workspaces\專題\multi-horizon-comparison`，branch 為 `multi-horizon-comparison`
- [ ] `scripts\run_feature_ablation.py` 存在
- [ ] `data\features` 有 16 份 CSV，列數與 0-5 一致
- [ ] `logs\feature_manifest.json` 存在
- [ ] `$py` 可以 import xgboost、sklearn、pandas
- [ ] 本文件已 commit 在新 branch 上

### 步驟 1 之後：寫程式與執行實驗

第七節的程式修改完成後，在新 worktree 中執行：

```powershell
cd "C:\Users\user\orca\workspaces\專題\multi-horizon-comparison"
$py = "C:\Users\user\專題\.venv-transformer\Scripts\python.exe"

# 1. 單元測試
& $py -m pytest tests\test_btc_4h_evaluation.py tests\test_multi_horizon_comparison.py tests\test_feature_ablation.py -q

# 2. 只跑 BTC，確認切分、對齊列數、輸出格式（約 10～15 分鐘）
& $py scripts\run_multi_horizon_comparison.py --symbols BTC

# 3. 檢查 BTC 輸出（見下方檢查清單）；無誤後跑四幣種
& $py scripts\run_multi_horizon_comparison.py
```

### BTC 試跑後的檢查清單

- [ ] 四個視窗的對齊列數相同，且等於第四節的計算值
- [ ] 每個 fold 的 train 結束、validation 開始、test 開始之間相隔正好 24 列
- [ ] 最後一個 test 的結束時間 = 交集的最後一列
- [ ] 4h 的 XGBoost AUC 與舊 4h 結果差距在合理範圍（約 ±0.01）；差距太大要先查原因再往下跑
- [ ] 三顆 seed 的預測檔都存在，列數 = 12,000

**只看 BTC 的檢查清單，不看 BTC 的判定結果之後再改設計。** 如果檢查清單沒過需要改程式，只修 bug，不動判準。

## 十、報告撰寫規則

1. 先列 16 格訊號關卡表，再列各幣種 3 個對 4h 的比較。
2. 每個數字都附上 fold 內的數值，不只寫平均。
3. 「沒有證據」不等於「沒有用」，這句話要出現在結論中。
4. 固定包含「限制」段落（見下節）。

## 十一、本實驗無法補足的限制（報告中須明寫）

1. **Test 期間重複使用。** 4h 的 test 期間在先前實驗已經看過，本實驗只是「在同樣期間比較不同視窗」，不是獨立確認。
2. **長視窗的有效樣本比較少。** 24h 標籤相鄰列高度重疊，4,000 列 test 大約只有 167 個不重疊的 24h 區間，所以信賴區間會比 1h 寬很多，同樣的 ΔAUC 在 24h 比較難顯著。
3. **只測 XGBoost 與預設參數。** 預設參數原本是為 4h 設定的，對其他視窗不一定合適；這是刻意的取捨，目的是避免調參造成的資料探勘。
4. **AUC 不代表可交易。** 即使某視窗通過，也還沒有扣除成本，不能做獲利宣稱。

## 已確認的操作細節（2026-09-27）

六項都採用建議。

| # | 問題 | 決定 | 理由 |
| --- | --- | --- | --- |
| Q1 | Purge 列數 | 四個視窗一律 24 列 | 四個視窗的切分完全相同，比較的差異只來自視窗本身。 |
| Q2 | Bootstrap block 長度 | 四個視窗一律 48 列 | 至少是 24h 標籤重疊長度的兩倍，避免信賴區間過窄；四個視窗用同一把尺。 |
| Q3 | 深度模型 | 本輪只跑 XGBoost + 基準，全部使用 CPU；LSTM／Transformer 與 GPU 支援等複驗時再做 | 4h 階段深度模型沒有穩定勝過 XGBoost；本實驗比較的是視窗，不是模型。 |
| Q4 | 比較的參考視窗 | 以 4h 為參考，1h／12h／24h 各對 4h | 3 組比較的 Holm 懲罰比 6 組兩兩比較小，檢定力較高。 |
| Q5 | 新資料獨立確認集 | 本輪不抓新資料，照停止條件處理 | 08-09 到 09-27 只有約 1,180 列，24h 只有約 49 個不重疊區間，作為確認集太小。 |
| Q6 | 執行位置 | 從最新的 main 開新 branch `multi-horizon-comparison`，另開新 worktree | 與主資料夾其他工作分開；`soapfish` 落後 main。 |

本文件目前放在 `soapfish` worktree（`fix-usage-typo` branch），尚未 commit；照第九節步驟 0-6 搬到新 branch 後再 commit。
