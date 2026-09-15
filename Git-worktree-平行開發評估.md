# Git worktree 與平行開發評估

> 盤點日期：2026-08-11  
> 專案位置：`C:\Users\user\專題 new (五年)`  
> 本文件根據唯讀盤點結果整理；建立當下未建立 worktree、分支，也未修改既有檔案。

## 一、目前專案進度

資料準備主線已完成到「可開始建模」的階段：

- 已取得 BTC、ETH、SOL、XRP 四種幣別的五年期、每小時資料。
- 已完成資料品質檢查：四個幣別沒有結構性錯誤，但皆有缺失時段警告。
- 已完成缺值補齊：共補齊 28 筆資料。
- 已完成特徵工程：每筆資料包含 53 個特徵，並產生 1、4、12、24 小時四種預測視窗，共 16 份特徵資料集。
- 合理的下一階段是建立 XGBoost、LSTM、Transformer 基準模型，之後進行 walk-forward 驗證與模型整合。

### 現有產物

| 項目 | 狀態 |
| --- | --- |
| 原始資料 | 4 份 CSV |
| 處理後資料 | 4 份 CSV |
| 特徵資料 | 16 份 CSV |
| 品質報告 | 無 errors、4 項 missing-interval warnings |
| 測試程式 | 27 個單元測試案例（本次未執行） |

## 二、Git 現況與 worktree 阻礙

目前尚**不適合直接開始平行 worktree 開發**，原因如下：

- 目前分支為 `master`，但 Git 歷史是 **0 commits**。
- `.gitignore` 是唯一已暫存檔案；程式、測試、文件、資料與執行產物都仍是未追蹤檔案。
- 現有 linked worktree：`.worktrees/codex-initial-setup`，但其中只有 Git 指標、沒有專案檔案。這是尚未有初始提交時，worktree 無法取得專案內容的直接結果。
- `.gitignore` 目前只忽略 `.worktrees/`；資料、報告與 Python 快取都會被 Git 視為待加入內容。
- `data/` 約 772.3 MB，其中 `data/features/` 約 727.4 MB。因此必須先決定大型資料是否要提交到 Git，或採取外部保存／下載重建的做法。
- 尚未設定 Git 遠端，也沒有 stash。

此外，專案中另有 `C:\Users\user\專題`，它是另一個同樣尚未提交的 Git 專案；本評估以名稱明確對應的 `專題 new (五年)` 為準。

## 三、最適合的起點

在建立任何新 worktree 前，先完成一次「安全的專案基線整理」：

1. 確認原始資料、處理後資料、特徵資料與 logs 的版本管理策略。
2. 補齊 `.gitignore`，至少避免追蹤 `__pycache__/`、虛擬環境與不應納入 Git 的可再生產物。
3. 將應保留的程式、測試、設定與文件建立第一筆可還原的提交。
4. 確認或修正既有空白的 `codex/initial-setup` worktree。
5. 固定資料切分、評估指標與結果輸出格式。

完成後，最適合平行處理的是模型實作；因為不同模型可各自放在獨立檔案與 worktree 中，但使用共同且已固定的評估規格。

## 四、可安全平行的任務

以下任務以前述基線整理和共用評估規格完成為前提。

| 任務 | 目標 | 預計修改／新增檔案或模組 | 相依關係 | 驗收方式 |
| --- | --- | --- | --- | --- |
| XGBoost 基準模型 | 建立可比較的傳統模型 | 建議新增 `models/xgboost_baseline.py`、對應測試、模型說明 | 固定資料切分與指標 | 可輸出預測、評估指標與可重現設定 |
| LSTM 基準模型 | 建立序列模型 | 建議新增 `models/lstm_baseline.py`、對應測試、設定檔 | 同上；可與 XGBoost 平行 | 使用相同資料切分與輸出格式 |
| Transformer 基準模型 | 建立 Transformer 序列模型 | 建議新增 `models/transformer_baseline.py`、對應測試、設定檔 | 同上；可與其他模型平行 | 可重現訓練，且輸出共同指標 |
| 穩定幣脫鉤風險模組 | 對 USDC/USDT、TUSD/USDT 建立異常與風險分數流程 | 建議新增獨立的 `configs/`、`scripts/`、`models/` 子模組 | 應採獨立設定與資料路徑 | 可產生風險分數、事件警示與自動化測試 |
| 資料字典與方法文件 | 補齊資料來源、特徵定義與重現步驟 | 新建專屬文件，避免多人同改根目錄 README | 可先與模型平行；結果章節須待模型完成 | 第三人可依文件理解與重建流程 |

## 五、不適合平行的工作

| 工作 | 原因 |
| --- | --- |
| 初始提交與 `.gitignore` 策略 | 所有後續分支都依賴此基線；同時修改容易造成遺漏或大型資料誤入 Git。 |
| 資料抓取、品質檢查、補值、特徵工程 | 這是有順序的資料流水線，且共用 `config.json`、資料夾與 manifest。 |
| 共用切分與評估規格 | 若不同模型各自決定資料切法，結果無法公平比較。 |
| Walk-forward 驗證、stacking、最終比較 | 需要等各模型依同一格式完成輸出後才能進行。 |
| 多人同時修改 `README.md`、`agement.md`、`config.json` | 檔案雖少但共用度高，合併衝突機率高。 |

## 六、建議命名方式

### 分支

- `codex/chore-repo-baseline`
- `codex/feat-evaluation-contract`
- `codex/feat-xgboost-baseline`
- `codex/feat-lstm-baseline`
- `codex/feat-transformer-baseline`
- `codex/feat-stablecoin-depeg-risk`
- `codex/fix-<具體問題>`

### Worktree 目錄

- `.worktrees/codex-chore-repo-baseline`
- `.worktrees/codex-feat-xgboost-baseline`
- `.worktrees/codex-feat-lstm-baseline`
- `.worktrees/codex-feat-transformer-baseline`

`.worktrees/` 已被忽略。既有的 `codex/initial-setup` worktree 暫時不應刪除；待初始提交完成後，再決定修復或重建。

## 七、建議執行順序

1. 進行基線整理：確認資料管理策略、補齊忽略規則，建立第一筆提交。
2. 確認既有 worktree 可以取得基線內容。
3. 建立 `codex/feat-evaluation-contract`，固定資料切分、指標與結果輸出格式。
4. 同時建立 XGBoost、LSTM、Transformer 三個獨立 worktree 進行模型實作。
5. 模型結果齊備後，以單一分支進行 walk-forward 驗證、比較與 stacking。
6. 穩定幣風險模組可在模型實作期間平行進行，但必須使用獨立設定與資料路徑。

## 八、需先確認的事項

- 約 772 MB 的資料檔案是否要放入 Git？若要放入，應先確認遠端平台的檔案大小限制；若不放入，需定義可重建或外部保存方式。
- 品質檢查與特徵工程的計畫文件仍顯示未勾選，但其程式、測試與產物已存在；建議基線整理時同步校正文件進度。
- `data_quality_report.json` 有 4 項缺失時段警告；補值流程已處理對應缺口，但建模前應確認是否接受「補值資料不進特徵」的現有策略。
