# 專題交接說明

> 交接日期：2026-09-30  
> 交接方式：程式碼與文件在 GitHub（https://github.com/anbvjkannv12/Independent-Study），資料、實驗結果與凍結模型在資料壓縮檔 `專題_資料_2026-09-30.zip`（由原作者另外提供）  
> 建議閱讀順序：根目錄的 `交接使用說明.md`（怎麼開始、怎麼操作）→ 本文件（專案現況與完整清單）→ `docs/report/final_project_report.md`

---

## 一、一句話現況

研究主線已經收尾：**用 K 線衍生特徵預測 BTC／ETH／SOL／XRP 未來漲跌方向，統計上有微弱訊號（ROC-AUC 約 0.52），但遠不足以支付交易手續費。**報告圖表、報告初稿、簡報大綱都已完成；剩下的工作主要是確認繳交規定、把初稿轉成正式格式、準備口試，以及 2027-01-25 之後的新資料驗證。

---

## 二、取得專案之後要做的事

### 2.1 專案分成兩部分

| 部分 | 內容 | 取得方式 |
|---|---|---|
| **GitHub repo** | 所有程式碼、文件、報告初稿、圖表、4h 實驗程式、四幣種 4h Transformer 權重與 `.ptl`、完整 git 歷史 | `git clone` |
| **資料壓縮檔** `專題_資料_2026-09-30.zip`（約 667 MB） | `data/`（原始、補值後、特徵、外部、即時更新資料，解壓後約 1.8 GB）、`logs/`（所有實驗結果，約 210 MB）、`models/live/`（凍結與滾動模型）、`data_manifest.json`（每個檔案的 SHA-256） | 原作者另外提供（例如雲端硬碟） |

為什麼分開：`data/`、`logs/`、`models/live/` 合計約 2 GB，放進 git 會讓每次 clone 都要下載，而且這些資料已經定案、不需要版本控制，所以 `.gitignore` 排除了它們。**只 clone 程式碼的話，看不到任何實驗結果，驗證程式也跑不了**，一定要解壓資料壓縮檔。

資料壓縮檔的 SHA-256：`c4853476d46f634178c437fdc16c2b5d5251c273ea6caed01d67c06ad3927cfb`。收到後可以先用 `Get-FileHash 專題_資料_2026-09-30.zip -Algorithm SHA256` 比對，確認檔案沒有傳壞。


不需要、也不在任何地方提供的東西：

| 項目 | 原因與替代方式 |
|---|---|
| Python 虛擬環境（原電腦約 6 GB） | 綁定原電腦，請依 2.2 重新建立 |
| 離線套件快取（`deps/` 等） | 有網路就不需要 |

### 2.2 還原步驟（Windows PowerShell）

1. **clone 專案**到任何資料夾，以下假設是 `D:\專題`。路徑不要有空格：

   ```powershell
   git clone https://github.com/anbvjkannv12/Independent-Study.git D:\專題
   cd D:\專題
   ```

2. **解壓資料壓縮檔到專案根目錄**。解壓後，專案根目錄會多出 `data/`、`logs/`、`models/live/` 與 `data_manifest.json`：

   ```powershell
   Expand-Archive -Path <下載位置>\專題_資料_2026-09-30.zip -DestinationPath D:\專題
   Test-Path data\features\btc_4h.csv, logs\multi_horizon\verification.json, models\live\frozen_manifest.json
   ```

   三個都應該是 `True`。如果 `Expand-Archive` 遇到中文路徑或大檔案出錯，改用 7-Zip 解壓。

3. **安裝 Python 3.12**（原環境是 3.12.14），然後建立虛擬環境：

   ```powershell
   py -3.12 -m venv .venv-transformer
   .\.venv-transformer\Scripts\python.exe -m pip install -r requirements-lock.txt
   ```

   `requirements-lock.txt` 是原電腦的完整套件版本清單（含 torch、xgboost 3.2.0）。

4. **跑完整測試**，應該全部通過：

   ```powershell
   $py = ".\.venv-transformer\Scripts\python.exe"
   & $py -m unittest discover -s tests
   ```

5. **核對資料完整**：跑多視窗實驗的獨立驗證程式，應該顯示 V1～V8 全部 PASS。

   ```powershell
   & $py scripts\verify_multi_horizon_outputs.py
   ```

6. **（可選）核對 SHA-256**：從 `data_manifest.json` 抽幾個檔案，用 `Get-FileHash <檔案> -Algorithm SHA256` 比對。

如果第 4 或第 5 步失敗，先不要改任何程式，把錯誤訊息記下來。最常見的原因是套件版本不同、資料壓縮檔沒有解壓到專案根目錄，或解壓不完整。

### 2.3 只做前端網站的話

不需要整個專案，只要 `專題_前端資料包_2026-09-30.zip`（約 78 MB，由原作者另外提供），說明見 `docs/handover/frontend_data_guide.md`。

### 2.4 之後修改專案

- 程式碼與文件的修改：開分支、commit、push 到 GitHub，再開 Pull Request 合併到 `main`。
- **不要把 `data/`、`logs/`、`models/live/` 加進 git**（`.gitignore` 已排除）。資料有新增時，另外打包並更新 `data_manifest.json`。

---

## 三、交接清單（東西在哪裡）

以下路徑都是相對於專案根目錄（clone 下來的資料夾）。`data/`、`logs/`、`models/live/` 底下的檔案要先解壓資料壓縮檔才會有。

### 3.1 報告與結果

| 內容 | 位置 |
|---|---|
| 專題報告初稿（繳交用） | `docs/report/final_project_report.md` |
| 簡報大綱與 Q&A | `docs/report/final_presentation_outline.md` |
| 階段總報告 | `docs/report/five_year_4h_final_report.md` |
| 各實驗結果與結論 | `docs/experiments/<實驗>/`（對照見第四節研究結論總表） |
| 預先登記文件 | `docs/superpowers/specs/` |
| 報告圖 F1～F10 | `docs/figures/F*.png` |
| 圖表使用的數字 | `docs/figures/figure_data.json` |
| 圖表產生程式 | `scripts/make_report_figures.py` |

### 3.2 實驗結果原始資料（logs）

| 實驗 | 位置 |
|---|---|
| 4h 三模型比較 | `logs/*_4h_model_comparison*.json` |
| 外部特徵 A/B | `logs/*_4h_model_comparison_{base,ext}_2026-09-16.json` |
| 特徵消融 v1、v2 | `logs/ab/`、`logs/ab_v2/` |
| 多視窗比較與驗證 | `logs/multi_horizon/`（含 `verification.json`） |
| 重訓頻率 | `logs/retrain_frequency/` |
| 三段式目標 | `logs/three_class_target/` |
| 波動大小延伸研究 | `logs/volatility_target/` |
| 前向測試預測紀錄 | `logs/live/predictions.csv`、`logs/live/run_log.jsonl` |
| 最終檢查紀錄 | `logs/final_full_test_*.log`（當時 107 項測試）、`logs/final_verify_multi_horizon_*.log`（V1～V8）、`logs/final_three_class_recompute_*.log`、`logs/volatility_target_rerun_*.log` |

### 3.3 模型權重

| 模型 | 位置 | 格式 | 說明 |
|---|---|---|---|
| 4h Transformer（5 年資料，四幣種各一） | `models/<幣種>_4h_transformer/` | `.pth`（權重＋標準化參數）、`.ptl`（行動端）、`.ptl.json`（規格） | 由 `scripts/build_4h_transformer_ptl.py` 產生；說明見 `docs/handover/ptl_guide.md` |
| 凍結 4h XGBoost（四幣種） | `models/live/*_4h_frozen.json` | XGBoost JSON | **工作 8 的驗證依據，不可重訓或覆寫**；SHA-256 在 `models/live/frozen_manifest.json` |
| 滾動 4h XGBoost（四幣種） | `models/live/*_4h_rolling_*.json` ＋ `.meta.json` | XGBoost JSON | 前向測試用，meta 記錄訓練範圍 |

**注意**：4h 三模型比較（XGBoost／LSTM／Transformer 的 walk-forward）與之後的預先登記實驗，**程式設計上不儲存模型權重**，只儲存逐列預測與結果 JSON（在 `logs/`）。要取得權重必須重新訓練；結果可重現，因為 seed 固定。

### 3.4 模型程式碼

| 內容 | 位置 |
|---|---|
| 4h 模型定義（XGBoost、LSTM、Transformer） | `scripts/btc_4h_model_adapters.py` |
| walk-forward 切分、校準、評估 | `scripts/btc_4h_evaluation.py` |
| 4h 三模型比較主程式 | `scripts/train_btc_4h_model_comparison.py` |
| 4h Transformer 訓練、存檔與 `.ptl` 匯出（四幣種） | `scripts/build_4h_transformer_ptl.py` |
| 各實驗主程式 | `scripts/run_*.py`（見第五節 5.3） |
| 前向測試流程 | `scripts/live_pipeline.py` |


---

## 四、研究結論總表

| 實驗 | 日期 | 結果 | 結論文件 |
|---|---|---|---|
| 4h 三模型比較（XGBoost、LSTM、Transformer） | 08-19～09-16 | 三 fold test 平均 AUC：BTC 0.529～0.532、ETH 0.522～0.526、SOL 0.503～0.511、XRP 0.515～0.524；沒有可靠的優勝模型 | `docs/experiments/4h_model_comparison/multi_asset_4h_model_comparison.md` |
| 外部特徵 A/B（鏈上、情緒、衍生品） | 09-16 | 沒有穩定改善 | `docs/experiments/feature_tests/external_features_ab_comparison.md` |
| 特徵消融 v1、v2 | 09-25、09-26 | v2：20 個組合中 0 個通過 | `docs/experiments/feature_tests/feature_ab_test_results.md`；v2 分析文件只在原作者電腦的 `feature-ab-test-v2` 分支（未 push） |
| 多視窗比較（1h／4h／12h／24h） | 09-28 | 只有 XRP 1h 穩定強過 4h；12h、24h 沒有更好 | `docs/experiments/multi_horizon/multi_horizon_conclusions.md` |
| 重訓頻率 | 09-28 | 12 個組合中 0 個通過 | `docs/experiments/retrain_frequency/retrain_frequency_conclusions.md` |
| 三段式目標（扣 0.2% 手續費） | 09-28 | 每筆淨報酬 −0.15%～−0.22%，0 個幣種通過；統計錯誤已依 v2 勘誤修正，結論不變 | `docs/experiments/three_class_target/three_class_target_conclusions.md` |
| 波動大小預測（延伸研究） | 09-30 | 四幣種 AUC 0.64～0.68，都通過預定判準（**注意事項見第七節第 5 點**） | `docs/experiments/volatility_target/volatility_target_extension_results.md` |

**核心結論**（報告第 5 章的重點）：
1. 瓶頸在資料的資訊量，不在模型：換模型、加特徵、刪特徵都停在同一個 AUC 上限。
2. 每筆交易的預期毛利 ≈ (2 × 命中率 − 1) × 平均波動。依 F10 的數字，要打平 0.2% 的來回手續費，命中率需約 56%～63%（依幣種），但實測只有約 50%～54%；扣手續費前每筆毛利只有約 0.01%～0.05%（見圖 F10 與 `docs/figures/figure_data.json`）。
3. 「方向」很難預測，但「波動大小」相對好預測，這是延伸研究的方向。

---

## 五、檔案地圖

### 5.1 先讀這些文件

| 文件 | 用途 |
|---|---|
| `README.md` | 專案總覽、目前結果、執行方式 |
| `agement.md` | **研究範圍的權威文件**：做什麼、不做什麼 |
| `交接使用說明.md` | 拿到專案後怎麼開始、日常怎麼操作 |
| `docs/report/final_project_report.md` | 繳交用報告初稿 |
| `docs/report/final_presentation_outline.md` | 口試簡報大綱與 Q&A |
| `docs/report/five_year_4h_final_report.md` | 階段總報告（第八節有所有文件與產物的清單） |

### 5.2 資料夾

| 資料夾 | 內容 | 注意 |
|---|---|---|
| `scripts/` | 所有資料流程、實驗、驗證、圖表程式 | 主要入口見 5.3 |
| `tests/` | 單元測試 | 修改程式後一定要跑 |
| `docs/handover/` | 交接文件、前端資料說明、`.ptl` 說明、git 歷史改寫對照表 | |
| `docs/report/` | 繳交用報告初稿、簡報大綱、階段總報告、研究計畫書 | |
| `docs/experiments/` | 各實驗的結果與結論，一個實驗一個子資料夾 | |
| `docs/data_features/` | 特徵工程與外部訊號說明 | |
| `docs/superpowers/specs/` | 每個實驗的預先登記文件 | **不要修改已定案的文件**；要改就另開 v2 |
| `docs/figures/` | 報告圖 F1～F10 與 `figure_data.json` | 由 `scripts/make_report_figures.py` 產生，不要手動改圖 |
| `data/raw/`、`data/processed/`、`data/features/` | 實驗的輸入資料 | **不要覆寫**，實驗結果 JSON 裡記錄了它們的 SHA-256 |
| `data/live/` | 即時更新流程抓的新資料 | 可重新產生 |
| `logs/` | 所有實驗結果 | 報告的每個數字都來自這裡 |
| `models/live/` | 凍結模型與滾動模型 | **凍結模型永遠不可重新訓練或覆寫** |
| `run/` | 早期的執行用 notebook | |

### 5.3 主要程式

| 程式 | 用途 |
|---|---|
| `scripts/fetch_binance_5y.py` → `check_data_quality.py` → `impute_missing_klines.py` → `feature_engineering.py` | 資料流程（抓取 → 品質檢查 → 補值 → 特徵） |
| `scripts/train_btc_4h_model_comparison.py` | 4h 三模型比較 |
| `scripts/run_multi_horizon_comparison.py`、`verify_multi_horizon_outputs.py` | 多視窗實驗與獨立驗證 |
| `scripts/run_retrain_frequency_comparison.py` | 重訓頻率實驗 |
| `scripts/run_three_class_target_comparison.py` | 三段式目標實驗（`--recompute-from-predictions` 可從既有預測重算） |
| `scripts/run_volatility_target_extension.py` | 波動大小延伸研究 |
| `scripts/make_report_figures.py` | 產生報告圖 F1～F10 |
| `scripts/live_pipeline.py` | 即時更新與前向測試：`init`、`update`、`retrain`、`status` |

---

## 六、剩下的工作

截至 2026-09-30：

| 工作 | 狀態 | 要做什麼 |
|---|---|---|
| 確認繳交規定 | **未完成，最優先** | 向老師確認：繳交期限、格式、頁數、是否口試、是否需要系統展示、是否繳交程式碼。填進附錄 A 的表 |
| 報告轉成正式格式 | 未完成 | 把 `docs/report/final_project_report.md` 依系上範本轉成 Word／PDF；圖用 `docs/figures/`；數字不要手打，從結果文件或 `figure_data.json` 複製 |
| 報告內容補強 | 待確認 | 決定波動延伸研究要不要寫進報告；若要，務必同時寫第七節第 5 點的限制 |
| 口試準備 | 大綱已完成 | 依 `docs/report/final_presentation_outline.md` 做成簡報並練習 |
| 備份 | 部分完成 | 程式碼在 GitHub；資料壓縮檔請至少存兩個地方（例如雲端硬碟與外接硬碟） |
| 遠端 repo | 已設定 | https://github.com/anbvjkannv12/Independent-Study |
| 即時預測展示 | 決定不做 | 需要時見附錄 D 啟用排程 |
| `.ptl` 模型檔（PyTorch Lite） | 已完成 | 5 年資料 4h Transformer，BTC／ETH／SOL／XRP 各一；見下方 6.1 |
| 前端呈現 | 網站由同學保管 | 資料說明見 `docs/handover/frontend_data_guide.md`（6.2） |
| 新資料獨立驗證 | **2027-01-25 之後** | 見附錄 C；波動大小預測也應一起驗證 |

### 6.1 `.ptl` 模型檔（已完成，2026-09-30）

完整說明見 **[`docs/handover/ptl_guide.md`](ptl_guide.md)**。重點：

- 四個幣種都用 4h Transformer，結構與設定都和 4h 三模型比較相同（四幣種設定一致），只用 2026-08-09 06:00 UTC 以前的資料訓練。資料用 `data/features/<幣種>_4h.csv`（與預先登記實驗、凍結模型相同）。
- 檔案：`models/<幣種>_4h_transformer/<幣種>_4h_transformer.{pth,ptl,ptl.json}`，由 `scripts/build_4h_transformer_ptl.py` 產生（可用 `--symbols` 指定幣種）。
- `.ptl` 已包含標準化與 sigmoid：輸入原始特徵 `(1, 24, 53)`，輸出 4 小時後上漲分數（未校準）。
- 驗證：四個幣種都在驗證期 4,000 個視窗上比對，`.ptl` 和訓練結果最大差距 1.19e-07，batch 與單筆一致，全部 PASS（`logs/ptl_<幣種>_4h_verify.json`）。驗證期 ROC-AUC：BTC 0.540、ETH 0.532、SOL 0.500、XRP 0.526，屬微弱訊號，只能當展示。

### 6.2 前端呈現

> 前端網站由同學保管（不在本專案中）。網站需要的資料、格式與更新方式見 **[`docs/handover/frontend_data_guide.md`](frontend_data_guide.md)**，對應的精簡資料包是 `專題_前端資料包_2026-09-30.zip`。以下為原本的規劃建議。

目前專案沒有任何前端程式。建議前端只做「讀取與顯示」，不在網頁伺服器上跑模型。可以呈現的內容與資料來源：

| 區塊 | 顯示內容 | 資料來源 |
|---|---|---|
| 研究結果 | 四幣種 AUC、多視窗熱圖、三段式淨報酬、手續費損益兩平 | `docs/figures/F3、F4、F9、F10`，數字在 `docs/figures/figure_data.json` |
| 即時預測 | 各幣種最近 48 小時的上漲機率（`p_up`）與實際漲跌 | `logs/live/predictions.csv`（先執行 `live_pipeline.py update` 更新） |
| 模型狀態 | 最新資料時間、凍結模型近 30 天 AUC | `live_pipeline.py status` 的輸出 |
| 聲明 | 「本系統為研究展示，訊號微弱，不構成投資建議」 | 固定文字 |

做法建議（擇一）：
- **最簡單**：靜態 HTML 加一個 JavaScript 圖表庫，直接讀 `figure_data.json` 與 `predictions.csv`。不需要後端。
- **Python**：Streamlit，約 100 行就能做出上表內容；需要把 `streamlit` 加進 `requirements-lock.txt`。
- 若要公開上線，排程與部署注意事項見附錄 D（雲端主機要選非美國區域）。

---

## 七、一定要知道的規則與注意事項

1. **2027-01-25 以前，不可用 2026-08-09 之後的資料做任何調整。**包括調參、選特徵、改判定規則、根據 `live_pipeline.py status` 顯示的 AUC 做決定。否則新資料驗證就失去獨立性。
2. **凍結模型（`models/live/*_4h_frozen.json`）永遠不可重新訓練或覆寫。**它的 SHA-256 記在 `models/live/frozen_manifest.json`，`live_pipeline.py update` 每次都會核對，不一致就停止。
3. **預先登記的規則**：新實驗要先寫 `docs/superpowers/specs/<日期>-<名稱>-design.md` 並**單獨 commit**，之後才能執行會產生結果的程式。發現錯誤時不改原文件，另開勘誤或 v2 文件（範例：`2026-09-28-three-class-target-v2-errata.md`）。
4. **報告用詞**：未通過判定寫「沒有證據顯示……能改善」，不寫「沒用」或「無效」；不寫「準確率高達」「可獲利」。引用 4h 主結果時，使用 `README.md`「目前結果」表格的數字。
5. **波動大小延伸研究的兩個限制**（寫進報告時必須一起寫）：
   - 它的預先登記文件、程式與結果在**同一個 commit**（`08e8681`）裡，無法用 git 歷史證明「先登記、後執行」，和其他實驗不同。
   - test 期間與之前的實驗重複使用。結果很可能是真的（波動有明顯的聚集現象，本來就比方向好預測），但仍需要 2027-01-25 之後的新資料獨立驗證，才能稱為「確認」。
6. **`data/features` 的版本差異**：特徵檔由較舊版的特徵程式產生，零成交量補值列的處理和現行程式不同（影響約 40 列，都在 2021～2023 年；2026-08-02 之後完全一致）。報告的研究限制要寫，重跑實驗時**請直接用現有的 `data/features`，不要重新產生**。
7. **Binance 連線**：`live_pipeline.py` 預設用 `api.binance.com`。它會擋美國 IP；如果連不上，加上 `--base-url https://data-api.binance.vision`。
8. **不使用任何 API key**：本專案只用公開端點，不要把任何金鑰放進專案。
9. **自動排程目前沒有啟用**：原電腦上的 Windows 排程已取消。需要時才手動執行 `live_pipeline.py update`；到 2027 年執行一次，就會自動補齊所有缺少的資料與預測。

---

## 八、git 分支

GitHub 上只有 `main`，所有成果都在這裡，從這裡開始工作。原作者電腦上還有幾個舊分支（例如 `feature-ab-test-v2`，內有特徵消融 v2 的分析文件），它們指向改寫前的舊歷史，沒有 push，也不需要。

2026-09-30 曾改寫 git 歷史以移除 3 年資料階段的檔案；commit 編號因此改變，新舊編號對照見 `docs/handover/history_rewrite_commit_map.md`。

---

## 九、常見問題

**Q：測試有幾項失敗怎麼辦？**
先確認套件版本和 `requirements-lock.txt` 一致（`pip freeze` 比對）。xgboost 版本不同，會讓模型結果有細微差異。

**Q：可以重新產生 `data/features` 嗎？**
不要。見第七節第 6 點。之前所有實驗都用現有檔案，重新產生會讓結果無法對應。

**Q：想重跑某個實驗驗證數字？**
不需要重新訓練。多視窗用 `verify_multi_horizon_outputs.py`；三段式用 `run_three_class_target_comparison.py --recompute-from-predictions --multi-horizon-json logs\multi_horizon\multi_asset_multi_horizon.json`，都只讀既有結果。

**Q：文件裡有 `C:\Users\user\...` 或 `C:\Users\user\orca\workspaces\...` 的路徑？**
那是原電腦的路徑，指的是當時實驗用的工作資料夾。對應的結果都已經收進資料壓縮檔的 `logs/`，可以忽略這些路徑。

**Q：報告的圖要改樣式？**
改 `scripts/make_report_figures.py` 再重新執行，不要直接修改 PNG，也不要手打數字。

---

## 附錄 A：需要先向老師確認的繳交規定

撰寫正式報告之前，先確認以下事項並填進本表：

| 項目 | 答案 |
|---|---|
| 報告繳交期限 | （待填）原訂 2026-09-30，已到期，需確認新期限 |
| 報告格式（Word／PDF／系上範本） | （待填） |
| 頁數或字數限制 | （待填） |
| 是否需要口試或簡報，時間多長 | （待填） |
| 是否需要展示系統（即時預測、網頁） | （待填） |
| 是否需要繳交程式碼或資料 | （待填） |

## 附錄 B：報告寫作原則

1. **誠實呈現弱訊號與負結果。**引用 4h 主結果時，使用 `README.md`「目前結果」表格的數字，不得以其他章節的較好數字替代。
2. **用詞一致**：
   - 未通過判定的，寫「沒有證據顯示……能改善」，不寫「沒有用」或「無效」。
   - 通過訊號關卡但未通過比較關卡的，寫「統計上有微弱訊號，但不足以……」。
   - 不寫「準確率高達」「可獲利」「可用於交易」。
3. **每個數字都能追溯**：報告中的數字來自 `docs/experiments/` 的結果文件或 `docs/figures/figure_data.json`，並在附錄列出對應的 JSON 檔。
4. **區分三種結果**：統計上顯著、描述性觀察（不參與判定）、事後觀察（例如 best-direction persistence，只能當上限參考）。
5. **BTC「定案 Transformer」的處理**：BTC 在 2026-08-19 依三 fold 平均 AUC 最高選定 Transformer；後續更嚴格的比較顯示三模型差距在雜訊範圍內，整體結論是「沒有可靠的優勝模型」。這段歷史要保留，但不要寫得像結論。
6. **注意 F3 的數字**：`figure_data.json` 的 F3 讀的是較早的結果檔，ETH、SOL、XRP 的 AUC 和 README 官方數字不同（例如 ETH XGBoost 0.5275 vs 0.5221）。修正 `scripts/make_report_figures.py` 的資料來源之前，報告請使用 README 的數字。

## 附錄 C：新資料獨立驗證（2027-01-25 之後）

**驗證對象**：
1. 1h 候選：XRP 1h 通過，BTC、ETH 1h 接近通過。
2. BTC G4 時間特徵候選。
3. 波動大小預測（延伸研究）。
4. 凍結 4h 模型在新資料上的 AUC（確認弱訊號是否延續）。

**步驟**：
1. **補齊資料**：執行 `scripts\live_pipeline.py update`，程式會自動補抓 2026-08-09 之後的所有 K 線，並補算凍結模型的預測。
2. **核對凍結模型**：`update` 每次都會比對 `models/live/*_4h_frozen.json` 的 SHA-256 與 `models/live/frozen_manifest.json`，不一致就停止。
3. **寫預先登記文件** `docs/superpowers/specs/<日期>-new-data-validation-design.md`，**單獨 commit 後才能執行**，內容包括：
   - test 區間：2026-08-09 06:00 之後，扣掉 24 列 purge 的第一段 4,000 列。
   - 判定規則沿用原實驗，不重新調整。
   - 1h 要改成主要視窗，必須至少 2 個幣種通過「比 4h 穩定地強」。
   - 1h 必須另外報告 XGBoost 是否勝過用**訓練資料**決定方向的 persistence，排除「只是短期反轉」。
   - 註明 `data/features` 與現行特徵程式的版本差異（第七節第 6 點）。
4. **實作、測試、執行、寫結果與結論**，流程同之前的實驗（範例：`docs/experiments/retrain_frequency/`）。
5. **把結果補進專題報告**；若報告已繳交，另寫一份補充報告。

**在此之前**：不對任何候選做調參或特徵修改，也不根據 `live_pipeline.py status` 顯示的 AUC 做任何決定。

## 附錄 D：即時更新流程（排程與雲端）

`scripts/live_pipeline.py` 的四個子指令：`init`（只做一次，已在原電腦完成）、`update`（每小時）、`retrain`（每月）、`status`（檢查）。資料寫在 `data/live/`、模型在 `models/live/`、預測在 `logs/live/predictions.csv`。

**在 Windows 啟用自動排程**（目前沒有啟用；路徑請換成你的專案位置）：

```powershell
$ps1 = "D:\專題\scripts\live_run.ps1"
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 50)
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ps1`" update"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date -Hour 0 -Minute 5 -Second 0) -RepetitionInterval (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "crypto-live-hourly" -Action $action -Trigger $trigger -Settings $settings
schtasks /Create /TN "crypto-live-monthly" /SC MONTHLY /D 1 /ST 09:30 /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \`"$ps1\`" retrain" /F
Set-ScheduledTask -TaskName "crypto-live-monthly" -Settings $settings
```

- 每小時第 5 分鐘執行，讓 Binance 完成上一根 K 線；錯過的時段下次執行會自動補抓資料、補算預測，並標記 `backfilled=1`。
- 排程只在登入 Windows 時執行。
- 取消排程：`Unregister-ScheduledTask -TaskName "crypto-live-hourly","crypto-live-monthly" -Confirm:$false`。
- 每小時預測（而不是每 4 小時）是為了和離線實驗的評估方式一致。

**搬到雲端（網頁上線時）**：
1. 網頁只讀 `predictions.csv`（或改存資料庫），不在網頁伺服器上跑模型。
2. 用雲端主機的 cron 或平台排程執行同樣的 `live_pipeline.py update`／`retrain`。
3. 主機選**非美國區域**（例如新加坡、東京、法蘭克福），或加上 `--base-url https://data-api.binance.vision`；Binance 會拒絕美國 IP。
4. `models/live/` 與 `logs/live/` 放在持久化儲存，避免主機重啟後遺失。
5. 運算量很小（訓練約 5 秒、預測不到 1 秒），最便宜的方案就夠用。
