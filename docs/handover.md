# 專題交接說明

> 交接日期：2026-09-30  
> 交接內容：整包壓縮檔 `專題_交接_2026-09-30.zip`（程式碼、git 歷史、資料、實驗結果、模型）  
> 建議閱讀順序：本文件 → `docs/final_stage_completion_status.md` → `docs/final_stage_work_plan.md` → `docs/final_project_report.md`

---

## 一、一句話現況

研究主線已經收尾：**用 K 線衍生特徵預測 BTC／ETH／SOL／XRP 未來漲跌方向，統計上有微弱訊號（ROC-AUC 約 0.52），但遠不足以支付交易手續費。**報告圖表、報告初稿、簡報大綱都已完成；剩下的工作主要是確認繳交規定、把初稿轉成正式格式、準備口試，以及 2027-01-25 之後的新資料驗證。

---

## 二、拿到壓縮檔之後要做的事

### 2.1 壓縮檔內容

| 包含 | 說明 |
|---|---|
| 所有程式碼與文件 | 和 git 的 `main` 分支一致 |
| `.git/` | 完整 git 歷史與所有分支 |
| `data/` | 原始、補值後、特徵、外部資料、即時更新資料（約 1.8 GB，解壓後） |
| `logs/` | 所有實驗結果 JSON、逐列預測、驗證紀錄（約 210 MB） |
| `models/` | 模型，包含 `models/live/` 的凍結模型與 `frozen_manifest.json` |
| `backup_manifest.json` | `data/features`、`logs`、`models` 每個檔案的 SHA-256 |

| **不包含** | 原因與替代方式 |
|---|---|
| `.venv-transformer/`（約 6 GB） | Python 環境綁定原電腦，請依 2.2 重新建立 |
| `deps/`、`.xgb_deps/`、`.codex_deps/` | 離線安裝用的套件快取，有網路就不需要 |
| `.claude/` | 原電腦的工具設定與暫存 worktree |

### 2.2 還原步驟（Windows PowerShell）

1. **解壓縮**到任何資料夾，以下假設是 `D:\專題`。路徑可以有中文，但不要有空格。
2. **安裝 Python 3.12**（原環境是 3.12.14），然後建立虛擬環境：

   ```powershell
   cd D:\專題
   py -3.12 -m venv .venv-transformer
   .\.venv-transformer\Scripts\python.exe -m pip install -r requirements-lock.txt
   ```

   `requirements-lock.txt` 是原電腦的完整套件版本清單（含 torch、xgboost 3.2.0）。

3. **清掉原電腦的 worktree 紀錄**（它們指向原電腦的路徑，在你的電腦上不存在）：

   ```powershell
   git worktree prune
   git status
   ```

   `git status` 應該顯示 working tree clean。

4. **跑完整測試**，應該全部通過（原電腦為 107 項）：

   ```powershell
   $py = ".\.venv-transformer\Scripts\python.exe"
   & $py -m unittest discover -s tests
   ```

5. **核對資料完整**：跑多視窗實驗的獨立驗證程式，應該顯示 V1～V8 全部 PASS。

   ```powershell
   & $py scripts\verify_multi_horizon_outputs.py
   ```

6. **（可選）核對 SHA-256**：從 `backup_manifest.json` 抽幾個檔案，用 `Get-FileHash <檔案> -Algorithm SHA256` 比對。

如果第 4 或第 5 步失敗，先不要改任何程式，把錯誤訊息記下來。最常見的原因是套件版本不同，或解壓縮不完整。

---

## 三、交接清單（東西在哪裡）

壓縮檔內的 `專題/` 資料夾就是原電腦的 `C:\Users\user\專題`。以下路徑都是相對於它。

### 3.1 報告與結果

| 內容 | 位置 |
|---|---|
| 專題報告初稿（繳交用） | `docs/final_project_report.md` |
| 簡報大綱與 Q&A | `docs/final_presentation_outline.md` |
| 階段總報告 | `docs/five_year_4h_final_report.md` |
| 最後階段工作計畫與完成狀態 | `docs/final_stage_work_plan.md`、`docs/final_stage_completion_status.md` |
| 各實驗結果與結論 | `docs/*_results.md`、`docs/*_conclusions.md`、`docs/*_comparison.md`（對照見第四節研究結論總表） |
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
| 最終檢查紀錄 | `logs/final_full_test_*.log`（107 項測試）、`logs/final_verify_multi_horizon_*.log`（V1～V8）、`logs/final_three_class_recompute_*.log`、`logs/volatility_target_rerun_*.log` |

### 3.3 模型權重

| 模型 | 位置 | 格式 | 說明 |
|---|---|---|---|
| BTC 24h PyTorch Transformer | `models/btc_24h_pytorch_transformer/checkpoints/` | `.pth`（權重）＋ `.pkl`（前處理設定） | 5 個 fold 與 final，各一組；`smoke_test/` 是測試用小模型。載入方式見 `src/pytorch_transformer_artifacts.py` |
| 凍結 4h XGBoost（四幣種） | `models/live/*_4h_frozen.json` | XGBoost JSON | **工作 8 的驗證依據，不可重訓或覆寫**；SHA-256 在 `models/live/frozen_manifest.json` |
| 滾動 4h XGBoost（四幣種） | `models/live/*_4h_rolling_*.json` ＋ `.meta.json` | XGBoost JSON | 前向測試用，meta 記錄訓練範圍 |
| 熱門幣模型（舊專題流程） | `models/popular_coins/` | `.npz` ＋ metrics JSON | 由 `src/train_popular_coins.py` 產生 |
| 3 年資料階段模型與結果 | `models/week3_naive_xgboost/`、`models/week4_lstm/`、`models/week5_transformer/`、`models/調整後(預測4,12,24小時)/` | 多為結果 CSV／JSON | 歷史參考 |

**注意**：4h 三模型比較（XGBoost／LSTM／Transformer 的 walk-forward）與之後的預先登記實驗，**程式設計上不儲存模型權重**，只儲存逐列預測與結果 JSON（在 `logs/`）。要取得權重必須重新訓練；結果可重現，因為 seed 固定。

### 3.4 模型程式碼

| 內容 | 位置 |
|---|---|
| 4h 模型定義（XGBoost、LSTM、Transformer） | `scripts/btc_4h_model_adapters.py` |
| walk-forward 切分、校準、評估 | `scripts/btc_4h_evaluation.py` |
| 4h 三模型比較主程式 | `scripts/train_btc_4h_model_comparison.py` |
| BTC 24h PyTorch Transformer 訓練與權重存取 | `src/train_btc_24h_pytorch_transformer.py`、`src/pytorch_transformer_artifacts.py` |
| 3 年資料階段的訓練程式 | `src/` 其他檔案 |
| 各實驗主程式 | `scripts/run_*.py`（見第五節 5.3） |
| 前向測試流程 | `scripts/live_pipeline.py` |


---

## 四、研究結論總表

| 實驗 | 日期 | 結果 | 結論文件 |
|---|---|---|---|
| 4h 三模型比較（XGBoost、LSTM、Transformer） | 08-19～09-16 | 三 fold test 平均 AUC：BTC 0.529～0.532、ETH 0.522～0.526、SOL 0.503～0.511、XRP 0.515～0.524；沒有可靠的優勝模型 | `docs/multi_asset_4h_model_comparison.md` |
| 外部特徵 A/B（鏈上、情緒、衍生品） | 09-16 | 沒有穩定改善 | `docs/external_features_ab_comparison.md` |
| 特徵消融 v1、v2 | 09-25、09-26 | v2：20 個組合中 0 個通過 | `docs/feature_ab_test_results.md`；v2 分析在 `feature-ab-test-v2` 分支 |
| 多視窗比較（1h／4h／12h／24h） | 09-28 | 只有 XRP 1h 穩定強過 4h；12h、24h 沒有更好 | `docs/multi_horizon_conclusions.md` |
| 重訓頻率 | 09-28 | 12 個組合中 0 個通過 | `docs/retrain_frequency_conclusions.md` |
| 三段式目標（扣 0.2% 手續費） | 09-28 | 每筆淨報酬 −0.15%～−0.22%，0 個幣種通過；統計錯誤已依 v2 勘誤修正，結論不變 | `docs/three_class_target_conclusions.md` |
| 波動大小預測（延伸研究） | 09-30 | 四幣種 AUC 0.64～0.68，都通過預定判準（**注意事項見第七節第 5 點**） | `docs/volatility_target_extension_results.md` |

**核心結論**（報告第 5 章的重點）：
1. 瓶頸在資料的資訊量，不在模型：換模型、加特徵、刪特徵都停在同一個 AUC 上限。
2. 每筆交易的預期毛利 ≈ (2 × 命中率 − 1) × 平均波動。命中率約 53%、4h 平均波動約 0.5% 時，毛利只有約 0.03%，遠小於 0.2% 的手續費；要打平，命中率約需 65～70%（見圖 F10）。
3. 「方向」很難預測，但「波動大小」相對好預測，這是延伸研究的方向。

---

## 五、檔案地圖

### 5.1 先讀這些文件

| 文件 | 用途 |
|---|---|
| `README.md` | 專案總覽、目前結果、執行方式 |
| `agement.md` | **研究範圍的權威文件**：做什麼、不做什麼 |
| `時程.md` | 各階段進度 |
| `docs/final_stage_completion_status.md` | 最後階段哪些完成、哪些還沒 |
| `docs/final_stage_work_plan.md` | 最後階段每項工作的詳細說明 |
| `docs/final_project_report.md` | 繳交用報告初稿 |
| `docs/final_presentation_outline.md` | 口試簡報大綱與 Q&A |
| `docs/five_year_4h_final_report.md` | 階段總報告（第八節有所有文件與產物的清單） |

### 5.2 資料夾

| 資料夾 | 內容 | 注意 |
|---|---|---|
| `scripts/` | 所有資料流程、實驗、驗證、圖表程式 | 主要入口見 4.3 |
| `tests/` | 單元測試 | 修改程式後一定要跑 |
| `docs/` | 所有報告與說明 | |
| `docs/superpowers/specs/` | 每個實驗的預先登記文件 | **不要修改已定案的文件**；要改就另開 v2 |
| `docs/figures/` | 報告圖 F1～F10 與 `figure_data.json` | 由 `scripts/make_report_figures.py` 產生，不要手動改圖 |
| `data/raw/`、`data/processed/`、`data/features/` | 實驗的輸入資料 | **不要覆寫**，實驗結果 JSON 裡記錄了它們的 SHA-256 |
| `data/live/` | 即時更新流程抓的新資料 | 可重新產生 |
| `logs/` | 所有實驗結果 | 報告的每個數字都來自這裡 |
| `models/live/` | 凍結模型與滾動模型 | **凍結模型永遠不可重新訓練或覆寫** |
| `run/` | 早期的執行用 notebook | |
| `舊專題_3年資料/` | 前一階段（3 年資料）的 notebook | 歷史參考 |

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

依 `docs/final_stage_completion_status.md`：

| 工作 | 狀態 | 要做什麼 |
|---|---|---|
| 確認繳交規定 | **未完成，最優先** | 向老師確認：繳交期限、格式、頁數、是否口試、是否需要系統展示、是否繳交程式碼。填進 `final_stage_work_plan.md` 第零節的表 |
| 報告轉成正式格式 | 未完成 | 把 `docs/final_project_report.md` 依系上範本轉成 Word／PDF；圖用 `docs/figures/`；數字不要手打，從結果文件或 `figure_data.json` 複製 |
| 報告內容補強 | 待確認 | 決定波動延伸研究要不要寫進報告；若要，務必同時寫第七節第 5 點的限制 |
| 口試準備 | 大綱已完成 | 依 `docs/final_presentation_outline.md` 做成簡報並練習 |
| 備份 | 部分完成 | 壓縮檔本身就是一份備份；請再存一份到雲端硬碟 |
| 遠端 repo | 未設定 | 若要用 GitHub，建 **private** repo 後 push `main`（push 前用 `git grep -n -i "api_key\|secret\|password"` 確認沒有金鑰） |
| 即時預測展示 | 決定不做 | 需要時見 `final_stage_work_plan.md` 工作 7 |
| `.ptl` 模型檔（PyTorch Lite） | 已完成 | 見下方 6.1；`max|ptl - pth|=5.96e-08`，batch_vs_single=0 |
| 前端呈現 | **尚未製作** | 見下方 6.2 |
| 新資料獨立驗證 | **2027-01-25 之後** | 見 `final_stage_work_plan.md` 工作 8；波動大小預測也應一起驗證 |

### 6.1 `.ptl` 模型檔（已完成，2026-09-30）

完整步驟見 **[`docs/ptl_guide.md`](ptl_guide.md)**（匯出程式、驗證程式、Python／Android／iOS 執行方式、輸入資料準備、常見問題）。重點：

- 只有 BTC 24h Transformer（`models/btc_24h_pytorch_transformer/checkpoints/btc_24h_final.pth`）可以轉；它是 3 年資料階段的模型，不是 5 年 4h 主實驗的模型。
- 匯出的 `.ptl` 會把標準化與 sigmoid 包進去：輸入原始特徵 `(1, 24, 53)`，輸出上漲機率。
- 已產生 `models/btc_24h_pytorch_transformer/checkpoints/btc_24h_final.ptl` 與 `btc_24h_final.ptl.json`。
- 驗證結果：`windows=200 max|ptl - pth|=5.96e-08 batch_vs_single=0.00e+00`，`PASS`。紀錄在 `logs/ptl_export_*.log` 與 `logs/ptl_verify_*.log`。
- 由於目前 Windows torch build 沒有 XNNPACK，`optimize_for_mobile` 已自動跳過；模型仍可由 lite interpreter 載入並通過一致性驗證。

### 6.2 前端呈現（尚未製作）

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
- 若要公開上線，排程與部署注意事項見 `docs/next_phase_work_plan.md` B.5（雲端主機要選非美國區域）。

---

## 七、一定要知道的規則與注意事項

1. **2027-01-25 以前，不可用 2026-08-09 之後的資料做任何調整。**包括調參、選特徵、改判定規則、根據 `live_pipeline.py status` 顯示的 AUC 做決定。否則新資料驗證就失去獨立性。
2. **凍結模型（`models/live/*_4h_frozen.json`）永遠不可重新訓練或覆寫。**它的 SHA-256 記在 `models/live/frozen_manifest.json`，`live_pipeline.py update` 每次都會核對，不一致就停止。
3. **預先登記的規則**：新實驗要先寫 `docs/superpowers/specs/<日期>-<名稱>-design.md` 並**單獨 commit**，之後才能執行會產生結果的程式。發現錯誤時不改原文件，另開勘誤或 v2 文件（範例：`2026-09-28-three-class-target-v2-errata.md`）。
4. **報告用詞**：未通過判定寫「沒有證據顯示……能改善」，不寫「沒用」或「無效」；不寫「準確率高達」「可獲利」。引用 4h 主結果時，使用 `README.md`「目前結果」表格的數字。
5. **波動大小延伸研究的兩個限制**（寫進報告時必須一起寫）：
   - 它的預先登記文件、程式與結果在**同一個 commit**（`76ae7c4`）裡，無法用 git 歷史證明「先登記、後執行」，和其他實驗不同。
   - test 期間與之前的實驗重複使用。結果很可能是真的（波動有明顯的聚集現象，本來就比方向好預測），但仍需要 2027-01-25 之後的新資料獨立驗證，才能稱為「確認」。
6. **`data/features` 的版本差異**：特徵檔由較舊版的特徵程式產生，零成交量補值列的處理和現行程式不同（影響約 40 列，都在 2021～2023 年；2026-08-02 之後完全一致）。報告的研究限制要寫，重跑實驗時**請直接用現有的 `data/features`，不要重新產生**。
7. **Binance 連線**：`live_pipeline.py` 預設用 `api.binance.com`。它會擋美國 IP；如果連不上，加上 `--base-url https://data-api.binance.vision`。
8. **不使用任何 API key**：本專案只用公開端點，不要把任何金鑰放進專案。
9. **自動排程目前沒有啟用**：原電腦上的 Windows 排程已取消。需要時才手動執行 `live_pipeline.py update`；到 2027 年執行一次，就會自動補齊所有缺少的資料與預測。

---

## 八、git 分支

| 分支 | 狀態 |
|---|---|
| `main` | 所有成果都在這裡，從這裡開始工作 |
| `feature-ab-test-v2` | **未合併**，內有特徵消融 v2 的結果與分析文件（`docs/feature_ab_test_results_v2.md`、`docs/feature_ab_test_v2_analysis.md`）；v2 的 logs 已收回 main 的 `logs/ab_v2/` |
| `claude/ab-test-planning-81e7f5` | **未合併**，早期的 A/B 設計草稿，內容已被後續文件取代 |
| 其他分支 | 都已合併進 `main`，可以刪除（`git branch -d <名稱>`） |

---

## 九、常見問題

**Q：測試有幾項失敗怎麼辦？**
先確認套件版本和 `requirements-lock.txt` 一致（`pip freeze` 比對）。xgboost 版本不同，會讓模型結果有細微差異。

**Q：可以重新產生 `data/features` 嗎？**
不要。見第七節第 6 點。之前所有實驗都用現有檔案，重新產生會讓結果無法對應。

**Q：想重跑某個實驗驗證數字？**
不需要重新訓練。多視窗用 `verify_multi_horizon_outputs.py`；三段式用 `run_three_class_target_comparison.py --recompute-from-predictions --multi-horizon-json logs\multi_horizon\multi_asset_multi_horizon.json`，都只讀既有結果。

**Q：文件裡有 `C:\Users\user\...` 或 `C:\Users\user\orca\workspaces\...` 的路徑？**
那是原電腦的路徑，指的是當時實驗用的工作資料夾。對應的結果都已經收進壓縮檔的 `logs/`，可以忽略這些路徑。

**Q：報告的圖要改樣式？**
改 `scripts/make_report_figures.py` 再重新執行，不要直接修改 PNG，也不要手打數字。
