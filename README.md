# 基於機器學習與深度學習之加密貨幣走勢預測

以 Binance Spot 五年小時級 K 線，比較 XGBoost、LSTM、Transformer 對 **BTC／ETH／SOL／XRP 未來 4 小時漲跌方向**的預測能力。整個流程（抓取 → 品質檢查 → 補值 → 特徵工程 → walk-forward 驗證）皆可重現，並保留 manifest 供追溯。

- 研究範圍的權威文件：[agement.md](agement.md)　｜　進度與排程：[時程.md](時程.md)
- 本階段**不含**穩定幣脫鉤預警、市場連動分析與風險分數設計（屬後續延伸）。

## 目前結果（誠實現況）

四幣種 4 小時方向分類，3 個 expanding-window folds 的 test 平均 ROC-AUC：

| 幣種 | XGBoost | LSTM | Transformer |
| --- | ---: | ---: | ---: |
| BTCUSDT | 0.5289 | 0.5302 | **0.5318** |
| ETHUSDT | 0.5221 | 0.5232 | **0.5257** |
| SOLUSDT | 0.5034 | 0.5089 | **0.5106** |
| XRPUSDT | 0.5149 | **0.5236** | 0.5162 |

**訊號強度尚不足以支撐任何交易或風險預警宣稱**：SOL 近乎隨機，BTC／ETH 也僅約 0.53，深度模型未穩定勝過 XGBoost。引用本專題結果時請一律使用上述數字，勿以其他章節績效替代。詳見 [docs/multi_asset_4h_model_comparison.md](docs/multi_asset_4h_model_comparison.md)。

BTC 已定案為 Transformer，不再重新訓練；ETH／SOL／XRP 的 4 小時三模型重跑與定案檢視已完成，結論皆偏向「無可靠優勝模型」。四幣種總報告、多視窗比較（僅 XRP 1h 通過，列為候選假說）、重訓頻率實驗（無任何組通過）與三段式目標實驗（四幣種扣手續費後皆為負報酬，無幣種通過）皆已完成，詳見 [docs/five_year_4h_final_report.md](docs/five_year_4h_final_report.md)。

## 持續更新資料與前向測試（程式已完成，排程尚未啟用）

完整執行步驟見 [docs/next_phase_work_plan.md](docs/next_phase_work_plan.md)（工作 A：三段式目標；工作 B：本節流程）。目標是讓資料與模型自動跟上新行情，同時累積 2026-08-09 之後、模型從未看過的預測紀錄，供新資料獨立驗證（最早 2027-01-25）使用。

1. **每小時**：增量抓取新 K 線 → 品質檢查與補值 → 特徵工程 → 以凍結模型（只用 2026-08-09 前資料訓練，供獨立驗證）與每月滾動模型各預測一次，寫入 `logs/live/predictions.csv`；標籤到期後回填真實漲跌。每小時預測才和離線實驗的評估方式一致。
2. **每月重新訓練一次**：expanding 訓練窗、24 列 purge、預設 XGBoost。重訓頻率實驗顯示更頻繁重訓沒有穩定改善，故取成本較低的頻率。
3. **排程**：本地用 Windows 工作排程器；日後架網頁時，排程搬到雲端（選非美國區域，Binance 會擋美國 IP），網頁只讀預測結果、不跑模型。
4. **獨立性規則**：2027-01-25 前只收資料與記錄預測，**不得**用新資料調參、換特徵或改判定規則。
5. **1h 候選的檢查重點**：XGBoost 是否勝過反向 persistence（排除只是短期反轉）、扣手續費後是否仍有優勢。

實作在 `scripts/live_pipeline.py`（資料寫入 `data/live/`、模型 `models/live/`、預測與執行紀錄 `logs/live/`，皆不進 git；不會改動 `data/raw`、`data/processed`、`data/features`）：

```powershell
& ".venv-transformer\Scripts\python.exe" scripts\live_pipeline.py init     # 只做一次：複製歷史資料、訓練凍結模型與第一個滾動模型
& ".venv-transformer\Scripts\python.exe" scripts\live_pipeline.py update   # 每小時：抓新 K 線、補值、特徵一致性檢查、預測、回填標籤
& ".venv-transformer\Scripts\python.exe" scripts\live_pipeline.py retrain  # 每月：訓練新的滾動模型
& ".venv-transformer\Scripts\python.exe" scripts\live_pipeline.py status   # 檢查資料新鮮度與凍結模型近 30 天 AUC（僅描述）
```

目前未設定自動排程，需要時手動執行上述指令；日後啟用時由 Windows 工作排程器呼叫 `scripts\live_run.ps1 update`（每小時）與 `scripts\live_run.ps1 retrain`（每月 1 日）。Binance 端點被擋時加 `--base-url https://data-api.binance.vision`。

## 快速開始

Python 環境（含 torch）固定使用 `.venv-transformer`：

```powershell
& ".venv-transformer\Scripts\python.exe" -m pip install -r requirements-model-comparison.txt
```

依序執行資料流程（全部從專案根目錄執行）：

```powershell
& ".venv-transformer\Scripts\python.exe" scripts\fetch_binance_5y.py --dry-run --config config.json
& ".venv-transformer\Scripts\python.exe" scripts\fetch_binance_5y.py --config config.json
& ".venv-transformer\Scripts\python.exe" scripts\check_data_quality.py --config config.json --manifest logs\fetch_manifest.json --report logs\data_quality_report.json
& ".venv-transformer\Scripts\python.exe" scripts\impute_missing_klines.py --config config.json
& ".venv-transformer\Scripts\python.exe" scripts\feature_engineering.py
& ".venv-transformer\Scripts\python.exe" scripts\train_btc_4h_model_comparison.py --symbol BTC --data data\features\btc_4h.csv --output logs\btc_4h_model_comparison.json
```

跑測試：

```powershell
& ".venv-transformer\Scripts\python.exe" -m unittest discover -s tests -v
```

## 流程說明

| 階段 | 程式 | 輸入 → 輸出 | 紀錄 |
| --- | --- | --- | --- |
| 1. 抓取 | `scripts/fetch_binance_5y.py` | Binance API → `data/raw/*_5y.csv` | `logs/fetch_manifest.json` |
| 2. 品質檢查 | `scripts/check_data_quality.py` | `data/raw/` | `logs/data_quality_report.json` |
| 3. 補值 | `scripts/impute_missing_klines.py` | `data/raw/` → `data/processed/` | `logs/imputation_manifest.json` |
| 4. 特徵工程 | `scripts/feature_engineering.py` | `data/processed/` → `data/features/` | `logs/feature_manifest.json` |
| 5. Baseline | `scripts/train_btc_4h_baseline.py` | `data/features/btc_4h.csv` | `docs/btc_4h_baseline.md` |
| 6. 三模型比較 | `scripts/train_btc_4h_model_comparison.py`（`--symbol BTC/ETH/SOL/XRP`） | `data/features/*_4h.csv` | `logs/*_4h_model_comparison.json` |
| 選配 A. 外部訊號抓取 | `scripts/fetch_external_signals.py` | 公開 API → `data/external/` | `logs/external_manifest.json` |
| 選配 B. 外部特徵合併 | `scripts/feature_engineering.py --external-dir data\external` | `data/processed/` + `data/external/` → `data/features_ext/` | `logs/feature_ext_manifest*.json` |

設定集中在 [config.json](config.json)：四個交易對、`1h` 週期、5 年、分頁 1000 筆、重試 5 次。

**資料契約**：53 個特徵欄位（順序固定），標籤 `future_log_return = log(close[t+h] / close[t])`、`target_up = future_log_return > 0`，支援 `1h`／`4h`／`12h`／`24h`。`is_imputed` 只是補值追蹤欄，**不是模型特徵**。缺口佔比約 0.016%，補值規則見 [agement.md](agement.md)。

**外部訊號（選配）**：鏈上指標（交易所流入流出、活躍地址數、MVRV）、市場情緒（Fear & Greed，可另接 FinBERT／Twitter-RoBERTa 語料）與衍生品（資金費率、未平倉量、多空比）已可接入，全部使用免金鑰公開來源，日頻資料加 1 天發布延遲、一律 backward 對齊。SOL 無鏈上資料、XRP 無交易所流量、未平倉量封存檔起點較晚——限制與指令見 [docs/external_signals.md](docs/external_signals.md)。目前尚未有證據顯示這些因子能改善上表 ROC-AUC。

walk-forward 與校準邏輯放在 `scripts/btc_4h_evaluation.py`、模型介面在 `scripts/btc_4h_model_adapters.py`（兩者為模組，由上表第 6 步呼叫）。

**防洩漏**：時間順序切分、expanding-window walk-forward；Platt 校準與門檻選擇只用 validation 期，test 完全不參與。基準組包含永遠上漲、訓練集多數類別、前一期方向、Logistic Regression、零交易。

## 目錄結構

```
config.json          抓取設定
agement.md           研究範圍（權威文件）
時程.md               階段排程與進度
scripts/             資料管線與訓練程式
src/                 沿用的訓練程式與實驗工具（LSTM／Transformer／multi-horizon）
tests/               單元測試，對應 scripts/ 與 src/
run/                 實際執行的 notebook（見 run/README.md）
data/raw/            原始 K 線（*_5y.csv，另含舊專題 *_3y.csv）
data/processed/      補值後的連續資料
data/features/       4 幣種 × 4 horizon 共 16 份特徵集
data/external/       鏈上／情緒／衍生品原始序列（選配）
data/features_ext/   特徵集 + 外部訊號欄位（選配）
models/              模型、指標、validation report 與 manifest
logs/                下載、品質、補值、特徵與比較結果 JSON
docs/                方法論、各幣種比較文件與專題報告
deps/                離線 wheel 與本地 XGBoost runtime
舊專題_3年資料/        前一階段（3 年資料）的 notebook
```

## 文件索引

| 文件 | 內容 |
| --- | --- |
| [docs/feature_engineering.md](docs/feature_engineering.md) | 特徵與標籤定義、執行指令、輸出格式 |
| [docs/btc_4h_baseline.md](docs/btc_4h_baseline.md)・[review](docs/btc_4h_baseline_review_2026-08-19.md) | BTC baseline 與評估檢討 |
| [docs/multi_asset_4h_model_comparison.md](docs/multi_asset_4h_model_comparison.md) | 四幣種彙整比較（主要結果） |
| [docs/five_year_4h_final_report.md](docs/five_year_4h_final_report.md) | 五年資料 4 小時方向分類階段報告 |
| [docs/btc_4h_model_comparison.md](docs/btc_4h_model_comparison.md)・[eth](docs/eth_4h_model_comparison.md)・[sol](docs/sol_4h_model_comparison.md)・[xrp](docs/xrp_4h_model_comparison.md) | 各幣種三模型比較 |
| [docs/external_signals.md](docs/external_signals.md) | 鏈上／情緒／衍生品外部訊號來源、特徵、防洩漏規則與限制 |
| [docs/整合說明_舊專題參考.md](docs/整合說明_舊專題參考.md) | 新舊專題關係、哪些舊成果可沿用 |
| [專題研究方法論審查報告.md](專題研究方法論審查報告.md) | 方法論審查 |
| `docs/7_...docx` | 研究計畫書（長期願景，範圍以 `agement.md` 為準） |

## 舊專題（3 年資料）

同一研究的前一階段（2026-06～08），以 3 年資料跑過 XGBoost／LSTM／Transformer 在 1h／4h／12h／24h 的分類與回歸實驗。Notebook 全數集中在 `舊專題_3年資料/notebooks/`（含 `調整後(預測4,12,24小時)/` 多 horizon 比較），結論文件與訓練程式已併入 `docs/`、`src/`。兩階段結論方向一致，可作為報告的前期成果對照——詳見 [docs/整合說明_舊專題參考.md](docs/整合說明_舊專題參考.md)。

## 安全說明

只使用公開端點（Binance K 線／資金費率與封存檔、Coin Metrics community、alternative.me），**不使用**、不讀取也不保存 API key 或 secret。先前外流的 API key 請在 Binance 撤銷重建，且不要放進本專案。
