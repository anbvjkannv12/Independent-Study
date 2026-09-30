# 基於機器學習與深度學習之加密貨幣走勢預測

以 Binance Spot 五年小時級 K 線，比較 XGBoost、LSTM、Transformer 對 **BTC／ETH／SOL／XRP 未來 4 小時漲跌方向**的預測能力。整個流程（抓取 → 品質檢查 → 補值 → 特徵工程 → walk-forward 驗證）皆可重現，並保留 manifest 供追溯。

## 交接：從這裡開始

1. **先讀 [交接使用說明.md](交接使用說明.md)**：怎麼取得資料、建立環境、確認一切正常，以及日常操作。
2. **再讀 [docs/handover/handover.md](docs/handover/handover.md)**：專案現況、所有檔案在哪裡、剩下的工作、一定要遵守的規則。
3. **資料不在 GitHub 上**：`data/`、`logs/`、`models/live/` 在另外提供的資料壓縮檔 `專題_資料_2026-09-30.zip`。只 clone 的話看不到任何實驗結果。
4. 研究範圍的權威文件是 [agement.md](agement.md)。本階段**不含**穩定幣脫鉤預警、市場連動分析與風險分數設計（屬後續延伸）。

## 目前結果（誠實現況）

四幣種 4 小時方向分類，3 個 expanding-window folds 的 test 平均 ROC-AUC：

| 幣種 | XGBoost | LSTM | Transformer |
| --- | ---: | ---: | ---: |
| BTCUSDT | 0.5289 | 0.5302 | **0.5318** |
| ETHUSDT | 0.5221 | 0.5232 | **0.5257** |
| SOLUSDT | 0.5034 | 0.5089 | **0.5106** |
| XRPUSDT | 0.5149 | **0.5236** | 0.5162 |

**訊號強度尚不足以支撐任何交易或風險預警宣稱**：SOL 近乎隨機，BTC／ETH 也僅約 0.53，深度模型未穩定勝過 XGBoost。引用本專題結果時請一律使用上述數字，勿以其他章節績效替代。詳見 [multi_asset_4h_model_comparison.md](docs/experiments/4h_model_comparison/multi_asset_4h_model_comparison.md)。

後續的預先登記實驗都沒有找到穩定的改善：

| 實驗 | 結果 | 結論文件 |
| --- | --- | --- |
| 外部特徵 A/B、特徵消融 v1／v2 | 沒有穩定改善 | [feature_tests/](docs/experiments/feature_tests/) |
| 多視窗（1h／4h／12h／24h） | 只有 XRP 1h 穩定強過 4h，列為候選 | [multi_horizon_conclusions.md](docs/experiments/multi_horizon/multi_horizon_conclusions.md) |
| 重訓頻率 | 12 個組合中 0 個通過 | [retrain_frequency_conclusions.md](docs/experiments/retrain_frequency/retrain_frequency_conclusions.md) |
| 三段式目標（扣 0.2% 手續費） | 四幣種每筆淨報酬皆為負 | [three_class_target_conclusions.md](docs/experiments/three_class_target/three_class_target_conclusions.md) |
| 波動大小預測（延伸研究） | 四幣種 AUC 0.64～0.68，初步通過，待新資料驗證 | [volatility_target_extension_results.md](docs/experiments/volatility_target/volatility_target_extension_results.md) |

整合結論見階段總報告 [five_year_4h_final_report.md](docs/report/five_year_4h_final_report.md) 與報告初稿 [final_project_report.md](docs/report/final_project_report.md)。

## 常用指令

全部從專案根目錄執行，Python 環境固定使用 `.venv-transformer`（建立方式見 [交接使用說明.md](交接使用說明.md)）：

```powershell
$py = ".\.venv-transformer\Scripts\python.exe"
& $py -m pip install -r requirements-lock.txt          # 安裝套件（完整鎖定版本）
& $py -m unittest discover -s tests                     # 跑全部測試
& $py scripts\verify_multi_horizon_outputs.py           # 獨立驗證多視窗實驗結果（V1～V8）
& $py scripts\make_report_figures.py                    # 重新產生報告圖 F1～F10
& $py scripts\live_pipeline.py update                   # 抓新 K 線並補上預測（需要網路）
& $py scripts\live_pipeline.py status                   # 查看資料與預測是否最新
```

從頭重建資料（通常不需要；**不要覆寫實驗用的 `data/features`**）：

```powershell
& $py scripts\fetch_binance_5y.py --dry-run --config config.json
& $py scripts\fetch_binance_5y.py --config config.json
& $py scripts\check_data_quality.py --config config.json --manifest logs\fetch_manifest.json --report logs\data_quality_report.json
& $py scripts\impute_missing_klines.py --config config.json
& $py scripts\feature_engineering.py
& $py scripts\train_btc_4h_model_comparison.py --symbol BTC --data data\features\btc_4h.csv --output logs\btc_4h_model_comparison.json
```

## 流程說明

| 階段 | 程式 | 輸入 → 輸出 | 紀錄 |
| --- | --- | --- | --- |
| 1. 抓取 | `scripts/fetch_binance_5y.py` | Binance API → `data/raw/*_5y.csv` | `logs/fetch_manifest.json` |
| 2. 品質檢查 | `scripts/check_data_quality.py` | `data/raw/` | `logs/data_quality_report.json` |
| 3. 補值 | `scripts/impute_missing_klines.py` | `data/raw/` → `data/processed/` | `logs/imputation_manifest.json` |
| 4. 特徵工程 | `scripts/feature_engineering.py` | `data/processed/` → `data/features/` | `logs/feature_manifest.json` |
| 5. Baseline | `scripts/train_btc_4h_baseline.py` | `data/features/btc_4h.csv` | `docs/experiments/4h_model_comparison/btc_4h_baseline.md` |
| 6. 三模型比較 | `scripts/train_btc_4h_model_comparison.py`（`--symbol BTC/ETH/SOL/XRP`） | `data/features/*_4h.csv` | `logs/*_4h_model_comparison.json` |
| 7. 預先登記實驗 | `scripts/run_*.py` | `data/features/` | `logs/<實驗>/`、`docs/experiments/<實驗>/` |
| 8. 前向測試 | `scripts/live_pipeline.py` | Binance API → `data/live/` | `logs/live/predictions.csv` |
| 選配 A. 外部訊號抓取 | `scripts/fetch_external_signals.py` | 公開 API → `data/external/` | `logs/external_manifest.json` |
| 選配 B. 外部特徵合併 | `scripts/feature_engineering.py --external-dir data\external` | `data/processed/` + `data/external/` → `data/features_ext/` | `logs/feature_ext_manifest*.json` |

設定集中在 [config.json](config.json)：四個交易對、`1h` 週期、5 年、分頁 1000 筆、重試 5 次。

**資料契約**：53 個特徵欄位（順序固定），標籤 `future_log_return = log(close[t+h] / close[t])`、`target_up = future_log_return > 0`，支援 `1h`／`4h`／`12h`／`24h`。`is_imputed` 只是補值追蹤欄，**不是模型特徵**。缺口佔比約 0.016%，補值規則見 [agement.md](agement.md)。

**外部訊號（選配）**：鏈上指標、市場情緒（Fear & Greed）與衍生品（資金費率、未平倉量、多空比）已可接入，全部使用免金鑰公開來源，日頻資料加 1 天發布延遲、一律 backward 對齊。限制與指令見 [external_signals.md](docs/data_features/external_signals.md)。

**防洩漏**：時間順序切分、expanding-window walk-forward、24 列 purge；Platt 校準與門檻選擇只用 validation 期，test 完全不參與。每個實驗先寫預先登記文件（`docs/superpowers/specs/`）並 commit，才執行會產生結果的程式。

## 目錄結構

```
交接使用說明.md       交接後的第一份文件：怎麼開始、怎麼操作
agement.md           研究範圍（權威文件）
DATA_STRUCTURE.md    資料夾結構說明
config.json          抓取設定
requirements-lock.txt 完整套件版本
scripts/             資料流程、實驗、驗證、圖表、前向測試、.ptl 產生程式
tests/               單元測試
run/                 早期實際執行的 notebook（見 run/README.md）
models/btc_4h_transformer/  BTC 4h Transformer 權重與 .ptl
docs/
  handover/          交接文件、前端資料說明、.ptl 說明、git 歷史改寫對照表
  report/            專題報告初稿、簡報大綱、階段總報告、研究計畫書
  experiments/       各實驗結果與結論（一個實驗一個資料夾）
  data_features/     特徵工程與外部訊號說明
  figures/           報告圖 F1～F10 與圖表數字
  superpowers/       預先登記文件（定案後不修改）

以下在資料壓縮檔裡，不在 GitHub：
data/                原始、補值後、特徵、外部、即時更新資料
logs/                所有實驗結果 JSON、逐列預測、驗證紀錄
models/live/         前向測試的凍結模型與滾動模型
```

## 文件索引

| 文件 | 內容 |
| --- | --- |
| [交接使用說明.md](交接使用說明.md) | 交接後怎麼開始、日常操作 |
| [docs/handover/handover.md](docs/handover/handover.md) | 專案現況、檔案清單、剩下的工作、規則、附錄（繳交規定、寫作原則、新資料驗證、排程） |
| [docs/handover/frontend_data_guide.md](docs/handover/frontend_data_guide.md) | 前端網站需要的資料與格式 |
| [docs/handover/ptl_guide.md](docs/handover/ptl_guide.md) | BTC 4h Transformer 的 `.ptl` 行動端模型說明 |
| [docs/report/final_project_report.md](docs/report/final_project_report.md) | 專題報告初稿 |
| [docs/report/final_presentation_outline.md](docs/report/final_presentation_outline.md) | 口試簡報大綱與 Q&A |
| [docs/report/five_year_4h_final_report.md](docs/report/five_year_4h_final_report.md) | 五年資料 4 小時方向分類階段報告 |
| [docs/experiments/4h_model_comparison/](docs/experiments/4h_model_comparison/) | BTC baseline、各幣種與四幣種彙整的三模型比較 |
| [docs/data_features/feature_engineering.md](docs/data_features/feature_engineering.md) | 特徵與標籤定義、執行指令、輸出格式 |
| `docs/report/7_...docx` | 研究計畫書（長期願景，範圍以 `agement.md` 為準） |

## 前一階段

本研究前一階段曾以 3 年資料做初步測試；本專案與報告僅使用 5 年資料與嚴格評估流程。

## 安全說明

只使用公開端點（Binance K 線／資金費率與封存檔、Coin Metrics community、alternative.me），**不使用**、不讀取也不保存 API key 或 secret。先前外流的 API key 請在 Binance 撤銷重建，且不要放進本專案。
