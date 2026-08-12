# 五年資料專題工作說明

## 當前目標

本專案依據研究計畫書「基於機器學習與深度學習之加密貨幣走勢預測與穩定幣脫鉤風險預警系統」調整執行順序。

目前**只需要先完成主流加密貨幣價格預測部分**，研究標的以 `BTC`、`ETH`、`SOL`、`XRP` 為主。原計畫中的穩定幣脫鉤風險預警屬於後續延伸工作，**現在不列入本階段必要範圍**。

## 明確排除項目

以下內容先不做，除非之後另行追加需求：

- `USDC/USDT`、`TUSD/USDT` 等穩定幣交易對分析
- 穩定幣脫鉤（de-pegging）風險偵測
- `Isolation Forest` 異常偵測流程
- 風險分數（risk score）與預警門檻設計
- 主流幣與穩定幣之市場連動驗證

## 研究範圍

本階段聚焦於主流幣種的價格走勢預測，使用 Binance Spot 的小時級歷史資料，建立可重現的資料整理、特徵工程、模型訓練與評估流程。研究方向以機器學習與深度學習模型比較為主，後續可視進度再加入模型融合。

## 研究標的與資料基準

- `BTCUSDT`、`ETHUSDT`、`SOLUSDT`、`XRPUSDT`
- 1 小時 K 線（OHLCV）
- 目前資料基準為最近五年
- 所有時間以 UTC 表示
- CSV 欄位需與既有資料流程相容

## 本階段工作重點

1. 完成四個主流幣種的五年資料蒐集與品質檢查。
2. 處理缺失值、時間戳同步與資料清理，建立可訓練版本資料集。
3. 完成探索性資料分析與特徵工程，例如 RSI、MACD、Bollinger Bands、ATR、lag features 與時間特徵。
4. 明確定義預測目標，例如下一小時價格、報酬率或漲跌方向，並固定評估方式。
5. 以時間序列切分與 walk-forward validation 進行模型訓練與驗證，避免資料洩漏。
6. 優先完成價格預測模型，可先從 `XGBoost`、`LSTM`、`Transformer` 中挑選可落地模型逐步實作。
7. 若基礎模型表現穩定，再考慮 `Stacking` 融合作為加強版，而不是本階段硬性完成項目。

## 工作原則

1. 先確認資料來源、時間範圍與欄位契約，再進行模型訓練。
2. 所有訓練、驗證與測試切分都必須遵守時間先後順序。
3. 保留資料缺漏、重試、補值與下載時間紀錄，讓結果可追溯。
4. 模型評估至少區分預測誤差與交易應用限制，不以單一指標宣稱模型有效。
5. 實際執行的 notebook 放在 `run/`，避免研究過程散落。

## 完成條件

- 四個交易對均有五年 CSV，或失敗原因完整記錄於 manifest。
- 每個 CSV 通過欄位、型別、重複與時間間隔檢查。
- 完成模型使用資料集的缺漏摘要、起訖時間與補值說明。
- 完成至少一套可重現的主流幣種價格預測流程。
- 完成模型評估報告，清楚說明資料限制、特徵設計、驗證方式與結果解讀。
- `agement.md` 與後續實作內容均以「先完成價格預測」為主，不要求納入穩定幣脫鉤模組。

## 時程

詳細安排請見 [時程.md](時程.md)。

## 2026-08-10 資料時間缺口處理決策

目前 4 個幣種的原始 1 小時 K 線資料各有 7 筆缺口，約占每個幣種 43,818 筆資料的 0.016%。這個比例很低，對整體模型訓練的影響預期很小，但如果直接忽略或偷偷補值，專題方法會比較難交代。

本專題決定保留 `data/raw/` 原始資料不動，另外產生 `data/processed/` 訓練用資料。缺少的時間點會補成一筆新的 K 線，並新增 `is_imputed` 欄位標記來源：

- 原始資料列：`is_imputed = 0`
- 補值資料列：`is_imputed = 1`

補值規則如下：

- `open`、`high`、`low`、`close` 使用上一根 K 線的 `close`，代表該缺口暫時視為平盤。
- `volume`、`quote_asset_volume`、`number_of_trades`、`taker_buy_base`、`taker_buy_quote` 補 `0`，避免製造不存在的交易量。
- `open_time` 使用缺少的 1 小時時間點，`close_time` 使用該小時結束前 1 毫秒。

這樣處理的好處是模型可以使用連續時間序列，不會因少量缺口造成技術指標或時間序列切窗中斷；同時 `is_imputed` 讓模型與報告都能辨識哪些資料是補出來的。專題報告中應註明：缺口比例極低，補值只用於維持時間序列完整性，不是主要偏差來源。

執行方式：

```powershell
C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\impute_missing_klines.py --config config.json
```

輸出檔案：

- `data/processed/*_5y.csv`
- `logs/imputation_manifest.json`

## 2026-08-11 今日進度整理

今天已經把資料整理階段往特徵工程階段接上，確認目前可以進入「特徵方程」與模型資料集準備的階段。整體方向維持原專題設定：本階段只做 `BTCUSDT`、`ETHUSDT`、`SOLUSDT`、`XRPUSDT` 四個主流幣種的價格走勢預測，不加入穩定幣脫鉤風險預警。

完成內容如下：

- 確認並沿用原專題 `C:\Users\user\專題` 中完全相同的特徵與標籤定義。
- 建立特徵工程流程 `scripts/feature_engineering.py`，由 `data/processed/*_5y.csv` 產生模型可用資料集。
- 固定原專題 53 個特徵欄位與欄位順序，沒有把 `is_imputed` 放進模型特徵。
- 固定標籤定義為 `future_log_return = log(close[t+h] / close[t])`，以及 `target_up = future_log_return > 0`。
- 目前支援 `1h`、`4h`、`12h`、`24h` 四種預測 horizon。
- 產生正式特徵輸出於 `data/features/`，共 16 份 CSV：4 個幣種乘上 4 個 horizon。
- 產生特徵 manifest：`logs/feature_manifest.json`，記錄特徵名稱、標籤定義、horizon、輸入輸出路徑、列數與補值列數。
- 新增測試 `tests/test_feature_engineering.py`，檢查 53 個特徵契約、標籤公式、欄位順序、缺漏欄位防呆，以及輸出 manifest。
- 新增使用說明 `docs/feature_engineering.md`，記錄特徵流程執行指令與輸出格式。

重要決策如下：

- `is_imputed` 只作為資料品質追蹤欄保留在輸出最後一欄，不納入 `feature_names`，避免改變原專題特徵定義。
- 資料品質檢查中的 `warning` 是已知時間缺口提示，不代表特徵流程失敗；正常模式仍可繼續執行。
- 特徵流程輸出的欄位順序固定為 `open_time`、53 個特徵、`future_log_return`、`target_up`、`is_imputed`。

今日驗證結果：

- 新產生的 53 個特徵名稱已與原專題 manifest 比對，結果完全一致。
- `data/features/` 下 16 份 CSV 全部存在，標頭格式一致。
- 正式輸出總列數為 699,984。
- 單元測試已執行 `python -m unittest discover -s tests -v`，共 27 個測試通過。

下一步建議：

- 先使用 `data/features/*_1h.csv` 或 `data/features/*_4h.csv` 建立第一版 baseline 模型。
- 模型訓練時只讀取 manifest 中的 53 個 `feature_names`，標籤則使用 `target_up` 或 `future_log_return`，依分類或回歸任務決定。
- 報告中需說明 `is_imputed` 是補值追蹤欄，不屬於模型特徵。
