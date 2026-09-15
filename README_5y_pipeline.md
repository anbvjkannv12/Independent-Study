# 專題 new (五年)

這是一個獨立於原專題的五年加密貨幣資料專案，資料來源為 Binance Spot 公開行情 API。

## 資料範圍

- 交易對：`BTCUSDT`、`ETHUSDT`、`SOLUSDT`、`XRPUSDT`
- 週期：`1h`
- 期間：執行當日往前五年，結束於最新完整小時 K 線
- 輸出：`data/raw/*_5y.csv`
- 品質紀錄：`logs/fetch_manifest.json`

## 安全說明

本專案使用 Binance 公開 K 線端點，不使用、不讀取也不保存 API key 或 secret。公開歷史行情不需要帳戶權限。你先前貼出的 API key 已暴露，請在 Binance 撤銷並重新建立，不要將新金鑰放入本專案。

## 執行方式

從專案根目錄執行：

```powershell
& "C:\Users\user\專題\.venv-transformer\Scripts\python.exe" scripts\fetch_binance_5y.py --dry-run --config config.json
& "C:\Users\user\專題\.venv-transformer\Scripts\python.exe" scripts\fetch_binance_5y.py --config config.json
```

下載程式會逐頁抓取，每頁最多 1000 根 K 線；下載完成後會檢查重複時間、空值、數值欄位與一小時間隔。只有通過驗證的暫存 CSV 才會取代正式輸出。

## 目錄

- `data/raw/`：四個交易對的五年原始 K 線 CSV
- `scripts/fetch_binance_5y.py`：下載、驗證、重試與 manifest 程式
- `tests/`：抓取器單元測試
- `run/`：實際執行的 Jupyter notebook 與執行說明
- `logs/`：下載狀態與資料品質摘要
- `agement.md`：五年研究範圍與工作說明
- `時程.md`：資料、模型與驗證安排

## 研究銜接

資料品質確認後，才將 `data/raw/` 接到原專題的特徵工程與模型流程。原專題檔案不在本專案內，也不會被此下載器修改。
## 資料品質檢查

下載完成後，可執行以下指令檢查 CSV 的欄位、時間排序、重複資料、OHLCV 合理性，以及 CSV 與 manifest 是否一致：

```powershell
C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\check_data_quality.py --config config.json --manifest logs\fetch_manifest.json --report logs\data_quality_report.json
```

檢查結果會寫入 `logs/data_quality_report.json`。已知的時間缺口會列為警告，正常模式仍會回傳成功；若要讓任何警告都使指令失敗，可加上 `--fail-on-warnings`。

## 資料缺口補值

原始資料保留在 `data/raw/`，訓練用的連續資料會輸出到 `data/processed/`。缺少的 1 小時 K 線會用上一根 K 線的 `close` 補價格欄位，成交量相關欄位補 `0`，並用 `is_imputed` 標記是否為補值列。

```powershell
C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\impute_missing_klines.py --config config.json
```

補值摘要會寫到 `logs/imputation_manifest.json`。
