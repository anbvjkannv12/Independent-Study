# Binance 五年資料專案設計規格

## 目標

建立一個獨立於目前專題的資料專案，位於 `C:\Users\user\專題 new (五年)`，抓取 Binance 現貨 BTC、ETH、SOL、XRP 的最近五年 1 小時 K 線資料，並保留可重複執行、可驗證與可增量更新的流程。

## 範圍

- 交易對：`BTCUSDT`、`ETHUSDT`、`SOLUSDT`、`XRPUSDT`
- 市場：Binance Spot
- 時間週期：`1h`
- 時間範圍：執行時間往前五年至執行時的最新完整 K 線
- 輸出格式：每個交易對一個 UTF-8 CSV
- API：使用 Binance 公開行情 REST API，不使用或保存 API key
- 原專題：不修改原有 `data`、`src`、`models` 與 notebooks

## 目錄設計

```text
C:\Users\user\專題 new (五年)\
├── data\raw\
│   ├── btc_5y.csv
│   ├── eth_5y.csv
│   ├── sol_5y.csv
│   └── xrp_5y.csv
├── scripts\
│   └── fetch_binance_5y.py
├── run\
│   └── README.md
├── logs\
│   └── fetch_manifest.json
├── config.json
├── README.md
├── agement.md
└── 時程.md
```

`agement.md` 以目前專題的 `docs\agent.md` 為基礎，改寫研究資料範圍與五年資料工作內容；`時程.md` 另外整理抓取、品質檢查、特徵工程、模型重訓與驗證階段。

`run` 專門存放這個五年資料專案實際執行的 `.ipynb` notebook。Notebook 會使用專案內的相對路徑，並以 `run\README.md` 記錄執行順序與輸出位置；原專題的 notebook 不會移動或覆蓋。

## 資料流程

1. 讀取 `config.json` 的交易對、週期與時間範圍設定。
2. 對每個交易對呼叫 `/api/v3/klines`，以最多 1000 根 K 線為一頁逐段取得資料。
3. 每次請求使用 `startTime` 與最後一筆 `open_time` 推進，避免重複抓取。
4. 將 Binance 回傳的欄位轉成既有模型使用的欄位名稱與資料型別。
5. 依 `open_time` 排序、去除重複時間戳，並檢查每筆間隔是否為一小時。
6. 先寫入同一資料夾的暫存 CSV，通過驗證後再以正式檔名取代既有輸出。
7. 將每個交易對的狀態、筆數、起訖時間、缺漏區段與錯誤寫入 `logs\fetch_manifest.json`。

## CSV 欄位

```text
open_time,open,high,low,close,volume,close_time,
quote_asset_volume,number_of_trades,taker_buy_base,taker_buy_quote
```

時間欄位使用 UTC ISO 8601 字串；價格、成交量與交易額使用數值欄位；交易筆數使用整數欄位。

## 錯誤處理與安全性

- HTTP 429、5xx 與網路暫時錯誤採指數退避重試。
- 超過重試次數時停止該交易對，保留錯誤訊息與最後成功游標。
- 不把 API key、secret、簽名或帳戶資料放入程式、設定檔、CSV、日誌或文件。
- 完整下載前先執行小範圍測試，確認端點、欄位與寫入權限。
- 暫存檔驗證失敗時不覆蓋既有正式 CSV。

## 驗證與驗收

每個輸出檔案必須通過以下檢查：

- 欄位完整且順序固定。
- `open_time` 無重複、可解析且排序遞增。
- OHLCV 欄位沒有空值、非數值或不合理的負值。
- 相鄰 K 線間隔為一小時；缺漏區段必須列入 manifest。
- `close_time` 與 Binance K 線週期一致。
- 筆數、起訖時間與資料品質摘要已寫入 manifest。
- 重新執行程式不會製造重複資料，且可安全更新輸出。
- `run` 內的 notebook 可在新專案根目錄執行，且不依賴原專題的相對路徑。

## 後續模型銜接

新資料專案先專注於資料取得與品質驗證，不直接複製或執行原專題模型。資料驗證完成後，再依 `agement.md` 與 `時程.md` 的安排，將資料路徑接到既有的特徵工程與模型訓練流程。
