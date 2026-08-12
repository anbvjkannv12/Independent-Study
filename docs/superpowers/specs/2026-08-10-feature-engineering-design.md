# 原專題特徵與標籤流程整合規格

## 目標

將 `C:\Users\user\專題` 的既有特徵工程與標籤定義接到目前五年 Binance 1h 資料流程，產生可供後續模型直接使用的特徵資料集。

## 固定相容性

- 特徵欄位必須與原專題 `models/popular_coins/feature_manifest.json` 完全一致，共 53 欄，順序也一致。
- 分類標籤：`target_up = 1` when `future_log_return > 0`, otherwise `0`。
- 回歸標籤：`future_log_return = log(close[t+h] / close[t])`。
- 預設 horizon：`1`、`4`、`12`、`24` 小時。
- 資料先依 UTC `open_time` 排序並去除重複時間；lag、rolling 與指標只能使用目前或過去資料。
- `future_log_return` 與 `target_up` 不得出現在特徵欄位。
- 新流程的 `is_imputed` 僅作為資料品質追蹤欄，不納入特徵，避免改變原專題的 53 特徵定義。

## 輸入與輸出

- 輸入：`data/processed/{btc,eth,sol,xrp}_5y.csv`。
- 輸出：`data/features/{symbol}_{horizon}h.csv`，包含 `open_time`、53 個特徵、`future_log_return`、`target_up` 與 `is_imputed`。
- 輸出 manifest：`logs/feature_manifest.json`，記錄版本、幣種、horizon、特徵名稱、標籤定義、每個輸出檔案的列數與時間範圍。
- CLI：`scripts/feature_engineering.py --config config.json`，可用 `--input-dir`、`--output-dir`、`--manifest`、`--horizons` 覆寫預設值。

## 驗證

流程必須在寫檔前驗證：

- 每個輸出資料集都有精確 53 個特徵，且欄位順序與原 manifest 相同。
- `target_up` 與 `future_log_return` 逐列符合公式。
- `open_time` 單調遞增且不重複。
- 特徵與標籤沒有 NaN 或無限值。
- 缺少必要輸入欄位時，以清楚的錯誤訊息停止，不產生部分成功的 manifest。

## 不在本次範圍

- 不修改原專題檔案。
- 不在本次接模型訓練、scaler 或 walk-forward evaluation。
- 不把 `is_imputed` 轉成模型特徵，也不改寫既有 `data/processed` 檔案。
