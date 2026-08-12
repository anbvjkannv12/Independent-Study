# 特徵工程流程

目前特徵流程已沿用原專題的完整定義，輸入為 `data/processed/` 的四個五年 1h CSV，輸出為 `data/features/`。

執行完整流程：

```powershell
$py = "C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py scripts\feature_engineering.py --config config.json
```

預設會產生 BTC、ETH、SOL、XRP 的 `1h`、`4h`、`12h`、`24h` 資料集，例如 `data/features/btc_12h.csv`。每份資料包含 53 個原專題特徵、`future_log_return`、`target_up` 與 `is_imputed`；最後一欄只用於資料品質稽核，不會進入模型特徵。

特徵與標籤定義記錄在 `logs/feature_manifest.json`：

- 分類：`target_up = future_log_return > 0`
- 回歸：`future_log_return = log(close[t+h] / close[t])`
- 特徵欄位數：53

如只需特定 horizon，可使用 `--horizons 1` 或 `--horizons 4,12`。也可以用 `--input-dir`、`--output-dir` 與 `--manifest` 指定路徑。
