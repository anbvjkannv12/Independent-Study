# 專題資料夾結構說明

本專題目前分成兩部分：舊專題（3 年資料）與新專題（5 年資料）。

## 舊專題：3 年資料

位置：

```text
舊專題_3年資料/
├── data/
│   ├── btc_3y.csv
│   ├── eth_3years.csv
│   ├── sol_3years.csv
│   └── xrp_3y_hourly.csv
├── notebooks/
│   └── 舊專題使用的 Jupyter Notebook
└── archive/
    └── Jupyter 自動產生的 checkpoint 備份
```

說明：

- 3 年的 BTC、ETH、XRP、SOL 原始資料已集中放在 `舊專題_3年資料/data/`。
- 舊專題 Notebook 保留在 `舊專題_3年資料/notebooks/`。
- `.ipynb_checkpoints` 已移到 `舊專題_3年資料/archive/`，避免干擾正式 Notebook。

## 新專題：5 年資料

位置：

```text
data/
├── raw/
│   ├── btc_5y.csv
│   ├── eth_5y.csv
│   ├── sol_5y.csv
│   └── xrp_5y.csv
├── processed/
│   ├── btc_5y.csv
│   ├── eth_5y.csv
│   ├── sol_5y.csv
│   └── xrp_5y.csv
├── features/
│   ├── btc_1h.csv / btc_4h.csv / btc_12h.csv / btc_24h.csv
│   ├── eth_1h.csv / eth_4h.csv / eth_12h.csv / eth_24h.csv
│   ├── sol_1h.csv / sol_4h.csv / sol_12h.csv / sol_24h.csv
│   └── xrp_1h.csv / xrp_4h.csv / xrp_12h.csv / xrp_24h.csv
├── features_ext/
│   └── 加入外部特徵後的特徵資料
└── external/
    └── funding、open interest、fear greed、on-chain 等外部資料
```

說明：

- `data/raw/` 現在只保留新專題的 5 年四幣種原始資料。
- `data/processed/` 是處理後的 5 年資料。
- `data/features/` 是模型訓練用特徵資料。
- `data/features_ext/` 是加入外部因子後的特徵資料。
- `data/external/` 是外部訊號資料。

## 整理原則

1. 舊專題 3 年資料集中放到 `舊專題_3年資料/`。
2. 新專題 5 年資料集中放到 `data/`。
3. `data/raw/` 不再混放 3 年舊資料，只保留 5 年新資料。
4. 暫不搬動 `scripts/`、`src/`、`models/`、`runs/`、`logs/`，避免影響程式路徑與模型結果。
