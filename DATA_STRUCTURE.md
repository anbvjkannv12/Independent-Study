# 專題資料夾結構說明

本專題使用 5 年資料。本研究前一階段曾以 3 年資料做初步測試；本專案與報告僅使用 5 年資料與嚴格評估流程。

## 5 年資料

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

- `data/raw/` 是 5 年四幣種原始資料。
- `data/processed/` 是處理後的 5 年資料。
- `data/features/` 是模型訓練用特徵資料。
- `data/features_ext/` 是加入外部因子後的特徵資料。
- `data/external/` 是外部訊號資料。

## 整理原則

1. 5 年資料集中放到 `data/`，不進 git，另以資料壓縮檔交接（見 `docs/handover.md`）。
2. 實驗結果放在 `logs/`，模型放在 `models/`。
