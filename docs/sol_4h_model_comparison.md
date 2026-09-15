# SOLUSDT 4 小時三模型 Walk-forward 比較

結果位於 `logs/sol_4h_model_comparison.json`。此檔使用與 BTC 相同的 3 個 expanding-window folds、53 項特徵、validation-only Platt 校準與門檻選擇，比較 XGBoost、LSTM、Transformer 與共同基準。

## 重跑

```powershell
$py = "C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py scripts\train_btc_4h_model_comparison.py --symbol SOL --data data\features\sol_4h.csv --manifest logs\feature_manifest.json --output logs\sol_4h_model_comparison.json
```

## 結果摘要

| 模型 | 平均 test ROC-AUC | 樣本標準差 | 平均 Brier |
| --- | ---: | ---: | ---: |
| XGBoost | 0.5023 | 0.0153 | 0.2502 |
| LSTM | 0.5088 | 0.0243 | 0.2500 |
| Transformer | 0.5067 | 0.0271 | 0.2499 |

三模型皆接近隨機排序，且跨 fold 波動較大；目前沒有可支持進入成本與持倉驗證的穩定訊號。詳見跨幣種比較 [multi_asset_4h_model_comparison.md](multi_asset_4h_model_comparison.md)。
