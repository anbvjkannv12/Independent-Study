# XRPUSDT 4 小時三模型 Walk-forward 比較

結果位於 `logs/xrp_4h_model_comparison.json`。此檔使用與 BTC 相同的 3 個 expanding-window folds、53 項特徵、validation-only Platt 校準與門檻選擇，比較 XGBoost、LSTM、Transformer 與共同基準。

## 重跑

```powershell
$py = "C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py scripts\train_btc_4h_model_comparison.py --symbol XRP --data data\features\xrp_4h.csv --manifest logs\feature_manifest.json --output logs\xrp_4h_model_comparison.json
```

## 結果摘要

| 模型 | 平均 test ROC-AUC | 樣本標準差 | 平均 Brier |
| --- | ---: | ---: | ---: |
| XGBoost | 0.5164 | 0.0064 | 0.2505 |
| LSTM | 0.5231 | 0.0030 | 0.2507 |
| Transformer | 0.5157 | 0.0276 | 0.2505 |

訊號很弱且 Brier score 未顯示可靠的機率優勢；不可由此推論交易獲利。詳見跨幣種比較 [multi_asset_4h_model_comparison.md](multi_asset_4h_model_comparison.md)。
