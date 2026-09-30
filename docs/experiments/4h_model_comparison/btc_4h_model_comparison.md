# BTCUSDT 4 小時三模型 Walk-forward 比較

此流程以相同的時間 folds 比較 XGBoost、LSTM 與 Transformer，並同時列出永遠上漲、訓練集多數類別、前一期方向、Logistic Regression 與零交易基準。結果只用於研究比較；方向分類指標不代表交易成本後獲利。

## 安裝與執行

在專案根目錄執行：

```powershell
$py = "C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$sourceRoot = "C:\Users\user\專題 new (五年)"
& $py -m pip install -r requirements-lock.txt
& $py scripts\train_btc_4h_model_comparison.py --data "$sourceRoot\data\features\btc_4h.csv" --manifest "$sourceRoot\logs\feature_manifest.json"
```

預設會使用 43,755 列 BTC 四小時資料建立三個 expanding-window folds：初始訓練 19,755 列，接著每個 fold 使用 4,000 列 validation 與 4,000 列 test。完整成功後才會原子寫入 `logs/btc_4h_model_comparison.json`。

可在快速驗證時降低 epoch：

```powershell
& $py scripts\train_btc_4h_model_comparison.py --data "$sourceRoot\data\features\btc_4h.csv" --manifest "$sourceRoot\logs\feature_manifest.json" --max-epochs 5 --patience 2
```

## 如何閱讀結果

- `models.<name>.test_aggregate`：每個模型跨 folds 的測試指標平均與樣本標準差。
- `models.<name>.folds`：每個 fold 的 validation-only Platt 校準、選定門檻、未校準 ROC-AUC 與測試結果。
- `baselines`：所有共同基準在相同 folds 的結果；零交易不會被偽裝成分類指標。
- `probability_quintiles`：校準後測試機率由低到高五等分的預測機率與實際上漲率，用於檢查機率區隔與校準。

模型只有在大多數 folds 的未校準 test ROC-AUC 高於 0.5、Brier score 不劣於適當常數基準，且結果未集中於單一市場期時，才值得進行成本與持倉規則驗證。

使用 [03_btc_4h_model_comparison.ipynb](../../../run/03_btc_4h_model_comparison.ipynb) 檢視已儲存結果；notebook 不會訓練模型。
