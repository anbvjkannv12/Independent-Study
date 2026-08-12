# Run Notebook

這個資料夾只存放五年資料專案實際執行的 `.ipynb` 檔案。

## 建議執行順序

1. 從 `C:\Users\user\專題 new (五年)` 開啟 `run\01_fetch_and_validate.ipynb`。
2. 先執行專案根目錄檢查與 dry-run cell。
3. 確認交易對、`1h` 與五年時間窗正確後，再將 `RUN_FULL_DOWNLOAD` 設為 `True`。
4. 執行下載 cell，完成後讀取 `logs\fetch_manifest.json`。
5. 執行 CSV 摘要 cell，確認四個檔案的列數、起訖時間與缺漏區段。

完整下載也可以直接從專案根目錄執行：

```powershell
& "C:\Users\user\專題\.venv-transformer\Scripts\python.exe" scripts\fetch_binance_5y.py --config config.json
```

Notebook 不包含 API key，也不會從原專題讀取資料。
