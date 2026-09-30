# 外部訊號：鏈上指標、市場情緒與衍生品

在既有 53 個 K 線特徵之外，額外接上三類外部因子。全部使用**免金鑰的公開來源**，維持「不讀取也不保存 API key」的原則。

| 類別 | 特徵 | 來源 | 原始頻率 |
| --- | --- | --- | --- |
| 鏈上 | 交易所流入／流出、活躍地址數、MVRV | Coin Metrics community API v4 | 1 天 |
| 情緒 | Fear & Greed 指數 | alternative.me | 1 天 |
| 情緒（選配） | FinBERT／Twitter-RoBERTa 文本情緒 | 自備新聞／社群語料 | 依語料 |
| 衍生品 | 資金費率 | Binance USDT-M `fapi` | 8 小時 |
| 衍生品 | 未平倉量、多空比 | `data.binance.vision` 每日 metrics 封存檔 | 5 分鐘 → 取每小時 |

## 執行方式

```powershell
& ".venv-transformer\Scripts\python.exe" scripts\fetch_external_signals.py --config config.json --start 2021-08-01 --end 2026-08-10
& ".venv-transformer\Scripts\python.exe" scripts\feature_engineering.py --external-dir data\external --output-dir data\features_ext --manifest logs\feature_ext_manifest.json
```

第一步輸出到 `data/external/`，並寫入 `logs/external_manifest.json`（記錄每個來源實際抓到幾列、檔案在哪、未平倉量涵蓋起點）。未平倉量要逐日下載封存檔（四幣種約 7,000 個檔案），第一次執行需要數十分鐘；檔案會快取在 `data/external/cache/open_interest/`，重跑時直接沿用。只想抓某幾類時用 `--sources onchain,funding,fear_greed`。

第二步在原有特徵後面接上外部欄位，輸出到 `data/features_ext/`。因為每個幣種可用的外部欄位不同，除了合併 manifest 之外，另外會產生 `logs/feature_ext_manifest_btc.json` 等**各幣種 manifest**，訓練時搭配使用：

```powershell
& ".venv-transformer\Scripts\python.exe" scripts\train_btc_4h_model_comparison.py --symbol BTC --data data\features_ext\btc_4h.csv --manifest logs\feature_ext_manifest_btc.json --output logs\btc_4h_model_comparison_ext.json
```

## 特徵定義

鏈上（`scripts/external_features.py`）：

- `onchain_active_addr_chg`、`onchain_active_addr_z30`：活躍地址數的對數變化與 30 日 z-score
- `onchain_mvrv`、`onchain_mvrv_z90`：MVRV 比率與 90 日 z-score
- `onchain_exch_netflow_z30`：交易所淨流入（流入−流出）的 30 日 z-score
- `onchain_exch_flow_ratio`、`onchain_exch_inflow_chg`：流入／流出比值、流入的對數變化

衍生品：

- `funding_rate`、`funding_rate_mean_3`（24 小時）、`funding_rate_mean_9`（3 天）、`funding_rate_z90`
- `oi_log_change_1h`、`oi_log_change_24h`、`oi_notional_z168`
- `oi_toptrader_ls_ratio`、`oi_account_ls_ratio`、`oi_taker_ls_vol_ratio`（槓桿部位擁擠度）

情緒：

- `sent_fng`、`sent_fng_chg_1d`、`sent_fng_z30`
- 有語料時另加 `sent_news_mean`、`sent_news_mean_3`、`sent_news_pos_ratio`、`sent_news_volume_z30`

## 防洩漏規則

1. 所有外部序列一律以 `merge_asof(direction="backward")` 對齊，只會取到**該 K 棒開盤時間之前**已經存在的值。
2. 日頻資料（鏈上、Fear & Greed）額外加上 **1 天發布延遲**：D 日的值最早只能在 D+1 00:00 UTC 之後被使用。
3. 每個來源都有過期容忍上限（鏈上／情緒 3 天、資金費率 1 天、未平倉量 6 小時），超過就視為沒有資料。
4. 涵蓋範圍外的 K 棒**直接刪除，不做回填**，所以 `data/features_ext/` 的列數會比 `data/features/` 少。
5. 上述規則由 `tests/test_external_features.py` 驗證。

## 已知限制（請照實寫進報告）

- **SOL 沒有鏈上資料**：Coin Metrics community 版不提供 SOL 的 `AdrActCnt`／`CapMVRVCur`／交易所流量，因此 SOL 只有衍生品與情緒特徵。
- **XRP 沒有交易所流量**：只有活躍地址數與 MVRV。
- **資料列數會變少**：未平倉量封存檔 BTCUSDT 自 2021-08-01 起、ETH／SOL／XRP 自 2021-12-01 起（REST 端點只提供近 30 天，補不了這段歷史）；再加上 MVRV 90 日、資金費率 90 期、未平倉量 168 小時的 rolling 暖機期（不用不完整視窗算 z-score），4 小時資料集實際結果如下：

| 幣種 | `data/features/` | `data/features_ext/` | 起點 | 外部欄位數 |
| --- | ---: | ---: | --- | ---: |
| BTC | 43,755 | 34,172 | 2021-10-30 | 20 |
| ETH | 43,756 | 33,238 | 2021-12-07 | 20 |
| SOL | 43,755 | 33,237 | 2021-12-07 | 13 |
| XRP | 43,755 | 33,261 | 2021-12-07 | 17 |
- **FinBERT／Twitter-RoBERTa 需要自備語料**：沒有免費來源提供五年份的加密貨幣新聞全文。`scripts/score_news_sentiment.py` 接受任一份 `time,text` 的 CSV（例如 Kaggle 的加密貨幣新聞資料集），用 FinBERT 或 Twitter-RoBERTa 打分後輸出 `data/external/news_sentiment.csv`，特徵流程會自動接上。需先 `pip install transformers`。在補上語料之前，情緒因子只有 Fear & Greed 指數。
- **已檢驗，結論是沒有改善**（2026-09-16）：在相同資料列與相同 folds 上做「有／無外部特徵」的受控比較，12 組（4 幣種 × 3 模型）的平均 test ROC-AUC 差異為 −0.0017，變好 6/12，Brier 變化都在 ±0.0007 內。詳見 [external_features_ab_comparison.md](../experiments/feature_tests/external_features_ab_comparison.md)。文獻上鏈上因子把方向預測 AUC 拉到 0.76 的結果，在本專題的 4 小時視窗下沒有重現。
