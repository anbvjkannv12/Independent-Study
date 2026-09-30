# ETHUSDT 4 小時三模型 Walk-forward 比較

結果位於 `logs/eth_4h_model_comparison.json`。此檔使用與 BTC 相同的 3 個 expanding-window folds、53 項特徵、validation-only Platt 校準與門檻選擇，比較 XGBoost、LSTM、Transformer 與共同基準。

## 重跑

```powershell
$py = "C:\Users\user\專題\.venv-transformer\Scripts\python.exe"
& $py "C:\Users\user\專題 new (五年)\scripts\train_btc_4h_model_comparison.py" --symbol ETH --data "C:\Users\user\專題 new (五年)\data\features\revisions\2026-09-10\eth_4h.csv" --manifest "C:\Users\user\專題 new (五年)\logs\eth_4h_feature_manifest_2026-09-10.json" --output "C:\Users\user\專題 new (五年)\logs\eth_4h_model_comparison_2026-09-10.json" --initial-train-rows 19795 --validation-rows 4000 --test-rows 4000 --fold-count 3
```

此指令固定使用 `C:\Users\user\專題\.venv-transformer` 的模型執行環境；輸入資料、manifest 與結果輸出均固定留在 `C:\Users\user\專題 new (五年)`，不讀取或覆寫舊專題的資料或模型產物。日期版產物不覆寫 2026-08-27 的原始 ETH 結果；`initial_train_rows=19795` 補回原特徵檔被移除的 40 個小時，因此三個 validation/test 的日曆時間仍與原結果一致。

## 結果摘要

| 模型 | 平均 test ROC-AUC | 樣本標準差 | 平均 Brier |
| --- | ---: | ---: | ---: |
| XGBoost | 0.5221 | 0.0028 | 0.2499 |
| LSTM | 0.5232 | 0.0079 | 0.2499 |
| Transformer | 0.5257 | 0.0173 | 0.2495 |

三模型皆只有微弱排序訊號，尚不足以推論交易成本後的獲利。Transformer 雖有最高平均 ROC-AUC，但 Fold 3 降至 0.5063，沒有穩定優勢；ETH 的定案結論是「無可靠優勝模型」，而不是選擇一個可供交易的模型。詳見跨幣種比較 [multi_asset_4h_model_comparison.md](multi_asset_4h_model_comparison.md)。

## 2026-09-10 校正後重跑產物

- 特徵資料：`data/features/revisions/2026-09-10/eth_4h.csv`，43,796 列、53 項特徵、7 個 `is_imputed=1` 列，實測沒有時間跳點；SHA-256 為 `D9FE91EB2ADD309F7EBE49E26B6460FDA429F1CEADA85E450BF192C01B730CBE`。
- 特徵 manifest：`logs/eth_4h_feature_manifest_2026-09-10.json`；SHA-256 為 `2FF2BA910EF87403A47652EEDA20D5625424B8FA0848C290D3566CF528E054BF`。
- 比較結果：`logs/eth_4h_model_comparison_2026-09-10.json`；程式版本為 Git revision `20786630e69eade4fd1bb494cab239f13cadd9be`，執行環境為 `C:\Users\user\專題\.venv-transformer`（Python 3.12.14、Torch 2.14.0+cu126、Pandas 2.3.3、NumPy 2.5.0、XGBoost 3.2.0、scikit-learn 1.9.0）。
- 三個 test 期各 4,000 列，日期與校正前結果相同。三模型共 9 個 model-fold 報告與五個共同基準組皆完整保存；全部 test 的 sample count 均為 4,000。

## 2026-09-10 定案前可追溯性檢視

本節僅核對既有 ETH 4 小時結果，沒有重新訓練、改寫模型或啟動其他 horizon 的建模。

### 已確認的結果鏈結

- 結果檔 `logs/eth_4h_model_comparison.json` 指向 `data/features/eth_4h.csv` 與 `logs/feature_manifest.json`；結果與 manifest 的 53 項特徵名稱及順序一致。
- manifest 記錄 ETH 4 小時資料為 43,756 列，時間範圍為 2021-08-10 11:00 UTC 至 2026-08-09 06:00 UTC，標籤為 `target_up = future_log_return > 0`。
- 結果檔記錄三個 expanding-window folds；每個 validation、test 各 4,000 列，且 train < validation < test，三段時間皆連續銜接。三個 test 期依序為 2024-04-27 22:00 至 2024-10-11 13:00 UTC、2025-03-27 06:00 至 2025-09-09 21:00 UTC、2026-02-23 14:00 至 2026-08-09 05:00 UTC。
- 每個模型 fold 都保存 validation-only Platt calibration、validation 選出的門檻、test 指標與 test 機率五分位；共同基準組也完整保存（all-up、training-majority、previous-direction、logistic-regression、zero-trade）。流程程式碼明確將 test 排除在校準與門檻選擇之外。
- 共享 walk-forward 評估單元測試已於 2026-09-10 通過 5/5；它驗證 expanding folds 不重疊、門檻僅依 validation 選擇、zero-trade 不被誤報為分類器，以及基準組完整性。未重做 BTC 資料或模型驗證。

### 三個 test folds 的 ROC-AUC

| 模型 | Fold 1 | Fold 2 | Fold 3 | 平均 | 判讀 |
| --- | ---: | ---: | ---: | ---: | --- |
| XGBoost | 0.5200 | 0.5253 | 0.5211 | 0.5221 | 三 fold 都僅有微弱訊號。 |
| LSTM | 0.5282 | 0.5272 | 0.5141 | 0.5232 | Fold 3 更接近隨機。 |
| Transformer | 0.5317 | 0.5392 | 0.5063 | 0.5257 | 平均最高但 fold 間波動最大，沒有穩定勝出。 |

三者的平均 Brier score 約為 0.2495--0.2499，接近二元目標下的弱辨識情境。即使 Transformer 的平均 ROC-AUC 略高於 LSTM（差 0.0025），這個差距和各 fold 的表現均不足以支持交易、持倉、獲利或風險預警的宣稱。ETH 的本輪結果確認結論為：資料與流程可追溯，但不存在可誠實稱為優勝或可部署的單一模型。

### 定案前需處理的資料限制

`logs/imputation_manifest.json` 記錄 ETH 原始資料的 7 個缺失小時已補入 `data/processed/eth_5y.csv`。校正前的 `data/features/eth_4h.csv` 因補值列成交量為 0，使 `trade_intensity` 與 `taker_buy_ratio` 成為非有限值並在 `dropna()` 時移除，造成 11 個時間跳點、合計跳過 40 小時。此問題已在 `scripts/feature_engineering.py` 修正：零成交量補值列的兩個活動比率特徵固定為 0，並以新增回歸測試驗證該列與 `is_imputed=1` 會被保留。校正後日期版資料有 7 個補值列且沒有時間跳點；原始校正前資料與結果仍保留，供歷史可追溯與敏感度比較。

此外，既有結果檔保存了資料與 manifest 的絕對路徑、特徵、fold 與模型設定，但沒有資料雜湊、程式版本／Git revision、套件 lockfile 或實際執行環境快照。五年專題隨附的 Codex runtime 缺少 `torch`，但 `C:\Users\user\專題\.venv-transformer` 已於 2026-09-10 驗證可載入 Torch 2.14.0+cu126、Pandas 2.3.3、NumPy 2.5.0、XGBoost 3.2.0 與 scikit-learn 1.9.0，並可成功載入比較腳本說明；本次仍未重跑結果。

## 2026-09-11 結果確認與模型定案待辦

1. 已完成：校正零成交量補值列、僅重跑 ETH 4 小時比較，且保留舊版產物。
2. 已完成：固定使用 `C:\Users\user\專題\.venv-transformer\Scripts\python.exe`；輸入與輸出都在五年專題，未啟動其他幣種或 horizon。
3. 已完成：記錄資料／manifest 雜湊、Git revision、套件版本與三個 folds、基準組、validation-only calibration/threshold 核對結果。
4. 已完成：依一致性、平均 ROC-AUC 與 Brier score 確認無穩定優勢，ETH 結果以「無可靠優勝模型」呈現。
5. 報告撰寫時仍須保留 ROC-AUC 約 0.50--0.53、Brier 約 0.25、未納入交易成本／持倉測試，以及不可支撐交易決策等限制。
