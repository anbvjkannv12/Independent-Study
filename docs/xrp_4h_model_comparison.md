# XRPUSDT 4 小時三模型 Walk-forward 比較

結果位於 `logs/xrp_4h_model_comparison_2026-09-16.json`。本次使用校正後特徵資料 `data/features/revisions/2026-09-16/xrp_4h.csv`、manifest `logs/sol_xrp_4h_feature_manifest_2026-09-16.json`，沿用與 BTC 相同的 3 個 expanding-window folds、53 項特徵、validation-only Platt 校準與門檻選擇，比較 XGBoost、LSTM、Transformer 與共同基準。

歷史原始版結果保留於 `logs/xrp_4h_model_comparison.json`；本文以 2026-09-16 校正後重跑結果作為本輪定案依據。

## 重跑

```powershell
& ".venv-transformer\Scripts\python.exe" scripts\train_btc_4h_model_comparison.py --symbol XRP --data data\features\revisions\2026-09-16\xrp_4h.csv --manifest logs\sol_xrp_4h_feature_manifest_2026-09-16.json --output logs\xrp_4h_model_comparison_2026-09-16.json --initial-train-rows 19795 --validation-rows 4000 --test-rows 4000 --fold-count 3
```

本指令固定使用 `.venv-transformer`，因五年專題原 Codex runtime 不含 `torch`，不可再使用舊文件中的 Codex runtime 指令。

## 結果摘要

| 模型 | 平均 test ROC-AUC | 樣本標準差 | 平均 Brier | 平均 Accuracy | 平均 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.5149 | 0.0130 | 0.2507 | 0.5124 | 0.4612 |
| LSTM | **0.5236** | 0.0037 | 0.2507 | **0.5219** | 0.2552 |
| Transformer | 0.5162 | 0.0279 | 0.2508 | 0.5093 | 0.3043 |

LSTM 的平均 ROC-AUC 最高且跨 fold 標準差最小，但 F1 明顯偏低，代表門檻後的上漲預測比例過低；因此不能只用平均 ROC-AUC 宣稱 LSTM 是可靠優勝模型。

## 三個 test folds 的 ROC-AUC

| 模型 | Fold 1 | Fold 2 | Fold 3 | 判讀 |
| --- | ---: | ---: | ---: | --- |
| XGBoost | 0.5265 | 0.5172 | 0.5009 | Fold 3 幾乎回到隨機。 |
| LSTM | 0.5199 | 0.5236 | 0.5272 | 排序表現最穩，但訊號仍很弱。 |
| Transformer | 0.5358 | 0.4842 | 0.5286 | Fold 2 低於隨機，跨期穩定性不足。 |

三個 test 期皆為 4,000 列：

| Fold | Train | Validation | Test |
| --- | --- | --- | --- |
| 1 | 2021-08-10 11:00 ～ 2023-11-13 06:00 UTC | 2023-11-13 07:00 ～ 2024-04-27 22:00 UTC | 2024-04-27 23:00 ～ 2024-10-11 14:00 UTC |
| 2 | 2021-08-10 11:00 ～ 2024-10-11 14:00 UTC | 2024-10-11 15:00 ～ 2025-03-27 06:00 UTC | 2025-03-27 07:00 ～ 2025-09-09 22:00 UTC |
| 3 | 2021-08-10 11:00 ～ 2025-09-09 22:00 UTC | 2025-09-09 23:00 ～ 2026-02-23 14:00 UTC | 2026-02-23 15:00 ～ 2026-08-09 06:00 UTC |

## 門檻後診斷

| 模型 | 平均 predicted up rate | 主要問題 |
| --- | ---: | --- |
| XGBoost | 0.4127 | 預測上漲比例較低，但仍可形成一定覆蓋。 |
| LSTM | 0.1877 | 平均 ROC-AUC 最高，但 Fold 2 predicted up rate 只有 0.0300，F1 被嚴重壓低。 |
| Transformer | 0.2487 | Fold 2 ROC-AUC 低於隨機，且預測上漲比例偏低。 |

LSTM 各 fold 的 predicted up rate 為 0.4170／0.0300／0.1160，F1 為 0.4821／0.0695／0.2139。這代表 LSTM 可能有微弱排序能力，但 validation 選出的門檻在 test 期並不穩定，不能視為可直接部署的方向分類器。

## 共同基準

| 基準 | Fold 1 | Fold 2 | Fold 3 | 平均 ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| all_up | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| training_majority | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| previous_direction | 0.4793 | 0.4890 | 0.4812 | 0.4832 |
| logistic_regression | 0.5095 | 0.4911 | 0.5263 | 0.5090 |

LSTM 平均 ROC-AUC 比 Logistic Regression 高約 0.0146，但 F1 與 predicted up rate 的異常使其無法被解讀為穩定可用的優勝模型。

## 2026-09-16 重跑產物與可追溯性

- 特徵資料：`data/features/revisions/2026-09-16/xrp_4h.csv`
- 特徵資料 SHA-256：`5f3deb9566b782027065c2c044c9fd89807497319658ff2adc05e17fffd0b84f`
- 特徵 manifest：`logs/sol_xrp_4h_feature_manifest_2026-09-16.json`
- Manifest SHA-256：`8d4673ebc6d2f5ab25b3cb979ef90c7c36a3575704e5220065b3df25ef7cae0f`
- 比較結果：`logs/xrp_4h_model_comparison_2026-09-16.json`
- 比較結果 SHA-256：`9adf1c7646186f745a4a1aaafc5d6fea75c428dcdd4151e8012f33ef4021a57d`
- 可追溯紀錄：`logs/xrp_4h_training_trace_2026-09-16.json`
- Git revision：`f475505435c05ec1a41a242382bbd352bea7bafb`
- 執行環境：Python 3.12.14、Torch 2.14.0+cu126、Pandas 2.3.3、NumPy 2.5.0、XGBoost 3.2.0、scikit-learn 1.9.0

## 定案結論

XRP 本輪不選單一可部署模型，定案結論為：**無可靠優勝模型**。

理由如下：

1. 三模型平均 ROC-AUC 只在 0.5149～0.5236，訊號非常弱。
2. LSTM 雖然平均 ROC-AUC 最高，但 F1 只有 0.2552，且 predicted up rate 嚴重偏低。
3. Transformer 在 Fold 2 低於隨機，顯示跨事件期間穩定性不足。
4. XRP 的 53 個特徵全部來自 K 線，沒有訴訟、上架、清算等外生事件資訊，無法充分表達 XRP 的事件驅動特性。
5. Brier score 約 0.2507～0.2508，未顯示可靠機率校準優勢。

因此，本結果只能作為研究比較，不可宣稱能穩定交易、獲利或作為風險預警系統。詳見跨幣種比較 [multi_asset_4h_model_comparison.md](multi_asset_4h_model_comparison.md)。
