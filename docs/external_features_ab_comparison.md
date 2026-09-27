# 外部特徵有效性檢驗：鏈上／情緒／衍生品 A/B 比較

> 執行日期：2026-09-16　｜　結果檔：`logs/{btc,eth,sol,xrp}_4h_model_comparison_{ext,base}_2026-09-16.json`

## 研究問題

在既有 53 個 K 線特徵之外，加入鏈上指標、市場情緒與衍生品訊號（定義見 [external_signals.md](external_signals.md)），能否改善四個主流幣 4 小時漲跌方向的預測能力？

## 實驗設計

兩組執行使用**完全相同的資料列與 folds，唯一差異是特徵集**：

| 組別 | 資料 | manifest | 特徵數 |
| --- | --- | --- | ---: |
| `ext`（有外部） | `data/features_ext/<幣種>_4h.csv` | `logs/feature_ext_manifest_<幣種>.json` | 66–73 |
| `base`（無外部） | **同一個檔案** | `logs/feature_manifest.json` | 53 |

關鍵設計決定：`base` 組是在 `data/features_ext/` 的縮減列數上**重新訓練**，而不是沿用既有的 43,755 列結果。外部資料的涵蓋範圍使列數降到 33–34k，若直接與全歷史結果相比，差異會混入「訓練資料量」這個變因，無法歸因於特徵。

其餘設定與先前所有執行一致：3 個 expanding-window folds、validation 與 test 各 4,000 列、Platt 校準與門檻僅用 validation、五種共同基準、模型超參數全部沿用預設。

| 幣種 | 列數 | 起點 | 外部欄位 | `initial_train_rows` |
| --- | ---: | --- | ---: | ---: |
| BTC | 34,172 | 2021-10-30 | 20 | 10,172 |
| ETH | 33,238 | 2021-12-07 | 20 | 9,238 |
| SOL | 33,237 | 2021-12-07 | **13**（無鏈上） | 9,237 |
| XRP | 33,261 | 2021-12-07 | 17（無交易所流量） | 9,261 |

三個 test 期間與原始全歷史結果**完全對齊**（2024-04-27～2024-10-11、2025-03-27～2025-09-09、2026-02-23～2026-08-09），因為資料尾端與 validation／test 長度相同。

## 結果：平均 test ROC-AUC

| 幣種 | 模型 | 無外部 | 有外部 | 差異 | fold 標準差 |
| --- | --- | ---: | ---: | ---: | ---: |
| BTC | XGBoost | 0.5336 | 0.5293 | −0.0043 | 0.0151 |
| | LSTM | 0.5257 | 0.5253 | −0.0004 | 0.0093 |
| | Transformer | 0.5294 | 0.5323 | +0.0029 | 0.0158 |
| ETH | XGBoost | 0.5253 | 0.5366 | **+0.0113** | 0.0060 |
| | LSTM | 0.5281 | 0.5203 | −0.0078 | 0.0154 |
| | Transformer | 0.5322 | 0.5325 | +0.0003 | 0.0210 |
| SOL | XGBoost | 0.5011 | 0.5028 | +0.0017 | 0.0278 |
| | LSTM | 0.5103 | 0.4974 | −0.0129 | 0.0194 |
| | Transformer | 0.5082 | 0.5050 | −0.0032 | 0.0204 |
| XRP | XGBoost | 0.5209 | 0.5216 | +0.0007 | 0.0188 |
| | LSTM | 0.5266 | 0.5102 | **−0.0164** | 0.0101 |
| | Transformer | 0.5146 | 0.5225 | +0.0078 | 0.0223 |

12 組比較：差異平均 **−0.0017**、中位數 **−0.0001**、變好 **6/12**。

只有 2 組的差異超出 fold 間標準差，且**一好一壞**（ETH XGBoost +0.0113、XRP LSTM −0.0164）。在 12 次比較中出現兩個離群值，與純雜訊的預期一致，不構成效果證據。

Brier score 全部 12 組的變化都在 ±0.0007 以內（皆約 0.2495–0.2503），機率品質沒有改善。

## 結論

**沒有證據顯示這三類外部因子能改善本專題的 4 小時方向預測。** 加入 13–20 個外部欄位後，排序能力仍停留在 ROC-AUC 0.50–0.53，與只用 K 線特徵時相同；SOL 的有外部版甚至有兩個 fold 落在 0.5 以下（0.4744、0.4898）。

文獻中鏈上因子把方向預測 AUC 推到 0.76 的結果，**在本專題的設定下沒有重現**。差異可能來自預測視窗（本專題為 4 小時，文獻多為日級）、標的與期間、以及驗證方式的嚴格程度。本比較不足以否定該文獻，但足以說明那個數量級的績效無法直接移轉到這裡。

## 兩個附帶發現

**一、縮短歷史幾乎沒有代價。** 同為 53 特徵、同樣的 test 期間，全歷史（43,755 列）與截短版（33–34k 列）的差異在 ±0.0065 內且無系統性方向，12 組中 7 組反而略升。訊號薄弱不是訓練資料量造成的。

| 幣種 | XGBoost | LSTM | Transformer |
| --- | ---: | ---: | ---: |
| BTC | +0.0047 | −0.0045 | −0.0024 |
| ETH | +0.0031 | +0.0049 | +0.0065 |
| SOL | −0.0023 | +0.0014 | −0.0024 |
| XRP | +0.0045 | +0.0035 | −0.0011 |

**二、Logistic Regression 基準咬得很緊。** 在有外部特徵的執行中，同樣吃 53+N 個特徵的 Logistic Regression 基準，表現與三個模型相當，SOL 甚至勝過全部三個：

| 幣種 | Logistic Regression | 最佳模型 |
| --- | ---: | --- |
| BTC | 0.5302 | Transformer 0.5323 |
| ETH | 0.5305 | XGBoost 0.5366 |
| SOL | **0.5123** | Transformer 0.5050 |
| XRP | 0.5103 | Transformer 0.5225 |

這強化了先前的結論：XGBoost、LSTM、Transformer 的額外複雜度，在這個問題上沒有換到穩定的回報。

## 重現方式

```powershell
$py = ".venv-transformer\Scripts\python.exe"
# 有外部特徵
& $py scripts\train_btc_4h_model_comparison.py --symbol BTC `
    --data data\features_ext\btc_4h.csv --manifest logs\feature_ext_manifest_btc.json `
    --output logs\btc_4h_model_comparison_ext_2026-09-16.json `
    --initial-train-rows 10172 --validation-rows 4000 --test-rows 4000 --fold-count 3
# 無外部特徵，同一個資料檔、同樣的 folds
& $py scripts\train_btc_4h_model_comparison.py --symbol BTC `
    --data data\features_ext\btc_4h.csv --manifest logs\feature_manifest.json `
    --output logs\btc_4h_model_comparison_base_2026-09-16.json `
    --initial-train-rows 10172 --validation-rows 4000 --test-rows 4000 --fold-count 3
```

其餘幣種將 `--symbol`、`--data`、`--manifest`、`--output` 換成對應檔名，`--initial-train-rows` 依上表（ETH 9238、SOL 9237、XRP 9261）。八次執行在 CPU 上約需 10 分鐘。

## 報告撰寫注意事項

- 本節結果應寫成**否定結果**，不要包裝成「外部因子略有幫助」。12 組中 6 組變好就是擲硬幣。
- 引用時使用真實數字（ROC-AUC 0.50–0.53、Brier 約 0.25），並說明未納入交易成本與持倉規則。
- SOL 缺鏈上、XRP 缺交易所流量，跨幣種比較表必須註明各幣種外部欄位數不同。
- 這個否定結果本身是有價值的方法論貢獻：它示範了如何在相同資料列上做受控的特徵消融，而不是換一批資料就宣稱有改善。
