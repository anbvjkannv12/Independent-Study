# 四種主流幣 4 小時模型比較摘要

四個交易對皆使用相同資料契約、53 項特徵與 3 個 expanding-window folds。每個模型只以該 fold 的 validation 期做 Platt 校準及門檻選擇，然後在後續 test 期評估。數值為三個 test folds 的平均 ROC-AUC。

本頁以目前可用的定案／校正後結果為準：BTC 使用既有定案結果；ETH、SOL、XRP 使用 2026-09-16 校正後或重跑結果。

| 幣種 | XGBoost | LSTM | Transformer | 本輪定案判讀 |
| --- | ---: | ---: | ---: | --- |
| BTCUSDT | 0.5289 | 0.5302 | **0.5318** | 已選 Transformer 作為 BTC 定案模型，但訊號仍弱。 |
| ETHUSDT | 0.5221 | 0.5232 | **0.5257** | 無可靠優勝模型；Transformer 平均最高但 fold 間不穩。 |
| SOLUSDT | 0.5034 | 0.5089 | **0.5106** | 無可靠優勝模型；三者皆接近隨機。 |
| XRPUSDT | 0.5149 | **0.5236** | 0.5162 | 無可靠優勝模型；LSTM 平均最高但 F1 與 predicted up rate 異常。 |

## 逐幣種重點

### BTCUSDT

BTC 是四幣種中相對最高的一組，Transformer 平均 ROC-AUC 為 0.5318，略高於 LSTM 與 XGBoost。BTC 已於前階段選定 Transformer 作為定案模型，但此定案只代表研究比較中的相對選擇，不代表可交易或可部署。

### ETHUSDT

ETH 三模型平均 ROC-AUC 介於 0.5221～0.5257。Transformer 平均最高，但 Fold 3 降至約 0.5063，和 XGBoost、LSTM 的差距不足以稱為穩定優勢。本輪結論是「無可靠優勝模型」。

### SOLUSDT

SOL 三模型平均 ROC-AUC 介於 0.5034～0.5106，接近隨機，且 fold 間波動明顯。即使 Transformer 平均略高，也不具備可靠排序能力。本輪結論是「無可靠優勝模型」。

### XRPUSDT

XRP 的 LSTM 平均 ROC-AUC 最高，為 0.5236，且三 fold AUC 較穩定；但 LSTM 平均 F1 只有 0.2552，平均 predicted up rate 只有 0.1877，Fold 2 predicted up rate 更只有 0.0300。這表示模型可能只保留微弱排序能力，但門檻後分類行為不穩定。本輪結論是「無可靠優勝模型」。

## 結論

整體來看，四幣種 4 小時方向分類的排序能力大多停留在 0.50～0.53：

- BTC、ETH 約在 0.52～0.53，但仍是弱訊號。
- SOL 幾乎隨機。
- XRP 雖有 LSTM 排序訊號，但分類門檻後表現異常。
- 深度模型沒有在四個幣種中穩定勝過 XGBoost。
- 目前沒有形成能支持交易應用、持倉決策、獲利宣稱或風險預警的跨幣種穩定證據。

下一輪應優先處理：

1. 三段式目標，例如上漲／下跌／不交易，避免強迫所有樣本二元分類。
2. 加入手續費與滑價的策略回測，但只能在確認有穩定排序訊號後作為補充。
3. 1h／12h／24h horizon 的建模比較，列為本輪報告後的延伸實驗。
4. 特徵與標籤在不同市場狀態下的漂移檢查。
5. 若要重做 A/B test，應在主報告完成後再開新實驗，避免本輪報告擴張成資料探勘。

## 可追溯結果

- BTC：`logs/btc_4h_model_comparison.json`
- ETH：`logs/eth_4h_model_comparison_2026-09-16_rerun.json`
- SOL：`logs/sol_4h_model_comparison_2026-09-16.json`
- XRP：`logs/xrp_4h_model_comparison_2026-09-16.json`
- XRP 可追溯紀錄：`logs/xrp_4h_training_trace_2026-09-16.json`

## 範圍說明

本輪報告聚焦於 4 小時方向分類。1h／12h／24h 特徵資料雖已產出，但尚未納入本輪模型訓練與定案；外部特徵 A/B test 已有初步否定結果，列為補充實驗，不取代本頁的 4 小時 K 線特徵主線結果。
