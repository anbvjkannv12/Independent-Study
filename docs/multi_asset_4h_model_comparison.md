# 四種主流幣 4 小時模型比較摘要

四個交易對皆使用相同資料契約、53 項特徵與 3 個 expanding-window folds。每個模型只以該 fold 的 validation 期做 Platt 校準及門檻選擇，然後在後續 test 期評估。數值為三個 test folds 的平均 ROC-AUC。

| 幣種 | XGBoost | LSTM | Transformer |
| --- | ---: | ---: | ---: |
| BTCUSDT | 0.5289 | 0.5302 | 0.5318 |
| ETHUSDT | 0.5275 | 0.5226 | 0.5261 |
| SOLUSDT | 0.5023 | 0.5088 | 0.5067 |
| XRPUSDT | 0.5164 | 0.5231 | 0.5157 |

## 結論

BTC、ETH 的排序能力僅約 0.53；SOL 近乎隨機；XRP 僅有極微弱訊號。深度模型沒有在四個幣種中穩定勝過 XGBoost，也沒有形成能支持交易應用的跨幣種穩定性證據。

下一輪應先檢查目標定義與特徵在不同市場狀態下的漂移，或設計含「不交易」區間的三段式目標；在此之前不應以目前結果進行成本、持倉或獲利宣稱。

## 可追溯結果

- `logs/btc_4h_model_comparison.json`
- `logs/eth_4h_model_comparison.json`
- `logs/sol_4h_model_comparison.json`
- `logs/xrp_4h_model_comparison.json`
