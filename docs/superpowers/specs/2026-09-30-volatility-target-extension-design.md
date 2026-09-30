# 延伸研究預先登記：未來波動大小預測（定案）

> 定案日期：2026-09-30  
> 目的：依最後階段工作 9，將後續研究問題從「方向預測」改為「波動大小預測」。  
> 狀態：只建立延伸研究計畫與預先登記；不得使用 2026-08-09 之後資料調參或選規則。

## 研究問題

在相同資料、特徵與 walk-forward 評估框架下，OHLCV 衍生特徵是否能比方向分類更穩定地預測未來 4 小時的波動大小？

## 固定資料範圍

- 資料：`data/features/<symbol>_4h.csv`，BTC/ETH/SOL/XRP。
- 使用欄位：`logs/feature_manifest.json` 的 53 個特徵。
- 不使用 2026-08-09 之後資料做任何設計調整。
- 切分：沿用三段式實驗的 4h 單視窗 fold：`initial = 列數 − 3 × (4000 + 4000 + 2 × 24)`，validation/test 各 4000，purge 24，fold_count 3。

## 標籤

令 `a = abs(future_log_return)`。每個 fold 只用 train 區段決定門檻：

- `target_high_vol = 1{a > median_train_abs_return}`。
- validation/test 使用該 fold train 中位數，不使用 validation/test 重新估計門檻。

此定義使每個 fold 的 train 類別比例約 50/50，避免先看 test 的波動分布。

## 模型與基準

- 主模型：XGBoost，沿用方向分類的固定超參數與 seeds 42/43/44。
- 基準 1：train 高波動比例常數機率。
- 基準 2：persistence volatility，過去 4 小時 `abs(log_return).rolling(4).sum()`，方向只用 train AUC 決定是否反向；預期通常不需反向。
- 不加入新特徵、不調參、不搜尋其他 horizon。

## 主要指標與判定

主要指標：pooled test ROC-AUC，三 seed 平均。

訊號關卡：
1. 每個 fold 的 AUC 都 > 0.5。
2. moving block bootstrap（block length 48、2,000 次、seed 20261002）相對 0.5 的單尾 p，四幣種 Holm 校正後 p ≤ 0.05。

比較關卡：相對 persistence volatility：
1. 每個 fold 的 ΔAUC 都 > 0。
2. Holm 校正後 p ≤ 0.05。
3. ΔAUC 大於三 seed A/A 最大差異 ε。

## 停止條件

- ≥ 2 個幣種同時通過訊號與比較關卡：波動大小預測列為下一階段主研究方向。
- 只有 1 個幣種通過：列為候選假說，等待新資料驗證。
- 0 個幣種通過：不繼續在 OHLCV 特徵上調整；改評估跨幣種或訂單簿資料。

## 描述性檢查

- validation vs test AUC。
- 各 fold train/test 高波動比例。
- precision/recall at validation-selected thresholds。
- 和方向模型 AUC 的對照。

## 未納入本實驗的變化

- 不搜尋不同分位數（例如 top 30%）。
- 不改 horizon。
- 不加入跨幣種、訂單簿或新聞資料。
- 不使用 2026-08-09 之後資料。
