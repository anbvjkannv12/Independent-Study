# 三段式目標實驗 v2 勘誤（定案）

> 日期：2026-09-28  
> 原預先登記：[`2026-09-28-three-class-target-design.md`](2026-09-28-three-class-target-design.md)  
> 原實作 commit：`d99dccc`；原結果 commit：`261b7e3`

本文件只修正**統計分析程式與原預先登記不一致之處**。k、c、q、切分、模型、seed、bootstrap 設定、判定規則全部不變；模型不重新訓練，逐列預測 CSV 沿用原結果。

## 發現的錯誤

1. **比較關卡的 bootstrap 統計量錯誤。**原程式先把三個 seed 的逐列淨報酬平均，再逐列計算「T − B」後取平均。T 或 B 有一方沒交易的列在相減後會變成 NaN 而被略過，所以實際算的是「T 與 B 同時交易的列」上的平均差，而不是預先登記 A.8 的 Δ = m_T − m_B。症狀：BTC 的 Δ < 0，原始單尾 p 卻是 0.108。
2. **主要指標 m 的計算順序與預先登記不同。**預先登記 A.7 是「各 seed 先算 m，再取三 seed 平均」；原程式是先逐列平均三個 seed 的淨報酬，再取平均。關卡一的 bootstrap 也受影響。
3. **漏做的預先登記項目：**
   - A.13 第 4 項：B 組 test AUC 三 seed 平均與多視窗 4h 結果差距 ≤ 0.01 的一致性檢查。
   - A.9 第 7 項：不重疊版本（只取 `open_time` 小時數為 0、4、8、12、16、20 的列）的 m。
   - A.14：報告缺少逐 fold m、95% CI、三類比例、T／B 方向 AUC 對照與不重疊版本表格。

## 修正方式

1. m：T、B 各 seed 用自己的 `net_return` 算平均（只算有交易的列），再取三 seed 平均；逐 fold 同理。
2. 兩道關卡的 bootstrap 共用同一組重抽索引（區塊長度 48、各 fold 內重抽、2,000 次、seed 20261001，與原設定相同）：
   - 關卡一重抽統計量：三 seed 平均的 m_T。
   - 關卡二重抽統計量：三 seed 平均的 m_T − 三 seed 平均的 m_B。
   - 單尾 p = (1 + #{重抽統計量 ≤ 0}) / 2001；四幣種各自 Holm 校正（兩道關卡分開校正），與原設定相同。
3. 補做 A.13 第 4 項一致性檢查，參照 `logs/multi_horizon/multi_asset_multi_horizon.json` 中 `symbols.<幣種>.horizons.4.mean_pooled_auc`；差距 > 0.01 即報錯停止。
4. 補做不重疊版本的 m（描述性，不參與判定）。
5. 報告補齊 A.14 所列表格。

## 執行

新增 `--recompute-from-predictions` 模式：只讀取既有的 `logs/three_class_target/<幣種>/*_predictions.csv` 與 JSON，重算上述統計並重寫 JSON 與報告，不重新訓練模型。原始預測 CSV 不修改。

## 對結論的影響

判定規則不變；若修正後的判定與原結論不同，以修正後為準，並在結論文件註明。
