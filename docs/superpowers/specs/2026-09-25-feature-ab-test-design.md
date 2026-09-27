# BTC 4 小時特徵 A/B test（消融實驗）設計

## 目標

BTC 4 小時三模型比較（`logs/btc_4h_model_comparison.json`）中，XGBoost 的 test ROC-AUC 平均約 0.529，訊號很弱。本實驗要回答：**哪一組特徵對方向預測真的有貢獻？** 做法是一次移除一個特徵群，看表現有沒有顯著變差。

本文件只定義實驗設計與判定規則，並且在看到任何 test 結果之前就先定案。

## 範圍與非目標

- 資料、標籤、folds 都跟 `2026-08-19-btc-4h-model-comparison-design.md` 相同：`data/features/btc_4h.csv`、`target_up`、53 個特徵。
- 只比較「特徵組合」，模型架構與超參數一律不動。
- 不調整超參數、不新增特徵，也不涉及交易成本或獲利主張。

## 假設

對每個特徵群 G：

- **A（對照組）**：全部 53 個特徵。
- **B（實驗組）**：53 個特徵扣掉 G。
- **H0**：AUC_A = AUC_B（G 沒有貢獻）。
- **H1**：AUC_A > AUC_B（移除 G 會讓表現變差）。採單尾檢定。

## 特徵群

各特徵群互不重疊，一次只移除其中一群。未列出的 11 個基礎特徵（`quote_asset_volume`、`number_of_trades`、`log_return`、`range_pct`、`body_pct`、`upper_shadow_pct`、`lower_shadow_pct`、`log_volume`、`volume_change`、`trade_intensity`、`quote_volume_change`）所有組別都保留。

| 代號 | 特徵群 | 欄位 | 數量 |
|---|---|---|---:|
| G1 | 技術指標 | `rsi_14`, `macd_pct`, `macd_signal_pct`, `bollinger_z_20`, `atr_14_pct` | 5 |
| G2 | 報酬／量能滯後 | `log_return_lag_{1,2,3,6,12,24}`, `volume_change_lag_{1,2,3,6,12,24}` | 12 |
| G3 | 滾動統計 | `return_mean_{6,12,24}`, `return_std_{6,12,24}`, `volume_mean_{6,12,24}`, `range_mean_{6,12,24}` | 12 |
| G4 | 時間週期 | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` | 4 |
| G5 | 主動買盤 | `taker_buy_base`, `taker_buy_quote`, `taker_buy_ratio`, `taker_buy_ratio_lag_{1,2,3,6,12,24}` | 9 |

5 + 12 + 12 + 4 + 9 + 11 = 53。

## 固定不變的條件

A、B 兩組只有 `feature_names` 不同，其餘一律相同：

- Folds：`scripts/btc_4h_evaluation.py` 的 `build_expanding_folds`，共 3 個 expanding folds（初始訓練 19,755 列，每個 fold 有 validation 4,000 列、test 4,000 列）。
- 模型：`scripts/btc_4h_model_adapters.py` 的 `fit_predict_xgboost`，使用相同的 `ModelSettings`。
- 校準與門檻：只用 validation 做 Platt 校準，門檻依 balanced accuracy 選定（沿用 `_evaluate_model_fold`）。
- 進入點：`run_model_comparison(dataset, feature_names, ...)`（`scripts/train_btc_4h_model_comparison.py`）。B 組只要傳入縮減後的 `feature_names` 即可。

**先只做 XGBoost**：跑得快，而且可以直接看 feature 影響。有顯著結果的特徵群，再用 LSTM／Transformer 複驗（見時程）。

## 指標

| 角色 | 指標 | 理由 |
|---|---|---|
| 主要 | 未校準 test ROC-AUC | 不受門檻影響，也跟既有比較流程的判準一致 |
| 次要 | Brier score、balanced accuracy | 分別檢查機率品質與實際分類效果 |
| 護欄 | B 的 Brier 不可比常數基準差 | 避免 AUC 看起來沒變，但機率已經壞掉 |

## 統計檢定

1. **配對設計**：A、B 是在完全相同的 test 列上預測，所以比較的是逐列配對的差異，不是兩組各自的平均數。
2. **Block bootstrap 求 ΔAUC 信賴區間**
   - ΔAUC = AUC_A − AUC_B，在 3 個 fold 合併的 12,000 列 test 上計算。
   - 相鄰列的 4 小時標的時間重疊，存在自相關，所以不能逐列獨立重抽。改用長度 24 列的 moving block，在每個 fold 內各自重抽，重抽 2,000 次。
   - 每次重抽時，A、B 使用同一組抽樣索引，保持配對。
3. **A/A 雜訊基準**：XGBoost 有 `subsample=0.8`、`colsample_bytree=0.8`，seed 不同結果就會不同。用 seed 42、43、44 各跑一次 A，取兩兩配對 |ΔAUC| 的最大值作為雜訊門檻 ε。
4. **多重比較**：G1–G5 共 5 個單尾檢定，以 Holm 法校正，控制整體 α = 0.05。
5. **可偵測的效果大小**：AUC ≈ 0.53、n = 4,000 時，單一 fold 的 AUC 標準誤約 0.009；3 個 fold 合併後約 0.005。配對設計會讓誤差再小一點，但保守估計 ΔAUC 要 **≥ 0.01** 才可靠偵測得到。比這更小的效果，本設計無法判斷。

## 判定規則（事先定案）

特徵群 G 判定為「**有貢獻**」，必須同時符合以下四點：

1. 3 個 fold 的 ΔAUC 都 > 0。
2. Holm 校正後，bootstrap 信賴區間的下界 > 0。
3. 合併的 ΔAUC > A/A 雜訊門檻 ε。
4. 符合 Brier 護欄。

其他結果的寫法：

- 不符合上述條件時，結論寫「**沒有證據顯示 G 有貢獻**」，不能寫成「G 沒有用」或「兩組一樣」。
- ΔAUC 顯著 < 0（移除 G 反而變好）時，另外標記為「可能是雜訊特徵」，但只作為後續假設，不在本實驗內直接刪除該特徵。

## 防止資料洩漏

- 特徵群的劃分在本文件中已經固定，不根據 test 結果調整。
- 任何探索與除錯都只看 validation 指標；test 只在最後判定時使用一次。
- 實驗跑完後如果想加測新的特徵群，要另開新文件，並納入重新計算的多重比較校正。

## 實作前需要補的東西（下一階段）

- 目前的 `logs/btc_4h_model_comparison.json` 只存了彙總指標，**沒有存逐列的 test 預測**，但 bootstrap 需要逐列預測。
  - 需要讓每個組別都輸出逐列預測，至少包含 `fold`、`open_time`、`target_up`、`test_raw`，存到 `logs/ab/<arm>_predictions.csv`。
- 彙總結果寫入 `logs/ab/feature_ablation.json`。結構包含：每個組別的 fold 指標、ΔAUC 點估計與 CI、Holm 校正後的 p 值、ε、判定結果。
- 結果報告：`docs/feature_ab_test_results.md`。

## 時程

| 步驟 | 內容 | 訓練次數（組別 × fold） |
|---|---|---:|
| 1 | A/A 雜訊基準：XGBoost × seed 42/43/44 | 3 × 3 |
| 2 | G1–G5 各自的 B 組 × XGBoost（A 沿用 seed 42） | 5 × 3 |
| 3 | 判定為有貢獻的特徵群 → 用 LSTM／Transformer 複驗，判定規則相同 | 視結果而定 |
| 4 | 撰寫結果報告 | — |

步驟 1、2 都是 XGBoost，訓練很快。步驟 3 的深度模型也受 seed 影響，所以複驗時同樣要先做 A/A 基準。
