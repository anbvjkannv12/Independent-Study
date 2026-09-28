# 五年資料 4 小時方向分類階段報告

更新日期：2026-09-26  
報告期限：2026-09-30  
研究範圍：BTCUSDT、ETHUSDT、SOLUSDT、XRPUSDT；Binance Spot 五年 1 小時 K 線；預測未來 4 小時漲跌方向。  
模型：XGBoost、LSTM、Transformer。  
驗證：3 個 expanding-window walk-forward folds，validation-only Platt 校準與門檻選擇。

## 一、研究範圍與本輪定位

本輪聚焦於 **未來 4 小時二元方向分類**：

```text
target_up = future_log_return > 0
future_log_return = log(close[t+4] / close[t])
```

本輪不處理：

- 1h／12h／24h 模型訓練與定案。
- 報酬率回歸任務。
- 三段式目標，例如上漲／下跌／不交易。
- Stacking 融合。
- 新一輪外部特徵 A/B test。
- 手續費、滑價、持倉限制後的正式交易策略回測。

上述項目列為後續研究方向，避免本輪報告在訊號偏弱的情況下擴大成資料探勘。

## 二、主要結果

四幣種 4 小時方向分類，三個 test folds 的平均 ROC-AUC 如下：

| 幣種 | XGBoost | LSTM | Transformer | 本輪定案判讀 |
| --- | ---: | ---: | ---: | --- |
| BTCUSDT | 0.5289 | 0.5302 | **0.5318** | BTC 選 Transformer 作為研究定案模型，但不可解讀為可交易。 |
| ETHUSDT | 0.5221 | 0.5232 | **0.5257** | 無可靠優勝模型。 |
| SOLUSDT | 0.5034 | 0.5089 | **0.5106** | 無可靠優勝模型，幾乎隨機。 |
| XRPUSDT | 0.5149 | **0.5236** | 0.5162 | 無可靠優勝模型；LSTM 的 F1 與 predicted up rate 異常。 |

## 三、逐幣種結論

### BTCUSDT

BTC 的 Transformer 平均 ROC-AUC 為 0.5318，是四幣種中相對最高且已定案的模型。但 0.53 仍屬弱訊號，只能作為研究比較中的相對最佳模型，不代表已能支撐交易決策。

### ETHUSDT

ETH 的 Transformer 平均 ROC-AUC 為 0.5257，略高於 LSTM 與 XGBoost，但 Fold 3 降至接近隨機，跨 fold 穩定性不足。因此 ETH 不選單一可部署模型，結論為「無可靠優勝模型」。

### SOLUSDT

SOL 三模型平均 ROC-AUC 僅 0.5034～0.5106，接近隨機。即使 Transformer 平均略高，也沒有實質穩定排序能力。SOL 結論為「無可靠優勝模型」。

### XRPUSDT

XRP 的 LSTM 平均 ROC-AUC 為 0.5236，三 fold 排序結果相對穩定，但平均 F1 只有 0.2552，平均 predicted up rate 只有 0.1877，Fold 2 predicted up rate 更只有 0.0300。這代表 LSTM 可能有微弱排序能力，但門檻後分類行為不可靠。XRP 結論為「無可靠優勝模型」。

## 四、A/B test 狀態

### 4.1 外部特徵 A/B test（2026-09-16）

外部特徵 A/B test 已先做過一輪，產物包括：

- `docs/external_features_ab_comparison.md`
- `logs/btc_4h_model_comparison_base_2026-09-16.json`
- `logs/btc_4h_model_comparison_ext_2026-09-16.json`
- `logs/eth_4h_model_comparison_base_2026-09-16.json`
- `logs/eth_4h_model_comparison_ext_2026-09-16.json`
- `logs/sol_4h_model_comparison_base_2026-09-16.json`
- `logs/sol_4h_model_comparison_ext_2026-09-16.json`
- `logs/xrp_4h_model_comparison_base_2026-09-16.json`
- `logs/xrp_4h_model_comparison_ext_2026-09-16.json`

受控比較結果顯示，加入鏈上、情緒、衍生品欄位後，12 組模型比較中 6 組變好、平均差異 −0.0017、中位數 −0.0001；test ROC-AUC 仍停留在 0.50～0.53，Brier score 變化都在 ±0.0007 以內。目前沒有證據顯示外部特徵能穩定改善四幣種 4 小時方向分類；XRP 加入外部特徵後更出現 LSTM 明顯退步。因此外部特徵 A/B test 在本輪報告中定位為補充實驗與否定結果，不作為主線。

### 4.2 特徵消融 A/B test v1（2026-09-25）

v1 使用 XGBoost 進行一次移除一個特徵群的消融實驗，特徵群為 G1 技術指標、G2 報酬／量能滯後、G3 滾動統計、G4 時間週期、G5 主動買盤。此處需特別區分：消融實驗只測 XGBoost；BTC 的本輪定案模型仍是 Transformer，因此消融結論不能直接等同於 BTC 定案 Transformer 的特徵重要性。

v1 的設計包含 A/A 雜訊基準 ε、配對 moving block bootstrap 與幣種內 Holm 校正，但 A、B 組都只用 seed 42，且未設 validation 前置關卡。結果為 20 個「幣種 × 特徵群」組合中 0 個通過預定判準；依 `logs/ab/multi_asset_feature_ablation.json` 重算，負 ΔAUC 為 11/20。v1 因此不能寫成「特徵沒有用」，只能寫成「沒有證據顯示任一特徵群有貢獻」。v1 報告結尾已補上限制段落，說明單一 seed、control 訊號偏弱、檢定力與 Brier 護欄等問題。

### 4.3 特徵消融 A/B test v2（2026-09-26）

v2 是修正設計後的重新分析，主要依據為 worktree `C:/Users/user/orca/workspaces/專題/feature-ab-test-v2/docs/feature_ab_test_v2_analysis.md` 與 JSON `C:/Users/user/orca/workspaces/專題/feature-ab-test-v2/logs/ab_v2/multi_asset_feature_ablation.json`。v2 在執行前先寫下設計文件 `docs/superpowers/specs/2026-09-26-feature-ab-test-v2-design.md`，重點包括：

1. 預先登記研究問題、判準與輸出路徑。
2. 以 A/A 測試估計模型隨機性的雜訊門檻 ε。
3. A、B 組都跑 seed 42／43／44，ΔAUC 取三 seed 平均。
4. 新增 validation 前置關卡：control validation AUC 須顯著大於 0.5，避免在 control 本身無訊號時解讀消融。
5. 使用配對 moving block bootstrap，保留時間序列區塊結構。
6. 對每個幣種內五項比較做 Holm 校正。

v2 結果仍為 20 個組合中 0 個通過全部預定判準；負 ΔAUC 為 10/20。四幣種 validation 前置關卡皆通過，但 test AUC 皆低於 validation：BTC −0.006、ETH −0.014、SOL −0.030、XRP −0.028。這表示 validation 表現會高估後續 test 表現，4 小時方向訊號可能隨時間衰退。

最接近通過的是 **BTC G4（時間週期）**，但只能列為候選假說：三個 fold ΔAUC 皆為正，是 20 組中唯一三 fold 皆正者；原始單尾 p = 0.018，Holm 校正 p = 0.092；合併 ΔAUC = 0.0046，仍小於 BTC 的 A/A 雜訊門檻 ε = 0.0049。因此不能宣稱 BTC G4 有貢獻，只能寫成下一階段可用新資料獨立驗證的候選假說。

綜合外部特徵 A/B、特徵消融 v1、特徵消融 v2，目前沒有證據顯示加入外部特徵或移除任一特徵群能穩定改善 4 小時方向分類。三個受控實驗共同指向同一件事：瓶頸更可能在 4h 方向訊號本身太弱，而不是特徵選擇。

## 五、1h／12h／24h 多視窗比較（已完成）

本輪不訓練 1h／12h／24h，但已列為下一階段主要工作。

理由是目前三個受控實驗都顯示，4h 方向分類的瓶頸在訊號本身太弱，而不在特徵選擇：外部特徵 A/B 沒有證據顯示能穩定改善，v1 消融沒有任何特徵群通過判準，v2 在三 seed 平均、validation 前置關卡與 Holm 校正後仍沒有任何特徵群通過判準。此時繼續微調 4h 特徵組合，不太可能帶來實質突破。

因此，下一階段應把問題改成「同一套嚴格流程下，1h／4h／12h／24h 哪個預測視窗有較穩定的方向訊號」。執行前會先撰寫預先登記文件，固定資料切分、模型、指標、比較規則與停止條件；同時補上 purge 間隔，降低 4 小時標籤在 train／validation／test 邊界重疊造成的洩漏疑慮。完成設計文件後，再開始多視窗實驗。

已於 2026-09-28 完成比較及輸出驗證：1h 四幣種都通過訊號關卡，但只有 XRP 1h 通過「比 4h 穩定地強」的完整判定；12h、24h 沒有任何幣種比 4h 穩定地強。依預先登記，4h 維持主要視窗，1h 僅列為待獨立驗證的候選假說。詳見 [`multi_horizon_conclusions.md`](multi_horizon_conclusions.md)。

## 六、整體主結論

本輪結果顯示，目前 53 個 K 線衍生特徵對未來 4 小時方向只有非常有限的預測訊號。四幣種平均 ROC-AUC 多落在 0.50～0.53，Brier score 也接近二元隨機情境下的 0.25。深度模型沒有穩定勝過 XGBoost，也沒有出現跨幣種、跨 fold 都可靠的優勝模型。

加入外部特徵或移除任何特徵群後，test AUC 仍維持在 0.50～0.53；三輪受控 A/B test 都沒有證據顯示特徵選擇是主要瓶頸。整體判讀是：目前限制主要在 4 小時方向訊號強度，而不是現有特徵群是否保留或外部特徵是否加入。

因此，本階段成果應定位為：

> 已建立可重現的五年資料處理、特徵工程與 walk-forward 評估流程，並誠實確認四幣種 4 小時方向分類只具有弱且不穩定的排序訊號，尚不足以支撐交易、獲利或風險預警宣稱。

## 七、後續工作建議

建議順序：

1. **多視窗 1h／4h／12h／24h 比較（預先登記）（已完成，2026-09-28）**：只有 XRP 1h 通過相對 4h 的完整判定，其餘維持原視窗；詳見 [`multi_horizon_conclusions.md`](multi_horizon_conclusions.md)。
2. **BTC G4 時間特徵獨立驗證**：將 BTC G4 寫成候選假說，等 2026-08-09 之後的新資料累積足夠後，用 v2 流程獨立驗證。
3. **定期重新訓練 vs 固定模型**：對應 v2 觀察到的 validation→test AUC 衰退，檢查重新訓練頻率是否比調整特徵更有效。（已完成，2026-09-28）：四幣種 × 三種重訓頻率沒有任何組合通過預定判準，維持現行標準；詳見 [`retrain_frequency_conclusions.md`](retrain_frequency_conclusions.md)。
4. **三段式目標**：若方向二分類訊號仍偏弱，改為上漲／下跌／不交易，檢查是否能提升可用性。
5. **交易成本回測**：在模型訊號被確認後，再加入手續費、滑價與持倉限制，避免用弱訊號直接宣稱交易價值。

## 八、主要文件與產物

- 多視窗預先登記：`docs/superpowers/specs/2026-09-27-multi-horizon-comparison-design.md`
- 多視窗結果：`docs/multi_horizon_comparison_results.md`
- 多視窗結論：`docs/multi_horizon_conclusions.md`
- 多視窗 logs：`logs/multi_horizon/`（不進 git；原始來源 worktree `C:/Users/user/orca/workspaces/專題/multi-horizon-comparison/logs/multi_horizon/`）
- 多視窗輸出驗證：`logs/multi_horizon/verification.json`（不進 git）
- 重訓頻率預先登記：`docs/superpowers/specs/2026-09-29-retrain-frequency-design.md`
- 重訓頻率結果：`docs/retrain_frequency_results.md`
- 重訓頻率結論：`docs/retrain_frequency_conclusions.md`
- 重訓頻率 logs：`logs/retrain_frequency/`（不進 git）

- 本階段報告：`docs/five_year_4h_final_report.md`
- 四幣種彙整：`docs/multi_asset_4h_model_comparison.md`
- BTC：`logs/btc_4h_model_comparison.json`
- ETH：`logs/eth_4h_model_comparison_2026-09-16_rerun.json`
- SOL：`logs/sol_4h_model_comparison_2026-09-16.json`
- XRP：`logs/xrp_4h_model_comparison_2026-09-16.json`
- XRP 可追溯紀錄：`logs/xrp_4h_training_trace_2026-09-16.json`
- XRP 定案文件：`docs/xrp_4h_model_comparison.md`
- 外部特徵 A/B：`docs/external_features_ab_comparison.md`
- 外部特徵 A/B logs：`logs/{btc,eth,sol,xrp}_4h_model_comparison_{base,ext}_2026-09-16.json`
- 特徵消融 v1 設計：`docs/superpowers/specs/2026-09-25-feature-ab-test-design.md`
- 特徵消融 v1 彙整：`docs/feature_ab_test_results.md`
- 特徵消融 v1 logs：`logs/ab/`、`logs/ab/multi_asset_feature_ablation.json`
- 特徵消融 v2 設計：主資料夾 `docs/superpowers/specs/2026-09-26-feature-ab-test-v2-design.md`
- 特徵消融 v2 彙整與分析：worktree `C:/Users/user/orca/workspaces/專題/feature-ab-test-v2/docs/feature_ab_test_results_v2.md`、`C:/Users/user/orca/workspaces/專題/feature-ab-test-v2/docs/feature_ab_test_v2_analysis.md`
- 特徵消融 v2 logs：worktree `C:/Users/user/orca/workspaces/專題/feature-ab-test-v2/logs/ab_v2/`、`C:/Users/user/orca/workspaces/專題/feature-ab-test-v2/logs/ab_v2/multi_asset_feature_ablation.json`
