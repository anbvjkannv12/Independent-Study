# 四幣種 4 小時特徵 A/B test（消融實驗）v2 設計

> 前一版：[`2026-09-25-feature-ab-test-design.md`](2026-09-25-feature-ab-test-design.md)（v1）  
> v1 結果與限制：[`../../feature_ab_test_results.md`](../../feature_ab_test_results.md)  
> 本文件在執行 v2 之前定案。未列在「與 v1 的差異」中的項目，一律沿用 v1。

## 目標

研究問題與 v1 相同：**五個特徵群（G1–G5）中，哪一群對 4 小時方向預測有貢獻？** v2 不新增特徵群、不調超參數，只修正 v1 事後檢視發現的設計缺陷，讓「有／沒有證據」的結論可以被信任。

## 與 v1 的差異（本版補足的不足）

| # | v1 的不足 | 後果 | v2 的修正 |
|---|---|---|---|
| 1 | A、B 都只用 seed 42，而 seed 42 在 ETH／SOL／XRP 剛好是最差的一顆 | ΔAUC 被系統性拉負，XRP 五組全負看起來像「特徵有害」 | **A、B 都跑 seed 42／43／44**，同 seed 配對，ΔAUC 取三顆 seed 的平均 |
| 2 | 沒有檢查 control 本身有沒有訊號 | SOL control AUC = 0.499，消融結果沒有資訊量，卻仍列入判定 | 新增**前置關卡**：control AUC 須顯著 > 0.5，未通過的幣種不做消融，直接標記「無可消融訊號」 |
| 3 | Brier 護欄套在 B 組上 | Control 本身的 Brier 就在常數基準邊緣，「未通過」與移除特徵無關，判準失去意義 | 護欄**改為檢查 control**，作為描述性指標列在關卡區，不參與特徵群判定 |
| 4 | 個別幣種報告檔名固定，重跑會覆蓋 | v1 結果可能被 v2 覆蓋，無法比對 | 輸出改到 `logs/ab_v2/` 與 `docs/*_feature_ab_test_results_v2.md`，v1 檔案保持不動 |
| 5 | 限制沒有寫進報告 | 讀者可能把「沒有證據」讀成「沒有用」 | v1 報告已補「限制」段落；v2 報告固定包含「限制」段落 |

**v2 無法補足、會在報告中明寫的限制**見文末。

## 固定不變的條件（沿用 v1）

- 資料：`data/features/<幣種>_4h.csv`，標籤 `target_up`，manifest `logs/feature_manifest.json`（53 個特徵）。
- Folds：`build_expanding_folds`，初始訓練 19,755 列、3 個 expanding folds，每個 fold validation 4,000 列、test 4,000 列，合併 test 共 12,000 列。
- 模型：`fit_predict_xgboost` 與預設 `ModelSettings`，只改 `seed`。
- 校準與門檻：只用 validation 做 Platt 校準，門檻依 balanced accuracy 選定。
- 特徵群劃分（G1–G5）與 11 個永遠保留的基礎特徵：與 v1 完全相同。
- 主要指標：合併 test 列上的未校準 ROC-AUC（`test_raw`）。
- Bootstrap：moving block，block 長度 24 列，在每個 fold 內各自重抽，2,000 次，bootstrap seed 20260925。
- 多重比較：每個幣種內 G1–G5 五個單尾檢定做 Holm 校正，整體 α = 0.05。

## 實驗組別

每個幣種：

| 組別 | 特徵 | seeds | 訓練次數（組別 × seed × fold） |
|---|---|---|---:|
| A（control） | 53 個 | 42、43、44 | 1 × 3 × 3 = 9 |
| B_without_Gk（k = 1…5） | 53 − Gk | 42、43、44 | 5 × 3 × 3 = 45 |

四個幣種合計 216 次 XGBoost 訓練。依 v1 實測（每組約 3 秒），訓練約 3–4 分鐘；bootstrap 計算量是 v1 的 3 倍，全部預估 **20–30 分鐘**（CPU）。

## 流程與判定

### 步驟 1：前置關卡（每個幣種）

1. 訓練 A 組三顆 seed。
2. 計算 **control AUC** = 三顆 seed 各自合併 AUC 的平均。
3. 用同一套 block bootstrap，對「三 seed 平均 AUC − 0.5」重抽，單尾 p = (1 + #{重抽值 ≤ 0}) / (2,000 + 1)。
4. **通過條件**：p ≤ 0.05。
5. 未通過 → 該幣種**不訓練 B 組**，報告寫「control 無顯著訊號，消融不具資訊量」，不列入特徵群判定。
6. 另外計算描述性 Brier 指標：control 三 seed 平均的校準機率 Brier，與「各 fold 訓練集上漲率常數」的 Brier 相比。此項只描述，不擋關卡。

關卡不跨幣種做多重比較校正。它是篩選條件，不是研究結論；誤放一個無訊號的幣種進入消融，只會多出一組「沒有證據」，不會產生假陽性的特徵結論。

### 步驟 2：A/A 雜訊基準

與 v1 相同：三顆 A seed 兩兩配對的 |ΔAUC| 取最大值作為 ε。

v2 比較的是「三 seed 平均 vs 三 seed 平均」，雜訊比單 seed 小（理論上約 1/√3）。沿用單 seed 的 ε 是**刻意保守**的選擇，讓判準不會比 v1 寬鬆。

### 步驟 3：消融（僅通過關卡的幣種）

對每個特徵群 Gk：

- **ΔAUC** = mean_s AUC(A_s) − mean_s AUC(B_s)，s ∈ {42, 43, 44}，在合併的 12,000 列上計算。
- **逐 fold ΔAUC**：同樣的定義，只算單一 fold 的 4,000 列。
- **Bootstrap**：每次重抽產生一組列索引，六組預測（A×3、B×3）共用同一組索引後算上式，維持配對。
- 單尾 p 值與 Holm 校正方式同 v1。

### 判定規則（事先定案）

特徵群判定為「**有貢獻**」須同時符合：

1. 3 個 fold 的 ΔAUC 都 > 0。
2. Holm 校正後 p ≤ 0.05，且 Holm 水準下的 bootstrap 下界 > 0。
3. 合併 ΔAUC > ε。

v1 的第 4 項（B 組 Brier 護欄）移到步驟 1 的描述性指標，理由見「與 v1 的差異」第 3 點。

其他結果的寫法沿用 v1：

- 不符合 → 「沒有證據顯示此特徵群有貢獻」，不能寫成「沒有用」。
- ΔAUC < 0 且未校正的 95% CI 上界 < 0 → 標記「可能是雜訊特徵」，只作為後續假說。

## 輸出

| 內容 | 位置 |
|---|---|
| 逐列預測 | `logs/ab_v2/<幣種>/A_seed{42,43,44}_predictions.csv`、`B_without_G{k}_seed{42,43,44}_predictions.csv` |
| 各幣種完整結果 | `logs/ab_v2/<幣種>/feature_ablation.json`（含關卡、ε、各組 bootstrap 與判定） |
| 四幣種彙整 JSON | `logs/ab_v2/multi_asset_feature_ablation.json` |
| 彙整報告 | `docs/feature_ab_test_results_v2.md` |
| 各幣種報告 | `docs/<幣種>_feature_ab_test_results_v2.md` |

v1 的 `logs/ab/` 與 `docs/*_feature_ab_test_results.md` 不會被覆寫。

## 需要的程式修改（`scripts/run_feature_ablation.py`）

1. B 組改為三顆 seed，逐列預測檔名加上 seed。
2. AUC 與 bootstrap 函式改成接受「多顆 seed 的預測清單」，統計量取 seed 平均；傳入單一 DataFrame 時行為與 v1 相同，既有測試不需改。
3. 新增前置關卡（bootstrap 檢定 control AUC > 0.5），未通過則略過 B 組。
4. Brier 指標改算在 control 的 seed 平均機率上。
5. 預設輸出路徑改為 `logs/ab_v2/` 與 `docs/feature_ab_test_results_v2.md`；個別幣種報告檔名改由彙整報告檔名推導，避免覆蓋 v1。
6. 報告加入「前置關卡」表格與固定的「限制」段落。
7. `tests/test_feature_ablation.py` 新增兩個測試：多 seed 清單的 bootstrap 在配對下具決定性；有訊號的 control 通過關卡，隨機分數的 control 不通過。

## 執行步驟

```powershell
cd C:\Users\user\專題
$py = ".\.venv-transformer\Scripts\python.exe"

# 1. 單元測試
& $py -m pytest tests\test_feature_ablation.py -q

# 2. 先跑 BTC 確認流程（約 5–7 分鐘）
& $py scripts\run_feature_ablation.py --symbols BTC

# 3. 確認 BTC 輸出無誤後跑四個幣種（會重跑 BTC，結果相同）
& $py scripts\run_feature_ablation.py
```

`scripts/` 需要在 import 路徑上，指令從專案根目錄執行即可（與 v1 相同）。

## 報告撰寫規則

1. 先報告關卡：哪些幣種通過、哪些沒有，未通過的幣種不討論特徵群。
2. 與 v1 並列比較：v1 的 13/20 負值在 v2 剩多少，用來驗證 seed 偏誤的推論。
3. 對 G3（滾動統計）的觀察：v1 事後看到 G3 在三個幣種為負。v2 **不因此調整判準**，只照常報告 G3 的結果。
4. 「沒有證據」不等於「沒有用」，這句話要出現在結論中。

## v2 仍無法補足的限制（報告中須明寫）

1. **Test 集重複使用。** v2 使用與 v1 相同的 test 期間，而 v1 結果已經看過。v2 是「修正設計後的重新分析」，**不是獨立的確認實驗**。真正的確認需要 2026-08-09 之後的新資料，建議累積至少 4,000 列（約 22 個月）後再用 v2 流程跑一次。
2. **檢定力仍然有限。** 多 seed 平均只能降低模型隨機性造成的雜訊，無法降低 test 樣本本身的抽樣誤差；可偵測的 ΔAUC 仍約為 0.01。
3. **ε 仍只由三組配對估計**，數值不穩定；採用保守方向。
4. **只測 XGBoost。** 若有特徵群判定為有貢獻，LSTM／Transformer 複驗須另開文件，並先做深度模型的 A/A 基準（同 v1 時程第 3 步）。
