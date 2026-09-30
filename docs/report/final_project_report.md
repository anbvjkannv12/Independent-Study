# 專題報告初稿：加密貨幣 K 線方向預測的可重現評估

> 狀態：繳交用初稿，格式仍需依系上規定轉成 Word/PDF。  
> 圖表來源：`docs/figures/figure_data.json` 與對應 `logs/` JSON。  
> 注意：本文不宣稱可獲利，不建議作為交易系統使用。

## 摘要

本研究使用 BTC、ETH、SOL、XRP 四種主流加密貨幣約五年的 Binance Spot 1 小時 K 線資料，評估僅以 K 線衍生特徵預測未來價格方向的可行性。資料流程包含品質檢查、缺漏補值、53 個 OHLCV 衍生特徵、時間順序 expanding walk-forward 切分、24 列 purge、validation 前置門檻選擇，以及 block bootstrap 與 Holm 校正。主要模型包含 XGBoost、LSTM 與 Transformer，並進一步檢驗外部特徵、特徵消融、多視窗、重新訓練頻率與三段式目標。

結果顯示，4h 方向訊號在統計上存在但非常弱；四幣種三模型 test ROC-AUC 多落在 0.50～0.53。外部特徵、特徵選擇、改變預測視窗、提高重訓頻率與三段式不交易標籤，都沒有穩定改善。三段式目標在扣除 0.2% 來回手續費後，每筆平均淨報酬仍為負。整體結論是：K 線衍生特徵對短期方向有微弱排序訊號，但目前不足以支撐交易或獲利宣稱；本研究的主要貢獻在於建立可重現、防資料洩漏且誠實呈現負結果的評估框架。

## 第 1 章 緒論

加密貨幣市場具有 24 小時交易、高波動與資料可取得性高等特性，因此常被用於機器學習價格預測研究。本研究聚焦於一個明確問題：在嚴格避免資料洩漏的條件下，單一幣種的 K 線衍生特徵能否穩定預測未來短期漲跌方向？

研究範圍限於 BTC、ETH、SOL、XRP 四種熱門幣的方向分類。本研究前一階段曾以 3 年資料做初步測試；本報告僅使用 5 年資料與嚴格評估流程。本階段不執行穩定幣脫鉤預警、市場連動風險分數或實際交易系統；這些屬於未來工作。研究貢獻包含：一套五年資料處理流程、一套預先登記與 walk-forward 評估框架，以及一組可追溯的負結果。

## 第 2 章 文獻探討摘要

既有金融市場研究指出，短期價格方向通常接近效率市場假說下的難預測情境。加密貨幣市場雖然波動較高，但也容易因高頻雜訊、交易成本與市場型態改變而讓模型在測試期失效。機器學習研究常見問題包括隨機切分時間序列、使用未來資料做標準化或特徵、在 test 上反覆調參，以及多重比較造成偶然顯著。本研究以時間序列 walk-forward、purge、validation-only 門檻、預先登記與 Holm 校正降低上述風險。

## 第 3 章 資料與方法

### 3.1 資料

資料來自 Binance Spot 1 小時 K 線，涵蓋 BTC、ETH、SOL、XRP，時間約為 2021-08 至 2026-08，時區採 UTC。資料結構與檔案位置見 `DATA_STRUCTURE.md`、`logs/fetch_manifest.json` 與 `logs/imputation_manifest.json`。

### 3.2 資料品質與補值

資料先經品質檢查，再對缺漏 K 線進行補值。補值列保留 `is_imputed` 稽核欄位，但不作為模型特徵。缺漏比例很低，不是主要結果來源。

### 3.3 特徵工程

模型使用 `logs/feature_manifest.json` 中的 53 個 K 線衍生特徵，包含報酬、波動、成交量、影線與實體、技術指標、lag 特徵、rolling 統計與時間週期特徵。完整定義見 `docs/data_features/feature_engineering.md`。

### 3.4 標籤

方向標籤為 `target_up = 1{future_log_return > 0}`，其中 `future_log_return = log(close[t+h]/close[t])`。4h 主實驗使用 h=4。三段式目標另定義下跌／不交易／上漲：`r < -0.002`、`-0.002 <= r <= 0.002`、`r > 0.002`。

### 3.5 模型與驗證設計

模型包含 XGBoost、LSTM、Transformer，並使用 Logistic、persistence 等基準。評估採 expanding walk-forward，每個 fold 保留 train、validation、test，並在邊界加入 24 列 purge，以降低重疊標籤造成的洩漏。門檻與校準只使用 validation，test 不參與選擇。

### 3.6 統計檢定與預先登記

多個後續實驗都先撰寫預先登記文件，commit 後才執行。顯著性評估使用 moving block bootstrap 與 Holm 校正，並以 seed 間差異作為雜訊門檻 ε。預先登記文件保存在 `docs/superpowers/specs/`。

## 第 4 章 實驗結果

### 4.1 4h 三模型比較

四幣種 4h 三模型結果如 F3。AUC 多落在 0.50～0.53：BTC 約 0.529～0.532、ETH 約 0.522～0.526、SOL 約 0.503～0.511、XRP 約 0.515～0.524。早期 BTC 曾依三 fold 平均 AUC 選定 Transformer，但後續更嚴格比較顯示模型間差距在雜訊範圍內，因此整體結論是沒有可靠的優勝模型。

### 4.2 外部特徵與特徵消融

外部特徵 A/B 加入鏈上、情緒與衍生品資料後，沒有穩定改善。特徵消融 v1、v2 檢查不同特徵群是否有穩定貢獻；v2 的 20 個「幣種 × 特徵群」組合中沒有組合通過預定判準。這表示瓶頸不太可能只是某個特徵群是否保留。

### 4.3 多視窗比較

多視窗比較結果如 F4 與 F5。1h 四幣種通過訊號關卡，但只有 XRP 1h 通過「比 4h 穩定地強」的完整判定；12h、24h 沒有穩定勝過 4h。1h 的描述性結果與 best-direction persistence 接近，顯示部分優勢可能來自短期反轉，而非模型學到額外資訊。

### 4.4 重訓頻率

重訓頻率實驗比較固定模型與不同重訓間隔。12 個「幣種 × 頻率」組合中沒有任何組合通過預定判準，ΔAUC 的 95% CI 多跨過 0（F7）。固定模型分段 AUC（F8）也沒有顯示頻繁重訓能穩定解決訊號衰退。

### 4.5 三段式目標

三段式目標讓模型可以選擇不交易，並和二分類加信心門檻比較。結果四個幣種皆未通過兩道關卡。扣除 0.2% 來回手續費後，T 組每筆平均淨報酬皆為負；交易比例曲線見 F9。結論為：沒有證據顯示三段式目標能改善可用性。

## 第 5 章 討論：為什麼訊號這麼弱

### 5.1 瓶頸在資料資訊量，不在模型

多個模型、特徵組合與標籤設計最後都收斂到接近的 AUC 範圍，表示主要限制可能是 OHLCV 衍生特徵本身包含的方向資訊不足，而不是單一模型設定不佳。

### 5.2 市場效率、資訊重疊與時間變動

53 個特徵大多由同一組 OHLCV 推導，資訊高度重疊。外部資料多為日頻或低頻，對 4h 方向預測可能太慢。validation AUC 與 test AUC 的落差（F6）顯示訊號可能隨時間改變，導致歷史上看似可用的模式在後續期間衰退。

### 5.3 統計訊號 vs 經濟價值

AUC 0.52 可代表統計上的微弱排序能力，但不代表可交易。F10 顯示，要打平 0.2% 手續費，需要明顯高於實測約 53% 的命中率；實測點仍落在成本線下方。這也是本研究不進行完整交易回測的原因：固定手續費已使結果為負，更完整的滑價與限制只會更嚴格。

### 5.4 排除的解釋

本研究用時間順序切分、purge、validation-only 門檻與獨立驗證程式降低資料洩漏疑慮。多視窗輸出驗證 V1～V8 全部 PASS；三段式實驗也在勘誤後重算並保留紀錄。因此較合理的解釋不是單一程式錯誤，而是方向訊號本身弱且不穩定。

## 第 6 章 結論與未來工作

本研究結論有三點：第一，K 線衍生特徵對短期方向有微弱訊號；第二，更換模型、特徵、預測視窗、重訓頻率與三段式目標都沒有穩定改善；第三，扣除交易成本後不具交易價值，不能宣稱可獲利。

研究限制包含：test 期間被多個實驗重複使用，不是獨立確認；資料主要來自現貨 K 線；交易成本只扣固定手續費，未含滑價；4h 標籤逐小時重疊；`data/features` 由較舊版特徵程式產生，零成交量補值列處理差異約影響 40 列且位於早期資料。

未來工作包含：2027-01-25 之後執行新資料獨立驗證；延伸研究已初步改以預測波動大小為問題，四幣種在既有資料上皆通過預定判準（見 `docs/experiments/volatility_target/volatility_target_extension_results.md`），但仍需新資料驗證後才能作為正式結論；也可加入幣種間領先落後、訂單簿與成交明細，或另開穩定幣脫鉤預警研究。

## 附錄 A：可重現步驟

1. 建立 Python 環境並安裝依賴：`pip install -r requirements-lock.txt`。
2. 放回 `data/features/` 與 `logs/` 備份。
3. 執行完整測試：`python -m unittest discover -s tests -v`。本次紀錄：107 項通過。
4. 執行多視窗驗證：`python scripts/verify_multi_horizon_outputs.py`，V1～V8 應為 PASS。
5. 重新由三段式逐列預測重算主表：`python scripts/run_three_class_target_comparison.py --recompute-from-predictions --output-dir logs/three_class_target`。
6. 產生圖表：`python scripts/make_report_figures.py`。

## 附錄 B：主要結果檔案

- 4h 三模型：`logs/*_4h_model_comparison.json`
- 多視窗：`logs/multi_horizon/multi_asset_multi_horizon.json`
- 重訓頻率：`logs/retrain_frequency/multi_asset_retrain_frequency.json`
- 三段式目標：`logs/three_class_target/multi_asset_three_class_target.json`
- 圖表數字：`docs/figures/figure_data.json`
- 備份清單：`backup_manifest.json`
