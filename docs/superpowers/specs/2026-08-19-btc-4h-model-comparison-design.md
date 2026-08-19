# BTC 4 小時三模型 Walk-forward 比較設計

## 目標

建立 BTCUSDT 未來四小時漲跌的共用 walk-forward 評估流程，並以完全相同的資料範圍、時間 folds、機率校準與基準組，公平比較 XGBoost、LSTM 與 Transformer。此工作輸出研究評估結果，不構成交易策略或獲利主張。

## 範圍與非目標

- 範圍只包含 `data/features/btc_4h.csv` 的 BTCUSDT 四小時方向標籤與既有 53 個技術／交易特徵。
- 模型為 XGBoost、LSTM 與 Transformer；所有模型預測同一個 `target_up`。
- 不納入 ETH、SOL、XRP，亦不比較 1、12、24 小時視窗。
- 不實作下單、部位管理、交易成本、滑價或績效宣稱。

## 共用資料與時間切分

所有資料依 `open_time` UTC 嚴格遞增，`open_time`、`future_log_return`、`target_up`、`is_imputed` 不得作為模型特徵。資料須通過既有欄位、有限數值、唯一時間與排序檢查。

採 expanding-window walk-forward：每一 fold 均由連續且互不重疊的 train、validation、test 段組成；後續 fold 的 train 終點不得早於前一 fold。初始訓練段、validation 段、test 段與 fold 數由明確的 CLI 參數控制，預設使用至少三個完整 fold。任何 fold 若任一切分為空、時間重疊或單一類別，該執行應失敗且不寫出部分成果。

## 共用模型契約

每個模型實作一個共同介面：接收 fold 的 train／validation／test 資料與 53 個特徵名稱，輸出 validation 與 test 的正類機率。模型只能以 train 擬合；不得讀取 validation 或 test 的標籤、未來列或統計量。

- XGBoost 繼續使用既有樹模型設定作為傳統基準。
- LSTM 與 Transformer 由同一個固定長度的歷史序列輸入層產生樣本；每筆序列只含該時間點及更早的 53 個特徵，標籤仍為最後一筆時間點的四小時方向。
- LSTM 與 Transformer 的隨機種子、序列長度、batch size、epoch 上限與 early stopping 由 CLI 設定並寫入結果。

## 基準組與量測

每個 fold 的 validation 與 test 都必須列出：

1. 永遠預測上漲。
2. 訓練集多數類別。
3. 前一期方向（僅使用該列之前可得的方向）。
4. 使用同一 53 個特徵的 Logistic Regression。
5. 零交易訊號：不發出方向交易訊號，僅報告 coverage 為 0，不以 Accuracy、F1 或 AUC 偽裝為模型表現。

所有方向模型都報告 ROC-AUC、Brier score、accuracy、balanced accuracy、precision、recall、F1、2×2 混淆矩陣、實際上漲率、預測上漲率與樣本數。ROC-AUC 以未校準機率計算；決策指標以選定門檻後的類別計算。

## 校準與決策規則

每一 fold 的模型先只以 train 資料擬合。Platt scaling（logistic calibration）只在 validation 預測與標籤上擬合；校準器不得見到 test。門檻只由 validation 依最大 balanced accuracy 決定，並套用至該 fold test。若 validation 單一類別、校準失敗或所有候選門檻無效，fold 失敗。

結果同時保留未校準排序指標與校準後決策指標，並依校準後 test 機率五分位輸出平均預測機率、實際上漲率與樣本數，用來檢查校準品質。

## 輸出與可重現性

單一命令產生原子寫入的 JSON，包含資料／manifest 路徑、資料時間範圍、全部設定、每個 fold 的時間範圍及樣本數、各模型與基準的 validation／test 指標、校準與門檻、機率五分位、模型種子與版本資訊。只有所有指定模型完成後才可取代正式結果檔。

另提供只讀結果的 notebook／文件，顯示各模型跨 fold 的平均、標準差與逐 fold 比較；notebook 不可重新訓練模型。

## 測試與驗收

- 單元測試覆蓋 fold 無重疊、時間順序、序列無未來資訊、校準器與門檻不讀取 test、單一類別與非有限數拒絕。
- 整合測試以小型合成資料驗證三個模型和全部基準可產出相同 JSON schema。
- 結果測試驗證 zero-trade 不被當成方向分類器、所有模型都有相同必要欄位，且部分失敗不覆蓋既有結果檔。
- 研究結論僅在多數 fold 的 test ROC-AUC 高於 0.5、Brier score 不劣於適當常數基準、且結果未集中於單一市場期時，才稱為「值得進一步驗證」。否則結論必須是未證實穩定預測價值。

## 風險與限制

五年期單一資產資料仍可能存在市場 regime shift；walk-forward 降低但無法消除這個風險。校準與門檻會增加實驗自由度，因此僅能在 validation 做選擇。LSTM／Transformer 參數量與訓練時間較高，若現有 Python 環境缺少可用的深度學習框架，須先記錄明確相依性與可重現安裝方法，不能以未驗證替代實作冒充比較結果。
