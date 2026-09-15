# 加密貨幣走勢預測專題

這個資料夾已依照專題工作流程整理成資料、程式、notebook、文件、模型輸出與本地依賴幾個區塊。

## 資料夾結構

- `data/raw/`：原始加密貨幣 OHLCV CSV 資料。
- `src/`：可重複執行的 Python 訓練程式。
- `notebooks/`：探索、模型實驗與週進度 notebook。
- `docs/`：專題報告與進度紀錄文件。
- `models/`：模型、指標、validation report 與 manifest 輸出。
- `deps/`：離線 wheel 與本地 XGBoost runtime。

## 常用檔案

- `src/train_popular_coins.py`：訓練 BTC、ETH、SOL、XRP 的 popular coin 模型。
- `notebooks/train_popular_coins.ipynb`：從 notebook 重新執行訓練與檢視輸出。
- `notebooks/week3_naive_xgboost.ipynb`：Week 3 naive baseline 與 XGBoost walk-forward validation。
- `notebooks/week5_transformer.ipynb`：Week 5 Transformer 分類與回歸 walk-forward 實驗（已保留完整執行輸出）。
- `notebooks/調整後(預測4,12,24小時)/`：依序比較 XGBoost、LSTM、Transformer 在 4／12／24 小時預測 horizon 的調整實驗。
- `docs/7_基於機器學習與深度學習之加密貨幣走勢預測與穩定幣脫鉤風險預警系統.docx`：專題文件。
- `src/train_btc_24h_pytorch_transformer.py`：BTC 24 小時方向分類的 PyTorch Transformer 訓練與 checkpoint 輸出。
- `docs/btc_24h_pytorch_transformer.md`：PyTorch `.pth`／`.pkl` artifact 的訓練與推論說明。

## 路徑備註

Notebook 內的資料路徑已改成從 `notebooks/` 出發，例如 `../data/raw/btc_3y.csv`。`src/train_popular_coins.py` 則會自動從專題根目錄尋找 `data/raw/` 和 `models/`。
