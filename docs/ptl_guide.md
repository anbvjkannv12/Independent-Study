# BTC 4h Transformer `.ptl` 行動端模型說明

> 更新日期：2026-09-30  
> 狀態：**已在本機產生並通過驗證**（見第六節）。  
> 對象：要把模型放進 App 或做展示的同學

---

## 一、先知道這幾件事

1. **模型來源**：這是 5 年資料研究中 BTC 4h 的定案模型（Transformer）。資料、53 個特徵、模型結構與訓練設定，都和 4h 三模型比較（`scripts/train_btc_4h_model_comparison.py`、`logs/btc_4h_model_comparison.json`）完全相同。差別只在於這次用全部資料訓練一次，並把權重存了下來，原本的比較實驗不存權重。
2. **資料範圍**：只用到 2026-08-09 06:00 UTC 為止的資料，沒有用到之後的新資料，不影響 2027 年的新資料獨立驗證。
3. **訊號很弱**：驗證期 ROC-AUC 約 0.540，和整個研究的結論一致（4h 方向訊號約 0.52～0.53，扣手續費後不具交易價值）。**只能當技術展示**，不能宣稱能預測價格或獲利。
4. **輸出沒有經過機率校準**：輸出值可以用來排序（越高越偏向上漲），但不要解讀成「真實上漲機率」。
5. **PyTorch Mobile（`.ptl` 使用的 lite interpreter）目前處於維護狀態**，官方新的行動端方案是 ExecuTorch。本機的 Windows torch 沒有 XNNPACK，所以匯出時跳過了 `optimize_for_mobile`；模型仍可由 lite interpreter 正常載入，且已驗證結果正確。

---

## 二、檔案

| 檔案 | 說明 |
|---|---|
| `models/btc_4h_transformer/btc_4h_transformer.ptl` | 行動端模型（約 75 KB） |
| `models/btc_4h_transformer/btc_4h_transformer.ptl.json` | 輸入輸出規格、特徵順序、訓練區間、驗證 AUC |
| `models/btc_4h_transformer/btc_4h_transformer.pth` | 原始 PyTorch 權重與標準化參數（約 53 KB） |
| `scripts/build_btc_4h_transformer_ptl.py` | 訓練、存檔、匯出、驗證的一支程式 |
| `tests/test_btc_4h_transformer_ptl.py` | 單元測試 |
| `logs/ptl_btc_4h_verify.json` | 驗證結果（在資料壓縮檔裡） |

---

## 三、模型規格

| 項目 | 值 |
|---|---|
| 預測目標 | `close[t + 4] > close[t]`（4 小時後收盤價高於現在） |
| 模型類別 | `TransformerClassifier`（`scripts/btc_4h_model_adapters.py`）：53 維 → 線性投影到 32 維 + 學習式位置參數 → 1 層 Transformer Encoder（4 heads，FFN 64）→ 取最後一個時間點 → 1 |
| 訓練設定 | `ModelSettings()` 預設值：seed 42、序列長度 24、batch 128、最多 30 epochs、patience 5、學習率 0.001、hidden 32、heads 4 |
| 訓練區間 | 2021-08-10 11:00 ～ 2026-02-23 14:00 UTC |
| 驗證區間（early stopping） | 2026-02-23 15:00 ～ 2026-08-09 06:00 UTC（最後 4,000 列） |
| 驗證期 ROC-AUC | 0.540（描述性） |

### 3.1 `.ptl` 的輸入與輸出

匯出時把「標準化」與「sigmoid」都包進 `.ptl`，使用端不需要自己處理：

| 項目 | 規格 |
|---|---|
| 輸入 | `(batch, 24, 53)`，float32，**原始特徵值（不要自己標準化）** |
| 列的順序 | 由舊到新；最後一列是最新一根**已收盤**的 1 小時 K 線 |
| 輸出 | `(batch,)`，上漲分數（0～1，未校準） |
| 顯示門檻 | 0.5（僅供顯示「偏多／偏空」，研究中沒有用它做判定） |

### 3.2 53 個特徵的順序（必須完全一致）

與 `scripts/feature_engineering.py` 的 `EXPECTED_FEATURE_NAMES`、`logs/feature_manifest.json` 相同：

```
 1 quote_asset_volume      19 dow_cos                  37 taker_buy_ratio_lag_24
 2 number_of_trades        20 log_return_lag_1         38 return_mean_6
 3 taker_buy_base          21 volume_change_lag_1      39 return_std_6
 4 taker_buy_quote         22 taker_buy_ratio_lag_1    40 volume_mean_6
 5 log_return              23 log_return_lag_2         41 range_mean_6
 6 range_pct               24 volume_change_lag_2      42 return_mean_12
 7 body_pct                25 taker_buy_ratio_lag_2    43 return_std_12
 8 upper_shadow_pct        26 log_return_lag_3         44 volume_mean_12
 9 lower_shadow_pct        27 volume_change_lag_3      45 range_mean_12
10 atr_14_pct              28 taker_buy_ratio_lag_3    46 return_mean_24
11 log_volume              29 log_return_lag_6         47 return_std_24
12 volume_change           30 volume_change_lag_6      48 volume_mean_24
13 trade_intensity         31 taker_buy_ratio_lag_6    49 range_mean_24
14 taker_buy_ratio         32 log_return_lag_12        50 rsi_14
15 quote_volume_change     33 volume_change_lag_12     51 macd_pct
16 hour_sin                34 taker_buy_ratio_lag_12   52 macd_signal_pct
17 hour_cos                35 log_return_lag_24        53 bollinger_z_20
18 dow_sin                 36 volume_change_lag_24
```

使用端**不要手打這個順序**，一律從 `btc_4h_transformer.ptl.json` 的 `feature_columns` 讀取。

---

## 四、重新產生模型（通常不需要）

檔案已經在 repo 裡。只有在需要重現時才執行：

```powershell
$py = ".\.venv-transformer\Scripts\python.exe"
& $py scripts\build_btc_4h_transformer_ptl.py
```

- 需要 `data/features/btc_4h.csv` 與 `logs/feature_manifest.json`（在資料壓縮檔裡）。
- 程式會先確認特徵檔最後一列是 2026-08-09 06:00 UTC，不符就停止，避免用到之後的資料。
- CPU 約 40 秒。
- 產生 `.pth`、`.ptl`、`.ptl.json`，並自動執行第六節的驗證；驗證不通過就以錯誤結束。
- 同樣設定、同樣 seed 重跑，應該得到相同結果。

### 4.1 程式怎麼做

| 步驟 | 做法 | 為什麼 |
|---|---|---|
| 訓練 | 直接呼叫 `btc_4h_model_adapters._fit_predict_sequence_model`，它會把最佳 early-stopping 權重載回傳入的模型物件 | 重用實驗的訓練流程，不修改實驗程式 |
| 標準化參數 | `StandardScaler().fit(train)`，和訓練內部的計算完全相同 | 存進 `.pth` 並包進 `.ptl` |
| 包裝 | `ProbabilityModel`：`sigmoid(model((x − mean) / scale))` | 使用端不會忘記標準化或把 logit 當機率 |
| 匯出 | 關閉 Transformer fast path → `torch.jit.trace` → 嘗試 `optimize_for_mobile`（缺 XNNPACK 時跳過） | fast path 的融合運算子行動端不一定支援 |
| 存檔 | 先存到使用者目錄底下的英文暫存資料夾，再複製回專案 | lite interpreter 無法開啟含中文的路徑 |

---

## 五、在各平台執行 `.ptl`

### 5.1 Python（lite interpreter）

```python
import json, shutil, tempfile
from pathlib import Path
import numpy as np
import torch
from torch.jit.mobile import _load_for_lite_interpreter

ckpt = Path("models/btc_4h_transformer")
spec = json.loads((ckpt / "btc_4h_transformer.ptl.json").read_text(encoding="utf-8"))

# lite interpreter 不能開中文路徑：先複製到英文路徑
tmp = Path(tempfile.mkdtemp(dir=Path.home())) / "model.ptl"
shutil.copyfile(ckpt / "btc_4h_transformer.ptl", tmp)
model = _load_for_lite_interpreter(str(tmp))

# features: pandas DataFrame，由舊到新，至少 24 列，欄位包含 53 個特徵（見第七節）
x = features[spec["feature_columns"]].tail(24).to_numpy(dtype=np.float32)   # (24, 53) 原始值
score = float(model(torch.from_numpy(x).unsqueeze(0))[0])
label = "偏多" if score >= spec["display_threshold"] else "偏空"
```

### 5.2 Android（Kotlin）

```gradle
dependencies {
    implementation "org.pytorch:pytorch_android_lite:1.13.1"
}
```

把 `btc_4h_transformer.ptl` 放進 `app/src/main/assets/`，啟動時複製到內部儲存空間（`LiteModuleLoader` 需要實體檔案路徑）：

```kotlin
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor

val module = LiteModuleLoader.load(assetFilePath(context, "btc_4h_transformer.ptl"))
// features: FloatArray，長度 24 * 53；時間由舊到新，每列 53 個特徵依 .ptl.json 的順序
val input = Tensor.fromBlob(features, longArrayOf(1, 24, 53))
val score = module.forward(IValue.from(input)).toTensor().dataAsFloatArray[0]
```

`assetFilePath` 是 PyTorch Android 範例中常見的輔助函式：把 asset 複製到 `context.filesDir` 並回傳路徑。Android runtime 1.13.1 較舊，若載入時出現 bytecode 版本錯誤，見第八節。

### 5.3 iOS

使用 CocoaPods 的 `LibTorch-Lite`，以 Objective-C++ 呼叫 `torch::jit::_load_for_mobile`，輸入 `{1, 24, 53}` 的 float32 張量。一樣要處理 bytecode 版本相容性。

### 5.4 網頁

瀏覽器不能直接執行 `.ptl`。網頁展示請由 Python 後端執行，或直接讀取 `logs/live/predictions.csv`（見 `docs/frontend_data_guide.md`）。

---

## 六、驗證結果

`scripts/build_btc_4h_transformer_ptl.py` 匯出後，會用 lite interpreter 載入 `.ptl`，並和訓練時的結果比對：

| 項目 | 方法 | 結果 |
|---|---|---|
| 與訓練結果一致 | 驗證期全部 4,000 個原始特徵視窗送進 `.ptl`，比較訓練時算出的機率 | 最大差距 **1.19e-07**（容許 1e-5） |
| batch 一致 | 一次送 4 筆 vs 一筆一筆送 | 差距 **0** |
| 判定 | | **PASS** |

紀錄檔：`logs/ptl_btc_4h_verify.json`。

---

## 七、準備輸入資料

| 要求 | 說明 |
|---|---|
| 來源 | Binance Spot BTCUSDT 1 小時 K 線，UTC |
| 只用已收盤的 K 線 | 最後一列必須是已經收盤的那一根 |
| 歷史長度 | lag 與 rolling 最長用到 24～25 列，MACD 的指數平均需要更長暖機；**建議至少取最近 300 根 K 線**再算特徵 |
| 缺口 | 依專案規則補值（`scripts/impute_missing_klines.py`：價格沿用前一根收盤、成交量補 0） |
| 特徵函式 | 用 `scripts/feature_engineering.py` 的 `add_features(raw, prediction_horizon=4, keep_unlabeled=True)` |
| 不要自己標準化 | `.ptl` 內已包含標準化 |
| 型別 | float32，不能有 NaN 或無限大 |

**一定要用 `keep_unlabeled=True`**。預設的 `add_features` 會刪掉「4 小時後價格還不知道」的最新 4 列；即時預測需要的正是這幾列。`scripts/live_pipeline.py` 的 `live_features()` 就是這樣做的，可以直接參考或呼叫：

```python
import sys
import pandas as pd
sys.path.insert(0, "scripts")
from live_pipeline import live_features

processed = pd.read_csv("data/live/processed/btc_1h.csv")   # 補值後的連續 1 小時 K 線
features = live_features(processed)                           # 53 欄，保留最新的列
```

---

## 八、常見問題

| 症狀 | 可能原因 | 處理 |
|---|---|---|
| `RuntimeError` 開啟 `.ptl` 失敗 | 路徑含中文 | 先複製到英文路徑再載入（見 5.1） |
| 匯出時出現 XNNPACK 警告 | 本機 torch 沒有 XNNPACK | 可忽略；以驗證 PASS 為準 |
| Android 出現 `bytecode version` 錯誤 | 匯出的 bytecode 版本比 App runtime 新 | Python：`from torch.jit.mobile import _get_model_bytecode_version, _backport_for_mobile`；用 `_backport_for_mobile("in.ptl", "out.ptl", to_version=<runtime 支援的版本>)` 降版，再重新驗證 |
| 輸入形狀錯誤 | 少了 batch 維度或欄位數不是 53 | 輸入必須是 `(1, 24, 53)` |
| 輸出 NaN | 輸入有 NaN／無限大 | 檢查歷史長度與缺口補值 |
| 結果和 Python 端不同 | 特徵順序錯、用了未收盤 K 線、重複標準化、沒有用 `keep_unlabeled=True` | 對照第七節逐項檢查 |

---

## 九、展示時的說明文字（建議直接使用）

> 本展示模型為 5 年資料研究中的 BTC 4 小時方向分類 Transformer，驗證期 ROC-AUC 約 0.54，屬微弱訊號、未經機率校準，僅用於展示模型部署流程，不構成投資建議。
