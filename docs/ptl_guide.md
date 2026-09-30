# BTC 24h Transformer 轉成 `.ptl` 與執行說明
> **2026-09-30 注意**：本文件使用的模型檔（`btc_24h_final.pth`、`.pkl`、`.ptl`、`.ptl.json`）屬於 3 年資料階段，依原作者決定**不放在 GitHub，也不在資料壓縮檔**。從 GitHub clone 的專案沒有這些檔案；需要時請向原作者索取，放回 `models/btc_24h_pytorch_transformer/checkpoints/` 後再照本文件操作。

> 撰寫日期：2026-09-30  
> 狀態：**已於 2026-09-30 在本機執行完成並通過驗證。**由於目前 Windows torch build 沒有 XNNPACK，匯出程式會自動跳過 `optimize_for_mobile`，但 `.ptl` 已可由 lite interpreter 載入，且和原模型輸出一致。  
> 對象：要把模型放進 App 或前端展示的同學

---

## 一、先知道這幾件事

1. **只有一個模型可以轉**：專案裡唯一的 PyTorch 權重是 `models/btc_24h_pytorch_transformer/checkpoints/btc_24h_final.pth`。5 年資料 4h 主實驗的模型（XGBoost、LSTM、Transformer）在程式設計上不儲存權重，沒有東西可以轉。
2. **這個模型來自 3 年資料階段**，不是本專題最後報告的主模型：
   - 預測目標：**24 小時後**收盤價是否高於現在（`close[t + 24] > close[t]`）。
   - 訓練資料：`btc_3y.csv`（3 年資料），用舊版特徵函式 `src/lstm_experiment_utils.add_features` 計算特徵。`btc_3y.csv` 現在不在 `data/raw/`，如果需要，可以在 `舊專題_3年資料/` 或原本的舊專案資料夾裡找。
   - 訓練區間：fit 2022-10-17 ～ 2025-09-13，validation 2025-09-14 ～ 2025-10-14。
3. **訊號很弱**：這個模型在測試期的 accuracy 約 0.51。門檻 0.35 使 recall 高達 0.94，代表它幾乎每次都預測「上漲」。**只能當技術展示**，不能宣稱能預測價格或獲利。
4. **PyTorch Mobile（`.ptl` 使用的 lite interpreter）目前處於維護狀態**，官方新的行動端方案是 ExecuTorch。本文件用 `.ptl` 是因為專題需要這個格式；如果你的 PyTorch 版本已經不支援本文件的函式，見第九節。

---

## 二、`.ptl` 是什麼

| 格式 | 內容 | 誰可以讀 |
|---|---|---|
| `.pth` | 模型權重（`state_dict`）加上設定。必須有 Python 的模型類別定義，才能還原成模型 | 完整 PyTorch（Python） |
| `.ptl` | 把「模型結構 + 權重」一起編譯成 TorchScript，再轉成行動端 lite interpreter 格式。**不需要 Python 程式碼**就能執行 | PyTorch Android Lite、iOS LibTorch-Lite、Python 的 lite interpreter |

所以轉檔的本質是：用 Python 載入 `.pth` → 追蹤（trace）一次模型的計算過程 → 存成 `.ptl`。

---

## 三、模型規格

| 項目 | 值 |
|---|---|
| 來源權重 | `models/btc_24h_pytorch_transformer/checkpoints/btc_24h_final.pth` |
| 前處理設定 | 同資料夾的 `btc_24h_final.pkl`（標準化參數、特徵順序、門檻） |
| 模型類別 | `BtcTransformerClassifier`（`src/pytorch_transformer_artifacts.py`） |
| 架構 | 53 維特徵 → 線性投影到 64 維 + 位置嵌入 → 1 層 Transformer Encoder（4 heads，FFN 128，GELU）→ 時間平均 → LayerNorm → 32 → ReLU → 1 |
| 原模型輸入 | `(batch, 24, 53)`，float32，**已標準化**：`(x − scaler_mean) / scaler_std` |
| 原模型輸出 | `(batch,)` 的 **logit**；機率 = `sigmoid(logit)` |
| 判定門檻 | 機率 ≥ 0.35 視為「預測上漲」（`selected_threshold`） |
| 儲存時的 PyTorch 版本 | 2.13.0（目前環境是 2.14.0） |

### 3.1 本文件匯出的 `.ptl` 規格（和原模型不同）

為了避免使用端忘記標準化或忘記 sigmoid，匯出時會把這兩步**包進 `.ptl` 裡**：

| 項目 | `.ptl` 的規格 |
|---|---|
| 輸入 | `(batch, 24, 53)`，float32，**原始特徵值（不要自己標準化）** |
| 輸出 | `(batch,)`，**上漲機率**（0～1），不是 logit |
| 使用端還要做的事 | 算出 53 個特徵、取最近 24 列、依固定順序排好；拿到機率後和 0.35 比較 |

### 3.2 53 個特徵的順序（必須完全一致）

順序與 `btc_24h_final.pkl` 的 `feature_columns` 相同，也和 `scripts/feature_engineering.py` 的 `EXPECTED_FEATURE_NAMES` 相同（已核對）：

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

使用端**不要自己手打這個順序**，一律從匯出時產生的 `btc_24h_final.ptl.json`（見第五節）讀取。

---

## 四、環境需求

| 項目 | 需求 |
|---|---|
| Python | 3.12（原環境 3.12.14） |
| 套件 | `requirements-lock.txt`（含 torch 2.14.0、numpy、pandas） |
| 工作目錄 | 專案根目錄（`C:\Users\user\專題` 或你解壓縮的位置） |

```powershell
$py = ".\.venv-transformer\Scripts\python.exe"
& $py -c "import torch; print(torch.__version__); from torch.utils.mobile_optimizer import optimize_for_mobile; from torch.jit.mobile import _load_for_lite_interpreter; print('mobile OK')"
```

**預期結果**：印出 torch 版本與 `mobile OK`。如果出現 ImportError，代表這個 PyTorch 版本已移除行動端功能，見第九節。

---

## 五、步驟一：建立匯出程式

新增檔案 `scripts/export_btc_24h_ptl.py`。本機實作與下方原始範例相比，多了兩個相容性處理：Windows torch build 缺 XNNPACK 時會跳過 `optimize_for_mobile`，且因 lite interpreter 不能開中文路徑，會先存到英文暫存路徑再搬回專案。原始範例如下：

```python
"""Export the BTC 24h Transformer (.pth) to a PyTorch Lite (.ptl) model.

The .ptl takes RAW (unscaled) features of shape (batch, 24, 53) and returns the
up-probability, because scaling and sigmoid are baked in. See docs/ptl_guide.md.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pytorch_transformer_artifacts import load_artifact_pair  # noqa: E402

CHECKPOINTS = ROOT / "models" / "btc_24h_pytorch_transformer" / "checkpoints"
PTH = CHECKPOINTS / "btc_24h_final.pth"
PKL = CHECKPOINTS / "btc_24h_final.pkl"
PTL = CHECKPOINTS / "btc_24h_final.ptl"
SPEC = CHECKPOINTS / "btc_24h_final.ptl.json"


class ProbabilityModel(torch.nn.Module):
    """Raw features in, up-probability out: (x - mean) / std -> model -> sigmoid."""

    def __init__(self, model: torch.nn.Module, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, raw_sequences: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model((raw_sequences - self.mean) / self.std))


def main() -> None:
    model, metadata = load_artifact_pair(PTH, PKL)  # eval() mode
    if any(s <= 0 for s in metadata["scaler_std"]):
        raise ValueError("scaler_std must be positive")
    wrapper = ProbabilityModel(model, metadata["scaler_mean"], metadata["scaler_std"]).eval()

    # The Transformer "fast path" uses fused kernels that the mobile runtime may not support.
    torch.backends.mha.set_fastpath_enabled(False)
    seq_len, n_features = model.config.sequence_length, model.config.feature_count
    example = torch.zeros(1, seq_len, n_features, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
    optimize_for_mobile(traced)._save_for_lite_interpreter(str(PTL))

    spec = {
        "model_file": PTL.name,
        "source_checkpoint": PTH.name,
        "source_checkpoint_sha256": hashlib.sha256(PTH.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "input": {"shape": ["batch", seq_len, n_features], "dtype": "float32",
                  "scaling": "none - pass RAW feature values; scaling is inside the model",
                  "row_order": "oldest to newest; the last row is the most recent closed 1h candle"},
        "output": {"shape": ["batch"], "meaning": "probability that close[t + 24] > close[t]"},
        "threshold": float(metadata["selected_threshold"]),
        "feature_columns": list(metadata["feature_columns"]),
        "target_definition": metadata["target_definition"],
        "training_coverage": metadata["training_coverage"],
    }
    SPEC.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {PTL} ({PTL.stat().st_size} bytes) and {SPEC.name}")


if __name__ == "__main__":
    main()
```

### 設計說明

| 做法 | 原因 |
|---|---|
| 用 `load_artifact_pair` 載入 | 重用專案既有的載入函式，它會檢查格式版本、特徵數、序列長度是否一致 |
| 標準化與 sigmoid 包進模型 | 使用端最常犯的錯就是忘了標準化，或把 logit 當機率；包進去之後，這兩種錯誤就不可能發生 |
| 用 `trace` 而不是 `script` | 原模型的 `forward` 有 Python 字串錯誤訊息等 TorchScript 不支援的寫法；`trace` 只記錄實際執行的張量運算 |
| 關閉 Transformer fast path | eval 模式下 PyTorch 可能改走融合運算子，行動端 runtime 不一定支援 |
| 另存 `.ptl.json` | 特徵順序、門檻、輸入格式都寫在這裡，使用端讀這個檔，不手打 |

---

## 六、步驟二：執行匯出

```powershell
& $py scripts\export_btc_24h_ptl.py
```

**預期結果**：
- 印出 `wrote ...btc_24h_final.ptl (約數百 KB) and btc_24h_final.ptl.json`。
- `torch.jit.trace` 可能印出 `TracerWarning`（例如關於 shape 檢查的 Python 布林值）。這是因為原模型 `forward` 開頭的形狀檢查只在追蹤時執行一次，**可以忽略**；但如果出現 `Error` 就要停下來。

產生的檔案：

| 檔案 | 用途 |
|---|---|
| `models/btc_24h_pytorch_transformer/checkpoints/btc_24h_final.ptl` | 行動端模型 |
| `models/btc_24h_pytorch_transformer/checkpoints/btc_24h_final.ptl.json` | 輸入輸出規格、特徵順序、門檻 |

---

## 七、步驟三：驗證 `.ptl` 和原模型結果一致（一定要做）

轉檔可能在不報錯的情況下算錯（例如 fast path、運算子替換）。所以必須用**真實資料**比較 `.ptl` 和原本的推論函式 `predict_latest_probability`。

新增 `scripts/verify_btc_24h_ptl.py`：

```python
"""Check that the .ptl gives the same up-probability as the original .pth inference path."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.jit.mobile import _load_for_lite_interpreter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pytorch_transformer_artifacts import load_artifact_pair, predict_latest_probability  # noqa: E402

CHECKPOINTS = ROOT / "models" / "btc_24h_pytorch_transformer" / "checkpoints"
TOLERANCE = 1e-5
WINDOWS = 200


def main() -> None:
    spec = json.loads((CHECKPOINTS / "btc_24h_final.ptl.json").read_text(encoding="utf-8"))
    columns, seq_len = spec["feature_columns"], spec["input"]["shape"][1]
    model, metadata = load_artifact_pair(CHECKPOINTS / "btc_24h_final.pth", CHECKPOINTS / "btc_24h_final.pkl")
    lite = _load_for_lite_interpreter(str(CHECKPOINTS / "btc_24h_final.ptl"))

    # Any real feature rows work for an equivalence test; these have the same 53 columns.
    features = pd.read_csv(ROOT / "data" / "features" / "btc_24h.csv", usecols=columns)[columns]
    ends = np.linspace(seq_len, len(features), WINDOWS, dtype=int)

    worst = 0.0
    for end in ends:
        rows = features.iloc[end - seq_len:end]
        expected = predict_latest_probability(model, metadata, rows)["up_probability"]
        raw = torch.from_numpy(rows.to_numpy(dtype=np.float32)).unsqueeze(0)
        actual = float(lite(raw)[0])
        worst = max(worst, abs(actual - expected))

    batch = torch.from_numpy(np.stack([features.iloc[e - seq_len:e].to_numpy(dtype=np.float32) for e in ends[:4]]))
    batch_out = lite(batch)
    single_out = torch.stack([lite(batch[i:i + 1])[0] for i in range(4)])
    batch_diff = float((batch_out - single_out).abs().max())

    print(f"windows={len(ends)} max|ptl - pth|={worst:.2e} batch_vs_single={batch_diff:.2e}")
    if worst > TOLERANCE or batch_diff > TOLERANCE:
        raise SystemExit("FAIL: .ptl output differs from the original model")
    print("PASS")


if __name__ == "__main__":
    main()
```

執行：

```powershell
& $py scripts\verify_btc_24h_ptl.py
```

**本機結果**：`windows=200 max|ptl - pth|=5.96e-08 batch_vs_single=0.00e+00`，最後印出 `PASS`。

| 驗證項目 | 檢查什麼 | 通過條件 |
|---|---|---|
| 單筆一致 | 200 個真實視窗，`.ptl` 機率 vs 原推論函式機率 | 最大差距 ≤ 1e-5 |
| batch 一致 | 一次送 4 筆 vs 一筆一筆送 | 最大差距 ≤ 1e-5（確認 trace 沒有把 batch 大小寫死） |
| 能用 lite interpreter 載入 | `_load_for_lite_interpreter` 不報錯 | 能載入 |

**如果 FAIL**：不要調整容許誤差讓它通過。先看第十節的常見問題；最常見的原因是 fast path 沒有關閉，或 batch 大小被寫死。

---

## 八、在各平台執行 `.ptl`

### 8.1 Python（lite interpreter）

```python
import json
import numpy as np
import torch
from torch.jit.mobile import _load_for_lite_interpreter

ckpt = "models/btc_24h_pytorch_transformer/checkpoints/"
spec = json.load(open(ckpt + "btc_24h_final.ptl.json", encoding="utf-8"))
model = _load_for_lite_interpreter(ckpt + "btc_24h_final.ptl")

# rows: pandas DataFrame, 至少 24 列、欄位包含 53 個特徵、由舊到新排序
x = rows[spec["feature_columns"]].tail(24).to_numpy(dtype=np.float32)   # (24, 53) 原始值
prob = float(model(torch.from_numpy(x).unsqueeze(0))[0])              # 上漲機率
predict_up = prob >= spec["threshold"]
```

### 8.2 Android（Kotlin）

`build.gradle`：

```gradle
dependencies {
    implementation "org.pytorch:pytorch_android_lite:1.13.1"
}
```

把 `btc_24h_final.ptl` 放進 `app/src/main/assets/`，App 啟動時先複製到內部儲存空間（`LiteModuleLoader` 需要實體檔案路徑）：

```kotlin
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor

val module = LiteModuleLoader.load(assetFilePath(context, "btc_24h_final.ptl"))

// features: FloatArray，長度 24 * 53；依時間由舊到新，每列 53 個特徵依 .ptl.json 的順序
val input = Tensor.fromBlob(features, longArrayOf(1, 24, 53))
val prob = module.forward(IValue.from(input)).toTensor().dataAsFloatArray[0]
val predictUp = prob >= 0.35f
```

`assetFilePath` 是 PyTorch Android 範例中常見的輔助函式：把 asset 複製到 `context.filesDir` 並回傳路徑。

**版本相容性**：Android runtime 1.13.1 是較舊的版本，可能讀不了用 torch 2.14 匯出的較新 bytecode 版本。遇到載入錯誤時，見第十節「bytecode version」。

### 8.3 iOS

使用 CocoaPods 的 `LibTorch-Lite`，以 Objective-C++ 包一層呼叫 `torch::jit::_load_for_mobile`，輸入同樣是 `{1, 24, 53}` 的 float32 張量。iOS 的整合步驟較多，建議參考 PyTorch 官方 iOS 範例，並一樣先處理 bytecode 版本相容性。

### 8.4 網頁前端

瀏覽器**不能**直接執行 `.ptl`。網頁展示有兩種做法：
- 由 Python 後端執行 `.ptl`（或直接用 `.pth`），網頁只顯示結果。
- 另外把模型匯出成 ONNX，再用 ONNX Runtime Web 在瀏覽器執行。這是另一條轉檔流程，不在本文件範圍。

---

## 九、準備輸入資料（使用端最容易出錯的地方）

### 9.1 資料要求

| 要求 | 說明 |
|---|---|
| 來源 | Binance Spot BTCUSDT 1 小時 K 線，時間用 UTC |
| 只用已收盤的 K 線 | 最後一列必須是已經收盤的那一根 |
| 歷史長度 | 算特徵需要足夠的歷史：lag 與 rolling 最長用到 24～25 列；MACD 的指數平均需要更長的暖機。**建議至少取最近 300 根 K 線**（約 12.5 天）來算特徵，再取最後 24 列 |
| 缺口 | K 線有缺口時，依專案規則補值（`scripts/impute_missing_klines.py`：價格沿用前一根收盤、成交量補 0） |
| 特徵函式 | 模型是用 `src/lstm_experiment_utils.add_features` 訓練的，**使用端應該用同一個函式**計算特徵。`scripts/feature_engineering.py` 的欄位名稱和順序相同，但它是 5 年資料階段的版本，在零成交量等邊界情況的處理可能不同 |
| 不要自己標準化 | `.ptl` 內已包含標準化 |
| 型別 | float32；不能有 NaN 或無限大（出現時代表歷史不夠長或資料有問題） |

### 9.2 Python 範例：從原始 K 線到預測

```python
import sys
import pandas as pd
sys.path.insert(0, "src")
from lstm_experiment_utils import add_features

raw = pd.read_csv("data/processed/btc_5y.csv").tail(300)          # 或即時抓的最近 300 根 K 線
features = add_features(raw, prediction_horizon=24)
# 注意：add_features 會刪掉「標籤還未知」的最後 24 列。
# 即時預測需要最新的列，請改用保留未知標籤列的版本，
# 或參考 scripts/live_pipeline.py 的 live_features()（它對 5 年版特徵函式做了相同處理）。
```

**重要**：`add_features` 最後會 `dropna()`，把「24 小時後價格還不知道」的最新 24 列刪掉。離線驗證沒問題，但**即時預測時一定要保留最新的列**，否則預測的其實是 24 小時前的狀態。處理方式可參考 `scripts/feature_engineering.py` 中 `keep_unlabeled=True` 的寫法：只對特徵欄位做 `dropna`，不對標籤欄位做。

---

## 十、常見問題

| 症狀 | 可能原因 | 處理 |
|---|---|---|
| 匯出時 `ImportError: optimize_for_mobile` 或 `_load_for_lite_interpreter` | PyTorch 版本已移除行動端功能 | 改用 `requirements-lock.txt` 的 torch 版本；或改走 ExecuTorch（另一套流程，匯出成 `.pte`） |
| 匯出時出現 `TracerWarning` | 原模型 `forward` 的形狀檢查在追蹤時被轉成常數 | 可忽略；以第七節驗證是否 PASS 為準 |
| 驗證 `max|ptl - pth|` 很大 | Transformer fast path 沒關閉，或追蹤到錯誤的分支 | 確認匯出程式有 `torch.backends.mha.set_fastpath_enabled(False)`，並且模型在 `eval()` 模式 |
| `batch_vs_single` 很大，或 batch ≠ 1 時報錯 | trace 把 batch 大小寫死 | 改用 batch = 2 的 example 重新匯出，並在 App 端固定一次只送 1 筆 |
| Android 載入時出現 `bytecode version` 或 `Lite Interpreter version` 相關錯誤 | 匯出用的 bytecode 版本比 App 的 runtime 新 | 在 Python 查版本：`from torch.jit.mobile import _get_model_bytecode_version, _backport_for_mobile`，`_get_model_bytecode_version("...ptl")`；用 `_backport_for_mobile("in.ptl", "out.ptl", to_version=<runtime 支援的版本>)` 降版後，再重跑第七節驗證 |
| 輸入形狀錯誤 | 沒有 batch 維度，或欄位數不是 53 | 輸入必須是 `(1, 24, 53)` |
| 預測機率幾乎都大於 0.35 | 模型本身的特性（recall 0.94） | 不是錯誤，見第一節第 3 點 |
| 輸出 NaN | 輸入有 NaN 或無限大 | 檢查歷史長度是否足夠、缺口是否補值 |
| 預測結果和 Python 端不同 | 特徵順序錯、忘了只用已收盤 K 線、自己又做了一次標準化 | 從 `.ptl.json` 讀特徵順序；不要再標準化 |

---

## 十一、完成後

1. **commit**：`scripts/export_btc_24h_ptl.py`、`scripts/verify_btc_24h_ptl.py`、`.ptl`、`.ptl.json`。`models/` 沒有被 `.gitignore` 排除，`.ptl` 只有數百 KB，可以直接進 git。
2. **記錄驗證結果**：把第七節印出的 `max|ptl - pth|` 和 `PASS` 寫進 commit 訊息或 `docs/handover.md` 6.1。
3. **展示時的說明文字**（建議直接使用）：
   > 本展示模型為 3 年資料階段訓練的 BTC 24 小時方向分類 Transformer，測試期 accuracy 約 0.51，屬微弱訊號，僅用於展示模型部署流程，不構成投資建議。

## 十二、檢查清單

- [ ] 第四節：環境可以 import 行動端相關函式
- [ ] 第五、六節：`.ptl` 與 `.ptl.json` 已產生
- [ ] 第七節：驗證 `PASS`（單筆與 batch 都一致）
- [ ] 第八節：目標平台可以載入並執行（Android／iOS 需處理 bytecode 版本）
- [ ] 第九節：使用端的特徵計算保留了最新的列、只用已收盤 K 線、沒有重複標準化
- [ ] 第十一節：程式與模型已 commit，展示文字已加上
