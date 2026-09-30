# 前端網站資料包說明

> 撰寫日期：2026-09-30  
> 對象：負責前端網站的同學  
> 資料包：`專題_前端資料包_2026-09-30.zip`。它是完整專案的子集合，只放網站需要的模型、預測資料、圖表與更新程式。資料夾結構和專案相同，所以裡面的程式可以直接執行。完整專案請看 `專題_交接_2026-09-30.zip` 與 `docs/handover/handover.md`。

---

## 一、資料包內容

```
前端資料包/
├─ README_前端資料包.md          ← 本文件
├─ README.md                      專案總覽（「目前結果」表格是官方引用數字）
├─ requirements-lock.txt          Python 套件版本
├─ docs/
│  ├─ figures/F1～F10*.png        報告圖
│  ├─ figures/figure_data.json    圖表使用的數字
│  ├─ handover/frontend_data_guide.md   本文件
│  ├─ handover/ptl_guide.md             .ptl 模型說明
│  ├─ handover/handover.md              專案交接說明
│  ├─ report/final_project_report.md    專題報告初稿（網站文字可引用）
│  └─ experiments/*/                    各實驗結果與結論
├─ logs/
│  ├─ live/predictions.csv        即時預測紀錄（網站主要資料）
│  ├─ live/run_log.jsonl          每次更新的執行紀錄
│  └─ feature_manifest.json       53 個特徵名稱
├─ models/
│  ├─ live/                       4h XGBoost：凍結模型、滾動模型與 meta
│  └─ <幣種>_4h_transformer/        btc、eth、sol、xrp 各一個資料夾
│     ├─ <幣種>_4h_transformer.ptl        行動端模型
│     ├─ <幣種>_4h_transformer.ptl.json   .ptl 輸入輸出規格
│     └─ <幣種>_4h_transformer.pth        原始 PyTorch 權重與標準化參數
├─ data/
│  ├─ live/                       即時更新用的 K 線（raw、processed）
│  └─ features/*_4h.csv           實驗用的 4h 特徵檔（更新程式做一致性檢查時需要）
└─ scripts/                       更新預測資料的程式（live_pipeline.py 與相依模組）、.ptl 產生程式
```

---

## 二、網站可以呈現的內容

| 區塊 | 內容 | 資料來源 | 本文件章節 |
|---|---|---|---|
| 研究結果 | 四幣種模型 AUC、多視窗比較、三段式淨報酬、手續費損益兩平 | `docs/figures/*.png`、`figure_data.json` | 四 |
| 即時預測 | 各幣種每小時的上漲機率與實際結果 | `logs/live/predictions.csv` | 三 |
| 模型狀態 | 最新資料時間、凍結模型近 30 天 AUC | `live_pipeline.py status` | 六 |
| 模型展示 | 四幣種 4h Transformer 的 `.ptl` | `models/<幣種>_4h_transformer/` | 五 |
| 聲明 | 研究展示、訊號微弱、不構成投資建議 | 固定文字 | 七 |

---

## 三、即時預測資料：`logs/live/predictions.csv`

### 3.1 欄位

| 欄位 | 型別 | 說明 |
|---|---|---|
| `symbol` | 字串 | `BTC`、`ETH`、`SOL`、`XRP` |
| `open_time` | ISO 8601（UTC） | 被預測的那一根 1 小時 K 線的開盤時間 |
| `model_id` | 字串 | `frozen`（凍結模型）或 `rolling_YYYYMMDD`（該日期訓練的滾動模型） |
| `p_up` | 0～1 | 模型預測「4 小時後收盤價高於現在」的機率 |
| `predicted_at` | ISO 8601（UTC） | 實際產生這筆預測的時間 |
| `backfilled` | 0／1 | 1 = 事後補算（錯過即時時段）；0 = 即時預測 |
| `label_due_at` | ISO 8601（UTC） | 實際結果可以得知的時間（`open_time` + 5 小時） |
| `future_log_return` | 數字或空白 | 實際 4 小時對數報酬；到期前是空白 |
| `target_up` | 1／0 或空白 | 實際是否上漲；到期前是空白 |

### 3.2 使用方式

- **兩種模型要分開顯示**：
  - `frozen`：只用 2026-08-09 以前的資料訓練，永遠不更新，是**研究驗證用**的模型。從 2026-08-09 07:00 開始每小時都有預測。
  - `rolling_*`：每月重新訓練，只預測它生效之後的時段。目前的 `rolling_20260930` 在 2026-09-30 才建立，所以它的預測筆數還很少，甚至可能還沒有。
- **空白是正常的**：`target_up` 空白代表答案還沒揭曉，網站應顯示「待揭曉」，不要當成 0。
- **預測方向**：`p_up ≥ 0.5` 可以顯示為「偏多」，否則「偏空」。這個 0.5 只是顯示用的門檻，研究中沒有用它做任何判定。
- **資料截至**：目前檔案最新一列是 2026-09-30 00:00 UTC。自動排程已經取消，資料不會自己更新；要更新請見第六節。

### 3.3 JavaScript 讀取範例

```javascript
// 需要 CSV 解析套件，例如 PapaParse
const rows = Papa.parse(csvText, { header: true, dynamicTyping: true, skipEmptyLines: true }).data;
const btcFrozen = rows
  .filter(r => r.symbol === "BTC" && r.model_id === "frozen")
  .map(r => ({
    time: new Date(r.open_time),
    pUp: r.p_up,
    actual: r.target_up === null || r.target_up === "" ? null : Number(r.target_up),  // null = 待揭曉
  }));
```

---

## 四、研究結果圖表

### 4.1 圖檔

| 圖 | 內容 | 建議放在網站 |
|---|---|---|
| F1 | 資料流程 | 方法介紹 |
| F2 | walk-forward 切分與 purge | 方法介紹 |
| F3 | 四幣種 × 三模型 AUC | 主要結果（**見 4.3 的數字問題**） |
| F4 | 多視窗 AUC 熱圖 | 主要結果 |
| F5 | 1h XGBoost vs persistence | 補充 |
| F6 | validation vs test AUC | 補充 |
| F7 | 重訓頻率 ΔAUC | 補充 |
| F8 | 固定模型衰退曲線 | 補充 |
| F9 | 三段式交易比例 vs 淨報酬 | 主要結果 |
| F10 | 手續費損益兩平 | 主要結果（最能說明「為什麼不能交易」） |

### 4.2 `figure_data.json` 結構

最上層鍵值是 `F1`～`F10`。每張圖都有 `source`（原始結果檔），數字放在 `auc` 或 `rows` 裡：

| 鍵 | 結構 | 範例 |
|---|---|---|
| `F3.auc` | `{幣種: {xgboost, lstm, transformer}}` | `F3.auc.BTC.xgboost` |
| `F4.auc` | `{幣種: {"1", "4", "12", "24"}}`（視窗小時數為字串） | `F4.auc.XRP["1"]` |
| `F9.rows` | `{幣種: {T/B/P: [{q, threshold, trade_count, trade_ratio, mean_net_return, hit_rate}]}}` | `mean_net_return` 是對數報酬，−0.0018 ≈ −0.18% |
| `F10` | `formula`、`cost`（0.002）、`rows: {幣種: {train_mean_abs_return, breakeven_hit_rate, measured_T_hit_rate, measured_T_gross_return}}` | 損益兩平命中率約 0.63 |

網站要畫互動圖表時，**從這個 JSON 讀數字，不要手打**。

### 4.3 需要注意：F3 的數字和 README 不一致

`figure_data.json` 的 F3 讀的是較早的結果檔。ETH、SOL、XRP 的 AUC 和 README「目前結果」表格（官方引用數字，來自校正後重跑的結果）不同，例如：

| | README（官方） | F3 |
|---|---:|---:|
| ETH XGBoost | 0.5221 | 0.5275 |
| SOL Transformer | 0.5106 | 0.5067 |

在 `scripts/make_report_figures.py` 修正資料來源、重新產生 F3 之前，**網站顯示四幣種 AUC 時請使用 README 的表格數字**，或暫時不要放 F3。

---

## 五、模型檔

| 模型 | 檔案 | 輸入 | 輸出 | 能在瀏覽器跑嗎 |
|---|---|---|---|---|
| 4h XGBoost 凍結模型 | `models/live/<幣種>_4h_frozen.json` | 53 個特徵（單列，順序見 `logs/feature_manifest.json`） | 上漲機率 | 不行，需要 Python 後端 |
| 4h XGBoost 滾動模型 | `models/live/<幣種>_4h_rolling_*.json` ＋ `.meta.json` | 同上 | 上漲機率 | 不行 |
| 4h Transformer（四幣種各一） | `models/<幣種>_4h_transformer/<幣種>_4h_transformer.ptl` | 原始特徵 `(1, 24, 53)` | 4 小時後上漲分數（未校準） | 不行；`.ptl` 給 Android／iOS 或 Python lite interpreter 用 |

**建議**：網站不要自己跑模型，直接讀 `predictions.csv`。模型的預測已經由 `live_pipeline.py` 算好寫進去了。

如果一定要在網站後端即時推論：
- XGBoost：Python 用 `xgboost.XGBClassifier().load_model(路徑)`，特徵用 `scripts/live_pipeline.py` 的 `live_features()` 計算。
- `.ptl`：完整做法見 `docs/handover/ptl_guide.md`（已驗證和訓練結果一致，最大差距 1.19e-07）。注意 lite interpreter **不能開啟含中文的路徑**。

**不可以**：重新訓練或覆寫凍結模型。它是 2027 年新資料驗證的依據，SHA-256 記在 `models/live/frozen_manifest.json`。

---

## 六、更新預測資料

資料包內附有更新程式，可以抓最新 K 線並補上預測與答案：

```powershell
cd 前端資料包
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts\live_pipeline.py update   # 抓新資料、預測、回填答案（需要網路）
.\.venv\Scripts\python.exe scripts\live_pipeline.py status   # 顯示最新資料時間與凍結模型近 30 天 AUC
```

- `update` 會更新 `data/live/` 與 `logs/live/predictions.csv`。每次執行都會核對凍結模型的 SHA-256，並檢查線上特徵和實驗特徵是否一致，任何一項不符就停止，不會寫入錯誤的預測。
- 連不上 `api.binance.com`（例如在美國的主機）時，加上 `--base-url https://data-api.binance.vision`。
- `retrain`（每月重新訓練滾動模型）會記錄 git 版本，**請在完整專案中執行**，不要在資料包裡執行。
- 如果網站要定時更新，排程方式見 `docs/handover/handover.md` 附錄 D；目前原電腦上的排程已取消。

---

## 七、網站文字與聲明

### 7.1 必須放的聲明

> 本網站為大學專題研究展示。模型對加密貨幣短期漲跌只有統計上微弱的訊號（ROC-AUC 約 0.52），扣除交易手續費後不具獲利能力。所有預測僅供研究參考，不構成任何投資建議。

### 7.2 用詞規則（和專題報告一致）

| 不要寫 | 改成 |
|---|---|
| 「準確率高達…」「AI 精準預測」 | 「ROC-AUC 約 0.52，屬微弱訊號」 |
| 「可獲利」「建議買進」 | 「扣除手續費後每筆淨報酬為負」 |
| 「模型無效」「沒用」 | 「沒有證據顯示能穩定改善」 |
| 把凍結模型近 30 天 AUC 當成績效宣傳 | 「前向測試中，僅供描述，2027 年後才進行正式驗證」 |

### 7.3 可以引用的結論

- 4h 方向訊號在統計上存在，但非常弱（AUC 約 0.50～0.53）。
- 更換模型、特徵、預測視窗、重訓頻率、標籤設計都沒有穩定改善。
- 要打平 0.2% 來回手續費，命中率需約 56%～63%（依幣種），實測只有約 50%～54%（F10）。
- 延伸研究顯示「波動大小」比「漲跌方向」好預測（AUC 0.64～0.68）；但這個實驗的預先登記與結果在同一個 commit，且尚未用新資料驗證，引用時須註明「初步結果」。

---

## 八、常見問題

**Q：網站上可以即時顯示最新一小時的預測嗎？**
可以，但要先定時執行 `live_pipeline.py update`。目前沒有自動排程，資料停在 2026-09-30 00:00 UTC。

**Q：`rolling` 模型怎麼幾乎沒有預測？**
正常。它只預測自己生效（2026-09-30）之後的時段，要等更新程式持續執行才會累積。

**Q：可以把資料包的檔案直接放到公開網站上嗎？**
`predictions.csv`、圖檔、`figure_data.json` 可以。模型檔與 `data/` 不需要公開。本專案沒有任何 API key，但部署前仍請確認沒有放入任何帳號密碼。
