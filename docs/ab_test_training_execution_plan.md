# A/B Test 模型訓練執行規劃

> 目的：在獨立 worktree 中執行模型 A/B test，避免影響主資料夾目前未提交的工作。  
> Worktree：`C:/Users/user/orca/workspaces/專題/ab-test-training`  
> Branch：`ab-test-training`

## 1. 實驗目標

本次 A/B test 要回答：**新的訓練設定或特徵設計，是否在相同資料切分與評估流程下，穩定優於目前基準模型？**

建議先把 A/B 定義如下：

| 組別 | 定義 | 目的 |
| --- | --- | --- |
| A：Control | 既有穩定版本，例如只用 `logs/feature_manifest.json` 的 K 線技術特徵 | 建立可重現基準 |
| B：Treatment | 新方案，例如外部特徵、不同模型參數、不同 sequence length、不同 horizon 或新特徵工程 | 檢驗是否真的改善 |

核心原則：**A 與 B 除了要測試的變因之外，其餘條件必須完全相同**，包含資料列、fold、seed、validation/test 長度、校準方式與評估指標。

## 2. 開始前準備

因為 `data/`、`logs/`、`.venv-transformer/` 都被 `.gitignore` 排除，新 worktree 不會自動帶過去。第一次進入新 worktree 後，先同步必要資料：

```powershell
cd "C:/Users/user/orca/workspaces/專題/ab-test-training"

# 從主資料夾複製資料與既有 manifest/log
Copy-Item -Recurse "C:/Users/user/專題/data" .
Copy-Item -Recurse "C:/Users/user/專題/logs" .

# 如果不想重建環境，也可以複製既有 venv
Copy-Item -Recurse "C:/Users/user/專題/.venv-transformer" .
```

確認：

```powershell
.\.venv-transformer\Scripts\python.exe --version
Get-ChildItem data\features\btc_4h.csv
Get-ChildItem logs\feature_manifest.json
```

## 3. 建議實驗設計

### 3.1 固定切分

建議沿用目前 4 小時方向預測設定：

| 參數 | 建議值 |
| --- | ---: |
| `fold-count` | 3 |
| `validation-rows` | 4000 |
| `test-rows` | 4000 |
| `seed` | 42 |
| 校準 | Platt calibration |
| threshold 選擇 | 只用 validation |

如果使用外部特徵檔 `data/features_ext/`，為了公平，A 組也應該在同一個 `features_ext` 資料檔上訓練，只是使用 baseline manifest，以免資料列數不同造成偏誤。

### 3.2 建議比較對象

第一輪建議先做「外部特徵是否有效」的 A/B test：

| 組別 | data | manifest | 說明 |
| --- | --- | --- | --- |
| A：base | `data/features_ext/<coin>_4h.csv` | `logs/feature_manifest.json` | 同資料列，但只吃原始 K 線特徵 |
| B：ext | `data/features_ext/<coin>_4h.csv` | `logs/feature_ext_manifest_<coin>.json` | K 線 + 外部特徵 |

四個幣種的 `initial-train-rows`：

| 幣種 | initial-train-rows |
| --- | ---: |
| BTC | 10172 |
| ETH | 9238 |
| SOL | 9237 |
| XRP | 9261 |

## 4. 執行指令

以下以 PowerShell 為例。

### BTC

```powershell
$py = ".\.venv-transformer\Scripts\python.exe"

# A：base
& $py scripts\train_btc_4h_model_comparison.py --symbol BTC `
  --data data\features_ext\btc_4h.csv `
  --manifest logs\feature_manifest.json `
  --output logs\ab_btc_base.json `
  --initial-train-rows 10172 --validation-rows 4000 --test-rows 4000 --fold-count 3 --seed 42

# B：ext
& $py scripts\train_btc_4h_model_comparison.py --symbol BTC `
  --data data\features_ext\btc_4h.csv `
  --manifest logs\feature_ext_manifest_btc.json `
  --output logs\ab_btc_ext.json `
  --initial-train-rows 10172 --validation-rows 4000 --test-rows 4000 --fold-count 3 --seed 42
```

### ETH / SOL / XRP

把 `--symbol`、`--data`、`--manifest`、`--output` 與 `--initial-train-rows` 換成下列設定：

| 幣種 | A output | B output | data | B manifest | initial-train-rows |
| --- | --- | --- | --- | --- | ---: |
| ETH | `logs/ab_eth_base.json` | `logs/ab_eth_ext.json` | `data/features_ext/eth_4h.csv` | `logs/feature_ext_manifest_eth.json` | 9238 |
| SOL | `logs/ab_sol_base.json` | `logs/ab_sol_ext.json` | `data/features_ext/sol_4h.csv` | `logs/feature_ext_manifest_sol.json` | 9237 |
| XRP | `logs/ab_xrp_base.json` | `logs/ab_xrp_ext.json` | `data/features_ext/xrp_4h.csv` | `logs/feature_ext_manifest_xrp.json` | 9261 |

## 5. 評估指標與判定標準

主要指標建議使用：

1. `test_aggregate.roc_auc.mean`：主要排序能力。
2. `test_aggregate.brier_score.mean`：機率校準品質，越低越好。
3. `balanced_accuracy`：避免類別比例影響。
4. fold 間 `sample_std`：檢查改善是否穩定。

建議判定門檻：

| 結果 | 判定 |
| --- | --- |
| B 的 ROC-AUC 平均提升 < 0.005 | 視為無明確改善 |
| B 提升 0.005–0.01，但只出現在單一幣種或單一 fold | 視為弱證據，需要重跑或擴大實驗 |
| B 在多數幣種、多數模型提升 > 0.01，且 Brier 不惡化 | 可視為有效改善 |
| B 的平均提升小於 fold 標準差 | 不應宣稱穩定有效 |

## 6. 結果紀錄格式

完成訓練後，建議新增一份結果文件，例如：

```text
docs/ab_test_result_YYYY-MM-DD.md
```

表格建議格式：

| 幣種 | 模型 | A ROC-AUC | B ROC-AUC | 差異 | A Brier | B Brier | 結論 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| BTC | XGBoost |  |  |  |  |  |  |
| BTC | LSTM |  |  |  |  |  |  |
| BTC | Transformer |  |  |  |  |  |  |

結論撰寫要避免只挑好看的結果，應同時說明：

- 四個幣種整體平均改善多少。
- 12 組比較中幾組變好、幾組變差。
- 改善是否大於 fold 標準差。
- Brier score 是否惡化。
- 是否有資料列數、特徵缺漏或時間區間不一致問題。

## 7. 建議工作流程

1. 在主資料夾保留目前工作，不切分支、不清理未提交內容。
2. 在 `ab-test-training` worktree 複製 `data/`、`logs/`、`.venv-transformer/`。
3. 先跑 BTC 的 A/B，確認指令與環境正常。
4. 再跑 ETH、SOL、XRP。
5. 彙整 JSON 指標到結果文件。
6. 若 B 沒有穩定改善，只保留否定結果與方法論說明。
7. 若 B 穩定改善，再把程式修改整理成 commit，之後再決定是否 merge 回 main。

## 8. 注意事項

- 不要拿不同資料列數的 A 與 B 直接比較。
- 不要用 test set 選 threshold 或調參。
- 不要因為單一 fold 或單一幣種變好就宣稱有效。
- 如果改了程式碼，結果 JSON 檔名要包含實驗名稱與日期，避免覆蓋舊結果。
- `logs/` 被 git ignore，重要結果要另外整理成 `docs/` 文件，否則不會被版本控制保存。
