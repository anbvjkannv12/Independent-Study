# 多視窗方向分類比較結果

預先登記：`docs/superpowers/specs/2026-09-27-multi-horizon-comparison-design.md`。主要指標：未校準 test ROC-AUC。

## 訊號關卡（16 格一起 Holm 校正）

| 幣種 | 視窗 | 合併 AUC（三 seed 平均） | 各 fold AUC（三 seed 平均） | Holm p | 通過 |
|---|---:|---:|---|---:|---|
| BTC | 1h | 0.542365 | 0.550017, 0.543844, 0.531567 | 0.007996 | 是 |
| BTC | 4h | 0.527838 | 0.544536, 0.514151, 0.527067 | 0.007996 | 是 |
| BTC | 12h | 0.524988 | 0.559559, 0.494333, 0.521953 | 0.059970 | 否 |
| BTC | 24h | 0.519726 | 0.538644, 0.505817, 0.510219 | 0.410795 | 否 |
| ETH | 1h | 0.537146 | 0.547374, 0.533560, 0.532826 | 0.007996 | 是 |
| ETH | 4h | 0.522596 | 0.524692, 0.524981, 0.518163 | 0.010995 | 是 |
| ETH | 12h | 0.538823 | 0.548230, 0.546594, 0.518480 | 0.007996 | 是 |
| ETH | 24h | 0.537765 | 0.535288, 0.521736, 0.556266 | 0.031484 | 是 |
| SOL | 1h | 0.516126 | 0.524054, 0.513240, 0.507408 | 0.014993 | 是 |
| SOL | 4h | 0.500407 | 0.513702, 0.516024, 0.475136 | 1.000000 | 否 |
| SOL | 12h | 0.499839 | 0.521342, 0.517567, 0.464568 | 1.000000 | 否 |
| SOL | 24h | 0.501416 | 0.501774, 0.505322, 0.480675 | 1.000000 | 否 |
| XRP | 1h | 0.535749 | 0.551194, 0.530644, 0.526745 | 0.007996 | 是 |
| XRP | 4h | 0.516402 | 0.526088, 0.524214, 0.498866 | 0.073463 | 否 |
| XRP | 12h | 0.499253 | 0.504211, 0.498984, 0.498019 | 1.000000 | 否 |
| XRP | 24h | 0.501789 | 0.513738, 0.533561, 0.477140 | 1.000000 | 否 |

## 相對 4h 的比較（幣種內 Holm）

| 幣種 | h | ΔAUC | fold ΔAUC | ε | Holm p | 95% CI | 判定 |
|---|---:|---:|---|---:|---:|---|---|
| BTC | 1h | 0.014527 | 0.005481, 0.029693, 0.004500 | 0.004097 | 0.074963 | 0.000122, 0.028386 | 沒有證據顯示 h 比 4h 好 |
| BTC | 12h | -0.002850 | 0.015024, -0.019817, -0.005113 | 0.004262 | 1.000000 | -0.021414, 0.016445 | 沒有證據顯示 h 比 4h 好 |
| BTC | 24h | -0.008112 | -0.005891, -0.008333, -0.016848 | 0.008036 | 1.000000 | -0.033568, 0.018826 | 沒有證據顯示 h 比 4h 好 |
| ETH | 1h | 0.014550 | 0.022682, 0.008580, 0.014663 | 0.005418 | 0.085457 | -0.000290, 0.028098 | 沒有證據顯示 h 比 4h 好 |
| ETH | 12h | 0.016227 | 0.023538, 0.021613, 0.000317 | 0.007177 | 0.085457 | -0.001725, 0.036541 | 沒有證據顯示 h 比 4h 好 |
| ETH | 24h | 0.015169 | 0.010596, -0.003245, 0.038103 | 0.005243 | 0.192904 | -0.015675, 0.039675 | 沒有證據顯示 h 比 4h 好 |
| SOL | 1h | 0.015718 | 0.010353, -0.002785, 0.032272 | 0.002485 | 0.083958 | -0.000714, 0.030069 | 沒有證據顯示 h 比 4h 好 |
| SOL | 12h | -0.000568 | 0.007641, 0.001543, -0.010568 | 0.004885 | 1.000000 | -0.017927, 0.017726 | 沒有證據顯示 h 比 4h 好 |
| SOL | 24h | 0.001009 | -0.011928, -0.010702, 0.005539 | 0.002481 | 1.000000 | -0.027090, 0.026210 | 沒有證據顯示 h 比 4h 好 |
| XRP | 1h | 0.019347 | 0.025105, 0.006430, 0.027880 | 0.005554 | 0.004498 | 0.006701, 0.032255 | 較 4h 穩定地強 |
| XRP | 12h | -0.017149 | -0.021877, -0.025230, -0.000847 | 0.007602 | 1.000000 | -0.035453, 0.004171 | 沒有證據顯示 h 比 4h 好 |
| XRP | 24h | -0.014613 | -0.012350, 0.009347, -0.021725 | 0.005554 | 1.000000 | -0.045071, 0.016365 | 沒有證據顯示 h 比 4h 好 |

## 描述性檢查（不參與判定）

各 fold 依序列出 XGBoost AUC（三 seed 平均）、Logistic AUC、persistence AUC、best-direction persistence = max(p, 1 − p)、校準 Brier（三 seed 平均）、訓練上漲率常數 Brier、test 上漲率、validation AUC（三 seed 平均）；各項基準與逐 seed 數值見 JSON。best-direction persistence 是看過 test 才選方向的描述性上限參考，不能參與判定。

### BTC 1h
- fold 0: XGB=0.550017, Logistic=0.553011, persistence=0.465746, XGB−persistence=0.084271, best-direction persistence=0.534254, XGB−best-direction persistence=0.015763, Brier=0.248753 vs constant=0.250002, up=0.5015, validation AUC=0.560245 (test−validation=-0.010228)
- fold 1: XGB=0.543844, Logistic=0.544022, persistence=0.491499, XGB−persistence=0.052344, best-direction persistence=0.508501, XGB−best-direction persistence=0.035343, Brier=0.248834 vs constant=0.250020, up=0.5005, validation AUC=0.547846 (test−validation=-0.004002)
- fold 2: XGB=0.531567, Logistic=0.533168, persistence=0.479238, XGB−persistence=0.052329, best-direction persistence=0.520762, XGB−best-direction persistence=0.010805, Brier=0.249216 vs constant=0.250064, up=0.4975, validation AUC=0.543740 (test−validation=-0.012173)

### BTC 4h
- fold 0: XGB=0.544536, Logistic=0.543063, persistence=0.475942, XGB−persistence=0.068594, best-direction persistence=0.524058, XGB−best-direction persistence=0.020478, Brier=0.249288 vs constant=0.249991, up=0.5055, validation AUC=0.562143 (test−validation=-0.017608)
- fold 1: XGB=0.514151, Logistic=0.533043, persistence=0.490414, XGB−persistence=0.023737, best-direction persistence=0.509586, XGB−best-direction persistence=0.004564, Brier=0.250757 vs constant=0.249960, up=0.5065, validation AUC=0.547230 (test−validation=-0.033079)
- fold 2: XGB=0.527067, Logistic=0.530471, persistence=0.472962, XGB−persistence=0.054104, best-direction persistence=0.527038, XGB−best-direction persistence=0.000029, Brier=0.249594 vs constant=0.250116, up=0.4960, validation AUC=0.522958 (test−validation=0.004108)

### BTC 12h
- fold 0: XGB=0.559559, Logistic=0.545821, persistence=0.480484, XGB−persistence=0.079075, best-direction persistence=0.519516, XGB−best-direction persistence=0.040043, Brier=0.248107 vs constant=0.250292, up=0.5168, validation AUC=0.552810 (test−validation=0.006749)
- fold 1: XGB=0.494333, Logistic=0.464971, persistence=0.464053, XGB−persistence=0.030280, best-direction persistence=0.535947, XGB−best-direction persistence=-0.041614, Brier=0.249500 vs constant=0.249811, up=0.5282, validation AUC=0.510512 (test−validation=-0.016179)
- fold 2: XGB=0.521953, Logistic=0.509465, persistence=0.500495, XGB−persistence=0.021458, best-direction persistence=0.500495, XGB−best-direction persistence=0.021458, Brier=0.250028 vs constant=0.249854, up=0.5128, validation AUC=0.514084 (test−validation=0.007869)

### BTC 24h
- fold 0: XGB=0.538644, Logistic=0.552978, persistence=0.470210, XGB−persistence=0.068434, best-direction persistence=0.529790, XGB−best-direction persistence=0.008854, Brier=0.250834 vs constant=0.250035, up=0.5052, validation AUC=0.558881 (test−validation=-0.020237)
- fold 1: XGB=0.505817, Logistic=0.459257, persistence=0.470586, XGB−persistence=0.035231, best-direction persistence=0.529414, XGB−best-direction persistence=-0.023597, Brier=0.249449 vs constant=0.249544, up=0.5340, validation AUC=0.528639 (test−validation=-0.022821)
- fold 2: XGB=0.510219, Logistic=0.489951, persistence=0.507738, XGB−persistence=0.002480, best-direction persistence=0.507738, XGB−best-direction persistence=0.002480, Brier=0.251271 vs constant=0.249628, up=0.5212, validation AUC=0.491783 (test−validation=0.018436)

### ETH 1h
- fold 0: XGB=0.547374, Logistic=0.549926, persistence=0.462974, XGB−persistence=0.084400, best-direction persistence=0.537026, XGB−best-direction persistence=0.010348, Brier=0.248802 vs constant=0.249994, up=0.4965, validation AUC=0.556006 (test−validation=-0.008632)
- fold 1: XGB=0.533560, Logistic=0.550796, persistence=0.484536, XGB−persistence=0.049024, best-direction persistence=0.515464, XGB−best-direction persistence=0.018096, Brier=0.248835 vs constant=0.249942, up=0.5150, validation AUC=0.539841 (test−validation=-0.006281)
- fold 2: XGB=0.532826, Logistic=0.523870, persistence=0.472214, XGB−persistence=0.060612, best-direction persistence=0.527786, XGB−best-direction persistence=0.005040, Brier=0.249230 vs constant=0.250055, up=0.4958, validation AUC=0.538509 (test−validation=-0.005682)

### ETH 4h
- fold 0: XGB=0.524692, Logistic=0.528771, persistence=0.491223, XGB−persistence=0.033469, best-direction persistence=0.508777, XGB−best-direction persistence=0.015915, Brier=0.250164 vs constant=0.250061, up=0.4965, validation AUC=0.531289 (test−validation=-0.006597)
- fold 1: XGB=0.524981, Logistic=0.531733, persistence=0.494369, XGB−persistence=0.030612, best-direction persistence=0.505631, XGB−best-direction persistence=0.019349, Brier=0.249384 vs constant=0.249788, up=0.5210, validation AUC=0.536740 (test−validation=-0.011760)
- fold 2: XGB=0.518163, Logistic=0.521273, persistence=0.464750, XGB−persistence=0.053413, best-direction persistence=0.535250, XGB−best-direction persistence=-0.017087, Brier=0.250029 vs constant=0.250062, up=0.4998, validation AUC=0.534059 (test−validation=-0.015895)

### ETH 12h
- fold 0: XGB=0.548230, Logistic=0.520035, persistence=0.474201, XGB−persistence=0.074029, best-direction persistence=0.525799, XGB−best-direction persistence=0.022430, Brier=0.249393 vs constant=0.250168, up=0.5058, validation AUC=0.553296 (test−validation=-0.005066)
- fold 1: XGB=0.546594, Logistic=0.536757, persistence=0.468881, XGB−persistence=0.077713, best-direction persistence=0.531119, XGB−best-direction persistence=0.015474, Brier=0.248075 vs constant=0.250017, up=0.5337, validation AUC=0.536770 (test−validation=0.009823)
- fold 2: XGB=0.518480, Logistic=0.512290, persistence=0.489700, XGB−persistence=0.028781, best-direction persistence=0.510300, XGB−best-direction persistence=0.008180, Brier=0.250104 vs constant=0.249960, up=0.5065, validation AUC=0.539143 (test−validation=-0.020663)

### ETH 24h
- fold 0: XGB=0.535288, Logistic=0.523534, persistence=0.487503, XGB−persistence=0.047786, best-direction persistence=0.512497, XGB−best-direction persistence=0.022791, Brier=0.251370 vs constant=0.250029, up=0.5008, validation AUC=0.546006 (test−validation=-0.010718)
- fold 1: XGB=0.521736, Logistic=0.536194, persistence=0.487008, XGB−persistence=0.034728, best-direction persistence=0.512992, XGB−best-direction persistence=0.008743, Brier=0.249266 vs constant=0.249870, up=0.5218, validation AUC=0.548702 (test−validation=-0.026966)
- fold 2: XGB=0.556266, Logistic=0.466813, persistence=0.477090, XGB−persistence=0.079176, best-direction persistence=0.522910, XGB−best-direction persistence=0.033357, Brier=0.249908 vs constant=0.249895, up=0.5110, validation AUC=0.513008 (test−validation=0.043258)

### SOL 1h
- fold 0: XGB=0.524054, Logistic=0.526626, persistence=0.472243, XGB−persistence=0.051812, best-direction persistence=0.527757, XGB−best-direction persistence=-0.003703, Brier=0.249652 vs constant=0.250128, up=0.4983, validation AUC=0.534216 (test−validation=-0.010162)
- fold 1: XGB=0.513240, Logistic=0.521715, persistence=0.492720, XGB−persistence=0.020520, best-direction persistence=0.507280, XGB−best-direction persistence=0.005959, Brier=0.249850 vs constant=0.250244, up=0.5118, validation AUC=0.537838 (test−validation=-0.024599)
- fold 2: XGB=0.507408, Logistic=0.503729, persistence=0.486660, XGB−persistence=0.020748, best-direction persistence=0.513340, XGB−best-direction persistence=-0.005932, Brier=0.250030 vs constant=0.249963, up=0.4935, validation AUC=0.527564 (test−validation=-0.020156)

### SOL 4h
- fold 0: XGB=0.513702, Logistic=0.518099, persistence=0.493964, XGB−persistence=0.019738, best-direction persistence=0.506036, XGB−best-direction persistence=0.007665, Brier=0.249951 vs constant=0.250287, up=0.5040, validation AUC=0.541303 (test−validation=-0.027602)
- fold 1: XGB=0.516024, Logistic=0.503039, persistence=0.499856, XGB−persistence=0.016169, best-direction persistence=0.500144, XGB−best-direction persistence=0.015880, Brier=0.249655 vs constant=0.250147, up=0.5082, validation AUC=0.533092 (test−validation=-0.017068)
- fold 2: XGB=0.475136, Logistic=0.497898, persistence=0.482486, XGB−persistence=-0.007350, best-direction persistence=0.517514, XGB−best-direction persistence=-0.042378, Brier=0.251113 vs constant=0.249902, up=0.4838, validation AUC=0.532101 (test−validation=-0.056966)

### SOL 12h
- fold 0: XGB=0.521342, Logistic=0.533558, persistence=0.493482, XGB−persistence=0.027860, best-direction persistence=0.506518, XGB−best-direction persistence=0.014825, Brier=0.250879 vs constant=0.250261, up=0.4975, validation AUC=0.501013 (test−validation=0.020329)
- fold 1: XGB=0.517567, Logistic=0.505587, persistence=0.499108, XGB−persistence=0.018459, best-direction persistence=0.500892, XGB−best-direction persistence=0.016675, Brier=0.250172 vs constant=0.250541, up=0.5235, validation AUC=0.519493 (test−validation=-0.001926)
- fold 2: XGB=0.464568, Logistic=0.492772, persistence=0.493273, XGB−persistence=-0.028705, best-direction persistence=0.506727, XGB−best-direction persistence=-0.042160, Brier=0.252782 vs constant=0.249840, up=0.4823, validation AUC=0.533337 (test−validation=-0.068769)

### SOL 24h
- fold 0: XGB=0.501774, Logistic=0.550437, persistence=0.492928, XGB−persistence=0.008845, best-direction persistence=0.507072, XGB−best-direction persistence=-0.005298, Brier=0.252540 vs constant=0.250313, up=0.4948, validation AUC=0.485497 (test−validation=0.016276)
- fold 1: XGB=0.505322, Logistic=0.488497, persistence=0.496656, XGB−persistence=0.008666, best-direction persistence=0.503344, XGB−best-direction persistence=0.001978, Brier=0.250201 vs constant=0.250257, up=0.5062, validation AUC=0.499139 (test−validation=0.006183)
- fold 2: XGB=0.480675, Logistic=0.505750, persistence=0.481244, XGB−persistence=-0.000568, best-direction persistence=0.518756, XGB−best-direction persistence=-0.038081, Brier=0.252893 vs constant=0.250191, up=0.5065, validation AUC=0.521250 (test−validation=-0.040575)

### XRP 1h
- fold 0: XGB=0.551194, Logistic=0.544725, persistence=0.469856, XGB−persistence=0.081338, best-direction persistence=0.530144, XGB−best-direction persistence=0.021049, Brier=0.248164 vs constant=0.250098, up=0.4918, validation AUC=0.573079 (test−validation=-0.021886)
- fold 1: XGB=0.530644, Logistic=0.527695, persistence=0.499472, XGB−persistence=0.031172, best-direction persistence=0.500528, XGB−best-direction persistence=0.030116, Brier=0.249290 vs constant=0.249986, up=0.5038, validation AUC=0.535778 (test−validation=-0.005134)
- fold 2: XGB=0.526745, Logistic=0.524075, persistence=0.483972, XGB−persistence=0.042774, best-direction persistence=0.516028, XGB−best-direction persistence=0.010717, Brier=0.249281 vs constant=0.250170, up=0.4840, validation AUC=0.528770 (test−validation=-0.002025)

### XRP 4h
- fold 0: XGB=0.526088, Logistic=0.511485, persistence=0.460729, XGB−persistence=0.065360, best-direction persistence=0.539271, XGB−best-direction persistence=-0.013183, Brier=0.251783 vs constant=0.250031, up=0.5012, validation AUC=0.589592 (test−validation=-0.063504)
- fold 1: XGB=0.524214, Logistic=0.500033, persistence=0.500000, XGB−persistence=0.024214, best-direction persistence=0.500000, XGB−best-direction persistence=0.024214, Brier=0.249815 vs constant=0.250003, up=0.4993, validation AUC=0.523419 (test−validation=0.000794)
- fold 2: XGB=0.498866, Logistic=0.525033, persistence=0.492909, XGB−persistence=0.005956, best-direction persistence=0.507091, XGB−best-direction persistence=-0.008225, Brier=0.249425 vs constant=0.249991, up=0.4718, validation AUC=0.510758 (test−validation=-0.011892)

### XRP 12h
- fold 0: XGB=0.504211, Logistic=0.496837, persistence=0.495935, XGB−persistence=0.008276, best-direction persistence=0.504065, XGB−best-direction persistence=0.000146, Brier=0.252968 vs constant=0.249969, up=0.4933, validation AUC=0.570557 (test−validation=-0.066346)
- fold 1: XGB=0.498984, Logistic=0.518148, persistence=0.485899, XGB−persistence=0.013085, best-direction persistence=0.514101, XGB−best-direction persistence=-0.015117, Brier=0.250933 vs constant=0.249952, up=0.4930, validation AUC=0.533311 (test−validation=-0.034327)
- fold 2: XGB=0.498019, Logistic=0.530555, persistence=0.473574, XGB−persistence=0.024445, best-direction persistence=0.526426, XGB−best-direction persistence=-0.028408, Brier=0.248560 vs constant=0.249615, up=0.4617, validation AUC=0.496559 (test−validation=0.001459)

### XRP 24h
- fold 0: XGB=0.513738, Logistic=0.541622, persistence=0.438478, XGB−persistence=0.075260, best-direction persistence=0.561522, XGB−best-direction persistence=-0.047785, Brier=0.254765 vs constant=0.250117, up=0.4973, validation AUC=0.584933 (test−validation=-0.071195)
- fold 1: XGB=0.533561, Logistic=0.563053, persistence=0.467728, XGB−persistence=0.065833, best-direction persistence=0.532272, XGB−best-direction persistence=0.001289, Brier=0.250318 vs constant=0.250075, up=0.4975, validation AUC=0.545428 (test−validation=-0.011867)
- fold 2: XGB=0.477140, Logistic=0.518652, persistence=0.480348, XGB−persistence=-0.003208, best-direction persistence=0.519652, XGB−best-direction persistence=-0.042512, Brier=0.248438 vs constant=0.249430, up=0.4557, validation AUC=0.469792 (test−validation=0.007348)

長視窗 XGBoost 不勝 persistence 的 folds：SOL 12h fold 2、SOL 24h fold 2、XRP 24h fold 2；這些訊號可能主要來自趨勢延續，不宜解讀為模型學到額外資訊。
persistence AUC < 0.5 表示該視窗短期反轉；比較 XGB 的額外資訊時應使用 best-direction persistence。這是事後選方向的描述性參考，不可用於判定。

新切分重新訓練的 BTC 4h 合併 AUC 約 0.52784；舊 4h 報告的 fold 平均 AUC 約 0.52889（切分及統計方式不同，僅供參考，不參與判定）。

## 結論

沒有證據不等於沒有用。
通過組合：1h: XRP。單幣種僅為候選假說，待 2026-08-09 後的新資料累積再獨立驗證，不宣稱發現；至少兩幣種通過才進入深度模型複驗。

## 限制

- 先前 4h 實驗已使用過部分 test 時段，本次不是獨立確認。
- 24h 相鄰標籤高度重疊；每 4,000 列約只有 167 個不重疊區間，CI 較寬。
- 只測預設 XGBoost 與基準；其他視窗可能不適合 4h 的參數。
- AUC 不代表可交易，未扣交易成本，不能宣稱獲利。
