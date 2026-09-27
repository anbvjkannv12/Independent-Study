# ETH 4h training rerun summary

Task: ETHUSDT 4-hour walk-forward model comparison

| Model | ROC-AUC mean | ROC-AUC std | Brier mean | Accuracy mean | Balanced Acc mean | F1 mean |
|---|---:|---:|---:|---:|---:|---:|
| xgboost | 0.5221 | 0.0028 | 0.2499 | 0.5123 | 0.5139 | 0.4798 |
| lstm | 0.5232 | 0.0079 | 0.2499 | 0.5155 | 0.5131 | 0.4157 |
| transformer | 0.5257 | 0.0173 | 0.2495 | 0.5133 | 0.5154 | 0.3882 |

## Fold ROC-AUC
| Model | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|
| xgboost | 0.5200 | 0.5253 | 0.5211 |
| lstm | 0.5282 | 0.5272 | 0.5141 |
| transformer | 0.5317 | 0.5392 | 0.5063 |

Conclusion: follow plan wording — ETH has no reliable winning model; Transformer has the highest mean ROC-AUC, but fold stability is weak.
