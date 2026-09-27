# SOL 4h training rerun summary

Task: SOLUSDT 4-hour walk-forward model comparison

| Model | ROC-AUC mean | ROC-AUC std | Brier mean | Accuracy mean | Balanced Acc mean | F1 mean |
|---|---:|---:|---:|---:|---:|---:|
| xgboost | 0.5034 | 0.0181 | 0.2502 | 0.5061 | 0.5041 | 0.4616 |
| lstm | 0.5089 | 0.0239 | 0.2500 | 0.5055 | 0.5095 | 0.3109 |
| transformer | 0.5106 | 0.0258 | 0.2497 | 0.5080 | 0.5039 | 0.3809 |

## Fold ROC-AUC
| Model | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|
| xgboost | 0.5181 | 0.5090 | 0.4832 |
| lstm | 0.4994 | 0.5361 | 0.4911 |
| transformer | 0.5377 | 0.5076 | 0.4863 |

## Baseline ROC-AUC
| Baseline | Mean | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|---:|
| all_up | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| training_majority | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| previous_direction | 0.4912 | 0.5013 | 0.4932 | 0.4793 |
| logistic_regression | 0.5048 | 0.5169 | 0.4993 | 0.4983 |

Conclusion: SOL remains near-random; Transformer has the highest mean ROC-AUC, but all models are unstable across folds and Fold 3 is below 0.5 for all three models.
