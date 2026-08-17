# BTCUSDT 4-hour XGBoost baseline

This baseline predicts the direction of the next four-hour BTCUSDT return.  It reads exactly the 53 features named by `logs/feature_manifest.json`, keeps the data in time order, and uses the first 70% for training, the next 15% for validation, and the final 15% for testing.

## Run it

```powershell
$py = "C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py -m pip install -r requirements-baseline.txt
& $py scripts\train_btc_4h_baseline.py
```

The command writes `logs/baseline_btc_4h.json` only after loading, validation, training, and evaluation all succeed.  The output is atomically replaced, so a failed run cannot leave a partial JSON report.

Optional paths and seed can be supplied explicitly:

```powershell
& $py scripts\train_btc_4h_baseline.py --data data\features\btc_4h.csv --manifest logs\feature_manifest.json --output logs\baseline_btc_4h.json --random-state 42
```

## Read the report

The report records its source paths, the 53 feature names, split counts and inclusive time ranges, fixed model settings, and validation/test results for the model and two constant benchmarks:

- `all_up`: always predicts an upward move.
- `training_majority`: predicts the training set's majority class.

Each evaluation has accuracy, precision, recall, F1, ROC-AUC (from positive-class probabilities), and a `[ [TN, FP], [FN, TP] ]` confusion matrix.  `gain_importance` is the model's gain-based XGBoost feature importance, sorted from greatest to least; absence of a feature means no tree used it.

Open `run/02_xgboost_baseline_btc_4h.ipynb` after running the command to inspect the saved report. The notebook does not train or fit a model.
