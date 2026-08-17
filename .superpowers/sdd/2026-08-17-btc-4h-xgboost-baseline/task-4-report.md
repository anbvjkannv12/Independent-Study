# Task 4 report

## Status

Implemented the baseline CLI, atomic JSON report, documentation, and read-only result notebook. Generated `logs/baseline_btc_4h.json` from the supplied five-year feature dataset.

## Verification

- `C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests -v`: 40 tests passed.
- Real-data CLI run completed with 30,628 train rows, 6,563 validation rows, and 6,564 test rows.
- Report validation confirmed 53 feature names, all three split summaries, model/benchmark validation and test reports, and 53 non-empty gain-importance values. Test ROC-AUC: `0.5159673890691272`.

## Concerns

The real CSV and feature manifest are present in the main workspace rather than the isolated worktree, so the real run used their explicit absolute paths. The committed report records those paths; normal project-root execution uses the documented relative defaults.
