"""Preregistered extension: predict high future 4h volatility instead of direction."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import roc_auc_score

from btc_4h_evaluation import build_expanding_folds
from btc_4h_model_adapters import ModelSettings, fit_predict_xgboost
from run_feature_ablation import holm_adjust, moving_block_bootstrap_deltas, _write_csv_atomically
from train_btc_4h_baseline import load_baseline_dataset
from train_btc_4h_model_comparison import write_report_atomically

SYMBOLS = ("BTC", "ETH", "SOL", "XRP")
SEEDS = (42, 43, 44)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261002
BLOCK_LENGTH = 48


def volatility_labels(abs_returns: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(abs_returns, dtype=float) > threshold).astype(int)


def volatility_persistence_signal(dataset: pd.DataFrame) -> np.ndarray:
    if "log_return" not in dataset.columns:
        raise ValueError("volatility persistence requires log_return")
    return dataset["log_return"].abs().rolling(4, min_periods=4).sum().to_numpy(dtype=float)


def _auc_safe(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    yy = np.asarray(y, dtype=int)[mask]
    ss = np.asarray(score, dtype=float)[mask]
    if len(np.unique(yy)) < 2:
        return 0.5
    return float(roc_auc_score(yy, ss))


def persistence_direction(train_signal: np.ndarray, train_target: np.ndarray) -> int:
    return 1 if _auc_safe(train_target, train_signal) >= 0.5 else -1


def joined_seed_predictions(predictions: dict[int, pd.DataFrame]) -> pd.DataFrame:
    first = predictions[SEEDS[0]][["fold", "open_time", "target_high_vol"]].copy()
    for seed in SEEDS:
        frame = predictions[seed]
        if not first[["fold", "open_time", "target_high_vol"]].equals(frame[["fold", "open_time", "target_high_vol"]]):
            raise ValueError("seed predictions are not paired")
        first[f"score_seed{seed}"] = frame["score"].to_numpy(dtype=float)
    return first


def mean_auc(frame: pd.DataFrame) -> float:
    return float(np.mean([_auc_safe(frame.target_high_vol.to_numpy(), frame[f"score_seed{s}"].to_numpy()) for s in SEEDS]))


def bootstrap_result(values: np.ndarray) -> dict[str, object]:
    return {
        "block_length": BLOCK_LENGTH,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "raw_one_sided_p_value": float((1 + np.count_nonzero(values <= 0)) / (len(values) + 1)),
        "unadjusted_95_percent_two_sided_ci": [float(x) for x in np.quantile(values, [0.025, 0.975])],
    }


def _fold_summary(fold) -> dict[str, object]:
    return {
        "index": fold.index,
        "train": {"count": len(fold.train), "start": fold.train.open_time.iloc[0].isoformat(), "end": fold.train.open_time.iloc[-1].isoformat()},
        "validation": {"count": len(fold.validation), "start": fold.validation.open_time.iloc[0].isoformat(), "end": fold.validation.open_time.iloc[-1].isoformat()},
        "test": {"count": len(fold.test), "start": fold.test.open_time.iloc[0].isoformat(), "end": fold.test.open_time.iloc[-1].isoformat()},
    }


def run_symbol(symbol: str, args: argparse.Namespace) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    start = time.perf_counter()
    data_path = args.data_dir / f"{symbol.lower()}_4h.csv"
    dataset, features = load_baseline_dataset(data_path, args.manifest)
    initial = len(dataset) - 3 * (4000 + 4000 + 2 * args.purge_rows)
    if initial < 1:
        raise ValueError("insufficient rows")
    folds_base = build_expanding_folds(dataset, initial, 4000, 4000, 3, args.purge_rows)
    psig = volatility_persistence_signal(dataset)
    seed_predictions: dict[int, pd.DataFrame] = {}
    seed_reports: dict[str, object] = {}
    p_frames = []
    thresholds = []
    for seed in SEEDS:
        frames = []
        fold_reports = []
        for fold in folds_base:
            threshold = float(fold.train.future_log_return.abs().median())
            thresholds.append(threshold)
            # Rebuild fold copies with fold-specific high-vol labels in target_up for existing adapter.
            train = fold.train.copy(); validation = fold.validation.copy(); test = fold.test.copy()
            for frame in (train, validation, test):
                frame["target_high_vol"] = volatility_labels(frame.future_log_return.abs().to_numpy(), threshold)
                frame["target_up"] = frame["target_high_vol"]
            val_prob, test_prob = fit_predict_xgboost(train, validation, test, features, ModelSettings(seed=seed))
            frames.append(pd.DataFrame({
                "fold": fold.index,
                "open_time": pd.to_datetime(test.open_time, utc=True),
                "future_log_return": test.future_log_return.to_numpy(dtype=float),
                "target_high_vol": test.target_high_vol.to_numpy(dtype=int),
                "score": test_prob,
            }))
            fold_reports.append({
                "fold": fold.index,
                "threshold_train_median_abs_return": threshold,
                "train_high_vol_rate": float(train.target_high_vol.mean()),
                "validation_high_vol_rate": float(validation.target_high_vol.mean()),
                "test_high_vol_rate": float(test.target_high_vol.mean()),
                "validation_auc": _auc_safe(validation.target_high_vol.to_numpy(), val_prob),
                "test_auc": _auc_safe(test.target_high_vol.to_numpy(), test_prob),
            })
        pred = pd.concat(frames, ignore_index=True)
        seed_predictions[seed] = pred
        seed_reports[str(seed)] = {"folds": fold_reports, "pooled_auc": _auc_safe(pred.target_high_vol.to_numpy(), pred.score.to_numpy())}
        _write_csv_atomically(pred, args.output_dir / symbol.lower() / f"xgb_seed{seed}_predictions.csv")

    # Persistence comparator uses seed42 labels/rows; score is deterministic.
    for fold in folds_base:
        threshold = float(fold.train.future_log_return.abs().median())
        test_target = volatility_labels(fold.test.future_log_return.abs().to_numpy(), threshold)
        train_target = volatility_labels(fold.train.future_log_return.abs().to_numpy(), threshold)
        direction = persistence_direction(psig[fold.train.index.to_numpy()], train_target)
        score = direction * psig[fold.test.index.to_numpy()]
        p_frames.append(pd.DataFrame({
            "fold": fold.index,
            "open_time": pd.to_datetime(fold.test.open_time, utc=True),
            "target_high_vol": test_target,
            "score_seed42": score,
            "score_seed43": score,
            "score_seed44": score,
        }))
    persistence = pd.concat(p_frames, ignore_index=True)
    joined = joined_seed_predictions(seed_predictions)
    p_auc = mean_auc(persistence)
    aa_pairs = [{"seeds": [a, b], "absolute_delta": abs(seed_reports[str(a)]["pooled_auc"] - seed_reports[str(b)]["pooled_auc"])} for a, b in itertools.combinations(SEEDS, 2)]
    result = {
        "symbol": symbol,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python_version": sys.version,
        "xgboost_version": xgboost.__version__,
        "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "settings": {"purge_rows": args.purge_rows, "seeds": list(SEEDS), "block_length": BLOCK_LENGTH, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED},
        "folds": [_fold_summary(f) for f in folds_base],
        "arms": {"xgboost": seed_reports, "persistence_volatility": {"pooled_auc": p_auc}},
        "mean_pooled_auc": mean_auc(joined),
        "persistence_pooled_auc": p_auc,
        "epsilon": max(p["absolute_delta"] for p in aa_pairs),
        "aa_pairs": aa_pairs,
        "runtime_seconds": float(time.perf_counter() - start),
    }
    return result, joined, persistence


def analyse(reports: dict[str, dict], joined: dict[str, pd.DataFrame], persistence: dict[str, pd.DataFrame]) -> None:
    columns = tuple(f"score_seed{s}" for s in SEEDS)
    signal_p, compare_p = {}, {}
    signal_boots, compare_boots = {}, {}
    for symbol, frame in joined.items():
        null = frame.copy(); null.loc[:, columns] = 0.5
        boot = moving_block_bootstrap_deltas(frame.rename(columns={"target_high_vol": "target_up"}), null.rename(columns={"target_high_vol": "target_up"}), BLOCK_LENGTH, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, score_columns=columns)
        signal_boots[symbol] = boot; signal_p[symbol] = bootstrap_result(boot)["raw_one_sided_p_value"]
        comp = moving_block_bootstrap_deltas(frame.rename(columns={"target_high_vol": "target_up"}), persistence[symbol].rename(columns={"target_high_vol": "target_up"}), BLOCK_LENGTH, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, score_columns=columns)
        compare_boots[symbol] = comp; compare_p[symbol] = bootstrap_result(comp)["raw_one_sided_p_value"]
    sig_adj, cmp_adj = holm_adjust(signal_p), holm_adjust(compare_p)
    for symbol, report in reports.items():
        sboot = bootstrap_result(signal_boots[symbol]); sboot.update(sig_adj[symbol])
        cboot = bootstrap_result(compare_boots[symbol]); cboot.update(cmp_adj[symbol])
        frame = joined[symbol]; per = persistence[symbol]
        fold_auc = [mean_auc(frame[frame.fold == i]) for i in range(3)]
        fold_delta = [mean_auc(frame[frame.fold == i]) - mean_auc(per[per.fold == i]) for i in range(3)]
        delta = report["mean_pooled_auc"] - report["persistence_pooled_auc"]
        report["signal_bootstrap"] = sboot
        report["comparison"] = {"delta_auc": delta, "fold_auc": fold_auc, "fold_delta_auc": fold_delta, "bootstrap": cboot,
                                  "criteria": {"all_folds_auc_above_half": all(x > 0.5 for x in fold_auc), "signal_holm_significant": sboot["adjusted_p_value"] <= 0.05,
                                               "all_fold_deltas_positive": all(x > 0 for x in fold_delta), "comparison_holm_significant": cboot["adjusted_p_value"] <= 0.05,
                                               "above_seed_noise": delta > report["epsilon"]}}
        crit = report["comparison"]["criteria"]
        report["decision"] = "波動大小預測通過預定判準" if all(crit.values()) else "沒有證據顯示波動大小模型穩定勝過基準"


def render(summary: dict[str, object]) -> str:
    lines = ["# 波動大小預測延伸研究結果", "", "預先登記：`docs/superpowers/specs/2026-09-30-volatility-target-extension-design.md`。", "", "| 幣種 | XGB AUC | persistence AUC | ΔAUC | ε | 訊號 Holm p | 比較 Holm p | 判定 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for s, r in summary["symbols"].items():
        lines.append(f"| {s} | {r['mean_pooled_auc']:.6f} | {r['persistence_pooled_auc']:.6f} | {r['comparison']['delta_auc']:.6f} | {r['epsilon']:.6f} | {r['signal_bootstrap']['adjusted_p_value']:.6f} | {r['comparison']['bootstrap']['adjusted_p_value']:.6f} | {r['decision']} |")
    passed = [s for s, r in summary["symbols"].items() if r["decision"] == "波動大小預測通過預定判準"]
    if len(passed) >= 2:
        conclusion = f"本延伸研究只使用 2026-08-09 前既有資料與預先登記規則。共有 {len(passed)} 個幣種通過兩道關卡，依停止條件，波動大小預測可列為下一階段主研究方向；後續仍需另以新資料獨立驗證。"
    elif len(passed) == 1:
        conclusion = "本延伸研究只有 1 個幣種通過，依停止條件僅列為候選假說，等待新資料驗證。"
    else:
        conclusion = "本延伸研究沒有幣種通過，依停止條件不繼續在 OHLCV 特徵上調整，改評估跨幣種或訂單簿資料。"
    lines += ["", "## 結論", "", conclusion, ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", choices=SYMBOLS, default=SYMBOLS)
    parser.add_argument("--data-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--manifest", type=Path, default=Path("logs/feature_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/volatility_target"))
    parser.add_argument("--report", type=Path, default=Path("docs/experiments/volatility_target/volatility_target_extension_results.md"))
    parser.add_argument("--purge-rows", type=int, default=24)
    args = parser.parse_args()
    if args.purge_rows < 24:
        parser.error("purge rows must be >=24")
    try:
        reports, joined, persistence = {}, {}, {}
        for symbol in args.symbols:
            print(f"Running {symbol}...", flush=True)
            reports[symbol], joined[symbol], persistence[symbol] = run_symbol(symbol, args)
        analyse(reports, joined, persistence)
        summary = {"task": "high future volatility extension", "symbols": reports}
        write_report_atomically(summary, args.output_dir / "multi_asset_volatility_target.json")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render(summary), encoding="utf-8")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Volatility target extension failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
