"""Preregistered three-class target vs binary confidence-threshold comparison."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from btc_4h_evaluation import build_expanding_folds, WalkForwardFold
from btc_4h_model_adapters import ModelSettings, fit_predict_xgboost
from run_feature_ablation import _write_csv_atomically, holm_adjust
from train_btc_4h_baseline import load_baseline_dataset
from train_btc_4h_model_comparison import write_report_atomically

SYMBOLS = ("BTC", "ETH", "SOL", "XRP")
SEEDS = (42, 43, 44)
DEFAULT_K = 0.002
DEFAULT_COST = 0.002
DEFAULT_COVERAGE = 0.20
Q_CURVE = (0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261001
BLOCK_LENGTH = 48


def three_class_labels(r: np.ndarray, k: float) -> np.ndarray:
    values = np.asarray(r, dtype=float)
    labels = np.ones(values.shape, dtype=int)
    labels[values < -k] = 0
    labels[values > k] = 2
    return labels


def fit_predict_xgboost_multiclass(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(train.loc[:, features], train["target_3c"])
    val = model.predict_proba(validation.loc[:, features])
    tst = model.predict_proba(test.loc[:, features])
    return np.asarray(val, dtype=float), np.asarray(tst, dtype=float)


def persistence_signal(dataset: pd.DataFrame) -> np.ndarray:
    if "log_return" not in dataset.columns:
        raise ValueError("persistence requires log_return")
    return dataset["log_return"].rolling(4, min_periods=4).sum().to_numpy(dtype=float)


def _auc_safe(y: np.ndarray, s: np.ndarray) -> float:
    mask = np.isfinite(s)
    y2 = np.asarray(y, dtype=int)[mask]
    s2 = np.asarray(s, dtype=float)[mask]
    if len(np.unique(y2)) < 2:
        return 0.5
    return float(roc_auc_score(y2, s2))


def persistence_direction(train_signal: np.ndarray, train_target: np.ndarray) -> int:
    return 1 if _auc_safe(np.asarray(train_target), np.asarray(train_signal)) >= 0.5 else -1


def validation_threshold(val_scores: np.ndarray, q: float) -> float:
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    return float(np.quantile(np.abs(np.asarray(val_scores, dtype=float)), 1 - q))


def trade_returns(scores: np.ndarray, r: np.ndarray, t: float, c: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    r = np.asarray(r, dtype=float)
    traded = (np.abs(scores) >= t) & (scores != 0)
    out = np.full(scores.shape, np.nan, dtype=float)
    out[traded] = np.sign(scores[traded]) * r[traded] - c
    return out


def mean_trade_return(g: np.ndarray) -> float:
    values = np.asarray(g, dtype=float)
    finite = np.isfinite(values)
    return float(np.mean(values[finite])) if finite.any() else float("nan")


def block_bootstrap_indices(
    folds: list[np.ndarray] | list[pd.DataFrame], block_length: int, replicates: int, seed: int
) -> list[np.ndarray]:
    if block_length < 1 or replicates < 1:
        raise ValueError("block_length and replicates must be positive")
    lengths = [len(fold) for fold in folds]
    if any(length < block_length for length in lengths):
        raise ValueError("every fold must contain at least block_length rows")
    rng = np.random.default_rng(seed)
    offsets = np.cumsum([0, *lengths[:-1]])
    samples: list[np.ndarray] = []
    for _ in range(replicates):
        parts: list[np.ndarray] = []
        for offset, length in zip(offsets, lengths):
            count = math.ceil(length / block_length)
            starts = rng.integers(0, length - block_length + 1, size=count)
            local = np.concatenate([np.arange(s, s + block_length, dtype=int) for s in starts])[:length]
            parts.append(local + int(offset))
        samples.append(np.concatenate(parts))
    return samples


def _bootstrap_mean(values: np.ndarray, fold_lengths: list[int], replicates: int, seed: int) -> np.ndarray:
    idxs = block_bootstrap_indices([np.empty(n) for n in fold_lengths], BLOCK_LENGTH, replicates, seed)
    vals = np.asarray(values, dtype=float)
    out = np.empty(replicates, dtype=float)
    for i, idx in enumerate(idxs):
        sample = vals[idx]
        out[i] = mean_trade_return(sample)
    return out


def _fold_summary(fold: WalkForwardFold) -> dict[str, object]:
    return {
        "fold": fold.index,
        "train_rows": len(fold.train),
        "validation_rows": len(fold.validation),
        "test_rows": len(fold.test),
        "train_start": fold.train.open_time.iloc[0].isoformat(),
        "train_end": fold.train.open_time.iloc[-1].isoformat(),
        "validation_start": fold.validation.open_time.iloc[0].isoformat(),
        "validation_end": fold.validation.open_time.iloc[-1].isoformat(),
        "test_start": fold.test.open_time.iloc[0].isoformat(),
        "test_end": fold.test.open_time.iloc[-1].isoformat(),
    }


def _class_proportions(frame: pd.DataFrame) -> dict[str, float]:
    counts = frame["target_3c"].value_counts(normalize=True).to_dict()
    return {"down": float(counts.get(0, 0.0)), "neutral": float(counts.get(1, 0.0)), "up": float(counts.get(2, 0.0))}


def _prediction_frame(fold: WalkForwardFold, score: np.ndarray, threshold: float, group: str,
                      probs: np.ndarray | None = None) -> pd.DataFrame:
    r = fold.test.future_log_return.to_numpy(dtype=float)
    g = trade_returns(score, r, threshold, DEFAULT_COST)
    frame = pd.DataFrame({
        "fold": fold.index,
        "open_time": pd.to_datetime(fold.test.open_time, utc=True),
        "future_log_return": r,
        "target_up": fold.test.target_up.to_numpy(dtype=int),
        "target_3c": fold.test.target_3c.to_numpy(dtype=int),
        "score": score,
        "threshold": threshold,
        "traded": np.isfinite(g).astype(int),
        "net_return": g,
    })
    if group == "T":
        frame["p_down"] = probs[:, 0]
        frame["p_neutral"] = probs[:, 1]
        frame["p_up"] = probs[:, 2]
    elif group == "B":
        frame["p_down"] = np.nan
        frame["p_neutral"] = np.nan
        frame["p_up"] = score + 0.5
    else:
        frame["p_down"] = np.nan
        frame["p_neutral"] = np.nan
        frame["p_up"] = np.nan
    cols = ["fold", "open_time", "future_log_return", "target_up", "target_3c", "score",
            "p_down", "p_neutral", "p_up", "threshold", "traded", "net_return"]
    return frame[cols]


def _summarize_predictions(frame: pd.DataFrame) -> dict[str, object]:
    traded = frame["traded"].to_numpy(dtype=bool)
    r = frame.future_log_return.to_numpy(dtype=float)
    score = frame.score.to_numpy(dtype=float)
    net = frame.net_return.to_numpy(dtype=float)
    return {
        "trade_count": int(traded.sum()),
        "test_trade_ratio": float(traded.mean()),
        "mean_net_return": mean_trade_return(net),
        "mean_net_return_percent": float(np.expm1(mean_trade_return(net)) * 100) if traded.any() else float("nan"),
        "hit_rate": float((np.sign(score[traded]) == np.sign(r[traded])).mean()) if traded.any() else float("nan"),
        "large_move_rate": float((np.abs(r[traded]) > DEFAULT_COST).mean()) if traded.any() else float("nan"),
        "fold_mean_net_returns": [mean_trade_return(frame.loc[frame.fold == i, "net_return"].to_numpy()) for i in sorted(frame.fold.unique())],
    }


def _coverage_curve(val_scores: np.ndarray, test_scores: np.ndarray, test_r: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for q in Q_CURVE:
        t = validation_threshold(val_scores, q)
        g = trade_returns(test_scores, test_r, t, DEFAULT_COST)
        traded = np.isfinite(g)
        rows.append({"q": q, "threshold": t, "trade_count": int(traded.sum()),
                     "trade_ratio": float(traded.mean()), "mean_net_return": mean_trade_return(g),
                     "hit_rate": float((np.sign(test_scores[traded]) == np.sign(test_r[traded])).mean()) if traded.any() else float("nan")})
    return rows


def _seed_noise(means: dict[int, float]) -> float:
    return float(max(abs(means[a] - means[b]) for a, b in itertools.combinations(means, 2)))


def _decision(gate1: bool, gate2: bool) -> str:
    if gate1 and gate2:
        return "三段式目標比二分類加門檻穩定地好，且扣手續費後仍有正報酬"
    if (not gate1) and gate2:
        return "三段式挑的交易比二分類好，但扣手續費後沒有證據顯示有正報酬"
    if gate1 and (not gate2):
        return "扣手續費後有正報酬，但沒有證據顯示比二分類加門檻好"
    return "沒有證據顯示三段式目標能改善可用性"


def run_symbol(symbol: str, args: argparse.Namespace) -> dict[str, object]:
    start = time.perf_counter()
    data_path = args.data_dir / f"{symbol.lower()}_4h.csv"
    dataset, features = load_baseline_dataset(data_path, args.manifest)
    if len(features) != 53:
        raise ValueError("three-class experiment requires exactly 53 manifest features")
    dataset = dataset.copy()
    dataset["target_3c"] = three_class_labels(dataset.future_log_return.to_numpy(), args.k)
    initial = len(dataset) - 3 * (4000 + 4000 + 2 * args.purge_rows)
    folds = build_expanding_folds(dataset, initial, 4000, 4000, 3, purge_rows=args.purge_rows)
    if any(len(f.test) != 4000 for f in folds):
        raise ValueError("each fold test must contain exactly 4000 rows")
    for f in folds:
        if f.train.target_3c.nunique() != 3:
            raise ValueError(f"{symbol} fold {f.index} train lacks at least one three-class label")

    output_dir = args.output_dir / symbol.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    psig = persistence_signal(dataset)
    arms: dict[str, dict[str, object]] = {"T": {"seeds": {}}, "B": {"seeds": {}}, "P": {}}
    test_r_all = np.concatenate([f.test.future_log_return.to_numpy(dtype=float) for f in folds])
    test_fold_lengths = [len(f.test) for f in folds]

    # P baseline once.
    p_frames = []
    p_val_scores_parts, p_test_scores_parts = [], []
    offset = 0
    for f in folds:
        d = persistence_direction(psig[f.train.index.to_numpy()], f.train.target_up.to_numpy())
        val_s = d * psig[f.validation.index.to_numpy()]
        test_s = d * psig[f.test.index.to_numpy()]
        t = validation_threshold(val_s[np.isfinite(val_s)], args.coverage)
        p_frames.append(_prediction_frame(f, test_s, t, "P"))
        p_val_scores_parts.append(val_s)
        p_test_scores_parts.append(test_s)
        offset += len(f.test)
    p_pred = pd.concat(p_frames, ignore_index=True)
    _write_csv_atomically(p_pred, output_dir / "P_predictions.csv")
    arms["P"].update(_summarize_predictions(p_pred))
    arms["P"]["coverage_curve"] = _coverage_curve(np.concatenate(p_val_scores_parts), np.concatenate(p_test_scores_parts), test_r_all)

    for seed in SEEDS:
        t_frames, b_frames = [], []
        val_t_all, test_t_all, val_b_all, test_b_all = [], [], [], []
        for f in folds:
            val_prob_t, test_prob_t = fit_predict_xgboost_multiclass(f.train, f.validation, f.test, features, seed)
            if not np.allclose(val_prob_t.sum(axis=1), 1.0, atol=1e-6) or not np.allclose(test_prob_t.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("three-class probabilities do not sum to 1")
            val_score_t = val_prob_t[:, 2] - val_prob_t[:, 0]
            test_score_t = test_prob_t[:, 2] - test_prob_t[:, 0]
            thresh_t = validation_threshold(val_score_t, args.coverage)
            t_frames.append(_prediction_frame(f, test_score_t, thresh_t, "T", test_prob_t))
            val_t_all.append(val_score_t); test_t_all.append(test_score_t)

            val_prob_b, test_prob_b = fit_predict_xgboost(f.train, f.validation, f.test, features, ModelSettings(seed=seed))
            val_score_b = val_prob_b - 0.5
            test_score_b = test_prob_b - 0.5
            thresh_b = validation_threshold(val_score_b, args.coverage)
            b_frames.append(_prediction_frame(f, test_score_b, thresh_b, "B"))
            val_b_all.append(val_score_b); test_b_all.append(test_score_b)

        t_pred = pd.concat(t_frames, ignore_index=True)
        b_pred = pd.concat(b_frames, ignore_index=True)
        if not t_pred[["fold", "open_time"]].equals(b_pred[["fold", "open_time"]]) or not t_pred[["fold", "open_time"]].equals(p_pred[["fold", "open_time"]]):
            raise ValueError("T, B, P rows are not paired")
        _write_csv_atomically(t_pred, output_dir / f"T_seed{seed}_predictions.csv")
        _write_csv_atomically(b_pred, output_dir / f"B_seed{seed}_predictions.csv")
        arms["T"]["seeds"][str(seed)] = {**_summarize_predictions(t_pred),
            "direction_auc": _auc_safe(t_pred.target_up.to_numpy(), t_pred.score.to_numpy()),
            "coverage_curve": _coverage_curve(np.concatenate(val_t_all), np.concatenate(test_t_all), test_r_all)}
        arms["B"]["seeds"][str(seed)] = {**_summarize_predictions(b_pred),
            "direction_auc": _auc_safe(b_pred.target_up.to_numpy(), b_pred.score.to_numpy()),
            "coverage_curve": _coverage_curve(np.concatenate(val_b_all), np.concatenate(test_b_all), test_r_all)}

    t_means = {s: arms["T"]["seeds"][str(s)]["mean_net_return"] for s in SEEDS}
    b_means = {s: arms["B"]["seeds"][str(s)]["mean_net_return"] for s in SEEDS}
    t_counts = [arms["T"]["seeds"][str(s)]["trade_count"] for s in SEEDS]
    b_counts = [arms["B"]["seeds"][str(s)]["trade_count"] for s in SEEDS]
    valid_counts = min(t_counts + b_counts) >= 200
    epsilon_t, epsilon_b = _seed_noise(t_means), _seed_noise(b_means)
    epsilon = max(epsilon_t, epsilon_b)
    t_net_mean = np.nanmean([pd.read_csv(output_dir / f"T_seed{s}_predictions.csv").net_return.to_numpy(dtype=float) for s in SEEDS], axis=0)
    b_net_mean = np.nanmean([pd.read_csv(output_dir / f"B_seed{s}_predictions.csv").net_return.to_numpy(dtype=float) for s in SEEDS], axis=0)
    t_boot = _bootstrap_mean(t_net_mean, test_fold_lengths, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    d_boot = _bootstrap_mean(t_net_mean - b_net_mean, test_fold_lengths, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    m_t = mean_trade_return(t_net_mean)
    m_b = mean_trade_return(b_net_mean)
    delta = m_t - m_b
    fold_t = [mean_trade_return(t_net_mean[sum(test_fold_lengths[:i]):sum(test_fold_lengths[:i+1])]) for i in range(3)]
    fold_b = [mean_trade_return(b_net_mean[sum(test_fold_lengths[:i]):sum(test_fold_lengths[:i+1])]) for i in range(3)]
    fold_delta = [a - b for a, b in zip(fold_t, fold_b)]
    result = {
        "task": f"{symbol} 4h three-class target comparison",
        "symbol": symbol,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python_version": sys.version,
        "xgboost_version": xgboost.__version__,
        "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "settings": {"k": args.k, "cost": args.cost, "coverage": args.coverage, "purge_rows": args.purge_rows,
                     "seeds": list(SEEDS), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                     "bootstrap_seed": BOOTSTRAP_SEED, "block_length": BLOCK_LENGTH},
        "folds": [_fold_summary(f) for f in folds],
        "class_proportions": {str(f.index): {"train": _class_proportions(f.train), "validation": _class_proportions(f.validation), "test": _class_proportions(f.test)} for f in folds},
        "arms": arms,
        "primary": {"m_T": m_t, "m_B": m_b, "delta": delta, "epsilon_T": epsilon_t, "epsilon_B": epsilon_b,
                    "epsilon": epsilon, "fold_m_T": fold_t, "fold_m_B": fold_b, "fold_delta": fold_delta,
                    "valid_trade_counts": valid_counts,
                    "signal_bootstrap": {"raw_one_sided_p_value": float((1 + np.count_nonzero(t_boot <= 0)) / (BOOTSTRAP_REPLICATES + 1)),
                                         "unadjusted_95_percent_two_sided_ci": [float(x) for x in np.quantile(t_boot, [0.025, 0.975])]},
                    "comparison_bootstrap": {"raw_one_sided_p_value": float((1 + np.count_nonzero(d_boot <= 0)) / (BOOTSTRAP_REPLICATES + 1)),
                                             "unadjusted_95_percent_two_sided_ci": [float(x) for x in np.quantile(d_boot, [0.025, 0.975])]}},
        "runtime_seconds": float(time.perf_counter() - start),
    }
    write_report_atomically(result, output_dir / "three_class_target_comparison.json")
    return result


def aggregate(output_dir: Path, report_path: Path = Path("docs/three_class_target_results.md")) -> dict[str, object]:
    reports = {}
    for sym in SYMBOLS:
        path = output_dir / sym.lower() / "three_class_target_comparison.json"
        if path.exists():
            reports[sym] = json.loads(path.read_text(encoding="utf-8"))
    if not reports:
        raise ValueError("no symbol reports found")
    sig_adj = holm_adjust({s: r["primary"]["signal_bootstrap"]["raw_one_sided_p_value"] for s, r in reports.items()})
    cmp_adj = holm_adjust({s: r["primary"]["comparison_bootstrap"]["raw_one_sided_p_value"] for s, r in reports.items()})
    for s, r in reports.items():
        r["primary"]["signal_bootstrap"].update(sig_adj[s])
        r["primary"]["comparison_bootstrap"].update(cmp_adj[s])
        p = r["primary"]
        gate1 = bool(p["valid_trade_counts"] and all(x > 0 for x in p["fold_m_T"]) and p["signal_bootstrap"]["adjusted_p_value"] <= 0.05)
        gate2 = bool(p["valid_trade_counts"] and all(x > 0 for x in p["fold_delta"]) and p["comparison_bootstrap"]["adjusted_p_value"] <= 0.05 and p["delta"] > p["epsilon"])
        p["gate1_passes"] = gate1; p["gate2_passes"] = gate2; p["decision"] = _decision(gate1, gate2)
        p["possible_fee_adjusted_loss"] = bool(p["m_T"] < 0 and p["signal_bootstrap"]["unadjusted_95_percent_two_sided_ci"][1] < 0)
    summary = {"task": "multi-asset three-class target summary", "symbols": reports,
               "holm_scope": "four symbols for signal gate and four symbols for comparison gate"}
    write_report_atomically(summary, output_dir / "multi_asset_three_class_target.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(summary: dict[str, object]) -> str:
    lines = ["# 三段式目標比較結果", "", "預先登記：`docs/superpowers/specs/2026-09-28-three-class-target-design.md`。", "", "## 主表", "", "| 幣種 | m_T | m_B | Δ | ε | T trades | B trades | 關卡一 Holm p | 關卡二 Holm p | 判定 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for sym, rep in summary["symbols"].items():
        p = rep["primary"]
        t_trades = int(np.mean([rep["arms"]["T"]["seeds"][str(s)]["trade_count"] for s in SEEDS]))
        b_trades = int(np.mean([rep["arms"]["B"]["seeds"][str(s)]["trade_count"] for s in SEEDS]))
        lines.append(f"| {sym} | {p['m_T']:.6f} | {p['m_B']:.6f} | {p['delta']:.6f} | {p['epsilon']:.6f} | {t_trades} | {b_trades} | {p['signal_bootstrap']['adjusted_p_value']:.6f} | {p['comparison_bootstrap']['adjusted_p_value']:.6f} | {p['decision']} |")
    lines += ["", "## 交易比例曲線（q=20% 主要，其他描述性）", ""]
    for sym, rep in summary["symbols"].items():
        lines.append(f"### {sym}")
        lines.append("| 組別 | q | 交易數 | 實際比例 | m | 命中率 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for group in ("T", "B"):
            curve = rep["arms"][group]["seeds"]["42"]["coverage_curve"]
            for row in curve:
                lines.append(f"| {group} seed42 | {row['q']:.2f} | {row['trade_count']} | {row['trade_ratio']:.4f} | {row['mean_net_return']:.6f} | {row['hit_rate']:.4f} |")
        for row in rep["arms"]["P"]["coverage_curve"]:
            lines.append(f"| P | {row['q']:.2f} | {row['trade_count']} | {row['trade_ratio']:.4f} | {row['mean_net_return']:.6f} | {row['hit_rate']:.4f} |")
        lines.append("")
    lines += ["## 三類比例、AUC 與不重疊版本", "", "完整三類比例、T/B 方向 AUC、逐 fold m、95% CI 與所有逐列預測請見 `logs/three_class_target/` JSON 與 CSV。本報告由程式依預先登記產生。", "", "## 限制", "", "- test 期間和先前多個實驗重疊，不是獨立確認。", "- 固定 k = 0.002 讓各幣種的不交易比例不同。", "- 只扣固定手續費，沒有滑價、資金費率、部位上限。", "- 4h 標籤逐小時重疊，交易不是互相獨立。", "- 只測 XGBoost。", ""]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", choices=SYMBOLS, default=SYMBOLS)
    parser.add_argument("--purge-rows", type=int, default=24)
    parser.add_argument("--k", type=float, default=DEFAULT_K)
    parser.add_argument("--cost", type=float, default=DEFAULT_COST)
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/three_class_target"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--manifest", type=Path, default=Path("logs/feature_manifest.json"))
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.purge_rows < 24:
        print("--purge-rows must be >= 24", file=sys.stderr); return 2
    if args.k != DEFAULT_K or args.cost != DEFAULT_COST or args.coverage != DEFAULT_COVERAGE:
        print("--k, --cost and --coverage are preregistered fixed defaults", file=sys.stderr); return 2
    try:
        if not args.aggregate_only:
            for sym in args.symbols:
                print(f"Running {sym}...", flush=True)
                run_symbol(sym, args)
        aggregate(args.output_dir)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Three-class target comparison failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
