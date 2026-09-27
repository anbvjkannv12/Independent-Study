"""Preregistered, aligned and purged multi-horizon directional comparison."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import roc_auc_score

from btc_4h_evaluation import build_expanding_folds, evaluate_baselines, evaluate_directional_predictions
from run_feature_ablation import (
    _write_csv_atomically, holm_adjust, moving_block_bootstrap_deltas,
    pooled_auc, run_xgboost_arm,
)
from train_btc_4h_baseline import load_baseline_dataset
from train_btc_4h_model_comparison import _fold_summary, write_report_atomically

SEEDS = (42, 43, 44)
HORIZONS = (1, 4, 12, 24)
REPLICATES = 2000
BOOTSTRAP_SEED = 20260927


def align_datasets(datasets: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """Keep only shared timestamps; never pair a row by position alone."""
    if not datasets:
        raise ValueError("no horizons supplied")
    common = set.intersection(*(set(frame.open_time) for frame in datasets.values()))
    if not common:
        raise ValueError("no shared open_time values")
    aligned = {
        h: frame.loc[frame.open_time.isin(common)].reset_index(drop=True)
        for h, frame in datasets.items()
    }
    reference = next(iter(aligned.values())).open_time
    if any(not frame.open_time.equals(reference) for frame in aligned.values()):
        raise ValueError("horizon open_time timestamps are inconsistent after alignment")
    return aligned


def persistence_scores(dataset: pd.DataFrame, horizon: int) -> np.ndarray:
    """Sum returns ending at t; never use future_log_return or target_up."""
    if horizon < 1 or "log_return" not in dataset:
        raise ValueError("persistence requires log_return and a positive horizon")
    # Compute before slicing into folds so the first validation/test row has past history.
    return (dataset.log_return.rolling(horizon, min_periods=horizon).sum() > 0).to_numpy(dtype=float)


def joined_predictions(predictions: dict[int, pd.DataFrame]) -> pd.DataFrame:
    first = predictions[SEEDS[0]][["fold", "open_time", "target_up"]].copy()
    for seed in SEEDS:
        frame = predictions[seed]
        if not first[["fold", "open_time", "target_up"]].equals(frame[["fold", "open_time", "target_up"]]):
            raise ValueError("seed predictions are not paired")
        first[f"test_raw_seed{seed}"] = frame.test_raw.to_numpy()
    return first


def mean_auc(frame: pd.DataFrame) -> float:
    return float(np.mean([roc_auc_score(frame.target_up, frame[f"test_raw_seed{s}"]) for s in SEEDS]))


def bootstrap_result(values: np.ndarray) -> dict[str, object]:
    return {
        "replicates": len(values), "block_length": None, "seed": BOOTSTRAP_SEED,
        "raw_one_sided_p_value": float((1 + np.count_nonzero(values <= 0)) / (len(values) + 1)),
        "unadjusted_95_percent_two_sided_ci": [float(x) for x in np.quantile(values, [0.025, 0.975])],
    }


def run_symbol(symbol: str, args: argparse.Namespace) -> tuple[dict, dict[int, pd.DataFrame]]:
    datasets = {}
    hashes = {}
    names = None
    for h in args.horizons:
        path = args.data_dir / f"{symbol.lower()}_{h}h.csv"
        hashes[f"{h}h"] = hashlib.sha256(path.read_bytes()).hexdigest()
        datasets[h], features = load_baseline_dataset(path, args.manifest)
        if len(features) != 53 or (names is not None and features != names):
            raise ValueError("expected identical 53 manifest features for all horizons")
        names = features
    aligned = align_datasets(datasets)
    count = len(next(iter(aligned.values())))
    initial = count - 3 * (8000 + 2 * args.purge_rows)
    if initial < 1:
        raise ValueError("insufficient aligned rows for three purged folds")
    result = {
        "symbol": symbol, "aligned_rows": count, "initial_train_rows": initial,
        "alignment_start": str(next(iter(aligned.values())).open_time.iloc[0]),
        "alignment_end": str(next(iter(aligned.values())).open_time.iloc[-1]),
        "fold_settings": {"initial_train_rows": initial, "validation_rows": 4000,
                          "test_rows": 4000, "fold_count": 3, "purge_rows": args.purge_rows},
        "input_sha256": hashes, "horizons": {}, "comparisons": {},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python_version": sys.version, "xgboost_version": xgboost.__version__,
        "primary_metric": "mean over three seeds of pooled uncalibrated test ROC-AUC",
    }
    joined = {}
    for h, dataset in aligned.items():
        folds = build_expanding_folds(dataset, initial, 4000, 4000, 3, args.purge_rows)
        scores = persistence_scores(dataset, h)
        # Map original aligned row positions; fold test slices do not contain imputed future labels.
        test_starts = [initial + i * (8000 + 2 * args.purge_rows) + args.purge_rows + 4000 + args.purge_rows for i in range(3)]
        baseline = []
        for fold, start in zip(folds, test_starts):
            entry = evaluate_baselines(fold, names)
            entry["persistence"] = {
                "test": evaluate_directional_predictions(fold.test.target_up, scores[start:start + 4000], 0.5),
                "validation": evaluate_directional_predictions(
                    fold.validation.target_up,
                    scores[start - 4000 - args.purge_rows:start - args.purge_rows], 0.5),
            }
            baseline.append(entry)
        predictions = {}
        arms = {}
        for seed in SEEDS:
            arms[seed], predictions[seed] = run_xgboost_arm(folds, names, seed)
            _write_csv_atomically(predictions[seed], args.output_dir / symbol.lower() / f"{h}h_seed{seed}_predictions.csv")
        joined[h] = joined_predictions(predictions)
        aucs = {str(seed): pooled_auc(predictions[seed]) for seed in SEEDS}
        aa_pairs = [{"seeds": [a, b], "absolute_delta": abs(aucs[str(a)] - aucs[str(b)])}
                    for a, b in itertools.combinations(SEEDS, 2)]
        result["horizons"][str(h)] = {
            "folds": [_fold_summary(fold) for fold in folds], "arms": {str(k): v for k, v in arms.items()},
            "baselines": baseline, "pooled_auc_by_seed": aucs, "mean_pooled_auc": mean_auc(joined[h]),
            "epsilon": max(p["absolute_delta"] for p in aa_pairs), "aa_pairs": aa_pairs,
            "test_up_rate": float(joined[h].target_up.mean()),
            "fold_test_up_rates": [float(fold.test.target_up.mean()) for fold in folds],
            "brier_vs_train_constant": [{"fold": fold.index, "constant_brier": float(np.mean(
                (fold.train.target_up.mean() - fold.test.target_up.to_numpy()) ** 2)),
                "model_brier_by_seed": {str(s): arms[s]["folds"][fold.index]["test"]["brier_score"] for s in SEEDS}}
                for fold in folds],
        }
    return result, joined


def analyse(reports: dict[str, dict], joined: dict[str, dict[int, pd.DataFrame]], args: argparse.Namespace) -> None:
    columns = tuple(f"test_raw_seed{s}" for s in SEEDS)
    signal_p = {}
    bootstraps = {}
    for symbol, report in reports.items():
        for h in args.horizons:
            frame = joined[symbol][h]
            # Identical scores in the comparator give AUC exactly 0.5 for each resample.
            null = frame.copy()
            null.loc[:, columns] = 0.5
            boot = moving_block_bootstrap_deltas(frame, null, args.block_length, REPLICATES,
                                                   BOOTSTRAP_SEED, score_columns=columns)
            bootstraps[symbol, h] = boot
            item = report["horizons"][str(h)]
            item["signal_bootstrap"] = bootstrap_result(boot)
            item["signal_bootstrap"]["block_length"] = args.block_length
            signal_p[f"{symbol}-{h}h"] = item["signal_bootstrap"]["raw_one_sided_p_value"]
    adjusted = holm_adjust(signal_p)
    for symbol, report in reports.items():
        for h in args.horizons:
            item = report["horizons"][str(h)]
            item["signal_bootstrap"].update(adjusted[f"{symbol}-{h}h"])
            item["signal_passes"] = adjusted[f"{symbol}-{h}h"]["adjusted_p_value"] <= 0.05
        if 4 not in args.horizons:
            continue
        p_values = {}
        for h in args.horizons:
            if h == 4:
                continue
            a, b = joined[symbol][h], joined[symbol][4]
            if not a[["fold", "open_time"]].equals(b[["fold", "open_time"]]):
                raise ValueError("comparison test timestamps are not paired")
            delta = mean_auc(a) - mean_auc(b)
            fold_deltas = [mean_auc(a[a.fold == i]) - mean_auc(b[b.fold == i]) for i in range(3)]
            # All signal resamples use the same fold-local indices and seed.
            # Subtracting them gives the paired cross-horizon delta distribution.
            boot = bootstraps[symbol, h] - bootstraps[symbol, 4]
            epsilon = max(report["horizons"][str(h)]["epsilon"], report["horizons"]["4"]["epsilon"])
            entry = {"mean_pooled_auc_delta": delta, "fold_auc_deltas": fold_deltas,
                     "epsilon": epsilon, "bootstrap": bootstrap_result(boot)}
            entry["bootstrap"]["block_length"] = args.block_length
            report["comparisons"][str(h)] = entry
            p_values[str(h)] = entry["bootstrap"]["raw_one_sided_p_value"]
        for h, correction in holm_adjust(p_values).items():
            entry = report["comparisons"][h]
            entry["bootstrap"].update(correction)
            entry["criteria"] = {"all_folds_positive": all(x > 0 for x in entry["fold_auc_deltas"]),
                                 "holm_significant": correction["adjusted_p_value"] <= 0.05,
                                 "above_seed_noise": entry["mean_pooled_auc_delta"] > entry["epsilon"],
                                 "signal_passes": report["horizons"][h]["signal_passes"]}
            entry["decision"] = "較 4h 穩定地強" if all(entry["criteria"].values()) else "沒有證據顯示 h 比 4h 好"
            entry["possibly_weaker"] = (entry["mean_pooled_auc_delta"] < 0 and
                entry["bootstrap"]["unadjusted_95_percent_two_sided_ci"][1] < 0)


def render_report(reports: dict[str, dict], args: argparse.Namespace) -> str:
    lines = ["# 多視窗方向分類比較結果", "", "預先登記：`docs/superpowers/specs/2026-09-27-multi-horizon-comparison-design.md`。主要指標：未校準 test ROC-AUC。", "", "## 訊號關卡（16 格一起 Holm 校正）", "", "| 幣種 | 視窗 | 合併 AUC（三 seed 平均） | 各 fold AUC（三 seed 平均） | Holm p | 通過 |", "|---|---:|---:|---|---:|---|"]
    for symbol, report in reports.items():
        for h, item in report["horizons"].items():
            folds = [np.mean([item["arms"][str(s)]["folds"][i]["test"]["uncalibrated_roc_auc"] for s in SEEDS]) for i in range(3)]
            lines.append(f"| {symbol} | {h}h | {item['mean_pooled_auc']:.6f} | {', '.join(f'{x:.6f}' for x in folds)} | {item['signal_bootstrap']['adjusted_p_value']:.6f} | {'是' if item['signal_passes'] else '否'} |")
    lines += ["", "## 相對 4h 的比較（幣種內 Holm）", "", "| 幣種 | h | ΔAUC | fold ΔAUC | ε | Holm p | 95% CI | 判定 |", "|---|---:|---:|---|---:|---:|---|---|"]
    winners = {}
    for symbol, report in reports.items():
        for h, item in report["comparisons"].items():
            if item["decision"] == "較 4h 穩定地強":
                winners.setdefault(h, []).append(symbol)
            lines.append(f"| {symbol} | {h}h | {item['mean_pooled_auc_delta']:.6f} | {', '.join(f'{x:.6f}' for x in item['fold_auc_deltas'])} | {item['epsilon']:.6f} | {item['bootstrap']['adjusted_p_value']:.6f} | {', '.join(f'{x:.6f}' for x in item['bootstrap']['unadjusted_95_percent_two_sided_ci'])} | {item['decision']}{'（可能較弱）' if item['possibly_weaker'] else ''} |")
    lines += ["", "## 描述性檢查（不參與判定）", "", "各 fold 依序列出 XGBoost AUC（三 seed 平均）、Logistic AUC、persistence AUC、best-direction persistence = max(p, 1 − p)、校準 Brier（三 seed 平均）、訓練上漲率常數 Brier、test 上漲率、validation AUC（三 seed 平均）；各項基準與逐 seed 數值見 JSON。best-direction persistence 是看過 test 才選方向的描述性上限參考，不能參與判定。"]
    for symbol, report in reports.items():
        for h, item in report["horizons"].items():
            lines.append(f"\n### {symbol} {h}h")
            for i in range(3):
                arms = [item["arms"][str(s)]["folds"][i] for s in SEEDS]
                base = item["baselines"][i]
                brier = item["brier_vs_train_constant"][i]
                xauc = np.mean([a["test"]["uncalibrated_roc_auc"] for a in arms])
                persistence = base['persistence']['test']['roc_auc']
                best_persistence = max(persistence, 1 - persistence)
                lines.append(f"- fold {i}: XGB={xauc:.6f}, Logistic={base['logistic_regression']['test']['roc_auc']:.6f}, persistence={persistence:.6f}, XGB−persistence={xauc-persistence:.6f}, best-direction persistence={best_persistence:.6f}, XGB−best-direction persistence={xauc-best_persistence:.6f}, Brier={np.mean(list(brier['model_brier_by_seed'].values())):.6f} vs constant={brier['constant_brier']:.6f}, up={item['fold_test_up_rates'][i]:.4f}, validation AUC={np.mean([a['validation']['uncalibrated_roc_auc'] for a in arms]):.6f} (test−validation={xauc-np.mean([a['validation']['uncalibrated_roc_auc'] for a in arms]):.6f})")
    long_trend = [f"{symbol} {h}h fold {i}" for symbol, report in reports.items()
                  for h in ("12", "24") if h in report["horizons"]
                  for i, base in enumerate(report["horizons"][h]["baselines"])
                  if np.mean([report["horizons"][h]["arms"][str(s)]["folds"][i]["test"]["uncalibrated_roc_auc"]
                              for s in SEEDS]) <= base["persistence"]["test"]["roc_auc"]]
    lines += ["", "長視窗 XGBoost 不勝 persistence 的 folds：" + ("、".join(long_trend) if long_trend else "無") +
              "；這些訊號可能主要來自趨勢延續，不宜解讀為模型學到額外資訊。",
              "persistence AUC < 0.5 表示該視窗短期反轉；比較 XGB 的額外資訊時應使用 best-direction persistence。這是事後選方向的描述性參考，不可用於判定。",
              "", "新切分重新訓練的 BTC 4h 合併 AUC 約 0.52784；舊 4h 報告的 fold 平均 AUC 約 0.52889（切分及統計方式不同，僅供參考，不參與判定）。",
              "", "## 結論", "", "沒有證據不等於沒有用。"]
    if not winners:
        lines.append("沒有任何幣種的視窗通過步驟 2；不調參、不換特徵或 fold。下一步比較重新訓練頻率與固定模型。")
    else:
        lines.append("通過組合：" + "; ".join(f"{h}h: {', '.join(v)}" for h, v in winners.items()) + "。單幣種僅為候選假說，待 2026-08-09 後的新資料累積再獨立驗證，不宣稱發現；至少兩幣種通過才進入深度模型複驗。")
    lines += ["", "## 限制", "", "- 先前 4h 實驗已使用過部分 test 時段，本次不是獨立確認。", "- 24h 相鄰標籤高度重疊；每 4,000 列約只有 167 個不重疊區間，CI 較寬。", "- 只測預設 XGBoost 與基準；其他視窗可能不適合 4h 的參數。", "- AUC 不代表可交易，未扣交易成本，不能宣稱獲利。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", choices=("BTC", "ETH", "SOL", "XRP"), default=("BTC", "ETH", "SOL", "XRP"))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=HORIZONS)
    parser.add_argument("--purge-rows", type=int, default=24)
    parser.add_argument("--block-length", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/multi_horizon"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--manifest", type=Path, default=Path("logs/feature_manifest.json"))
    args = parser.parse_args()
    if args.purge_rows < 24 or args.block_length < 1 or len(set(args.horizons)) != len(args.horizons):
        parser.error("purge must be >=24, block length positive, horizons unique")
    reports, joined = {}, {}
    for symbol in args.symbols:
        print(f"Training {symbol}...", flush=True)
        reports[symbol], joined[symbol] = run_symbol(symbol, args)
    print("Bootstrapping...", flush=True)
    analyse(reports, joined, args)
    for symbol, report in reports.items():
        write_report_atomically(report, args.output_dir / symbol.lower() / "multi_horizon_comparison.json")
    write_report_atomically({"symbols": reports, "signal_holm_scope": "all requested symbol-horizon cells",
                             "comparison_holm_scope": "three comparisons per symbol"},
                            args.output_dir / "multi_asset_multi_horizon.json")
    path = Path("docs/multi_horizon_comparison_results.md")
    path.write_text(render_report(reports, args), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
