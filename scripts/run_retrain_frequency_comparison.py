"""Preregistered 4h fixed-vs-retraining frequency experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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

from btc_4h_model_adapters import fit_predict_xgboost, ModelSettings
from run_feature_ablation import moving_block_bootstrap_deltas, holm_adjust, _write_csv_atomically
from train_btc_4h_baseline import load_baseline_dataset
from train_btc_4h_model_comparison import write_report_atomically

SEEDS = (42, 43, 44)
FREQUENCIES = {"F": None, "R4000": 4000, "R1000": 1000, "R168": 168}
REPLICATES = 2000
BOOTSTRAP_SEED = 20260929
COLUMNS = tuple(f"test_raw_seed{s}" for s in SEEDS)


def retrain_schedule(eval_start: int, eval_end: int, every: int | None) -> list[tuple[int, int]]:
    """Half-open prediction intervals covering [eval_start, eval_end) exactly once."""
    if eval_start < 1 or eval_end <= eval_start or (every is not None and every < 1):
        raise ValueError("invalid evaluation bounds or frequency")
    if every is None:
        return [(eval_start, eval_end)]
    return [(start, min(start + every, eval_end)) for start in range(eval_start, eval_end, every)]


def predict_schedule(dataset: pd.DataFrame, features: tuple[str, ...], seed: int,
                     schedule: list[tuple[int, int]], purge_rows: int, eval_start: int) -> tuple[pd.DataFrame, list[dict]]:
    frames, runs = [], []
    for start, end in schedule:
        train_end = start - purge_rows
        if train_end < 2:
            raise ValueError(f"too few training rows at prediction index {start}")
        train, test = dataset.iloc[:train_end], dataset.iloc[start:end]
        begin = time.monotonic()
        _, raw = fit_predict_xgboost(train, test, test, features, ModelSettings(seed=seed))
        runs.append({"prediction_start_index": start, "prediction_end_exclusive": end,
                     "train_rows": len(train), "last_train_index": train_end - 1,
                     "seconds": time.monotonic() - begin})
        frames.append(pd.DataFrame({"segment": (np.arange(start, end) - eval_start) // 4000,
                                    "open_time": test.open_time.to_numpy(),
                                    "target_up": test.target_up.to_numpy(dtype=int),
                                    "test_raw": np.asarray(raw, dtype=float),
                                    "train_rows": len(train)}))
    return pd.concat(frames, ignore_index=True), runs


def joined_predictions(predictions: dict[int, pd.DataFrame]) -> pd.DataFrame:
    result = predictions[42][["segment", "open_time", "target_up"]].copy()
    for seed in SEEDS:
        p = predictions[seed]
        if not p[["segment", "open_time", "target_up"]].equals(result[["segment", "open_time", "target_up"]]):
            raise ValueError("seed test rows differ")
        result[f"test_raw_seed{seed}"] = p.test_raw.to_numpy()
    return result.rename(columns={"segment": "fold"})


def mean_auc(frame: pd.DataFrame) -> float:
    return float(np.mean([roc_auc_score(frame.target_up, frame[column]) for column in COLUMNS]))


def run_symbol(symbol: str, args: argparse.Namespace) -> dict:
    start_time = time.monotonic()
    path = args.data_dir / f"{symbol.lower()}_4h.csv"
    dataset, features = load_baseline_dataset(path, args.manifest)
    if len(features) != 53 or args.eval_rows != 12000:
        raise ValueError("preregistered experiment requires 53 features and 12,000 evaluation rows")
    eval_start, eval_end = len(dataset) - args.eval_rows, len(dataset)
    output = args.output_dir / symbol.lower()
    result = {"symbol": symbol, "total_rows": len(dataset), "eval_start_index": eval_start,
              "eval_end_exclusive": eval_end, "purge_rows": args.purge_rows,
              "eval_start_time": str(dataset.open_time.iloc[eval_start]),
              "eval_end_time": str(dataset.open_time.iloc[-1]),
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "python_version": sys.version, "xgboost_version": xgboost.__version__, "arms": {}, "comparisons": {}}
    joined = {}
    for name, every in FREQUENCIES.items():
        schedule = retrain_schedule(eval_start, eval_end, every)
        predictions, schedules = {}, {}
        for seed in SEEDS:
            print(f"{symbol} {name} seed{seed}: {len(schedule)} fits", flush=True)
            predictions[seed], schedules[str(seed)] = predict_schedule(dataset, features, seed, schedule,
                                                                        args.purge_rows, eval_start)
            _write_csv_atomically(predictions[seed], output / f"{name}_seed{seed}_predictions.csv")
        joined[name] = joined_predictions(predictions)
        aucs = {str(seed): float(roc_auc_score(predictions[seed].target_up, predictions[seed].test_raw))
                for seed in SEEDS}
        result["arms"][name] = {"schedule": schedules, "training_count_per_seed": len(schedule),
                                "auc_by_seed": aucs, "pooled_mean_auc": mean_auc(joined[name]),
                                "segment_mean_auc": [mean_auc(joined[name].query("fold == @i")) for i in range(3)],
                                "epsilon": max(abs(aucs[str(a)] - aucs[str(b)])
                                               for a, b in itertools.combinations(SEEDS, 2)),
                                "first_train_rows": schedules["42"][0]["train_rows"],
                                "last_train_rows": schedules["42"][-1]["train_rows"]}
        # The shared initial fit must be bitwise reproducible for all three seeds.
        if name != "F":
            length = min(every, args.eval_rows)
            for seed in SEEDS:
                baseline = joined["F"][f"test_raw_seed{seed}"].to_numpy()[:length]
                candidate = joined[name][f"test_raw_seed{seed}"].to_numpy()[:length]
                if not np.array_equal(baseline, candidate):
                    raise RuntimeError(f"{symbol} {name} seed{seed}: initial predictions differ from F")
    # Explicitly confirm each saved model's segment-0 predictions equal F for R4000.
    for seed in SEEDS:
        if not np.array_equal(joined["F"][f"test_raw_seed{seed}"].to_numpy()[:4000],
                              joined["R4000"][f"test_raw_seed{seed}"].to_numpy()[:4000]):
            raise RuntimeError("F/R4000 first segment differs")
    p_values = {}
    for name in ("R4000", "R1000", "R168"):
        a, b = joined[name], joined["F"]
        if not a[["fold", "open_time", "target_up"]].equals(b[["fold", "open_time", "target_up"]]):
            raise ValueError("arms do not have identical evaluation rows")
        delta = mean_auc(a) - mean_auc(b)
        segments = [mean_auc(a[a.fold == i]) - mean_auc(b[b.fold == i]) for i in range(3)]
        bootstrap = moving_block_bootstrap_deltas(a, b, args.block_length, REPLICATES,
                                                   BOOTSTRAP_SEED, score_columns=COLUMNS)
        p = (1 + int(np.count_nonzero(bootstrap <= 0))) / (REPLICATES + 1)
        epsilon = max(result["arms"][name]["epsilon"], result["arms"]["F"]["epsilon"])
        result["comparisons"][name] = {"delta_auc": delta, "segment_deltas": segments, "epsilon": epsilon,
                                        "bootstrap": {"block_length": args.block_length, "replicates": REPLICATES,
                                                      "seed": BOOTSTRAP_SEED, "raw_one_sided_p_value": p,
                                                      "unadjusted_95_percent_two_sided_ci":
                                                      [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])]}}
        p_values[name] = p
    for name, correction in holm_adjust(p_values).items():
        entry = result["comparisons"][name]
        entry["bootstrap"].update(correction)
        examined = entry["segment_deltas"][1:] if name == "R4000" else entry["segment_deltas"]
        entry["criteria"] = {"all_changed_segments_positive": all(x > 0 for x in examined),
                             "holm_significant": correction["adjusted_p_value"] <= 0.05,
                             "above_seed_noise": entry["delta_auc"] > entry["epsilon"]}
        entry["decision"] = ("比固定模型穩定地好" if all(entry["criteria"].values())
                             else "沒有證據顯示該重新訓練頻率比固定模型好")
        entry["possibly_weaker"] = (entry["delta_auc"] < 0 and
                                     entry["bootstrap"]["unadjusted_95_percent_two_sided_ci"][1] < 0)
    fixed = joined["F"]
    result["fixed_decay_curve"] = [{"interval": i, "auc": mean_auc(fixed.iloc[i * 1000:(i + 1) * 1000])}
                                   for i in range(12)]
    result["segment_up_rates"] = [float(fixed[fixed.fold == i].target_up.mean()) for i in range(3)]
    result["runtime_seconds"] = time.monotonic() - start_time
    write_report_atomically(result, output / "retrain_frequency_comparison.json")
    return result


def render_report(reports: dict[str, dict]) -> str:
    lines = ["# 4h 重新訓練頻率 vs 固定模型", "", "依預先登記設計 `docs/superpowers/specs/2026-09-29-retrain-frequency-design.md` 判定；未校準 AUC，四幣種各自 Holm 校正。", "", "| 幣種 | 組別 | 合併平均 AUC | 段 0／1／2 AUC | ΔAUC | 段 0／1／2 ΔAUC | ε | Holm p | 95% CI | 判定 |", "|---|---|---:|---|---:|---|---:|---:|---|---|"]
    winners: dict[str, list[str]] = {}
    for symbol, report in reports.items():
        for name, arm in report["arms"].items():
            item = report["comparisons"].get(name)
            if item and item["decision"] == "比固定模型穩定地好":
                winners.setdefault(name, []).append(symbol)
            lines.append(f"| {symbol} | {name} | {arm['pooled_mean_auc']:.6f} | "
                         + ", ".join(f"{v:.6f}" for v in arm["segment_mean_auc"]) + " | "
                         + (f"{item['delta_auc']:.6f} | " + ", ".join(f"{v:.6f}" for v in item["segment_deltas"])
                            + f" | {item['epsilon']:.6f} | {item['bootstrap']['adjusted_p_value']:.6f} | "
                            + ", ".join(f"{v:.6f}" for v in item["bootstrap"]["unadjusted_95_percent_two_sided_ci"])
                            + f" | {item['decision']}{'（可能較弱）' if item['possibly_weaker'] else ''} |"
                            if item else "— | — | — | — | — | 固定參照 |"))
    lines += ["", "## 固定模型的衰退曲線（每段 1,000 列；描述性）", "",
              "| 幣種 | 第 0～11 個區間的 AUC（三 seed 平均） |", "|---|---|"]
    for symbol, report in reports.items():
        lines.append(f"| {symbol} | " + ", ".join(f"{x['auc']:.6f}" for x in report["fixed_decay_curve"]) + " |")
    lines += ["", "## 訓練量與類別比例", "", "| 幣種 | 組別 | 訓練次數／seed | 首次／末次訓練列數 | 段 0／1／2 上漲率 | 執行秒數（幣種） |", "|---|---|---:|---|---|---:|"]
    for symbol, report in reports.items():
        for name, arm in report["arms"].items():
            lines.append(f"| {symbol} | {name} | {arm['training_count_per_seed']} | {arm['first_train_rows']}／{arm['last_train_rows']} | "
                         + ", ".join(f"{x:.4f}" for x in report["segment_up_rates"]) +
                         f" | {report['runtime_seconds']:.1f} |")
    lines += ["", "## 結論", ""]
    adopted = next((name for name in ("R4000", "R1000", "R168") if len(winners.get(name, [])) >= 2), None)
    if adopted:
        lines.append(f"{adopted} 在至少兩幣種通過，採用成本最低的通過組別 {adopted} 作為後續實驗標準。")
    elif winners:
        lines.append("只有單幣種通過的組別：" + "; ".join(f"{k}: {', '.join(v)}" for k, v in winners.items()) + "。列為候選，保持原標準，等待新資料獨立驗證。")
    else:
        lines.append("沒有證據顯示重新訓練頻率能改善 4h 方向訊號；不因結果調整訓練窗、超參數或特徵，下一步先預先登記三段式目標。")
    lines += ["沒有證據不等於重新訓練沒有用。", "", "## 限制", "", "- 評估期間與先前 4h 及多視窗 test 重疊，不是獨立確認。", "- 只測 expanding 訓練窗，未測 sliding。", "- 只測預設 XGBoost；沒有 Logistic、LSTM 或 Transformer。", "- AUC 不代表可交易，沒有扣交易成本。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", choices=("BTC", "ETH", "SOL", "XRP"), default=("BTC", "ETH", "SOL", "XRP"))
    parser.add_argument("--purge-rows", type=int, default=24)
    parser.add_argument("--eval-rows", type=int, default=12000)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/retrain_frequency"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--manifest", type=Path, default=Path("logs/feature_manifest.json"))
    parser.add_argument("--block-length", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4, help="Independent symbols run in separate processes")
    args = parser.parse_args()
    if args.purge_rows < 24 or args.eval_rows != 12000 or args.block_length != 48 or args.workers < 1:
        parser.error("preregistered settings require purge>=24, eval=12000, block=48, workers>=1")
    if len(set(args.symbols)) != len(args.symbols):
        parser.error("duplicate symbols")
    start = time.monotonic()
    if len(args.symbols) > 1 and args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(args.symbols))) as pool:
            reports = dict(zip(args.symbols, pool.map(run_symbol, args.symbols, itertools.repeat(args))))
    else:
        reports = {symbol: run_symbol(symbol, args) for symbol in args.symbols}
    write_report_atomically({"symbols": reports, "total_wall_seconds": time.monotonic() - start,
                             "selection": "first group in R4000,R1000,R168 passing at least 2 assets"},
                            args.output_dir / "multi_asset_retrain_frequency.json")
    path = Path("docs/retrain_frequency_results.md")
    path.write_text(render_report(reports), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
