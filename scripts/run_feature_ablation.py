from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from btc_4h_evaluation import (
    WalkForwardFold,
    build_expanding_folds,
    evaluate_directional_predictions,
    fit_platt_calibrator,
    select_balanced_accuracy_threshold,
)
from btc_4h_model_adapters import ModelSettings, fit_predict_xgboost
from train_btc_4h_baseline import load_baseline_dataset
from train_btc_4h_model_comparison import _aggregate_test_metrics, _fold_summary, write_report_atomically


LAGS = (1, 2, 3, 6, 12, 24)
WINDOWS = (6, 12, 24)
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "G1": ("rsi_14", "macd_pct", "macd_signal_pct", "bollinger_z_20", "atr_14_pct"),
    "G2": tuple(
        name
        for lag in LAGS
        for name in (f"log_return_lag_{lag}", f"volume_change_lag_{lag}")
    ),
    "G3": tuple(
        name
        for window in WINDOWS
        for name in (
            f"return_mean_{window}",
            f"return_std_{window}",
            f"volume_mean_{window}",
            f"range_mean_{window}",
        )
    ),
    "G4": ("hour_sin", "hour_cos", "dow_sin", "dow_cos"),
    "G5": (
        "taker_buy_base",
        "taker_buy_quote",
        "taker_buy_ratio",
        *(f"taker_buy_ratio_lag_{lag}" for lag in LAGS),
    ),
}
GROUP_NAMES = {
    "G1": "技術指標",
    "G2": "報酬／量能滯後",
    "G3": "滾動統計",
    "G4": "時間週期",
    "G5": "主動買盤",
}
EXPECTED_REMAINDER = {
    "quote_asset_volume",
    "number_of_trades",
    "log_return",
    "range_pct",
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "log_volume",
    "volume_change",
    "trade_intensity",
    "quote_volume_change",
}


def validate_feature_groups(feature_names: tuple[str, ...]) -> None:
    if len(feature_names) != 53 or len(set(feature_names)) != 53:
        raise ValueError("feature ablation requires exactly 53 unique features")
    grouped = tuple(itertools.chain.from_iterable(FEATURE_GROUPS.values()))
    if len(grouped) != len(set(grouped)):
        raise ValueError("feature groups must not overlap")
    missing = sorted(set(grouped).difference(feature_names))
    remainder = set(feature_names).difference(grouped)
    if missing:
        raise ValueError(f"manifest is missing grouped features: {', '.join(missing)}")
    if remainder != EXPECTED_REMAINDER:
        raise ValueError(
            "the 11 always-retained features do not match the preregistered design; "
            f"found: {', '.join(sorted(remainder))}"
        )


def _write_csv_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(temporary_path, index=False)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def run_xgboost_arm(
    folds: list[WalkForwardFold],
    feature_names: tuple[str, ...],
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    settings = ModelSettings(seed=seed)
    fold_reports: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold in folds:
        validation_raw, test_raw = fit_predict_xgboost(
            fold.train, fold.validation, fold.test, feature_names, settings
        )
        calibrator = fit_platt_calibrator(validation_raw, fold.validation["target_up"])
        validation_probability = calibrator.predict_proba(validation_raw.reshape(-1, 1))[:, 1]
        test_probability = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
        threshold = select_balanced_accuracy_threshold(
            fold.validation["target_up"], validation_probability
        )
        fold_reports.append(
            {
                "threshold": threshold,
                "calibration": "platt",
                "validation": {
                    **evaluate_directional_predictions(
                        fold.validation["target_up"], validation_probability, threshold
                    ),
                    "uncalibrated_roc_auc": float(
                        roc_auc_score(fold.validation["target_up"], validation_raw)
                    ),
                },
                "test": {
                    **evaluate_directional_predictions(
                        fold.test["target_up"], test_probability, threshold
                    ),
                    "uncalibrated_roc_auc": float(
                        roc_auc_score(fold.test["target_up"], test_raw)
                    ),
                },
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold.index,
                    "open_time": pd.to_datetime(fold.test["open_time"], utc=True),
                    "target_up": fold.test["target_up"].to_numpy(dtype=int),
                    "test_raw": np.asarray(test_raw, dtype=float),
                    "test_probability": np.asarray(test_probability, dtype=float),
                    "predicted_up": (test_probability >= threshold).astype(int),
                }
            )
        )
    return (
        {
            "seed": seed,
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
            "folds": fold_reports,
            "test_aggregate": _aggregate_test_metrics(fold_reports),
        },
        pd.concat(prediction_frames, ignore_index=True),
    )


def _validate_paired_predictions(
    a: pd.DataFrame, b: pd.DataFrame, allow_different_targets: bool = False
) -> None:
    keys = ["fold", "open_time"] if allow_different_targets else ["fold", "open_time", "target_up"]
    if len(a) != len(b) or not a.loc[:, keys].reset_index(drop=True).equals(
        b.loc[:, keys].reset_index(drop=True)
    ):
        raise ValueError("A and B predictions are not paired on identical test rows")


def pooled_auc(predictions: pd.DataFrame) -> float:
    return float(roc_auc_score(predictions["target_up"], predictions["test_raw"]))


def paired_auc_delta(a: pd.DataFrame, b: pd.DataFrame) -> float:
    _validate_paired_predictions(a, b)
    return pooled_auc(a) - pooled_auc(b)


def moving_block_bootstrap_deltas(
    a: pd.DataFrame,
    b: pd.DataFrame,
    block_length: int,
    replicates: int,
    seed: int,
    allow_different_targets: bool = False,
    score_columns: tuple[str, ...] = ("test_raw",),
) -> np.ndarray:
    _validate_paired_predictions(a, b, allow_different_targets)
    if block_length < 1 or replicates < 1:
        raise ValueError("block_length and replicates must be positive")
    rng = np.random.default_rng(seed)
    fold_positions = [np.flatnonzero(a["fold"].to_numpy() == fold) for fold in a["fold"].unique()]
    if any(len(positions) < block_length for positions in fold_positions):
        raise ValueError("every fold must contain at least block_length rows")

    target = a["target_up"].to_numpy(dtype=int)
    target_b = b["target_up"].to_numpy(dtype=int)
    raw_a = a.loc[:, score_columns].to_numpy(dtype=float)
    raw_b = b.loc[:, score_columns].to_numpy(dtype=float)
    deltas = np.empty(replicates, dtype=float)
    def sampled_auc(labels: np.ndarray, scores: np.ndarray) -> float:
        if np.all(scores == scores[0]):
            return 0.5
        positives = int(labels.sum())
        ranks = rankdata(scores, method="average")
        return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                     / (positives * (len(labels) - positives)))
    for replicate in range(replicates):
        sampled_parts: list[np.ndarray] = []
        for positions in fold_positions:
            count = math.ceil(len(positions) / block_length)
            starts = rng.integers(0, len(positions) - block_length + 1, size=count)
            local = np.concatenate(
                [np.arange(start, start + block_length, dtype=int) for start in starts]
            )[: len(positions)]
            sampled_parts.append(positions[local])
        sampled = np.concatenate(sampled_parts)
        sampled_target = target[sampled]
        sampled_target_b = target_b[sampled]
        if np.unique(sampled_target).size < 2 or np.unique(sampled_target_b).size < 2:
            raise RuntimeError("bootstrap sample contains a single target class")
        deltas[replicate] = float(np.mean([
            sampled_auc(sampled_target, raw_a[sampled, col])
            - sampled_auc(sampled_target_b, raw_b[sampled, col])
            for col in range(len(score_columns))
        ]))
    return deltas


def holm_adjust(p_values: dict[str, float]) -> dict[str, dict[str, float | int]]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    result: dict[str, dict[str, float | int]] = {}
    running = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        multiplier = total - index
        running = max(running, min(1.0, multiplier * p_values[key]))
        result[key] = {
            "rank": index + 1,
            "holm_alpha": 0.05 / multiplier,
            "adjusted_p_value": min(1.0, running),
        }
    return result


def _constant_brier_guardrail(folds: list[WalkForwardFold], predictions: pd.DataFrame) -> dict[str, object]:
    model_squared_errors: list[np.ndarray] = []
    constant_squared_errors: list[np.ndarray] = []
    per_fold: list[dict[str, float | int | bool]] = []
    for fold in folds:
        arm = predictions.loc[predictions["fold"] == fold.index]
        y = arm["target_up"].to_numpy(dtype=float)
        probability = arm["test_probability"].to_numpy(dtype=float)
        constant_probability = float(fold.train["target_up"].mean())
        model_brier = float(np.mean((probability - y) ** 2))
        constant_brier = float(np.mean((constant_probability - y) ** 2))
        model_squared_errors.append((probability - y) ** 2)
        constant_squared_errors.append((constant_probability - y) ** 2)
        per_fold.append(
            {
                "fold": fold.index,
                "model_brier": model_brier,
                "constant_brier": constant_brier,
                "passes": model_brier <= constant_brier,
            }
        )
    pooled_model = float(np.mean(np.concatenate(model_squared_errors)))
    pooled_constant = float(np.mean(np.concatenate(constant_squared_errors)))
    return {
        "definition": "B calibrated Brier <= train-prevalence constant predictor Brier on pooled test rows",
        "model_brier": pooled_model,
        "constant_brier": pooled_constant,
        "passes": pooled_model <= pooled_constant,
        "folds": per_fold,
    }


def analyse_groups(
    folds: list[WalkForwardFold],
    control: pd.DataFrame,
    treatments: dict[str, pd.DataFrame],
    epsilon: float,
    block_length: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, object]]:
    provisional: dict[str, dict[str, object]] = {}
    raw_p_values: dict[str, float] = {}
    for group_index, (group, treatment) in enumerate(treatments.items()):
        _validate_paired_predictions(control, treatment)
        fold_deltas = [
            paired_auc_delta(
                control.loc[control["fold"] == fold].reset_index(drop=True),
                treatment.loc[treatment["fold"] == fold].reset_index(drop=True),
            )
            for fold in control["fold"].unique()
        ]
        delta = paired_auc_delta(control, treatment)
        bootstrap = moving_block_bootstrap_deltas(
            control,
            treatment,
            block_length,
            bootstrap_replicates,
            bootstrap_seed + group_index,
        )
        # The lower-tail bootstrap probability is the preregistered one-sided evidence measure.
        p_value = float((1 + np.count_nonzero(bootstrap <= 0.0)) / (bootstrap_replicates + 1))
        raw_p_values[group] = p_value
        provisional[group] = {
            "name": GROUP_NAMES[group],
            "removed_features": list(FEATURE_GROUPS[group]),
            "fold_auc_deltas": fold_deltas,
            "pooled_control_auc": pooled_auc(control),
            "pooled_treatment_auc": pooled_auc(treatment),
            "pooled_auc_delta": delta,
            "bootstrap": {
                "replicates": bootstrap_replicates,
                "block_length": block_length,
                "seed": bootstrap_seed + group_index,
                "raw_one_sided_p_value": p_value,
                "unadjusted_95_percent_lower_bound": float(np.quantile(bootstrap, 0.05)),
                "unadjusted_95_percent_two_sided_ci": [
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                ],
            },
            "brier_guardrail": _constant_brier_guardrail(folds, treatment),
            "above_aa_noise": delta > epsilon,
            "_bootstrap_values": bootstrap,
        }

    adjusted = holm_adjust(raw_p_values)
    for group, result in provisional.items():
        bootstrap = result.pop("_bootstrap_values")
        holm = adjusted[group]
        lower_bound = float(np.quantile(bootstrap, float(holm["holm_alpha"])))
        result["bootstrap"].update(holm)
        result["bootstrap"]["holm_adjusted_lower_bound"] = lower_bound
        all_folds_positive = all(delta > 0 for delta in result["fold_auc_deltas"])
        significant = bool(holm["adjusted_p_value"] <= 0.05 and lower_bound > 0)
        contributed = bool(
            all_folds_positive
            and significant
            and result["above_aa_noise"]
            and result["brier_guardrail"]["passes"]
        )
        result["criteria"] = {
            "all_fold_deltas_positive": all_folds_positive,
            "holm_significant_and_lower_bound_positive": significant,
            "above_aa_noise": result["above_aa_noise"],
            "brier_guardrail_passes": result["brier_guardrail"]["passes"],
        }
        result["decision"] = "有貢獻" if contributed else "沒有證據顯示此特徵群有貢獻"
        result["possible_noise_features"] = bool(
            result["pooled_auc_delta"] < 0
            and result["bootstrap"]["unadjusted_95_percent_two_sided_ci"][1] < 0
        )
    return provisional


def _render_report(report: dict[str, object]) -> str:
    symbol = str(report["symbol"])
    lines = [
        f"# {symbol} 4 小時特徵 A/B test 結果",
        "",
        "> 本報告由 `scripts/run_feature_ablation.py` 依預先定案規則產生。主要指標為未校準 test ROC-AUC。",
        "",
        "## A/A 雜訊基準",
        "",
        f"三個 control seeds 的兩兩最大 |ΔAUC|（ε）為 **{report['aa_noise']['epsilon']:.6f}**。",
        "",
        "## 消融結果",
        "",
        f"Control（seed 42）合併 AUC 為 **{next(iter(report['groups'].values()))['pooled_control_auc']:.6f}**。Brier 護欄使用各 fold 訓練集上漲率作為常數機率基準。",
        "",
        "| 群組 | 移除後 AUC | ΔAUC (A−B) | Holm p | Holm 下界 | Brier 護欄 | 判定 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for group, result in report["groups"].items():
        guardrail = "通過" if result["brier_guardrail"]["passes"] else "未通過"
        lines.append(
            f"| {group} {result['name']} | {result['pooled_treatment_auc']:.6f} | "
            f"{result['pooled_auc_delta']:.6f} | {result['bootstrap']['adjusted_p_value']:.6f} | "
            f"{result['bootstrap']['holm_adjusted_lower_bound']:.6f} | {guardrail} | {result['decision']} |"
        )
    lines.extend(
        [
            "",
            "## 解讀",
            "",
            "- 只有同時通過三個 fold ΔAUC > 0、Holm 校正、A/A ε 與 Brier 護欄，才標記為「有貢獻」。",
            "- 未通過不代表特徵無用，只能解讀為本實驗沒有找到足夠證據。",
            "- `logs/ab/feature_ablation.json` 保存完整 fold 指標、bootstrap 與判定欄位；逐列預測保存在同目錄 CSV。",
        ]
    )
    contributed = [group for group, value in report["groups"].items() if value["decision"] == "有貢獻"]
    lines.append(
        "- 需進入 LSTM／Transformer 複驗的群組："
        + ("、".join(contributed) if contributed else "無；依預先定案時程，不執行深度模型複驗。")
    )
    lines.append("")
    return "\n".join(lines)


def _render_aggregate_report(reports: dict[str, dict[str, object]]) -> str:
    lines = [
        "# 四幣種 4 小時特徵 A/B test 彙整",
        "",
        "> BTC 為原預先註冊實驗；ETH、SOL、XRP 是使用完全相同流程的延伸驗證。各幣種的五項 Holm 校正在幣種內分別進行。",
        "",
        "## 結果總表",
        "",
        "| 幣種 | Control AUC | A/A ε | 群組 | 移除後 AUC | ΔAUC (A−B) | Holm p | 三 fold 同向 | 超過 ε | Brier | 判定 |",
        "|---|---:|---:|---|---:|---:|---:|---|---|---|---|",
    ]
    contributed: list[str] = []
    for symbol, report in reports.items():
        for group, result in report["groups"].items():
            criteria = result["criteria"]
            is_contributed = result["decision"] == "有貢獻"
            if is_contributed:
                contributed.append(f"{symbol}-{group}")
            lines.append(
                f"| {symbol} | {result['pooled_control_auc']:.6f} | {report['aa_noise']['epsilon']:.6f} | "
                f"{group} {result['name']} | {result['pooled_treatment_auc']:.6f} | "
                f"{result['pooled_auc_delta']:.6f} | {result['bootstrap']['adjusted_p_value']:.6f} | "
                f"{'是' if criteria['all_fold_deltas_positive'] else '否'} | "
                f"{'是' if criteria['above_aa_noise'] else '否'} | "
                f"{'通過' if criteria['brier_guardrail_passes'] else '未通過'} | {result['decision']} |"
            )
    lines.extend(
        [
            "",
            "## 跨幣種摘要",
            "",
            f"- 共評估 {len(reports) * len(FEATURE_GROUPS)} 個「幣種 × 特徵群」組合。",
            f"- 符合該幣種全部四項預定判準者：{len(contributed)} 個"
            + (f"（{'、'.join(contributed)}）。" if contributed else "。"),
            "- 「沒有證據」不等於特徵無效；效果可能小於本設計的檢定力，或在不同時間區段不穩定。",
            "- 各幣種完整 fold、bootstrap、Brier 與逐列預測保存在 `logs/ab/<symbol>/`。",
        ]
    )
    if contributed:
        lines.append("- 上述達標組合依原時程應再使用 LSTM／Transformer 複驗。")
    else:
        lines.append("- 沒有組合達標，因此不執行 LSTM／Transformer 複驗。")
    lines.append("")
    return "\n".join(lines)


def run_symbol_experiment(
    symbol: str,
    data_path: Path,
    manifest_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    dataset, feature_names = load_baseline_dataset(data_path, manifest_path)
    validate_feature_groups(feature_names)
    folds = build_expanding_folds(
        dataset,
        args.initial_train_rows,
        args.validation_rows,
        args.test_rows,
        args.fold_count,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_reports: dict[str, dict[str, object]] = {}
    arm_predictions: dict[str, pd.DataFrame] = {}

    for seed in (42, 43, 44):
        arm = f"A_seed{seed}"
        arm_reports[arm], arm_predictions[arm] = run_xgboost_arm(folds, feature_names, seed)
        _write_csv_atomically(arm_predictions[arm], output_dir / f"{arm}_predictions.csv")

    for group, removed in FEATURE_GROUPS.items():
        arm = f"B_without_{group}"
        reduced = tuple(name for name in feature_names if name not in removed)
        arm_reports[arm], arm_predictions[arm] = run_xgboost_arm(folds, reduced, 42)
        _write_csv_atomically(arm_predictions[arm], output_dir / f"{arm}_predictions.csv")

    aa_pairs = []
    for left, right in itertools.combinations((42, 43, 44), 2):
        delta = paired_auc_delta(
            arm_predictions[f"A_seed{left}"], arm_predictions[f"A_seed{right}"]
        )
        aa_pairs.append({"seeds": [left, right], "auc_delta": delta, "absolute_delta": abs(delta)})
    epsilon = max(pair["absolute_delta"] for pair in aa_pairs)
    treatments = {group: arm_predictions[f"B_without_{group}"] for group in FEATURE_GROUPS}
    group_results = analyse_groups(
        folds,
        arm_predictions["A_seed42"],
        treatments,
        epsilon,
        args.block_length,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    report = {
        "task": f"{symbol}USDT 4-hour feature ablation",
        "symbol": symbol,
        "study_role": "preregistered" if symbol == "BTC" else "same-protocol extension",
        "paths": {
            "data": str(data_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "prediction_directory": str(output_dir.resolve()),
        },
        "fold_settings": {
            "initial_train_rows": args.initial_train_rows,
            "validation_rows": args.validation_rows,
            "test_rows": args.test_rows,
            "fold_count": args.fold_count,
        },
        "folds": [_fold_summary(fold) for fold in folds],
        "primary_metric": "pooled uncalibrated test ROC-AUC",
        "aa_noise": {
            "definition": "maximum pairwise absolute pooled AUC delta",
            "pairs": aa_pairs,
            "epsilon": epsilon,
        },
        "arms": arm_reports,
        "groups": group_results,
    }
    write_report_atomically(report, output_dir / "feature_ablation.json")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four-asset 4-hour feature ablation")
    parser.add_argument(
        "--symbols", nargs="+", choices=("BTC", "ETH", "SOL", "XRP"),
        default=("BTC", "ETH", "SOL", "XRP"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--manifest", type=Path, default=Path("logs/feature_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/ab"))
    parser.add_argument("--report", type=Path, default=Path("docs/experiments/feature_tests/feature_ab_test_results.md"))
    parser.add_argument("--initial-train-rows", type=int, default=19755)
    parser.add_argument("--validation-rows", type=int, default=4000)
    parser.add_argument("--test-rows", type=int, default=4000)
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--block-length", type=int, default=24)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260925)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        reports: dict[str, dict[str, object]] = {}
        for symbol in args.symbols:
            symbol_output = args.output_dir / symbol.lower()
            report = run_symbol_experiment(
                symbol,
                args.data_dir / f"{symbol.lower()}_4h.csv",
                args.manifest,
                symbol_output,
                args,
            )
            reports[symbol] = report
            individual_report = args.report.parent / f"{symbol.lower()}_feature_ab_test_results.md"
            individual_report.parent.mkdir(parents=True, exist_ok=True)
            individual_report.write_text(_render_report(report), encoding="utf-8")

        aggregate = {
            "task": "BTC/ETH/SOL/XRP 4-hour feature ablation summary",
            "multiple_comparison_scope": "Holm correction is applied separately to five groups within each asset",
            "symbols": reports,
        }
        write_report_atomically(aggregate, args.output_dir / "multi_asset_feature_ablation.json")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(_render_aggregate_report(reports), encoding="utf-8")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Feature ablation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
