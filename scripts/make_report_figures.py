"""Generate final-report figures from saved experiment JSON files only."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT = ROOT / "docs" / "figures"
SYMBOLS = ("BTC", "ETH", "SOL", "XRP")
MODELS = ("xgboost", "lstm", "transformer")
HORIZONS = ("1", "4", "12", "24")
COST = 0.002

plt.rcParams.update({
    "font.family": ["Microsoft JhengHei", "Noto Sans CJK TC", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
})


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def savefig(name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close()


def auc_axis(ax, low=0.48, high=0.56):
    ax.axhline(0.5, color="black", lw=1, ls="--", label="0.5 隨機基準")
    ax.set_ylim(low, high)
    ax.set_ylabel("ROC-AUC")


def f1_flow(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis("off")
    steps = ["Binance\n1h K 線", "品質檢查", "缺漏補值", "53 個特徵", "Walk-forward\n+ purge", "評估與校正"]
    xs = np.linspace(0.08, 0.92, len(steps))
    for i, (x, text) in enumerate(zip(xs, steps)):
        ax.text(x, 0.5, text, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="#eef5ff", ec="#3569a8"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i+1]-0.055, 0.5), xytext=(x+0.055, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"))
    ax.set_title("F1 資料與評估流程")
    data["F1"] = {"steps": steps, "source": "project pipeline summary"}
    savefig("F1_data_pipeline.png")


def f2_walk_forward(data: dict) -> None:
    mh = load_json(LOGS / "multi_horizon" / "multi_asset_multi_horizon.json")
    folds = mh["symbols"]["BTC"]["horizons"]["4"]["folds"]
    fig, ax = plt.subplots(figsize=(9, 2.8))
    colors = {"train": "#9ecae1", "validation": "#fdae6b", "test": "#a1d99b"}
    rows = []
    for y, fold in enumerate(folds):
        for part in ("train", "validation", "test"):
            start = pd.to_datetime(fold[part]["start"])
            end = pd.to_datetime(fold[part]["end"])
            ax.broken_barh([(mdates.date2num(start), mdates.date2num(end) - mdates.date2num(start))],
                           (y - 0.32, 0.24), facecolors=colors[part], label=part if y == 0 else None)
            rows.append({"fold": y, "part": part, "start": start.isoformat(), "end": end.isoformat()})
        ax.text(mdates.date2num(pd.to_datetime(fold["validation"]["start"])) - 7, y, "purge", ha="right", va="center", fontsize=8)
        ax.text(mdates.date2num(pd.to_datetime(fold["test"]["start"])) - 7, y, "purge", ha="right", va="center", fontsize=8)
    ax.set_yticks(range(len(folds)), [f"fold {i}" for i in range(len(folds))])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(ncol=3, loc="upper left")
    ax.set_title("F2 Walk-forward 切分與 24 列 purge 示意（BTC 4h）")
    data["F2"] = {"source": "logs/multi_horizon/multi_asset_multi_horizon.json", "rows": rows}
    savefig("F2_walk_forward_split.png")


def f3_model_auc(data: dict) -> None:
    vals = {s: {} for s in SYMBOLS}
    for s in SYMBOLS:
        rep = load_json(LOGS / f"{s.lower()}_4h_model_comparison.json")
        for m in MODELS:
            vals[s][m] = rep["models"][m]["test_aggregate"]["roc_auc"]["mean"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(SYMBOLS)); width = 0.24
    for i, m in enumerate(MODELS):
        ax.bar(x + (i-1)*width, [vals[s][m] for s in SYMBOLS], width, label=m)
    auc_axis(ax, 0.49, 0.54)
    ax.set_xticks(x, SYMBOLS)
    ax.set_title("F3 四幣種 × 三模型 4h test ROC-AUC")
    ax.legend()
    data["F3"] = {"source": "logs/*_4h_model_comparison.json", "auc": vals}
    savefig("F3_model_auc.png")


def f4_horizon_heatmap(data: dict) -> None:
    mh = load_json(LOGS / "multi_horizon" / "multi_asset_multi_horizon.json")
    mat = np.array([[mh["symbols"][s]["horizons"][h]["mean_pooled_auc"] for h in HORIZONS] for s in SYMBOLS])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    im = ax.imshow(mat, vmin=0.49, vmax=0.56, cmap="RdYlBu_r")
    ax.set_xticks(range(len(HORIZONS)), [h + "h" for h in HORIZONS])
    ax.set_yticks(range(len(SYMBOLS)), SYMBOLS)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="ROC-AUC")
    ax.set_title("F4 多視窗 XGBoost 平均 AUC")
    data["F4"] = {"source": "logs/multi_horizon/multi_asset_multi_horizon.json", "auc": {s: {h: float(mh["symbols"][s]["horizons"][h]["mean_pooled_auc"]) for h in HORIZONS} for s in SYMBOLS}}
    savefig("F4_multi_horizon_heatmap.png")


def f5_persistence(data: dict) -> None:
    mh = load_json(LOGS / "multi_horizon" / "multi_asset_multi_horizon.json")
    rows = {}
    for s in SYMBOLS:
        item = mh["symbols"][s]["horizons"]["1"]
        xgb = item["mean_pooled_auc"]
        p = float(np.mean([fold["persistence"]["test"]["roc_auc"] for fold in item["baselines"]]))
        rows[s] = {"xgboost": xgb, "persistence": p, "best_direction_persistence": max(p, 1-p)}
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(SYMBOLS)); width = 0.34
    ax.bar(x-width/2, [rows[s]["xgboost"] for s in SYMBOLS], width, label="XGBoost 1h")
    ax.bar(x+width/2, [rows[s]["best_direction_persistence"] for s in SYMBOLS], width, label="best-direction persistence")
    auc_axis(ax, 0.49, 0.58)
    ax.set_xticks(x, SYMBOLS)
    ax.set_title("F5 1h XGBoost vs 事後最佳方向 persistence")
    ax.legend()
    data["F5"] = {"source": "logs/multi_horizon/multi_asset_multi_horizon.json", "rows": rows}
    savefig("F5_1h_persistence_auc.png")


def f6_val_test_scatter(data: dict) -> None:
    mh = load_json(LOGS / "multi_horizon" / "multi_asset_multi_horizon.json")
    rows = []
    for s in SYMBOLS:
        for h in HORIZONS:
            item = mh["symbols"][s]["horizons"][h]
            for i in range(3):
                val = float(np.mean([item["arms"][str(seed)]["folds"][i]["validation"]["uncalibrated_roc_auc"] for seed in (42,43,44)]))
                test = float(np.mean([item["arms"][str(seed)]["folds"][i]["test"]["uncalibrated_roc_auc"] for seed in (42,43,44)]))
                rows.append({"symbol": s, "horizon": h, "fold": i, "validation_auc": val, "test_auc": test})
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for h, marker in zip(HORIZONS, ["o", "s", "^", "D"]):
        part = [r for r in rows if r["horizon"] == h]
        ax.scatter([r["validation_auc"] for r in part], [r["test_auc"] for r in part], marker=marker, label=h+"h", alpha=0.8)
    lo, hi = 0.47, 0.62
    ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1)
    ax.axhline(0.5, color="gray", ls=":"); ax.axvline(0.5, color="gray", ls=":")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("validation AUC"); ax.set_ylabel("test AUC")
    ax.set_title("F6 validation AUC vs test AUC")
    ax.legend(ncol=2)
    data["F6"] = {"source": "logs/multi_horizon/multi_asset_multi_horizon.json", "rows": rows}
    savefig("F6_validation_vs_test_auc.png")


def f7_retrain_delta(data: dict) -> None:
    rt = load_json(LOGS / "retrain_frequency" / "multi_asset_retrain_frequency.json")
    labels, deltas, lo, hi = [], [], [], []
    for s in SYMBOLS:
        for arm in ("R4000", "R1000", "R168"):
            c = rt["symbols"][s]["comparisons"][arm]
            labels.append(f"{s}\n{arm}")
            deltas.append(c["delta_auc"])
            ci = c["bootstrap"]["unadjusted_95_percent_two_sided_ci"]
            lo.append(c["delta_auc"] - ci[0]); hi.append(ci[1] - c["delta_auc"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(labels))
    ax.errorbar(x, deltas, yerr=[lo, hi], fmt="o", capsize=3, color="#2b6cb0")
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("ΔAUC vs fixed")
    ax.set_title("F7 重訓頻率相對固定模型的 ΔAUC 與 95% CI")
    data["F7"] = {"source": "logs/retrain_frequency/multi_asset_retrain_frequency.json", "rows": [{"label": l, "delta_auc": float(d), "err_low": float(a), "err_high": float(b)} for l,d,a,b in zip(labels,deltas,lo,hi)]}
    savefig("F7_retrain_delta_auc.png")


def f8_decay(data: dict) -> None:
    rt = load_json(LOGS / "retrain_frequency" / "multi_asset_retrain_frequency.json")
    fig, ax = plt.subplots(figsize=(8, 4))
    rows = {}
    for s in SYMBOLS:
        curve = rt["symbols"][s]["fixed_decay_curve"]
        rows[s] = curve
        ax.plot([r["interval"] for r in curve], [r["auc"] for r in curve], marker="o", label=s)
    auc_axis(ax, 0.45, 0.60)
    ax.set_xlabel("1,000 列區間")
    ax.set_title("F8 固定模型分段 AUC")
    ax.legend(ncol=4)
    data["F8"] = {"source": "logs/retrain_frequency/multi_asset_retrain_frequency.json", "rows": rows}
    savefig("F8_fixed_model_decay.png")


def f9_three_class_curve(data: dict) -> None:
    tc = load_json(LOGS / "three_class_target" / "multi_asset_three_class_target.json")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    rows = {}
    for ax, s in zip(axes.ravel(), SYMBOLS):
        rep = tc["symbols"][s]
        rows[s] = {}
        for group, style in (("T", "-o"), ("B", "--s")):
            curve = rep["arms"][group]["seeds"]["42"]["coverage_curve"]
            rows[s][group] = curve
            ax.plot([r["q"] for r in curve], [r["mean_net_return"]*100 for r in curve], style, label=group)
        curve = rep["arms"]["P"]["coverage_curve"]
        rows[s]["P"] = curve
        ax.plot([r["q"] for r in curve], [r["mean_net_return"]*100 for r in curve], ":^", label="P")
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.axhline(-0.2, color="gray", ls=":", lw=1)
        ax.set_title(s); ax.set_ylabel("每筆淨報酬 %")
    for ax in axes[-1]: ax.set_xlabel("交易比例 q")
    axes[0,0].legend(ncol=3, fontsize=8)
    fig.suptitle("F9 三段式交易比例 q 對每筆淨報酬")
    data["F9"] = {"source": "logs/three_class_target/multi_asset_three_class_target.json", "rows": rows}
    savefig("F9_three_class_coverage_curve.png")


def f10_fee_breakeven(data: dict) -> None:
    rows = {}
    tc = load_json(LOGS / "three_class_target" / "multi_asset_three_class_target.json")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ps = np.linspace(0.50, 0.75, 100)
    for s in SYMBOLS:
        frame = pd.read_csv(ROOT / "data" / "features" / f"{s.lower()}_4h.csv", usecols=["future_log_return"])
        initial = len(frame) - 3 * (4000 + 4000 + 2 * 24)
        mean_abs = float(frame.future_log_return.iloc[:initial].abs().mean())
        gross = (2 * ps - 1) * mean_abs
        ax.plot(ps, gross * 100, label=f"{s} E|r|={mean_abs:.3%}")
        pstar = 0.5 + COST / (2 * mean_abs)
        hit = float(np.mean([tc["symbols"][s]["arms"]["T"]["seeds"][str(seed)]["hit_rate"] for seed in (42,43,44)]))
        measured_gross = float(np.mean([tc["symbols"][s]["arms"]["T"]["seeds"][str(seed)]["mean_net_return"] + COST for seed in (42,43,44)]))
        ax.scatter([hit], [measured_gross * 100], s=45)
        rows[s] = {"train_mean_abs_return": mean_abs, "breakeven_hit_rate": pstar, "measured_T_hit_rate": hit, "measured_T_gross_return": measured_gross}
    ax.axhline(0, color="black", ls="--", lw=1, label="0")
    ax.axhline(COST * 100, color="gray", ls=":", lw=1.2, label="0.2% 手續費")
    ax.set_xlabel("命中率 p")
    ax.set_ylabel("每筆預期毛利 %")
    ax.set_title("F10 手續費損益兩平分析")
    ax.legend(fontsize=8, ncol=2)
    data["F10"] = {"source": "data/features/*_4h.csv and logs/three_class_target/multi_asset_three_class_target.json", "formula": "gross=(2p-1)*E|r|; p*=0.5+c/(2E|r|)", "cost": COST, "rows": rows}
    savefig("F10_fee_breakeven.png")


def main() -> int:
    data: dict = {}
    f1_flow(data)
    f2_walk_forward(data)
    f3_model_auc(data)
    f4_horizon_heatmap(data)
    f5_persistence(data)
    f6_val_test_scatter(data)
    f7_retrain_delta(data)
    f8_decay(data)
    f9_three_class_curve(data)
    f10_fee_breakeven(data)
    (OUT / "figure_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
