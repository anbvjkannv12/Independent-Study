"""Independently audit the saved multi-horizon experiment without retraining."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_multi_horizon_comparison import persistence_scores

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("BTC", "ETH", "SOL", "XRP")
HORIZONS = (1, 4, 12, 24)
SEEDS = (42, 43, 44)


def independent_holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    result, running = {}, 0.0
    for i, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p_values[key]))
        result[key] = running
    return result


def verify(root: Path = ROOT) -> dict:
    output = root / "logs/multi_horizon"
    report = json.loads((output / "multi_asset_multi_horizon.json").read_text(encoding="utf-8"))["symbols"]
    markdown = (root / "docs/experiments/multi_horizon/multi_horizon_comparison_results.md").read_text(encoding="utf-8")
    errors: dict[str, list[str]] = {f"V{i}": [] for i in range(1, 9)}
    gaps: dict[str, dict[str, dict[str, int]]] = {}

    def check(number: int, condition: bool, detail: str) -> None:
        if not condition:
            errors[f"V{number}"].append(detail)

    completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                               cwd=root, text=True, capture_output=True, encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    check(1, completed.returncode == 0, f"unittest returncode={completed.returncode}: {completed.stderr[-2500:]}")
    signal_p = {}
    for symbol in SYMBOLS:
        sr = report[symbol]
        processed = pd.read_csv(root / f"data/processed/{symbol.lower()}_5y.csv", usecols=["open_time", "close"])
        processed.open_time = pd.to_datetime(processed.open_time, utc=True)
        closes = processed.set_index("open_time").close
        feature = {}
        aligned = None
        predictions = {}
        for h in HORIZONS:
            frame = pd.read_csv(root / f"data/features/{symbol.lower()}_{h}h.csv",
                                usecols=["open_time", "future_log_return", "target_up", "log_return"])
            frame.open_time = pd.to_datetime(frame.open_time, utc=True)
            feature[h] = frame
        common = set.intersection(*(set(f.open_time) for f in feature.values()))
        aligned_frames = {h: f[f.open_time.isin(common)].reset_index(drop=True) for h, f in feature.items()}
        aligned = aligned_frames[1].open_time
        for h in HORIZONS:
            ar = sr["horizons"][str(h)]
            frame = feature[h]
            check(2, aligned_frames[h].open_time.equals(aligned), f"{symbol} {h}h: aligned timestamps differ")
            positions = pd.Series(np.arange(len(aligned)), index=aligned)
            preds = {}
            for seed in SEEDS:
                path = output / symbol.lower() / f"{h}h_seed{seed}_predictions.csv"
                p = pd.read_csv(path)
                p.open_time = pd.to_datetime(p.open_time, utc=True)
                preds[seed] = p
                counts = p.fold.value_counts().to_dict()
                check(2, len(p) == 12000 and counts == {0: 4000, 1: 4000, 2: 4000},
                      f"{symbol} {h}h seed{seed}: count={len(p)}, folds={counts}")
                if predictions:
                    reference = predictions[1][42][["fold", "open_time"]]
                    check(2, p[["fold", "open_time"]].equals(reference), f"{symbol} {h}h seed{seed}: pairing mismatch")
                elif seed != 42:
                    check(2, p[["fold", "open_time"]].equals(preds[42][["fold", "open_time"]]),
                          f"{symbol} {h}h seed{seed}: pairing mismatch")
                auc = roc_auc_score(p.target_up, p.test_raw)
                expected = ar["pooled_auc_by_seed"][str(seed)]
                check(6, abs(auc - expected) <= 1e-12,
                      f"{symbol} {h}h seed{seed}: AUC {auc} != {expected}")
                # Timestamp-based label reconstruction includes dropped feature rows.
                current = closes.reindex(p.open_time).to_numpy()
                future = closes.reindex(pd.DatetimeIndex(p.open_time) + pd.Timedelta(hours=h)).to_numpy()
                recalculated = np.log(future / current)
                check(5, np.isfinite(recalculated).all(), f"{symbol} {h}h seed{seed}: missing processed close")
                differences = int(np.count_nonzero((recalculated > 0) != p.target_up.to_numpy()))
                check(5, differences == 0, f"{symbol} {h}h seed{seed}: {differences} target mismatches")
                original = frame.set_index("open_time").future_log_return.reindex(p.open_time).to_numpy()
                max_error = float(np.max(np.abs(recalculated - original)))
                check(5, max_error <= 1e-9, f"{symbol} {h}h seed{seed}: max return error {max_error}")
            predictions[h] = preds
            mean = float(np.mean([roc_auc_score(preds[s].target_up, preds[s].test_raw) for s in SEEDS]))
            check(6, abs(mean - ar["mean_pooled_auc"]) <= 1e-12,
                  f"{symbol} {h}h: mean AUC {mean} != {ar['mean_pooled_auc']}")
            fold_aucs = []
            for i, fold in enumerate(ar["folds"]):
                subset = preds[42][preds[42].fold == i]
                start, end = subset.open_time.iloc[0], subset.open_time.iloc[-1]
                check(3, start == pd.Timestamp(fold["test"]["start"]) and end == pd.Timestamp(fold["test"]["end"]),
                      f"{symbol} {h}h fold{i}: CSV {start},{end} != JSON {fold['test']}")
                boundaries = {part: tuple(int(positions[pd.Timestamp(fold[part][boundary])])
                                          for boundary in ("start", "end"))
                              for part in ("train", "validation", "test")}
                left = boundaries["validation"][0] - boundaries["train"][1] - 1
                right = boundaries["test"][0] - boundaries["validation"][1] - 1
                check(4, left == 24 and right == 24,
                      f"{symbol} {h}h fold{i}: purge train/validation={left}, validation/test={right}")
                fold_mean = float(np.mean([roc_auc_score(preds[s].query('fold == @i').target_up,
                                                             preds[s].query('fold == @i').test_raw) for s in SEEDS]))
                fold_aucs.append(f"{fold_mean:.6f}")
            # Only the signal table (before the comparison section) is used here.
            signal_table = markdown.split("## 相對 4h")[0]
            pattern = rf"^\| {symbol} \| {h}h \| [^|]+ \| ([^|]+) \|"
            match = re.search(pattern, signal_table, re.MULTILINE)
            found = [] if match is None else [x.strip() for x in match.group(1).split(",")]
            check(6, found == fold_aucs, f"{symbol} {h}h: report folds={found} != {fold_aucs}")
            full_scores = persistence_scores(frame, h)
            rng = np.random.default_rng(20260928)
            test_indices = pd.Index(frame.open_time).get_indexer(preds[42].open_time)
            check(7, (test_indices >= 0).all(), f"{symbol} {h}h: missing test timestamps")
            for t in rng.choice(test_indices, size=200, replace=False):
                short = persistence_scores(frame.iloc[:int(t) + 1], h)[-1]
                check(7, short == full_scores[t], f"{symbol} {h}h row{t}: truncated={short}, full={full_scores[t]}")
            gap_flags = frame.open_time.diff().gt(pd.Timedelta(hours=1)).to_numpy()
            # A rolling window crossing any missing feature timestamp.
            affected = pd.Series(gap_flags).rolling(h, min_periods=1).max().fillna(0).to_numpy(dtype=bool)
            gaps[f"{symbol}-{h}h"] = {}
            for i in range(3):
                part = preds[42].query("fold == @i")
                idx = pd.Index(frame.open_time).get_indexer(part.open_time)
                auc = roc_auc_score(part.target_up, full_scores[idx])
                expected = ar["baselines"][i]["persistence"]["test"]["roc_auc"]
                check(7, abs(auc - expected) <= 1e-12,
                      f"{symbol} {h}h fold{i}: persistence AUC {auc} != {expected}")
                gaps[f"{symbol}-{h}h"][str(i)] = int(affected[idx].sum())
            signal_p[f"{symbol}-{h}h"] = ar["signal_bootstrap"]["raw_one_sided_p_value"]
    adjusted = independent_holm(signal_p)
    for symbol in SYMBOLS:
        comparisons = report[symbol]["comparisons"]
        local = independent_holm({h: v["bootstrap"]["raw_one_sided_p_value"] for h, v in comparisons.items()})
        for h in HORIZONS:
            item = report[symbol]["horizons"][str(h)]
            key = f"{symbol}-{h}h"
            check(8, abs(adjusted[key] - item["signal_bootstrap"]["adjusted_p_value"]) <= 1e-12,
                  f"{key}: Holm {adjusted[key]} != {item['signal_bootstrap']['adjusted_p_value']}")
            check(8, item["signal_passes"] == (adjusted[key] <= 0.05), f"{key}: signal_passes mismatch")
        for h, item in comparisons.items():
            check(8, abs(local[h] - item["bootstrap"]["adjusted_p_value"]) <= 1e-12,
                  f"{symbol} {h}h: Holm {local[h]} != {item['bootstrap']['adjusted_p_value']}")
            criteria = item["criteria"]
            passes = (all(x > 0 for x in item["fold_auc_deltas"]) and local[h] <= 0.05 and
                      item["mean_pooled_auc_delta"] > item["epsilon"] and
                      report[symbol]["horizons"][h]["signal_passes"])
            check(8, (item["decision"] == "較 4h 穩定地強") == passes,
                  f"{symbol} {h}h: decision={item['decision']} != {'pass' if passes else 'not pass'}")
    return {"checks": {key: {"status": "FAIL" if value else "PASS", "errors": value} for key, value in errors.items()},
            "persistence_gap_affected_test_rows": gaps,
            "notes": ["With 2000 resamples, minimum one-sided p is 1/2001; for 16 Holm tests it is 16/2001.",
                      "Five cells hit this floor: BTC 1h/4h, ETH 1h/12h, XRP 1h."]}


def main() -> int:
    result = verify()
    for key, value in result["checks"].items():
        print(f"{key}: {value['status']}")
        for detail in value["errors"][:20]:
            print("  ", detail)
    path = ROOT / "logs/multi_horizon/verification.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return int(any(value["status"] == "FAIL" for value in result["checks"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
