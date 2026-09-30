"""Train the 5-year BTC 4h Transformer, save its weights, export a PyTorch Lite (.ptl) model and verify it.

Uses the same data, features, model and settings as the 5-year 4h model comparison
(scripts/train_btc_4h_model_comparison.py), trained once on all rows up to 2026-08-09 06:00 UTC.
The .ptl takes RAW features of shape (batch, 24, 53) and returns the up-probability for
close[t + 4] > close[t]; scaling and sigmoid are baked in. See docs/handover/ptl_guide.md.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.jit.mobile import _load_for_lite_interpreter
from torch.utils.mobile_optimizer import optimize_for_mobile

from btc_4h_model_adapters import ModelSettings, TransformerClassifier, _fit_predict_sequence_model, _set_seed
from train_btc_4h_baseline import load_baseline_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "features" / "btc_4h.csv"
MANIFEST = ROOT / "logs" / "feature_manifest.json"
OUT_DIR = ROOT / "models" / "btc_4h_transformer"
PTH = OUT_DIR / "btc_4h_transformer.pth"
PTL = OUT_DIR / "btc_4h_transformer.ptl"
SPEC = OUT_DIR / "btc_4h_transformer.ptl.json"
VERIFY_LOG = ROOT / "logs" / "ptl_btc_4h_verify.json"
CUTOFF = pd.Timestamp("2026-08-09 06:00", tz="UTC")
VALIDATION_ROWS = 4000
TOLERANCE = 1e-5


class ProbabilityModel(torch.nn.Module):
    """Raw features in, up-probability out: (x - mean) / scale -> model -> sigmoid."""

    def __init__(self, model: torch.nn.Module, mean: np.ndarray, scale: np.ndarray) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, raw_sequences: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model((raw_sequences - self.mean) / self.scale))


def raw_windows(history: pd.DataFrame, features: tuple[str, ...], sequence_length: int) -> np.ndarray:
    """All windows of `sequence_length` rows ending at each row from index sequence_length - 1."""
    values = history.loc[:, features].to_numpy(dtype=np.float32)
    return np.stack([values[end - sequence_length + 1: end + 1] for end in range(sequence_length - 1, len(values))])


def save_lite(module: torch.jit.ScriptModule, destination: Path) -> str:
    """Save for the lite interpreter via an ASCII temp dir (it cannot open non-ASCII paths)."""
    try:
        module = optimize_for_mobile(module)
        optimization = "optimize_for_mobile"
    except RuntimeError as exc:
        # Some Windows torch builds lack XNNPACK; the unoptimized module still loads in the lite interpreter.
        if "XNNPACK" not in str(exc):
            raise
        optimization = "none - optimize_for_mobile skipped because XNNPACK is unavailable"
    with tempfile.TemporaryDirectory(dir=Path.home()) as tmp:
        tmp_path = Path(tmp) / "model.ptl"
        module._save_for_lite_interpreter(str(tmp_path))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_path, destination)
    return optimization


def load_lite(path: Path):
    with tempfile.TemporaryDirectory(dir=Path.home()) as tmp:
        tmp_path = Path(tmp) / "model.ptl"
        shutil.copyfile(path, tmp_path)
        return _load_for_lite_interpreter(str(tmp_path))


def main() -> None:
    dataset, features = load_baseline_dataset(DATA, MANIFEST)
    if len(features) != 53 or dataset.open_time.iloc[-1] != CUTOFF:
        raise ValueError("expected the frozen 53-feature BTC 4h dataset ending 2026-08-09 06:00 UTC")
    train, validation = dataset.iloc[:-VALIDATION_ROWS], dataset.iloc[-VALIDATION_ROWS:]
    settings = ModelSettings()  # identical to logs/btc_4h_model_comparison.json model_settings

    _set_seed(settings.seed)
    model = TransformerClassifier(len(features), settings.hidden_size, settings.num_heads, settings.sequence_length)
    validation_probability, _ = _fit_predict_sequence_model(model, train, validation, validation, features, settings)
    model.eval()
    scaler = StandardScaler().fit(train.loc[:, features])  # same deterministic fit as inside training
    validation_auc = float(roc_auc_score(validation.target_up, validation_probability))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage = {"train_start": str(train.open_time.iloc[0]), "train_end": str(train.open_time.iloc[-1]),
                "validation_start": str(validation.open_time.iloc[0]), "validation_end": str(validation.open_time.iloc[-1])}
    torch.save({
        "format_version": 1,
        "model_class": "scripts/btc_4h_model_adapters.py:TransformerClassifier",
        "model_config": {"input_size": len(features), "hidden_size": settings.hidden_size,
                         "num_heads": settings.num_heads, "sequence_length": settings.sequence_length},
        "state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_columns": list(features),
        "training_coverage": coverage,
        "validation_auc": validation_auc,
        "settings": vars(settings),
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
    }, PTH)

    wrapper = ProbabilityModel(model, scaler.mean_, scaler.scale_).eval()
    torch.backends.mha.set_fastpath_enabled(False)  # fused kernels may be unsupported by the mobile runtime
    example = torch.zeros(1, settings.sequence_length, len(features), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
    optimization = save_lite(traced, PTL)

    SPEC.write_text(json.dumps({
        "model_file": PTL.name,
        "source_checkpoint": PTH.name,
        "source_checkpoint_sha256": hashlib.sha256(PTH.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "mobile_optimization": optimization,
        "input": {"shape": ["batch", settings.sequence_length, len(features)], "dtype": "float32",
                  "scaling": "none - pass RAW feature values; scaling is inside the model",
                  "row_order": "oldest to newest; the last row is the most recent closed 1h candle"},
        "output": {"shape": ["batch"], "meaning": "uncalibrated probability that close[t + 4] > close[t]"},
        "display_threshold": 0.5,
        "feature_columns": list(features),
        "target_definition": "close[t + 4] > close[t]",
        "training_coverage": coverage,
        "validation_auc": validation_auc,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Verify: the .ptl on raw windows must reproduce the training-time validation probabilities.
    lite = load_lite(PTL)
    history = pd.concat([train.tail(settings.sequence_length - 1), validation], ignore_index=True)
    windows = torch.from_numpy(raw_windows(history, features, settings.sequence_length))
    with torch.no_grad():
        ptl_prob = np.concatenate([lite(windows[i:i + 500]).numpy() for i in range(0, len(windows), 500)])
        single = torch.stack([lite(windows[i:i + 1])[0] for i in range(4)]).numpy()
    max_diff = float(np.max(np.abs(ptl_prob - validation_probability)))
    batch_diff = float(np.max(np.abs(ptl_prob[:4] - single)))
    passed = len(ptl_prob) == len(validation_probability) and max_diff <= TOLERANCE and batch_diff <= TOLERANCE
    VERIFY_LOG.write_text(json.dumps({
        "windows": len(ptl_prob), "max_abs_diff_ptl_vs_training": max_diff, "batch_vs_single": batch_diff,
        "tolerance": TOLERANCE, "validation_auc": validation_auc, "mobile_optimization": optimization,
        "status": "PASS" if passed else "FAIL"}, indent=2), encoding="utf-8")
    print(f"validation_auc={validation_auc:.6f} windows={len(ptl_prob)} max|ptl - training|={max_diff:.2e} "
          f"batch_vs_single={batch_diff:.2e} optimization={optimization}")
    if not passed:
        raise SystemExit("FAIL: .ptl output differs from the trained model")
    print("PASS")


if __name__ == "__main__":
    sys.exit(main())
