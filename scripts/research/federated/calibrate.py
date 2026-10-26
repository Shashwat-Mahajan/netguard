"""
scripts/research/federated/calibrate.py
Post-hoc calibration of the federated ensembles. No retraining.

Adds one bias per class to the summed tree scores (margins), chosen to
maximise macro-F1 on the VALIDATION set. The test set is scored once at
the end, so there is no leakage.

  python scripts/research/federated/calibrate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402
from common import FED_DIR, MODELS_DIR, load_test  # noqa: E402

from config.settings import settings  # noqa: E402
from processor.feature_schema import FlowFeatures  # noqa: E402

NAMES = [FlowFeatures.LABEL_NAMES[i] for i in range(5)]
MODELS = ("iid", "noniid", "iid_noweights")
GRID = np.arange(-8.0, 8.01, 0.5)
PASSES = 3


def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.bincount(y * 5 + pred, minlength=25).reshape(5, 5)


def macro_f1(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    p = tp / np.maximum(cm.sum(axis=0), 1)
    r = tp / np.maximum(cm.sum(axis=1), 1)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return float(f.mean())


def per_class(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    return tp / np.maximum(cm.sum(axis=0), 1), tp / np.maximum(cm.sum(axis=1), 1)


def fit_bias(margins: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coordinate ascent on per-class biases (Benign stays at 0)."""
    bias = np.zeros(5)
    best = macro_f1(confusion(y, np.argmax(margins, axis=1)))
    for _ in range(PASSES):
        for c in range(1, 5):
            for g in GRID:
                trial = bias.copy()
                trial[c] = g
                pred = np.argmax(margins + trial, axis=1)
                f = macro_f1(confusion(y, pred))
                if f > best:
                    best, bias = f, trial
    return bias


def main() -> int:
    d = settings.data_processed_dir
    X_val = np.load(d / "X_val_cicids.npy")
    y_val = np.load(d / "y_val_cicids.npy").astype(np.int64)
    X_test, y_test = load_test()
    y_test = y_test.astype(np.int64)
    dval, dtest = xgb.DMatrix(X_val), xgb.DMatrix(X_test)

    results: dict[str, dict] = {}
    for name in MODELS:
        path = MODELS_DIR / f"{name}.json"
        if not path.is_file():
            print(f"skipping {name}: not found")
            continue
        bst = xgb.Booster()
        bst.load_model(str(path))
        m_val = bst.predict(dval, output_margin=True)
        m_test = bst.predict(dtest, output_margin=True)

        bias = fit_bias(m_val, y_val)
        before = confusion(y_test, np.argmax(m_test, axis=1))
        after = confusion(y_test, np.argmax(m_test + bias, axis=1))
        pb, rb = per_class(before)
        pa, ra = per_class(after)
        results[name] = {
            "bias": bias.tolist(),
            "macro_f1_before": macro_f1(before),
            "macro_f1_after": macro_f1(after),
            "precision_after": pa.tolist(),
            "recall_after": ra.tolist(),
        }

        print(f"\n=== {name} ===")
        print("bias per class:", {n: round(float(b), 2) for n, b in zip(NAMES, bias)})
        print(f"test macro-F1: {macro_f1(before):.4f} -> {macro_f1(after):.4f}")
        print(f"{'':<12}" + "".join(f"{n:>9}" for n in NAMES))
        print(f"{'prec before':<12}" + "".join(f"{v:>9.3f}" for v in pb))
        print(f"{'prec after':<12}" + "".join(f"{v:>9.3f}" for v in pa))
        print(f"{'rec before':<12}" + "".join(f"{v:>9.3f}" for v in rb))
        print(f"{'rec after':<12}" + "".join(f"{v:>9.3f}" for v in ra))

    out = FED_DIR / "calibration.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
