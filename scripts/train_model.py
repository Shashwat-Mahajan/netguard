"""
scripts/train_model.py
Phase 2 — Train XGBoost classifier + Isolation Forest + SHAP explainer.

Usage:
    python3 scripts/train_model.py --dataset nslkdd
    python3 scripts/train_model.py --dataset cicids
    python3 scripts/train_model.py --dataset nslkdd --skip-shap
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from config.logger import configure_logging, get_logger
from config.settings import settings
from models.xgboost_classifier import IntrusionClassifier
from models.anomaly_detector import AnomalyDetector
from models.explainer import SHAPExplainer
from processor.feature_schema import FlowFeatures

configure_logging()
log = get_logger("train")


def banner(text: str) -> None:
    print(f"\n{'─'*52}")
    print(f"  {text}")
    print(f"{'─'*52}")


def load_arrays(dataset: str) -> tuple[np.ndarray, ...]:
    """Load preprocessed numpy arrays from Phase 1."""
    out = settings.data_processed_dir
    suffix = dataset

    files = {
        "X_train": out / f"X_train_{suffix}.npy",
        "X_val":   out / f"X_val_{suffix}.npy",
        "X_test":  out / f"X_test_{suffix}.npy",
        "y_train": out / f"y_train_{suffix}.npy",
        "y_val":   out / f"y_val_{suffix}.npy",
        "y_test":  out / f"y_test_{suffix}.npy",
    }

    print("\nStep 1 — Loading Phase 1 artifacts")
    for name, path in files.items():
        if not path.exists():
            print(f"  ✗  {name} not found at {path}")
            print(f"     Run: python3 scripts/ingest_data.py --dataset {dataset}")
            sys.exit(1)
        print(f"  ✓  {name}: {np.load(path).shape}")

    return (
        np.load(files["X_train"]),
        np.load(files["X_val"]),
        np.load(files["X_test"]),
        np.load(files["y_train"]),
        np.load(files["y_val"]),
        np.load(files["y_test"]),
    )


def train_xgboost(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    dataset: str,
) -> IntrusionClassifier:
    banner("Step 2 — XGBoost classifier training")
    print(f"\n  Training on {len(X_train):,} samples with {X_train.shape[1]} features")
    print("  Early stopping on validation loss (patience=20 rounds)")
    print("  This takes ~1-3 minutes...\n")

    clf = IntrusionClassifier()
    t0 = time.time()
    val_metrics = clf.train(X_train, y_train, X_val, y_val)
    duration = time.time() - t0

    print(f"\n  ✓  Training complete in {duration:.1f}s")
    print(f"  ✓  Val Macro F1: {val_metrics['macro_f1']:.4f}")

    # Full test set evaluation
    banner("Step 3 — Test set evaluation")
    clf.evaluate(X_test, y_test, split="test")

    # Feature importance
    banner("Step 4 — Top 10 most important features")
    imps = clf.feature_importances
    sorted_imps = sorted(imps.items(), key=lambda x: x[1], reverse=True)[:10]
    print()
    for rank, (feat, score) in enumerate(sorted_imps, 1):
        bar = "█" * int(score * 300)
        print(f"  {rank:>2}. {feat:<35} {score:.4f}  {bar}")

    # Save model
    model_path = settings.data_processed_dir / f"xgboost_{dataset}.joblib"
    clf.save(model_path)
    # Also save as default
    clf.save(settings.data_processed_dir / "xgboost_model.joblib")

    return clf


def train_anomaly_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset: str,
) -> AnomalyDetector:
    banner("Step 5 — Isolation Forest (zero-day detector)")

    # Train ONLY on benign traffic
    benign_mask = y_train == 0
    X_benign = X_train[benign_mask]
    print(f"\n  Training on {len(X_benign):,} benign-only samples")
    print("  (Model learns what normal traffic looks like)")
    print("  This takes ~30 seconds...\n")

    det = AnomalyDetector()
    t0 = time.time()
    det.train(X_benign)
    print(f"  ✓  Training complete in {time.time()-t0:.1f}s")

    # Evaluate on test set
    det.evaluate(X_test, y_test)

    # Save
    det_path = settings.data_processed_dir / f"isolation_forest_{dataset}.joblib"
    det.save(det_path)
    det.save(settings.data_processed_dir / "isolation_forest.joblib")

    return det


def build_shap_explainer(
    clf: IntrusionClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset: str,
) -> None:
    banner("Step 6 — SHAP explainability")
    print("\n  Building SHAP TreeExplainer...")

    explainer = SHAPExplainer(clf)
    explainer.build()

    # Explain a sample of test flows (SHAP is slow on large batches)
    sample_size = min(500, len(X_test))
    idx = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[idx]

    print(f"  Computing SHAP values for {sample_size} test samples...")
    shap_values = explainer.explain_batch(X_sample)

    # Save SHAP summary plots for all 5 classes
    plots_dir = settings.data_processed_dir / "shap_plots"
    plots_dir.mkdir(exist_ok=True)

    for class_idx in range(5):
        class_name = FlowFeatures.LABEL_NAMES[class_idx]
        plot_path = plots_dir / f"shap_{class_name.lower()}_{dataset}.png"
        explainer.plot_summary(shap_values, X_sample, class_idx, save_path=plot_path)

    # Demo: explain one DoS sample
    attack_mask = y_test != 0
    if attack_mask.any():
        attack_idx = np.where(attack_mask)[0][0]
        pred_label, pred_int, confidence, _ = clf.predict_single(X_test[attack_idx])
        explanation = explainer.explain_single(X_test[attack_idx], pred_int)

        print(f"\n  Example explanation (actual label: {FlowFeatures.LABEL_NAMES[y_test[attack_idx]]}):")
        print(f"  {explainer.format_explanation(explanation)}")

    print(f"\n  ✓  SHAP plots saved → {plots_dir}/")


def final_summary(dataset: str) -> None:
    banner("Phase 2 complete")
    out = settings.data_processed_dir

    print(f"\n  Artifacts in {out}:")
    model_files = [
        f for f in sorted(out.iterdir())
        if f.suffix in (".joblib", ".png") or "shap" in f.name
    ]
    for f in model_files:
        size = f.stat().st_size / 1e6
        print(f"    {f.name:<45} {size:.2f} MB")

    print("""
  What you now have:
    ✓  XGBoost classifier  — classifies flows into 5 attack types
    ✓  Isolation Forest    — detects zero-day / unknown attacks
    ✓  SHAP explainer      — explains every prediction
    ✓  SHAP plots          — ready for research paper figures

  Next: Phase 3 — Real-time streaming pipeline (Redpanda + FastAPI)
    python3 scripts/train_model.py is complete.
    Start the API: uvicorn api.main:app --reload
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="NetGuard Phase 2 — model training")
    parser.add_argument("--dataset", choices=["nslkdd", "cicids"], default="nslkdd")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP (faster run, no explainability plots)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║   NetGuard — Phase 2 Model Training                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Dataset:    {args.dataset}")
    print(f"  SHAP:       {'disabled' if args.skip_shap else 'enabled'}")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_arrays(args.dataset)

    # Train XGBoost
    clf = train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test, args.dataset)

    # Train Isolation Forest
    train_anomaly_detector(X_train, y_train, X_test, y_test, args.dataset)

    # SHAP explainability
    if not args.skip_shap:
        try:
            build_shap_explainer(clf, X_test, y_test, args.dataset)
        except ImportError:
            print("\n  ⚠  SHAP not installed. Run: pip install shap matplotlib")
            print("     Then re-run with SHAP enabled.")
    else:
        print("\n  ⚠  SHAP skipped (--skip-shap flag)")

    final_summary(args.dataset)


if __name__ == "__main__":
    main()
