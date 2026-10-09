"""
scripts/research/r2_cicids.py
Research Phase R2 -- CICIDS2017 ingestion + cross-dataset evaluation.

Produces:
  1. CICIDS2017 ingestion (39-feature schema, scaled, split)
  2. XGBoost trained on CICIDS2017 -- compare F1 with NSL-KDD results
  3. Cross-dataset test: NSL-KDD model -> CICIDS2017 test set
  4. Cross-dataset test: CICIDS2017 model -> NSL-KDD test set
  5. Generalisation figures
  6. Table 3 LaTeX for the paper

Usage:
    python3 scripts/research/r2_cicids.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from config.settings import settings
from processor.feature_schema import FlowFeatures
from processor.cicids_parquet_loader import CICIDS2017ParquetLoader
from processor.preprocessor import DataPreprocessor

OUT = Path("data/research/r2_cicids")
OUT.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = [FlowFeatures.LABEL_NAMES[i] for i in range(5)]


def banner(text: str) -> None:
    print(f"\n{'='*56}\n  {text}\n{'='*56}")


def ingest_cicids() -> tuple:
    banner("Step 1 -- Ingesting CICIDS2017 (parquet format)")

    loader = CICIDS2017ParquetLoader()
    df = loader.load(use_cache=True)
    print(f"  Loaded: {len(df):,} rows")

    print("\n  Label distribution:")
    counts = df["label_encoded"].value_counts().sort_index()
    total  = len(df)
    for code, cnt in counts.items():
        name = FlowFeatures.LABEL_NAMES.get(int(code), str(code))
        pct  = cnt / total
        bar  = "#" * int(pct * 28)
        print(f"    {name:<10} {cnt:>8,}  ({pct*100:5.1f}%)  {bar}")

    print("\n  Preprocessing (scale + split)...")
    pp = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = pp.fit_transform(df)
    print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")

    out = settings.data_processed_dir
    np.save(out / "X_train_cicids.npy", X_train)
    np.save(out / "X_val_cicids.npy",   X_val)
    np.save(out / "X_test_cicids.npy",  X_test)
    np.save(out / "y_train_cicids.npy", y_train)
    np.save(out / "y_val_cicids.npy",   y_val)
    np.save(out / "y_test_cicids.npy",  y_test)
    pp.save_scaler(out / "scaler_cicids.joblib")
    print(f"  Saved arrays and scaler to {out}/")

    return X_train, X_val, X_test, y_train, y_val, y_test, pp


def _metrics(y_pred, y_test, name: str, train_time: float) -> dict:
    return {
        "model": name,
        "accuracy":     round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1":     round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "per_class_f1": {
            FlowFeatures.LABEL_NAMES[i]: round(float(v), 4)
            for i, v in enumerate(f1_score(y_test, y_pred, average=None, zero_division=0,
                                            labels=list(range(5))))
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=list(range(5))).tolist(),
        "train_time_sec": train_time,
    }


def train_xgboost_cicids(X_train, X_val, X_test, y_train, y_val, y_test) -> tuple:
    banner("Step 2 -- Training XGBoost on CICIDS2017")
    from models.xgboost_classifier import IntrusionClassifier

    clf = IntrusionClassifier()
    t0 = time.time()
    clf.train(X_train, y_train, X_val, y_val)
    train_time = round(time.time() - t0, 1)

    y_pred, _ = clf.predict_batch(X_test)
    result = _metrics(y_pred, y_test, "XGBoost-CICIDS2017", train_time)

    print(f"  Macro F1: {result['macro_f1']:.4f}   Train time: {train_time}s")

    clf.save(settings.data_processed_dir / "xgboost_cicids.joblib")
    print("  Model saved -> data/processed/xgboost_cicids.joblib")

    return result, clf


def cross_dataset_eval(cicids_clf) -> list[dict]:
    banner("Step 3 -- Cross-dataset generalisation test")
    from models.xgboost_classifier import IntrusionClassifier

    results = []
    out = settings.data_processed_dir

    print("\n  [A] NSL-KDD-trained model -> CICIDS2017 test set")
    nslkdd_clf = IntrusionClassifier()
    nslkdd_model_path = out / "xgboost_model.joblib"
    if nslkdd_model_path.exists():
        nslkdd_clf.load(nslkdd_model_path)
        X_test_cicids = np.load(out / "X_test_cicids.npy")
        y_test_cicids = np.load(out / "y_test_cicids.npy")
        try:
            y_pred, _ = nslkdd_clf.predict_batch(X_test_cicids)
            r = _metrics(y_pred, y_test_cicids, "NSL-KDD -> CICIDS2017", 0.0)
            results.append(r)
            print(f"      Macro F1: {r['macro_f1']:.4f}  (cross-dataset generalisation)")
        except Exception as e:
            print(f"      Skipped: {e}")
    else:
        print("      Skipped -- NSL-KDD model not found")

    print("\n  [B] CICIDS2017-trained model -> NSL-KDD test set")
    X_test_nslkdd = np.load(out / "X_test_nslkdd.npy")
    y_test_nslkdd = np.load(out / "y_test_nslkdd.npy")
    try:
        y_pred, _ = cicids_clf.predict_batch(X_test_nslkdd)
        r = _metrics(y_pred, y_test_nslkdd, "CICIDS2017 -> NSL-KDD", 0.0)
        results.append(r)
        print(f"      Macro F1: {r['macro_f1']:.4f}  (cross-dataset generalisation)")
    except Exception as e:
        print(f"      Skipped: {e}")

    return results


def plot_dataset_comparison(nslkdd_f1: float, cicids_result: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    datasets = ["NSL-KDD\n(Phase 2)", "CICIDS2017\n(R2)"]
    scores   = [nslkdd_f1, cicids_result["macro_f1"]]
    colors   = ["#2563EB", "#16A34A"]

    bars = ax.bar(datasets, scores, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Macro F1 Score", fontsize=11)
    ax.set_title("XGBoost Performance -- NSL-KDD vs CICIDS2017", fontsize=12)
    ax.set_ylim(0, 1.05)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.4f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = OUT / "fig7_dataset_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 7 saved -> {p}")


def plot_cross_dataset(cross_results: list[dict], nslkdd_native: float, cicids_native: float) -> None:
    if not cross_results:
        return
    fig, ax = plt.subplots(figsize=(9, 5))

    labels = ["NSL-KDD\n(native)"] + [r["model"] for r in cross_results] + ["CICIDS2017\n(native)"]
    values = [nslkdd_native] + [r["macro_f1"] for r in cross_results] + [cicids_native]
    colors = ["#16A34A"] + ["#DC2626"] * len(cross_results) + ["#16A34A"]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.55)
    ax.set_ylabel("Macro F1 Score", fontsize=11)
    ax.set_title("Cross-Dataset Generalisation -- Native vs Transfer", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.4f}",
                ha="center", fontsize=10, fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = OUT / "fig8_cross_dataset.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 8 saved -> {p}")


def latex_table3(cicids_result: dict, cross_results: list[dict], nslkdd_f1: float) -> None:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Cross-dataset generalisation results}",
        r"\label{tab:cross_dataset}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"\textbf{Train} $\rightarrow$ \textbf{Test} & \textbf{Macro F1} & \textbf{Accuracy} \\",
        r"\hline",
        f"NSL-KDD $\\rightarrow$ NSL-KDD (native) & {nslkdd_f1:.4f} & -- \\\\",
        f"CICIDS2017 $\\rightarrow$ CICIDS2017 (native) & {cicids_result['macro_f1']:.4f} & {cicids_result['accuracy']:.4f} \\\\",
    ]
    for r in cross_results:
        lines.append(f"{r['model']} & {r['macro_f1']:.4f} & {r['accuracy']:.4f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    tex = "\n".join(lines)
    p = OUT / "table3_latex.tex"
    p.write_text(tex)
    print(f"\n  LaTeX Table 3 saved -> {p}")
    print("  " + "-"*50)
    print(tex)


def save_json(cicids_result: dict, cross_results: list[dict]) -> None:
    data = {"cicids2017_native": cicids_result, "cross_dataset": cross_results}
    p = OUT / "results.json"
    p.write_text(json.dumps(data, indent=2))
    print(f"  Results JSON saved -> {p}")


def main() -> None:
    print("======================================================")
    print("   NetGuard Research -- R2: CICIDS2017 + Cross-Dataset")
    print("======================================================")
    print(f"  Output: {OUT}")

    X_train, X_val, X_test, y_train, y_val, y_test, pp = ingest_cicids()
    cicids_result, cicids_clf = train_xgboost_cicids(X_train, X_val, X_test, y_train, y_val, y_test)

    nslkdd_f1 = 0.9717
    r1_path = Path("data/research/r1_baselines/results.json")
    if r1_path.exists():
        r1_data = json.loads(r1_path.read_text())
        for r in r1_data:
            if r["model"] == "XGBoost":
                nslkdd_f1 = r["macro_f1"]

    cross_results = cross_dataset_eval(cicids_clf)

    banner("Generating figures")
    plot_dataset_comparison(nslkdd_f1, cicids_result)
    plot_cross_dataset(cross_results, nslkdd_f1, cicids_result["macro_f1"])

    banner("Saving artifacts")
    save_json(cicids_result, cross_results)
    latex_table3(cicids_result, cross_results, nslkdd_f1)

    banner("R2 COMPLETE")
    print(f"""
  Summary:
    NSL-KDD (native):     Macro F1 = {nslkdd_f1:.4f}
    CICIDS2017 (native):  Macro F1 = {cicids_result['macro_f1']:.4f}
""")
    for r in cross_results:
        print(f"    {r['model']:<28} Macro F1 = {r['macro_f1']:.4f}")

    print(f"""
  Files in {OUT}:
    fig7_dataset_comparison.png  -> Figure 7 (paper)
    fig8_cross_dataset.png       -> Figure 8 (paper)
    results.json                 -> all numbers
    table3_latex.tex             -> Table 3, paste into LaTeX

  Section 6.2 of your paper is now complete.
  Next: R3 -- Federated learning
""")


if __name__ == "__main__":
    main()
