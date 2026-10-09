"""
scripts/research/r1_baselines.py
Research Phase R1 — Baseline model comparisons.

Trains 4 baseline models on the same data as XGBoost and produces:
  1. Comparison table (Table 2 in paper)
  2. ROC curves figure (Figure 3 in paper)
  3. Per-class F1 heatmap (Figure 4 in paper)
  4. Training time vs F1 scatter (Figure 5 in paper)
  5. Confusion matrices for all models (Figure 6 in paper)
  6. LaTeX table ready to paste into paper

All figures saved to: data/research/r1_baselines/

Usage:
    python3 scripts/research/r1_baselines.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import StratifiedShuffleSplit

from config.settings import settings
from processor.feature_schema import FlowFeatures

OUT = Path("data/research/r1_baselines")
OUT.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = [FlowFeatures.LABEL_NAMES[i] for i in range(5)]

COLORS = {
    "XGBoost":       "#2563EB",
    "Random Forest": "#16A34A",
    "Decision Tree": "#CA8A04",
    "MLP":           "#9333EA",
    "SVM":           "#DC2626",
}


def banner(text):
    print(f"\n{'='*56}\n  {text}\n{'='*56}")


def load_data():
    banner("Loading preprocessed arrays")
    p = settings.data_processed_dir
    X_train = np.load(p / "X_train_nslkdd.npy")
    X_val   = np.load(p / "X_val_nslkdd.npy")
    X_test  = np.load(p / "X_test_nslkdd.npy")
    y_train = np.load(p / "y_train_nslkdd.npy")
    y_val   = np.load(p / "y_val_nslkdd.npy")
    y_test  = np.load(p / "y_test_nslkdd.npy")
    X_tr = np.vstack([X_train, X_val])
    y_tr = np.concatenate([y_train, y_val])
    print(f"  Training: {X_tr.shape}   Test: {X_test.shape}")
    return X_tr, X_test, y_tr, y_test


def metrics(y_pred, y_test, name, train_time, model_obj=None):
    return {
        "model":           name,
        "accuracy":        round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1":        round(float(f1_score(y_test, y_pred, average="macro",    zero_division=0)), 4),
        "weighted_f1":     round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall":    round(float(recall_score(y_test, y_pred, average="macro",    zero_division=0)), 4),
        "per_class_f1": {
            FlowFeatures.LABEL_NAMES[i]: round(float(v), 4)
            for i, v in enumerate(f1_score(y_test, y_pred, average=None, zero_division=0))
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_time_sec":  train_time,
        "y_pred":          y_pred.tolist(),
        "model_obj":       model_obj,
    }


def load_xgboost(X_test, y_test):
    banner("Loading XGBoost (Phase 2 trained model)")
    from models.xgboost_classifier import IntrusionClassifier
    clf = IntrusionClassifier()
    clf.load(settings.data_processed_dir / "xgboost_model.joblib")
    y_pred, _ = clf.predict_batch(X_test)
    r = metrics(y_pred, y_test, "XGBoost", 29.1, model_obj=clf._model)
    print(f"  XGBoost  Macro F1={r['macro_f1']:.4f}  (pre-trained, 29.1s)")
    return r


def train_baselines(X_tr, X_test, y_tr, y_test):
    results = []

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20, min_samples_split=5,
            class_weight="balanced", random_state=42),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            max_iter=100, early_stopping=True, validation_fraction=0.1,
            random_state=42, verbose=False),
        "SVM": SVC(
            kernel="rbf", C=10, gamma="scale",
            class_weight="balanced", probability=True, random_state=42),
    }

    sss = StratifiedShuffleSplit(n_splits=1, train_size=15000, random_state=42)
    svm_idx = next(sss.split(X_tr, y_tr))[0]

    for name, model in models.items():
        banner(f"Training {name}")
        if name == "SVM":
            print("  Using 15,000 sample (full SVM takes hours)")
            Xf, yf = X_tr[svm_idx], y_tr[svm_idx]
        else:
            Xf, yf = X_tr, y_tr

        print(f"  Fitting on {len(Xf):,} samples...")
        t0 = time.time()
        model.fit(Xf, yf)
        t = round(time.time() - t0, 1)
        y_pred = model.predict(X_test)
        r = metrics(y_pred, y_test, name, t, model_obj=model)
        results.append(r)
        print(f"  Macro F1={r['macro_f1']:.4f}  Accuracy={r['accuracy']:.4f}  Time={t}s")

    return results


def print_table(results):
    banner("TABLE 2 — Model Comparison (paper-ready)")
    hdrs = ["Model","Accuracy","Macro F1","Precision","Recall","Time(s)"]
    w    = [18,10,10,11,10,10]
    print("  " + "".join(f"{h:<{c}}" for h,c in zip(hdrs,w)))
    print("  " + "-"*sum(w))
    for r in sorted(results, key=lambda x: x["macro_f1"], reverse=True):
        tag = " <<" if r["model"]=="XGBoost" else ""
        print(f"  {r['model']:<{w[0]}}{r['accuracy']:<{w[1]}}{r['macro_f1']:<{w[2]}}"
              f"{r['macro_precision']:<{w[3]}}{r['macro_recall']:<{w[4]}}{r['train_time_sec']}{tag}")
    print()
    print("  Per-class F1:")
    print("  " + f"{'Model':<18}" + "".join(f"{n:<10}" for n in LABEL_NAMES))
    print("  " + "-"*(18+10*5))
    for r in sorted(results, key=lambda x: x["macro_f1"], reverse=True):
        row = f"  {r['model']:<18}"
        for n in LABEL_NAMES:
            row += f"{r['per_class_f1'].get(n,0):<10.4f}"
        print(row)


def plot_bar(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("NetGuard — Model Comparison (NSL-KDD)", fontsize=13, fontweight="bold")
    rs = sorted(results, key=lambda x: x["macro_f1"], reverse=True)
    names  = [r["model"] for r in rs]
    colors = [COLORS.get(n,"#888") for n in names]
    for ax, key, title in zip(axes,
        ["macro_f1","accuracy"],
        ["(a) Macro F1","(b) Accuracy"]):
        vals = [r[key] for r in rs]
        bars = ax.barh(names, vals, color=colors, edgecolor="white")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 1.08)
        for bar, v in zip(bars, vals):
            ax.text(v+0.005, bar.get_y()+bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    p = OUT / "fig2_bar_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 2 saved → {p}")


def plot_roc(results, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")

    for ax, class_idx, attack in zip(axes, [1, 2], ["DoS", "Probe"]):
        y_bin = (y_test == class_idx).astype(int)
        ax.plot([0,1],[0,1],"k--",lw=0.8, label="Random (AUC=0.50)")
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title(f"One-vs-Rest: {attack}", fontsize=11)
        ax.set_xlim(0,1); ax.set_ylim(0,1.05)
        ax.grid(True, alpha=0.3)

        for r in results:
            name  = r["model"]
            color = COLORS.get(name,"#888")
            obj   = r.get("model_obj")
            if obj is None:
                continue
            try:
                if name == "XGBoost":
                    probs = obj.predict_proba(X_test)
                else:
                    probs = obj.predict_proba(X_test)
                scores = probs[:, class_idx]
                fpr, tpr, _ = roc_curve(y_bin, scores)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=1.8,
                        label=f"{name} (AUC={roc_auc:.3f})")
            except Exception as e:
                print(f"  ROC skip {name}: {e}")

        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    p = OUT / "fig3_roc_curves.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 3 saved → {p}")


def plot_heatmap(results):
    rs = sorted(results, key=lambda x: x["macro_f1"], reverse=True)
    names = [r["model"] for r in rs]
    data  = np.array([[r["per_class_f1"].get(c,0) for c in LABEL_NAMES] for r in rs])

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="F1 Score")
    ax.set_xticks(range(5)); ax.set_xticklabels(LABEL_NAMES, fontsize=10)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=10)
    ax.set_title("Per-Class F1 Score Heatmap (NSL-KDD)", fontsize=12, pad=10)
    for i in range(len(names)):
        for j in range(5):
            v = data[i,j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if v<0.5 else "black", fontweight="bold")
    plt.tight_layout()
    p = OUT / "fig4_per_class_heatmap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 4 saved → {p}")


def plot_time_f1(results):
    fig, ax = plt.subplots(figsize=(8,5))
    for r in results:
        c = COLORS.get(r["model"],"#888")
        ax.scatter(r["train_time_sec"], r["macro_f1"], color=c, s=180,
                   zorder=5, edgecolors="white", lw=1.5)
        ax.annotate(r["model"], (r["train_time_sec"], r["macro_f1"]),
                    xytext=(8,4), textcoords="offset points", fontsize=9, color=c)
    ax.set_xlabel("Training Time (seconds)", fontsize=11)
    ax.set_ylabel("Macro F1 Score", fontsize=11)
    ax.set_title("Efficiency vs Accuracy Trade-off", fontsize=12)
    ax.axhline(0.95, color="green", ls="--", lw=0.8, alpha=0.5)
    ax.set_ylim(0.5, 1.02); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = OUT / "fig5_time_vs_f1.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 5 saved → {p}")


def plot_conf_matrices(results):
    rs    = sorted(results, key=lambda x: x["macro_f1"], reverse=True)
    n     = len(rs)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes  = axes.flatten()
    for idx, r in enumerate(rs):
        cm = np.array(r["confusion_matrix"])
        ax = axes[idx]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{r['model']} (F1={r['macro_f1']:.4f})", fontsize=10)
        ax.set_xticks(range(5)); ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(5)); ax.set_yticklabels(LABEL_NAMES, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        mx = cm.max()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        fontsize=8, color="white" if cm[i,j]>mx/2 else "black")
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    fig.suptitle("Confusion Matrices — All Models (NSL-KDD Test Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = OUT / "fig6_confusion_matrices.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure 6 saved → {p}")


def save_json(results):
    clean = [{k:v for k,v in r.items() if k not in ("model_obj","y_pred")}
             for r in results]
    p = OUT / "results.json"
    p.write_text(json.dumps(clean, indent=2))
    print(f"  Results JSON saved → {p}")


def latex_table(results):
    rs = sorted(results, key=lambda x: x["macro_f1"], reverse=True)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Model comparison on NSL-KDD dataset}",
        r"\label{tab:baseline}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{Precision} & \textbf{Recall} & \textbf{Time} \\",
        r"\hline",
    ]
    for r in rs:
        n = r["model"]
        if n == "XGBoost":
            n = r"\textbf{XGBoost (proposed)}"
        lines.append(
            f"{n} & {r['accuracy']:.4f} & {r['macro_f1']:.4f} & "
            f"{r['macro_precision']:.4f} & {r['macro_recall']:.4f} & {r['train_time_sec']}s \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    p   = OUT / "table2_latex.tex"
    p.write_text(tex)
    print(f"  LaTeX table saved → {p}")
    print("\n  Paste this into your paper:")
    print("  " + "-"*50)
    print(tex)


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   NetGuard Research — R1: Baseline Comparisons       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Output: {OUT}")

    X_tr, X_test, y_tr, y_test = load_data()

    xgb_result = load_xgboost(X_test, y_test)
    baseline_results = train_baselines(X_tr, X_test, y_tr, y_test)
    all_results = [xgb_result] + baseline_results

    print_table(all_results)

    banner("Generating figures")
    plot_bar(all_results)
    plot_roc(all_results, X_test, y_test)
    plot_heatmap(all_results)
    plot_time_f1(all_results)
    plot_conf_matrices(all_results)

    banner("Saving artifacts")
    save_json(all_results)
    latex_table(all_results)

    banner("R1 COMPLETE")
    print(f"""
  All files in {OUT}:
    fig2_bar_comparison.png     → Figure 2 (paper)
    fig3_roc_curves.png         → Figure 3 (paper)
    fig4_per_class_heatmap.png  → Figure 4 (paper)
    fig5_time_vs_f1.png         → Figure 5 (paper)
    fig6_confusion_matrices.png → Figure 6 (paper)
    results.json                → all numbers
    table2_latex.tex            → Table 2, paste into LaTeX

  Section 6.1 of your paper is now complete.
  Next: python3 scripts/research/r2_cicids.py
""")


if __name__ == "__main__":
    main()
