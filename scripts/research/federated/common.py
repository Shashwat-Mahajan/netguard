"""
scripts/research/federated/common.py
Shared helpers for the federated experiment (data loading, XGBoost params,
scoring). Uses the same preprocessed CICIDS arrays as the central model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from models.xgboost_classifier import IntrusionClassifier  # noqa: E402
from processor.feature_schema import FlowFeatures  # noqa: E402

FED_DIR = ROOT / "data" / "research" / "federated"
PARTS_DIR = FED_DIR / "partitions"
MODELS_DIR = FED_DIR / "models"

# Same settings as the central model (models/xgboost_classifier.py),
# written in xgboost.train() naming.
PARAMS = {
    "objective": "multi:softprob",
    "num_class": 5,
    "eval_metric": "mlogloss",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "alpha": 0.1,
    "lambda": 1.0,
    "tree_method": "hist",
    "seed": 42,
    "nthread": 2,
    "verbosity": 0,
}


def sample_weights(y: np.ndarray) -> np.ndarray:
    return IntrusionClassifier._compute_sample_weights(y)


def load_partition(mode: str, clients: int, cid: int):
    """Returns X_train, y_train, X_val, y_val for one client."""
    parts = np.load(PARTS_DIR / f"{mode}_{clients}.npz")
    d = settings.data_processed_dir
    X = np.load(d / "X_train_cicids.npy", mmap_mode="r")
    y = np.load(d / "y_train_cicids.npy")
    tr, va = parts[f"train_{cid}"], parts[f"val_{cid}"]
    return np.asarray(X[tr]), y[tr], np.asarray(X[va]), y[va]


def load_test():
    d = settings.data_processed_dir
    return np.load(d / "X_test_cicids.npy"), np.load(d / "y_test_cicids.npy")


def score(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """probs: (n, 5) class probabilities."""
    pred = np.argmax(probs, axis=1)
    recall = recall_score(
        y_true, pred, labels=list(range(5)), average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "recall": {FlowFeatures.LABEL_NAMES[i]: float(recall[i]) for i in range(5)},
    }
