"""
models/xgboost_classifier.py
XGBoost multi-class classifier for network intrusion detection.

Classifies each flow into one of 5 categories:
  0 = Benign
  1 = DoS
  2 = Probe
  3 = R2L
  4 = U2R

Design decisions:
  - scale_pos_weight not used here because XGBoost handles multi-class
    imbalance via sample_weight in fit()
  - early_stopping_rounds on validation set prevents overfitting
  - All hyperparameters are tunable via the config dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier

from config.logger import get_logger
from config.settings import settings
from processor.feature_schema import FlowFeatures

log = get_logger(__name__)

MODEL_PATH = settings.data_processed_dir / "xgboost_model.joblib"


# ── Default hyperparameters ───────────────────────────────────────────────────
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators":       500,
    "max_depth":          6,
    "learning_rate":      0.1,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   5,
    "gamma":              0.1,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "objective":          "multi:softprob",
    "num_class":          5,
    "eval_metric":        "mlogloss",
    "early_stopping_rounds": 20,
    "tree_method":        "hist",   # fast histogram method
    "random_state":       42,
    "n_jobs":             -1,       # use all CPU cores
    "verbosity":          0,
}


class IntrusionClassifier:
    """
    Wraps XGBClassifier with a clean train / evaluate / predict API.

    Usage:
        clf = IntrusionClassifier()
        clf.train(X_train, y_train, X_val, y_val)
        report = clf.evaluate(X_test, y_test)
        clf.save()

        # Later at inference:
        clf2 = IntrusionClassifier()
        clf2.load()
        label, confidence = clf2.predict_single(feature_vector)
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = {**DEFAULT_PARAMS, **(params or {})}
        self._model: XGBClassifier | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, float]:
        """
        Train the classifier.
        Returns dict of validation metrics after training.
        """
        log.info(
            "xgb.train.start",
            train_samples=len(X_train),
            val_samples=len(X_val),
            features=X_train.shape[1],
            params=self._params,
        )

        # Compute sample weights to handle class imbalance
        # Rare classes (U2R, R2L) get higher weight
        sample_weights = self._compute_sample_weights(y_train)

        self._model = XGBClassifier(**{
            k: v for k, v in self._params.items()
            if k != "early_stopping_rounds"
        })

        self._model.set_params(
            early_stopping_rounds=self._params.get("early_stopping_rounds", 20)
        )

        self._model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=50,   # print every 50 rounds
        )

        best_round = self._model.best_iteration
        log.info("xgb.train.done", best_round=best_round)

        # Return validation metrics
        return self.evaluate(X_val, y_val, split="val")

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split: str = "test",
    ) -> dict[str, Any]:
        """
        Full evaluation: per-class F1, macro F1, confusion matrix.
        Prints a human-readable report.
        """
        self._check_fitted()
        y_pred = self._model.predict(X)

        macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y, y_pred)

        report = classification_report(
            y,
            y_pred,
            target_names=[FlowFeatures.LABEL_NAMES[i] for i in range(5)],
            zero_division=0,
        )

        print(f"\n{'═'*52}")
        print(f"  Evaluation — {split.upper()} SET")
        print(f"{'═'*52}")
        print(report)
        print(f"  Macro F1:    {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
        labels = [FlowFeatures.LABEL_NAMES[i] for i in range(5)]
        col_w = 10
        print("  " + "".join(f"{l:>{col_w}}" for l in labels))
        for i, row in enumerate(cm):
            print(f"  {labels[i]:<8}" + "".join(f"{v:>{col_w}}" for v in row))

        log.info(
            "xgb.evaluate",
            split=split,
            macro_f1=round(macro_f1, 4),
            weighted_f1=round(weighted_f1, 4),
        )

        return {
            "macro_f1":    macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": cm.tolist(),
            "report":      report,
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_single(
        self,
        feature_vector: np.ndarray,
    ) -> tuple[str, int, float, dict[str, float]]:
        """
        Predict a single flow.
        Returns (label_name, label_int, confidence, all_class_probs).

        feature_vector: shape (39,) — already scaled by preprocessor
        """
        self._check_fitted()
        x = feature_vector.reshape(1, -1)
        probs = self._model.predict_proba(x)[0]
        label_int = int(np.argmax(probs))
        confidence = float(probs[label_int])
        label_name = FlowFeatures.LABEL_NAMES[label_int]

        all_probs = {
            FlowFeatures.LABEL_NAMES[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        }

        return label_name, label_int, confidence, all_probs

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict a batch of flows.
        Returns (labels_int, confidences).
        """
        self._check_fitted()
        probs = self._model.predict_proba(X)
        labels = np.argmax(probs, axis=1)
        confidences = probs[np.arange(len(probs)), labels]
        return labels, confidences

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH) -> None:
        self._check_fitted()
        joblib.dump(self._model, path)
        log.info("xgb.saved", path=str(path))
        print(f"\n  ✓  Model saved → {path}")

    def load(self, path: Path = MODEL_PATH) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self._model = joblib.load(path)
        log.info("xgb.loaded", path=str(path))

    def is_fitted(self) -> bool:
        return self._model is not None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

    @staticmethod
    def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
        """
        Inverse-frequency weighting so rare classes get more attention.
        U2R has ~52 samples vs 67k Benign — without this XGBoost ignores it.
        """
        classes, counts = np.unique(y, return_counts=True)
        freq = dict(zip(classes, counts))
        total = len(y)
        weights = np.array([
            total / (len(classes) * freq[label])
            for label in y
        ], dtype=np.float32)
        return weights

    @property
    def feature_importances(self) -> dict[str, float]:
        """Return feature importance scores (gain-based)."""
        self._check_fitted()
        imps = self._model.feature_importances_
        return {
            FlowFeatures.ML_FEATURES[i]: round(float(v), 6)
            for i, v in enumerate(imps)
        }
