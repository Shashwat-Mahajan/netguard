"""
models/anomaly_detector.py
Isolation Forest for zero-day / unknown attack detection.

Why Isolation Forest alongside XGBoost?
  - XGBoost only detects attack patterns it has seen in training data
  - Isolation Forest detects ANY anomalous flow, including zero-days
  - Together: XGBoost classifies known attacks, IsoForest flags unknowns

Scoring:
  - score < -0.1  → anomalous (flag for review)
  - score >= -0.1 → normal

The threshold is tunable — lower = more sensitive = more false positives.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from config.logger import get_logger
from config.settings import settings

log = get_logger(__name__)

ANOMALY_MODEL_PATH = settings.data_processed_dir / "isolation_forest.joblib"

# Flows with score below this are flagged as anomalous
DEFAULT_THRESHOLD = -0.3


class AnomalyDetector:
    """
    Isolation Forest wrapper for zero-day detection.

    Train on BENIGN traffic only — the model learns what normal looks like.
    At inference, anything that deviates from normal gets a low score.

    Usage:
        det = AnomalyDetector()
        det.train(X_train_benign)
        is_anomaly, score = det.predict_single(feature_vector)
        det.save()
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self._threshold = threshold
        self._model: IsolationForest | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X_benign: np.ndarray) -> None:
        """
        Train on benign-only traffic.
        X_benign: scaled feature matrix of normal flows only.
        """
        log.info(
            "isoforest.train.start",
            samples=len(X_benign),
            features=X_benign.shape[1],
        )

        self._model = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination=0.05,   # assume ~5% of traffic is anomalous
            max_features=1.0,
            bootstrap=False,
            random_state=42,
            n_jobs=-1,
        )

        self._model.fit(X_benign)
        # Flag roughly the lowest 1% of benign scores as anomalous
        benign_scores = self._model.score_samples(X_benign[:200_000])
        self._threshold = float(np.percentile(benign_scores, 1))
        log.info("isoforest.threshold_set", threshold=round(self._threshold, 4))
        log.info("isoforest.train.done")
        print(f"  ✓  Isolation Forest trained on {len(X_benign):,} benign samples")

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate anomaly detection performance.
        y: original labels (0=benign, anything else = attack)
        Treats this as binary: benign vs all attacks.
        """
        self._check_fitted()

        scores = self._model.score_samples(X)
        # Binary ground truth: 0 = benign, 1 = any attack
        y_binary = (y != 0).astype(int)

        # Negate scores because lower score = more anomalous
        try:
            auc = roc_auc_score(y_binary, -scores)
        except ValueError:
            auc = 0.0

        predictions = (scores < self._threshold).astype(int)
        tp = int(((predictions == 1) & (y_binary == 1)).sum())
        fp = int(((predictions == 1) & (y_binary == 0)).sum())
        tn = int(((predictions == 0) & (y_binary == 0)).sum())
        fn = int(((predictions == 0) & (y_binary == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        print(f"\n{'═'*52}")
        print("  Anomaly Detector Evaluation")
        print(f"{'═'*52}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Precision: {precision:.4f}  (of flagged flows, how many are real attacks)")
        print(f"  Recall:    {recall:.4f}  (of all attacks, how many did we catch)")
        print(f"  F1:        {f1:.4f}")
        print(f"  Threshold: {self._threshold}")
        print("\n  Confusion (binary — attack vs benign):")
        print("             Pred Normal  Pred Anomaly")
        print(f"  Actual Normal   {tn:>8}     {fp:>8}")
        print(f"  Actual Attack   {fn:>8}     {tp:>8}")

        log.info(
            "isoforest.evaluate",
            auc=round(auc, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
        )

        return {"auc": auc, "precision": precision, "recall": recall, "f1": f1}

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_single(
        self,
        feature_vector: np.ndarray,
    ) -> tuple[bool, float]:
        """
        Returns (is_anomaly, anomaly_score).
        is_anomaly=True means the flow is suspicious / zero-day.
        score: lower = more anomalous (typically -0.5 to 0.5)
        """
        self._check_fitted()
        x = feature_vector.reshape(1, -1)
        score = float(self._model.score_samples(x)[0])
        is_anomaly = score < self._threshold
        return is_anomaly, score

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores for a batch. Lower = more anomalous."""
        self._check_fitted()
        return self._model.score_samples(X)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = ANOMALY_MODEL_PATH) -> None:
        self._check_fitted()
        joblib.dump({"model": self._model, "threshold": self._threshold}, path)
        log.info("isoforest.saved", path=str(path))
        print(f"  ✓  Anomaly detector saved → {path}")

    def load(self, path: Path = ANOMALY_MODEL_PATH) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")
        obj = joblib.load(path)
        self._model = obj["model"]
        self._threshold = obj["threshold"]
        log.info("isoforest.loaded", path=str(path))

    def is_fitted(self) -> bool:
        return self._model is not None

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Detector not trained. Call train() or load() first.")
