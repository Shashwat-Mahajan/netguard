"""
models/explainer.py
SHAP-based explainability for the XGBoost classifier.

For every prediction, SHAP tells you WHICH features caused the classification
and HOW MUCH each feature contributed. This is critical for:
  1. Research paper — proves the model is not a black box
  2. SOC analyst trust — "flagged because SYN count=847, duration=0.001s"
  3. Compliance — regulators want explainable AI decisions

SHAP values interpretation:
  Positive value → pushed prediction toward this class
  Negative value → pushed prediction away from this class
  Magnitude      → how strongly it influenced the decision
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from config.logger import get_logger
from processor.feature_schema import FlowFeatures

if TYPE_CHECKING:
    from models.xgboost_classifier import IntrusionClassifier

log = get_logger(__name__)


class SHAPExplainer:
    """
    Wraps shap.TreeExplainer for the IntrusionClassifier.

    Usage:
        explainer = SHAPExplainer(classifier)
        explainer.build()

        # Explain one prediction
        explanation = explainer.explain_single(feature_vector, predicted_class=1)
        print(explanation["top_features"])

        # Explain a batch (for research paper plots)
        shap_values = explainer.explain_batch(X_test)
        explainer.plot_summary(shap_values, X_test)
    """

    def __init__(self, classifier: "IntrusionClassifier") -> None:
        self._classifier = classifier
        self._explainer = None

    def build(self) -> None:
        """Build the SHAP TreeExplainer from the trained XGBoost model."""
        try:
            import shap  # type: ignore
        except ImportError:
            raise ImportError("Run: pip install shap")

        if not self._classifier.is_fitted():
            raise RuntimeError("Train the classifier before building the explainer.")

        self._explainer = shap.TreeExplainer(self._classifier._model)
        log.info("shap.explainer.built")
        print("  ✓  SHAP explainer built")

    def explain_single(
        self,
        feature_vector: np.ndarray,
        predicted_class: int,
        top_n: int = 10,
    ) -> dict:
        """
        Explain a single prediction.

        Returns dict with:
          top_features: {feature_name: shap_value} for top N features
          base_value:   model's base (prior) probability
          all_shap:     full SHAP vector for this class
        """
        self._check_built()
        x = feature_vector.reshape(1, -1)
        # shap_values shape: (1, n_features, n_classes)
        shap_vals = self._explainer.shap_values(x)

        # Get SHAP values for the predicted class
        class_shap = shap_vals[0, :, predicted_class]

        # Rank by absolute importance
        indices = np.argsort(np.abs(class_shap))[::-1][:top_n]
        top_features = {
            FlowFeatures.ML_FEATURES[i]: round(float(class_shap[i]), 5)
            for i in indices
        }

        return {
            "predicted_class":    predicted_class,
            "predicted_label":    FlowFeatures.LABEL_NAMES[predicted_class],
            "top_features":       top_features,
            "base_value":         float(self._explainer.expected_value[predicted_class]),
            "all_shap":           class_shap.tolist(),
        }

    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a batch.
        Returns array of shape (n_samples, n_features, n_classes).
        Use for summary plots in the research paper.
        """
        self._check_built()
        log.info("shap.explain_batch.start", samples=len(X))
        shap_vals = self._explainer.shap_values(X)
        log.info("shap.explain_batch.done")
        return shap_vals

    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        class_idx: int = 1,
        save_path: Path | None = None,
    ) -> None:
        """
        Generate a SHAP summary bar plot for one class.
        Saves to file if save_path is provided, otherwise shows interactively.
        This plot goes directly into your research paper.
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Run: pip install shap matplotlib")

        class_name = FlowFeatures.LABEL_NAMES.get(class_idx, str(class_idx))
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values[:, :, class_idx],
            X,
            feature_names=FlowFeatures.ML_FEATURES,
            show=False,
            plot_type="bar",
            max_display=15,
        )
        plt.title(f"Feature Importance (SHAP) — Class: {class_name}", fontsize=13)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  ✓  SHAP plot saved → {save_path}")
            plt.close()
        else:
            plt.show()

    def format_explanation(self, explanation: dict) -> str:
        """
        Format an explanation as a human-readable string.
        Used by the LLM incident reporter in Phase 5.
        """
        lines = [
            f"Attack type: {explanation['predicted_label']}",
            f"Top contributing features:",
        ]
        for feat, val in explanation["top_features"].items():
            direction = "↑ increased" if val > 0 else "↓ decreased"
            lines.append(f"  {feat:<30} {direction} suspicion  (SHAP={val:+.4f})")
        return "\n".join(lines)

    def _check_built(self) -> None:
        if self._explainer is None:
            raise RuntimeError("Call build() first.")
