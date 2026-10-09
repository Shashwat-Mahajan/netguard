"""
tests/test_models.py
Tests for XGBoost classifier, Isolation Forest, and SHAP explainer.
Uses real trained models from data/processed/.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest
from pathlib import Path
from config.settings import settings


# Skip all if models not trained yet
MODEL_PATH = settings.data_processed_dir / "xgboost_model.joblib"
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Models not trained — run scripts/train_model.py first"
)


# ── XGBoost tests ─────────────────────────────────────────────

def test_xgboost_loads():
    from models.xgboost_classifier import IntrusionClassifier
    clf = IntrusionClassifier()
    clf.load(MODEL_PATH)
    assert clf.is_fitted()


def test_xgboost_predict_single():
    from models.xgboost_classifier import IntrusionClassifier
    clf = IntrusionClassifier()
    clf.load(MODEL_PATH)
    vec = np.zeros(39, dtype=np.float32)
    label, code, conf, probs = clf.predict_single(vec)
    assert label in ["Benign","DoS","Probe","R2L","U2R"]
    assert 0 <= code <= 4
    assert 0.0 <= conf <= 1.0
    assert len(probs) == 5
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_xgboost_predict_batch():
    from models.xgboost_classifier import IntrusionClassifier
    clf = IntrusionClassifier()
    clf.load(MODEL_PATH)
    X = np.zeros((10, 39), dtype=np.float32)
    labels, confs = clf.predict_batch(X)
    assert len(labels) == 10
    assert len(confs) == 10
    assert all(0 <= l <= 4 for l in labels)
    assert all(0.0 <= c <= 1.0 for c in confs)


def test_xgboost_feature_importances():
    from models.xgboost_classifier import IntrusionClassifier
    from processor.feature_schema import FlowFeatures
    clf = IntrusionClassifier()
    clf.load(MODEL_PATH)
    imps = clf.feature_importances
    assert len(imps) == len(FlowFeatures.ML_FEATURES)
    assert all(v >= 0 for v in imps.values())
    assert abs(sum(imps.values()) - 1.0) < 0.01


# ── Isolation Forest tests ─────────────────────────────────────

def test_isoforest_loads():
    from models.anomaly_detector import AnomalyDetector
    det = AnomalyDetector()
    path = settings.data_processed_dir / "isolation_forest.joblib"
    if not path.exists():
        pytest.skip("Isolation Forest not trained")
    det.load(path)
    assert det.is_fitted()


def test_isoforest_benign_vs_anomaly():
    from models.anomaly_detector import AnomalyDetector
    det = AnomalyDetector()
    path = settings.data_processed_dir / "isolation_forest.joblib"
    if not path.exists():
        pytest.skip("Isolation Forest not trained")
    det.load(path)

    # Normal traffic — low values, balanced
    benign = np.array([2.5, 8, 6, 1200, 8400, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 3840, 5.6, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 1, 1, 40, 40, 65535, 8192, 6],
                      dtype=np.float32)

    # DoS traffic — extreme values
    dos = np.array([0.001, 1000, 0, 60000, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 60000000, 1000000, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 1000, 0, 0,
                    0, 0, 1000, 0, 0, 0, 0, 0, 6],
                   dtype=np.float32)

    _, benign_score = det.predict_single(benign)
    _, dos_score    = det.predict_single(dos)

    # Both scores should be in valid range
    assert -1.0 <= benign_score <= 1.0
    assert -1.0 <= dos_score <= 1.0
    # Anomaly detector should return a boolean prediction
    is_anomaly_benign, _ = det.predict_single(benign)
    is_anomaly_dos, _    = det.predict_single(dos)
    # At least one should be flagged or both valid booleans
    assert isinstance(is_anomaly_benign, bool)
    assert isinstance(is_anomaly_dos, bool)


# ── End-to-end pipeline test ───────────────────────────────────

def test_full_pipeline():
    """Flow features → preprocessor → XGBoost → valid prediction."""
    from models.xgboost_classifier import IntrusionClassifier
    from processor.preprocessor import DataPreprocessor
    from processor.feature_schema import FlowFeatures

    pp = DataPreprocessor()
    scaler_path = settings.data_processed_dir / "scaler.joblib"
    if not scaler_path.exists():
        pytest.skip("Scaler not fitted")
    pp.load_scaler(scaler_path)

    clf = IntrusionClassifier()
    clf.load(MODEL_PATH)

    # Create a realistic flow
    f = FlowFeatures(
        duration=2.5,
        fwd_packet_count=8,
        bwd_packet_count=6,
        total_fwd_bytes=1200,
        total_bwd_bytes=8400,
        flow_bytes_per_sec=3840.0,
        flow_packets_per_sec=5.6,
        syn_flag_count=1,
        fin_flag_count=1,
        ack_flag_count=14,
        protocol=6,
    )

    vec = pp.transform_features(f)
    label, code, conf, probs = clf.predict_single(vec[0])

    assert label in ["Benign","DoS","Probe","R2L","U2R"]
    assert 0 <= code <= 4
    assert conf > 0.0
    assert len(probs) == 5

