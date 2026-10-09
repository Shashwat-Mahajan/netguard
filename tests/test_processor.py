"""
tests/test_processor.py
Tests for dataset loading and preprocessing logic.
Uses synthetic DataFrames — no actual dataset files needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from processor.feature_schema import FlowFeatures
from processor.preprocessor import DataPreprocessor


def _synthetic_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic DataFrame that mimics CICIDS2017 post-load structure.
    Labels are distributed 70% benign, 10% DoS, 10% Probe, 5% R2L, 5% U2R.
    """
    rng = np.random.default_rng(seed)
    n_features = len(FlowFeatures.ML_FEATURES)

    data = rng.exponential(scale=100, size=(n, n_features)).astype(np.float32)
    df = pd.DataFrame(data, columns=FlowFeatures.ML_FEATURES)

    labels = rng.choice(
        [0, 1, 2, 3, 4],
        size=n,
        p=[0.70, 0.10, 0.10, 0.05, 0.05],
    )
    df["label_encoded"] = labels
    df["label"] = df["label_encoded"].map(FlowFeatures.LABEL_NAMES)

    return df


# ── DataPreprocessor ──────────────────────────────────────────────────────────


def test_fit_transform_shapes():
    df = _synthetic_df(400)
    pp = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = pp.fit_transform(df)

    total = len(X_train) + len(X_val) + len(X_test)
    assert total == 400
    assert X_train.shape[1] == len(FlowFeatures.ML_FEATURES)


def test_fit_transform_no_nan_in_output():
    df = _synthetic_df(300)
    # Inject some NaN and Inf
    df.iloc[5, 3] = float("nan")
    df.iloc[10, 7] = float("inf")
    df.iloc[15, 2] = float("-inf")

    pp = DataPreprocessor()
    X_train, X_val, X_test, *_ = pp.fit_transform(df)

    assert not np.isnan(X_train).any()
    assert not np.isinf(X_train).any()
    assert not np.isnan(X_val).any()
    assert not np.isnan(X_test).any()


def test_fit_transform_labels_preserved():
    df = _synthetic_df(200)
    pp = DataPreprocessor()
    _, _, _, y_train, y_val, y_test = pp.fit_transform(df)

    all_labels = np.concatenate([y_train, y_val, y_test])
    assert set(all_labels).issubset({0, 1, 2, 3, 4})


def test_transform_without_fit_raises():
    pp = DataPreprocessor()
    df = _synthetic_df(50)
    with pytest.raises(RuntimeError, match="not fitted"):
        pp.transform(df)


def test_scaler_save_load(tmp_path):
    df = _synthetic_df(300)
    pp = DataPreprocessor()
    X_train, _, X_test, *_ = pp.fit_transform(df)

    scaler_path = tmp_path / "scaler.joblib"
    pp.save_scaler(scaler_path)

    pp2 = DataPreprocessor()
    pp2.load_scaler(scaler_path)

    # Re-transform the full (deduped) dataframe using the reloaded scaler.
    # Row count won't match X_test (that's only the 15% test split) — what
    # matters is that the reloaded scaler produces the correct feature
    # dimensionality and is numerically usable.
    X_reloaded = pp2.transform(df)
    assert X_reloaded.shape[1] == X_test.shape[1]
    assert X_reloaded.shape[0] == len(df)
    assert not np.isnan(X_reloaded).any()


def test_duplicate_removal():
    # Use enough rows that even the rarest class (5%) keeps >=2 members
    # in every stratified split after dedup.
    df = _synthetic_df(500)
    # Double every row
    df_dup = pd.concat([df, df], ignore_index=True)
    pp = DataPreprocessor()
    X_train, X_val, X_test, *_ = pp.fit_transform(df_dup)
    total = len(X_train) + len(X_val) + len(X_test)
    # After dedup, total should be ~500, not 1000
    assert total <= 500


def test_unknown_labels_dropped():
    df = _synthetic_df(500)
    # Add some -1 label rows
    bad_rows = df.iloc[:10].copy()
    bad_rows["label_encoded"] = -1
    df_with_bad = pd.concat([df, bad_rows], ignore_index=True)

    pp = DataPreprocessor()
    _, _, _, y_train, y_val, y_test = pp.fit_transform(df_with_bad)
    all_labels = np.concatenate([y_train, y_val, y_test])
    assert -1 not in all_labels


# ── FlowFeatures schema ────────────────────────────────────────────────────────


def test_feature_schema_ml_features_count():
    assert len(FlowFeatures.ML_FEATURES) == 39


def test_feature_schema_to_vector():
    f = FlowFeatures(duration=1.5, fwd_packet_count=10, total_fwd_bytes=1000)
    vec = f.to_vector()
    assert vec.dtype == np.float32
    assert len(vec) == len(FlowFeatures.ML_FEATURES)
    assert vec[FlowFeatures.ML_FEATURES.index("duration")] == pytest.approx(1.5)


def test_label_map_completeness():
    for k, v in FlowFeatures.LABEL_MAP.items():
        assert isinstance(k, str)
        assert v in {-1, 0, 1, 2, 3, 4}