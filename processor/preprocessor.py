"""
processor/preprocessor.py
Production-grade data preprocessing pipeline.

Responsibilities:
  - Remove duplicate rows
  - Handle NaN / Inf values
  - Scale features (RobustScaler — handles outliers better than StandardScaler
    for network traffic, which has extreme long tails)
  - Save / load scaler as a joblib artifact (used at inference time too)
  - Stratified train/val/test split
  - Class imbalance report
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from config.logger import get_logger
from config.settings import settings
from processor.feature_schema import FlowFeatures

logger = get_logger(__name__)

SCALER_PATH = settings.data_processed_dir / "scaler.joblib"


class DataPreprocessor:
    """
    Fit on training data; transform train / val / test consistently.

    Usage:
        pp = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = pp.fit_transform(df)
        pp.save_scaler()   # persist for inference

        # Later at inference time:
        pp2 = DataPreprocessor()
        pp2.load_scaler()
        X = pp2.transform(new_df)
    """

    def __init__(self) -> None:
        self._scaler: RobustScaler | None = None
        self._feature_cols: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Clean → split → scale.
        Returns (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        df = self._clean(df)
        self._feature_cols = self._select_features(df)

        X = df[self._feature_cols].values.astype(np.float32)
        y = df["label_encoded"].values.astype(np.int32)

        logger.info(
            "preprocessor.class_distribution",
            dist={
                FlowFeatures.LABEL_NAMES.get(int(k), str(k)): int(v)
                for k, v in zip(*np.unique(y, return_counts=True))
            },
        )

        # Stratified split: train | temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=val_size + test_size,
            stratify=y,
            random_state=random_state,
        )
        # Split temp into val | test
        relative_test = test_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test,
            stratify=y_temp,
            random_state=random_state,
        )

        # Fit scaler on training set only
        self._scaler = RobustScaler()
        X_train = self._scaler.fit_transform(X_train).astype(np.float32)
        X_val = self._scaler.transform(X_val).astype(np.float32)
        X_test = self._scaler.transform(X_test).astype(np.float32)

        logger.info(
            "preprocessor.split",
            train=len(X_train),
            val=len(X_val),
            test=len(X_test),
            features=len(self._feature_cols),
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the already-fitted scaler."""
        if self._scaler is None:
            raise RuntimeError(
                "Scaler not fitted. Call fit_transform() or load_scaler() first."
            )
        df = self._clean(df)
        X = df[self._feature_cols].values.astype(np.float32)
        return self._scaler.transform(X).astype(np.float32)

    def transform_features(self, features: FlowFeatures) -> np.ndarray:
        """Transform a single FlowFeatures object into a scaled vector."""
        if self._scaler is None:
            raise RuntimeError("Scaler not fitted.")
        vec = features.to_vector().reshape(1, -1)
        return self._scaler.transform(vec).astype(np.float32)

    def save_scaler(self, path: Path = SCALER_PATH) -> None:
        if self._scaler is None:
            raise RuntimeError("Scaler not fitted.")
        joblib.dump(
            {"scaler": self._scaler, "feature_cols": self._feature_cols},
            path,
        )
        logger.info("preprocessor.scaler_saved", path=str(path))

    def load_scaler(self, path: Path = SCALER_PATH) -> None:
        obj = joblib.load(path)
        self._scaler = obj["scaler"]
        self._feature_cols = obj["feature_cols"]
        logger.info(
            "preprocessor.scaler_loaded",
            path=str(path),
            features=len(self._feature_cols),
        )

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Replace inf
        df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)

        # Drop duplicate rows
        before = len(df)
        df.drop_duplicates(inplace=True)
        dropped = before - len(df)
        if dropped:
            logger.info("preprocessor.dedup", dropped=dropped)

        # Fill remaining NaN with column median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                fill = df[col].median()
                df[col] = df[col].fillna(fill if not pd.isna(fill) else 0.0)

        # Drop rows with unknown labels
        if "label_encoded" in df.columns:
            df = df[df["label_encoded"] >= 0].copy()

        return df

    @staticmethod
    def _select_features(df: pd.DataFrame) -> list[str]:
        """Return ML features that are present in the DataFrame."""
        available = [c for c in FlowFeatures.ML_FEATURES if c in df.columns]
        missing = set(FlowFeatures.ML_FEATURES) - set(available)
        if missing:
            logger.warning("preprocessor.missing_features", missing=sorted(missing))
        return available
