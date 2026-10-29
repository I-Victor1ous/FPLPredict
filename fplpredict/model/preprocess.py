"""Shared feature preprocessing: impute on the training rows, then scale."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# "median" (default) imputes from training rows and adds missing-value indicators;
# "zero" reproduces the old fillna(0) behaviour for comparisons.
# Default zero-fill: cross-league val log-loss check (scripts/compare_imputation.py) was
# slightly worse with median (+0.0025). Set FPL_IMPUTE=median to try median+indicators.
IMPUTE_STRATEGY = os.environ.get("FPL_IMPUTE", "zero")


def make_preprocessor(strategy: str | None = None) -> Pipeline:
    strategy = strategy or IMPUTE_STRATEGY
    if strategy == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
    elif strategy == "median":
        imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")
    return Pipeline([("impute", imputer), ("scale", StandardScaler())])


def feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Numeric float frame of ``features``; unparseable values become NaN."""
    return df[features].apply(pd.to_numeric, errors="coerce").astype(float)


def transform_features(preprocessor, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Transform with a fitted preprocessor (legacy artifacts are a bare StandardScaler)."""
    X = feature_frame(df, features)
    if isinstance(preprocessor, StandardScaler):
        X = X.fillna(0)
    return preprocessor.transform(X)
