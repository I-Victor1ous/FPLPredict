"""Greedy forward feature selection on the validation season."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from fplpredict.config import FEATURE_SELECTION_MAX, FEATURE_SELECTION_MIN_GAIN
from fplpredict.model.preprocess import feature_frame, make_preprocessor, transform_features
from fplpredict.model.sample_weights import training_sample_weights

log = logging.getLogger(__name__)

LABELS = [0, 1, 2]


def _val_log_loss(
    train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str]
) -> float:
    preprocessor = make_preprocessor()
    train_x = preprocessor.fit_transform(feature_frame(train_df, features))
    val_x = transform_features(preprocessor, val_df, features)
    train_y = train_df["Target"].astype(int).to_numpy()
    model = LogisticRegression(max_iter=2000)
    train_w = training_sample_weights(train_y)
    if train_w is None:
        model.fit(train_x, train_y)
    else:
        model.fit(train_x, train_y, sample_weight=train_w)
    proba = model.predict_proba(val_x)
    return float(log_loss(val_df["Target"].astype(int), proba, labels=LABELS))


def forward_select_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    candidates: list[str],
    *,
    max_features: int = FEATURE_SELECTION_MAX,
    min_gain: float = FEATURE_SELECTION_MIN_GAIN,
) -> tuple[list[str], list[dict]]:
    """Add the feature that most lowers validation log loss until the gain is too small."""
    pool = [
        c
        for c in candidates
        if c in train_df.columns
        and c in val_df.columns
        and train_df[c].fillna(0).std() > 0
    ]
    selected: list[str] = []
    history: list[dict] = []
    best_loss = np.inf
    while pool and len(selected) < max_features:
        scores = {c: _val_log_loss(train_df, val_df, selected + [c]) for c in pool}
        feature, loss = min(scores.items(), key=lambda kv: kv[1])
        gain = best_loss - loss
        if selected and gain < min_gain:
            break
        selected.append(feature)
        pool.remove(feature)
        best_loss = loss
        history.append({"feature": feature, "val_log_loss": round(loss, 5)})
        log.info("Feature selection: +%s (val log loss %.4f)", feature, loss)
    return selected, history
