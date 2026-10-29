"""Post-hoc probability calibration (temperature scaling on validation)."""

from __future__ import annotations

import os

import numpy as np
from sklearn.metrics import log_loss

TEMPERATURE_GRID = tuple(
    float(x)
    for x in os.environ.get(
        "FPL_TEMPERATURE_GRID",
        "0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0",
    ).split(",")
)

MARKET_IMPLIED_FEATURES = frozenset(
    {
        "Market_implied_win",
        "Market_implied_draw",
        "Market_implied_loss",
    }
)


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Sharpen (T<1) or soften (T>1) a probability matrix; T=1 is identity."""
    t = max(float(temperature), 1e-6)
    if abs(t - 1.0) < 1e-9:
        return np.asarray(proba, dtype=float)
    eps = 1e-15
    p = np.clip(proba, eps, 1.0)
    logits = np.log(p) / t
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), eps)


def tune_temperature(
    y_true: np.ndarray, proba: np.ndarray
) -> tuple[float, float]:
    """Pick T minimizing validation log loss."""
    best_t = 1.0
    best_loss = np.inf
    for t in TEMPERATURE_GRID:
        scaled = apply_temperature(proba, float(t))
        loss = float(log_loss(y_true, scaled, labels=[0, 1, 2]))
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t, best_loss


def calibration_temperature(meta: dict | None) -> float:
    if not meta:
        return 1.0
    cal = meta.get("calibration") or {}
    raw = cal.get("temperature")
    if raw is None:
        return 1.0
    return float(raw)


def apply_probability_calibration(
    proba: np.ndarray, meta: dict | None
) -> np.ndarray:
    return apply_temperature(proba, calibration_temperature(meta))


def modeling_feature_pool(columns: list[str] | set[str]) -> list[str]:
    """Candidate columns for training, excluding bookmaker-implied inputs."""
    return [c for c in columns if c not in MARKET_IMPLIED_FEATURES]
