"""Validation search for discrete draw pick thresholds."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
from sklearn.metrics import log_loss

_draw_weight_override: ContextVar[float | None] = ContextVar("draw_weight", default=None)

DRAW_THRESHOLD_GRID = np.linspace(0.20, 0.36, 17)


def draw_weight_mult() -> float | None:
    """Active draw class multiplier for training, if set via ``use_draw_weight``."""
    return _draw_weight_override.get()


@contextmanager
def use_draw_weight(draw_mult: float):
    token = _draw_weight_override.set(float(draw_mult))
    try:
        yield
    finally:
        _draw_weight_override.reset(token)


def predict_with_draw_threshold(proba: np.ndarray, t_draw: float) -> np.ndarray:
    """Pick Draw when draw prob clears threshold and is the top class."""
    pred = np.argmax(proba, axis=1)
    draw_ok = (proba[:, 0] >= t_draw) & (
        proba[:, 0] >= np.maximum(proba[:, 1], proba[:, 2])
    )
    pred = pred.copy()
    pred[draw_ok] = 0
    return pred


def threshold_val_log_loss(y_true: np.ndarray, proba: np.ndarray, t_draw: float) -> float:
    """Log loss of one-hot predictions from the threshold rule (pick tuning objective)."""
    pred = predict_with_draw_threshold(proba, t_draw)
    onehot = np.zeros_like(proba, dtype=float)
    onehot[np.arange(len(pred)), pred] = 1.0
    return float(log_loss(y_true, onehot, labels=[0, 1, 2]))


def tune_draw_threshold(y_val: np.ndarray, proba_val: np.ndarray) -> tuple[float, float]:
    best_t = float(DRAW_THRESHOLD_GRID[0])
    best_loss = np.inf
    for t in DRAW_THRESHOLD_GRID:
        loss = threshold_val_log_loss(y_val, proba_val, float(t))
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t, best_loss
