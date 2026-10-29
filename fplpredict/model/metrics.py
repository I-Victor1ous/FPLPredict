"""Forecast metrics (RPS, intervals, baselines)."""

from __future__ import annotations

import numpy as np

# sklearn Target / RESULT_LABELS: 0=Draw, 1=Loss, 2=Win
# RPS ordinal order: Loss < Draw < Win
TARGET_TO_ORDINAL = {1: 0, 0: 1, 2: 2}


def proba_to_ordinal(proba: np.ndarray) -> np.ndarray:
    """Reorder (n, 3) from [Draw, Loss, Win] to [Loss, Draw, Win]."""
    return proba[:, [1, 0, 2]]


def ranked_probability_score(
    y_true: np.ndarray, proba: np.ndarray, *, sklearn_order: bool = True
) -> float:
    """
    Mean RPS for multiclass outcomes with ranking Loss < Draw < Win.

    ``proba`` columns sklearn order (Draw, Loss, Win) when ``sklearn_order`` is True.
    """
    if len(y_true) == 0:
        return float("nan")
    p = proba_to_ordinal(proba) if sklearn_order else proba
    scores = []
    for i, y in enumerate(y_true):
        rank = TARGET_TO_ORDINAL[int(y)]
        f1 = p[i, 0]
        f2 = f1 + p[i, 1]
        o1 = 1.0 if rank <= 0 else 0.0
        o2 = 1.0 if rank <= 1 else 0.0
        scores.append(((f1 - o1) ** 2 + (f2 - o2) ** 2) / 2.0)
    return float(np.mean(scores))


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (float(max(0.0, centre - margin)), float(min(1.0, centre + margin)))


def accuracy_with_interval(correct: int, n: int) -> dict:
    if n <= 0:
        return {"n": 0, "accuracy": None, "ci95Low": None, "ci95High": None}
    low, high = wilson_interval(correct, n)
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "ci95Low": round(low, 4),
        "ci95High": round(high, 4),
    }


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot outcome (0 is perfect)."""
    if len(y_true) == 0:
        return float("nan")
    onehot = np.zeros_like(proba, dtype=float)
    onehot[np.arange(len(y_true)), np.asarray(y_true, dtype=int)] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def reliability_bins(
    y_true: np.ndarray, proba: np.ndarray, edges: tuple[float, ...] = (0.0, 0.4, 0.5, 0.6, 0.7, 1.0)
) -> list[dict]:
    """Group picks by top-pick probability; compare mean confidence with the hit rate."""
    if len(y_true) == 0:
        return []
    top = proba.max(axis=1)
    hit = np.argmax(proba, axis=1) == np.asarray(y_true, dtype=int)
    bins: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (top >= lo) & ((top < hi) if hi < 1.0 else (top <= hi))
        n = int(mask.sum())
        if not n:
            continue
        bins.append(
            {
                "low": lo,
                "high": hi,
                "n": n,
                "meanPredicted": round(float(top[mask].mean()), 4),
                "observed": round(float(hit[mask].mean()), 4),
            }
        )
    return bins


def home_win_baseline_accuracy(y_true: np.ndarray) -> float:
    """Always predict Win (class 2) from the team-row perspective."""
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true == 2))
