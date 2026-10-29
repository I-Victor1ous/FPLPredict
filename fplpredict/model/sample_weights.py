"""Training sample weights (optional draw-class upweight via ``use_draw_weight``)."""

from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from fplpredict.model.draw_tuning import draw_weight_mult


def training_sample_weights(y: np.ndarray) -> np.ndarray | None:
    """Per-sample weights for training, or ``None`` when all classes are weighted equally."""
    mult = draw_weight_mult()
    if mult is None:
        return None
    return compute_sample_weight(class_weight={0: float(mult), 1: 1.0, 2: 1.0}, y=y)
