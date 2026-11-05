import numpy as np

from fplpredict.model.draw_tuning import (
    predict_with_draw_threshold,
    threshold_val_log_loss,
    tune_draw_threshold,
)


def test_draw_threshold_can_force_draw_picks():
    proba = np.array(
        [
            [0.36, 0.32, 0.32],
            [0.35, 0.30, 0.35],
        ]
    )
    pred = predict_with_draw_threshold(proba, 0.28)
    assert pred[0] == 0
    assert pred[1] == 0  # draw ties loss at 0.35, threshold rule picks draw


def test_tune_draw_threshold_picks_best_grid_point():
    y = np.array([0, 2, 0, 1])
    proba = np.array(
        [
            [0.35, 0.30, 0.35],
            [0.10, 0.20, 0.70],
            [0.40, 0.30, 0.30],
            [0.15, 0.55, 0.30],
        ]
    )
    t, loss = tune_draw_threshold(y, proba)
    assert 0.20 <= t <= 0.36
    assert loss == threshold_val_log_loss(y, proba, t)
