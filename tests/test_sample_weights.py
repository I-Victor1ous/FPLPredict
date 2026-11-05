import numpy as np

from fplpredict.model.draw_tuning import use_draw_weight
from fplpredict.model.sample_weights import training_sample_weights


def test_training_sample_weights_none_by_default():
    y = np.array([0, 1, 2])
    assert training_sample_weights(y) is None


def test_training_sample_weights_with_draw_override():
    y = np.array([0, 1, 2])
    with use_draw_weight(2.0):
        w = training_sample_weights(y)
    assert w is not None
    assert w[0] == 2.0
    assert w[1] == 1.0
    assert w[2] == 1.0
