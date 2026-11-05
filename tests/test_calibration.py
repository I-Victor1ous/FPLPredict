import numpy as np

from fplpredict.model.calibration import (
    apply_probability_calibration,
    apply_temperature,
    tune_temperature,
)


def test_temperature_one_is_identity():
    p = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    out = apply_temperature(p, 1.0)
    np.testing.assert_allclose(out, p, rtol=1e-6)


def test_temperature_softens_peaked_distribution():
    p = np.array([[0.05, 0.05, 0.9]])
    soft = apply_temperature(p, 2.0)
    assert soft[0, 2] < 0.9
    assert np.isclose(soft.sum(), 1.0)


def test_tune_temperature_returns_grid_member():
    y = np.array([2, 1, 0, 2])
    proba = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.2, 0.5, 0.3],
            [0.5, 0.25, 0.25],
            [0.15, 0.15, 0.7],
        ]
    )
    t, loss = tune_temperature(y, proba)
    assert t >= 0.5
    assert loss < np.inf
    meta = {"calibration": {"temperature": t}}
    calibrated = apply_probability_calibration(proba, meta)
    assert calibrated.shape == proba.shape
