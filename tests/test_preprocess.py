import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from fplpredict.model.preprocess import feature_frame, make_preprocessor, transform_features

FEATURES = ["a", "b"]


def _frame(a, b):
    return pd.DataFrame({"a": a, "b": b})


def test_imputer_uses_training_median_and_adds_indicator():
    train = _frame([1.0, 2.0, 3.0, np.nan], [10.0, 10.0, 10.0, 10.0])
    prep = make_preprocessor("median").fit(feature_frame(train, FEATURES))
    imputer = prep.named_steps["impute"]
    assert imputer.statistics_[0] == pytest.approx(2.0)
    # Two features plus one indicator column for the feature that had gaps.
    assert prep.transform(feature_frame(train, FEATURES)).shape == (4, 3)


def test_imputer_ignores_values_seen_only_after_fit():
    train = _frame([1.0, 2.0, 3.0, np.nan], [0.0, 1.0, 2.0, 3.0])
    prep = make_preprocessor("median").fit(feature_frame(train, FEATURES))
    later = _frame([100.0, 200.0], [5.0, 6.0])
    prep.transform(feature_frame(later, FEATURES))
    assert prep.named_steps["impute"].statistics_[0] == pytest.approx(2.0)


def test_upcoming_rows_never_produce_nan():
    train = _frame([1.0, np.nan, 3.0, 4.0], [1.0, 2.0, np.nan, 4.0])
    prep = make_preprocessor("median").fit(feature_frame(train, FEATURES))
    upcoming = pd.DataFrame({"a": [np.nan, pd.NA], "b": ["2.5", None]}, dtype=object)
    out = transform_features(prep, upcoming, FEATURES)
    assert np.isfinite(out).all()


def test_zero_mode_matches_old_fillna_behaviour():
    train = _frame([1.0, np.nan, 3.0], [2.0, 2.0, np.nan])
    new = make_preprocessor("zero").fit_transform(feature_frame(train, FEATURES))
    old = StandardScaler().fit_transform(train.fillna(0))
    np.testing.assert_allclose(new, old)


def test_legacy_standard_scaler_artifacts_still_transform():
    train = _frame([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    legacy = StandardScaler().fit(train)
    out = transform_features(legacy, _frame([np.nan], [5.0]), FEATURES)
    assert np.isfinite(out).all()
