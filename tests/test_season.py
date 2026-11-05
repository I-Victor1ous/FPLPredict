from datetime import date

from fplpredict.config import MODEL_DATA_SEASONS, current_season_end, model_season_window


def test_season_rolls_over_on_first_of_july():
    assert current_season_end(date(2027, 6, 30)) == 2027
    assert current_season_end(date(2027, 7, 1)) == 2028


def test_mid_season_dates_map_to_end_year():
    assert current_season_end(date(2026, 10, 14)) == 2027
    assert current_season_end(date(2027, 2, 1)) == 2027


def test_model_window_ends_at_current_season():
    window = model_season_window(date(2027, 2, 1))
    assert window[-1] == 2027
    assert len(window) == MODEL_DATA_SEASONS
    assert window == sorted(window)
