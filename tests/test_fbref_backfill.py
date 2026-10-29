from fplpredict.scrape.fbref_backfill import parse_target_seasons


def test_parse_target_seasons():
    assert parse_target_seasons("2021,2022") == {2021, 2022}
    assert parse_target_seasons(" 2022 ") == {2022}
