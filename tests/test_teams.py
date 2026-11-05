from fplpredict.teams import normalize_team_name


def test_koln_alias():
    assert normalize_team_name("Koln") == "Köln"


def test_nottingham_aliases():
    assert normalize_team_name("Nott'ham Forest") == "Nottingham Forest"


def test_la_liga_accent_aliases():
    assert normalize_team_name("Alavés") == "Alaves"
    assert normalize_team_name("Atlético Madrid") == "Atletico Madrid"
    assert normalize_team_name("Málaga") == "Malaga"
    assert normalize_team_name("Dep. A Coruña") == "Dep. La Coruña"
    assert normalize_team_name("Deportivo A Coruna") == "Dep. La Coruña"
    assert normalize_team_name("Racing Sant") == "Racing Santander"


def test_bundesliga_frankfurt_alias():
    assert normalize_team_name("Frankfurt") == "Eintracht Frankfurt"


def test_serie_a_inter_alias():
    assert normalize_team_name("Internazionale") == "Inter"


def test_backfill_aliases():
    assert normalize_team_name("West Brom") == "West Bromwich Albion"
    assert normalize_team_name("Arminia") == "Arminia Bielefeld"
    assert normalize_team_name("Nimes") == "Nîmes"
    assert normalize_team_name("Greuther Furth") == "Greuther Fürth"
    assert normalize_team_name("Greuther Fuerth") == "Greuther Fürth"
    assert normalize_team_name("Leicester") == "Leicester City"
    assert normalize_team_name("SD Huesca") == "Huesca"
