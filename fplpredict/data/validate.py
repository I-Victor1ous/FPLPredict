"""Data quality checks after scrape / xG enrichment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from fplpredict.config import LEAGUE_INFO_URLS, MODEL_FEATURES
from fplpredict.data_io import read_csv
from fplpredict.model.features import build_feature_matrix

log = logging.getLogger(__name__)

MAX_FEATURE_MISSING_RATE = 0.25
MAX_XG_MISSING_RATE = 0.35
# football-data.co.uk only publishes HxG/AxG for the current season file.
MAX_XG_MISSING_RATE_HISTORICAL = 0.90
MAX_DAYS_SINCE_COMPLETED = 10
# June and July have no league fixtures, so stale data is expected then.
OFF_SEASON_MONTHS = {6, 7}


def stale_data_warning(newest_completed: date | None, today: date | None = None) -> str | None:
    today = today or date.today()
    if today.month in OFF_SEASON_MONTHS:
        return None
    if newest_completed is None:
        return "No completed matches found."
    age = (today - newest_completed).days
    if age > MAX_DAYS_SINCE_COMPLETED:
        return (
            f"Newest completed match is {age} days old ({newest_completed.isoformat()}); "
            "the scrape may be stale."
        )
    return None


@dataclass
class DataQualityReport:
    league: str
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues


def _league_matches_raw(league: str) -> pd.DataFrame:
    m = read_csv("matches.csv")
    return m[m["Comp"] == league].copy()


def check_global_csvs(leagues: list[str]) -> list[str]:
    """Return hard errors if core CSVs are missing or empty for active leagues."""
    issues: list[str] = []
    try:
        matches = read_csv("matches.csv")
    except FileNotFoundError:
        return ["matches.csv is missing."]
    if matches.empty:
        issues.append("matches.csv has no rows.")
    try:
        nextm = read_csv("next_matches.csv")
    except FileNotFoundError:
        issues.append("next_matches.csv is missing.")
        nextm = pd.DataFrame()
    try:
        hist = read_csv("history.csv")
        if hist.empty:
            issues.append("history.csv has no rows (h2h features will be empty).")
    except FileNotFoundError:
        issues.append("history.csv is missing (h2h features will be empty).")
    for league in leagues:
        sub = matches[matches["Comp"] == league] if "Comp" in matches.columns else matches
        if sub.empty:
            issues.append(f"No match rows for league: {league}.")
        if not nextm.empty and "Comp" in nextm.columns:
            up = nextm[nextm["Comp"] == league]
            if up.empty:
                issues.append(
                    f"No upcoming fixtures in next_matches.csv for {league}."
                )
    return issues


def check_league_raw(league: str) -> DataQualityReport:
    report = DataQualityReport(league=league)
    sub = _league_matches_raw(league)
    if sub.empty:
        report.issues.append("No rows in matches.csv for this league.")
        return report
    for col in ("xG", "xGA"):
        if col not in sub.columns:
            report.issues.append(f"Column {col} missing from matches.csv.")
            continue
        rate = float(sub[col].isna().mean())
        report.stats[f"{col}_missing_rate"] = round(rate, 4)
        hist_thresh = MAX_XG_MISSING_RATE_HISTORICAL
        if rate > hist_thresh:
            report.warnings.append(
                f"{col} missing on {rate:.0%} of league rows (historical threshold {hist_thresh:.0%})."
            )
    completed = sub
    if "Result" in sub.columns:
        completed = sub[sub["Result"].notna()]
    newest = pd.to_datetime(completed["Date"], errors="coerce").max() if not completed.empty else None
    newest_date = None if newest is None or pd.isna(newest) else newest.date()
    report.stats["newest_completed_date"] = newest_date.isoformat() if newest_date else None
    stale = stale_data_warning(newest_date)
    if stale:
        report.warnings.append(stale)
    if not completed.empty and "xG" in completed.columns:
        r_all = float(completed["xG"].isna().mean())
        report.stats["completed_xG_missing_rate"] = round(r_all, 4)
        if r_all > 0:
            report.warnings.append(
                f"Completed league matches without football-data xG: {r_all:.2%} "
                "(kept; xG features are blank for them)."
            )
    if "Season" in sub.columns:
        seasons = pd.to_numeric(sub["Season"], errors="coerce")
        recent = sub[seasons == seasons.max()]
        if "Result" in recent.columns:
            recent = recent[recent["Result"].notna()]
        if not recent.empty and "xG" in recent.columns:
            r = float(recent["xG"].isna().mean())
            report.stats["latest_season_xG_missing_rate"] = round(r, 4)
            if r > MAX_XG_MISSING_RATE:
                report.warnings.append(
                    f"Latest season completed xG missing rate {r:.0%} exceeds "
                    f"{MAX_XG_MISSING_RATE:.0%}."
                )
        report.warnings.extend(_season_coverage_warnings(league, sub))
    return report


def _season_coverage_warnings(league: str, sub: pd.DataFrame) -> list[str]:
    """Club count per season and fixtures where only one club's row exists."""
    from fplpredict.data.matches_clean import expected_club_count

    warnings: list[str] = []
    for season, grp in sub.groupby("Season", dropna=True):
        expected = expected_club_count(league, season)
        clubs = set(grp["Team"].dropna()) | set(grp["Opponent"].dropna())
        if len(clubs) != expected:
            warnings.append(
                f"Season {season}: {len(clubs)} clubs in fixtures (expected {expected})."
            )
        rows = set(zip(grp["Date"], grp["Team"], grp["Opponent"]))
        one_sided = sum(1 for d, t, o in rows if (d, o, t) not in rows)
        if one_sided:
            warnings.append(
                f"Season {season}: {one_sided} fixtures have only one club's row."
            )
    return warnings


def check_league_features(league: str) -> DataQualityReport:
    report = DataQualityReport(league=league)
    matches = read_csv("matches.csv")
    nextm = read_csv("next_matches.csv")
    try:
        hist = read_csv("history.csv")
    except FileNotFoundError:
        hist = pd.DataFrame()
    past, future, _ = build_feature_matrix(matches, nextm, hist, league=league)
    from fplpredict.model.ensemble import selected_features

    features = selected_features(league, list(past.columns) + MODEL_FEATURES)
    missing_feats = [f for f in features if f not in past.columns]
    if missing_feats:
        report.issues.append(f"Model features missing from matrix: {missing_feats}")
        return report
    for feat in features:
        rate = float(past[feat].isna().mean())
        report.stats[f"feat_{feat}_missing"] = round(rate, 4)
        if rate > MAX_FEATURE_MISSING_RATE:
            report.warnings.append(f"Feature {feat} is {rate:.0%} NaN in past matches.")
    report.stats["past_rows"] = len(past)
    report.stats["future_rows"] = len(future)
    return report


def run_data_quality_checks(
    leagues: list[str] | None = None,
    *,
    strict: bool = False,
) -> list[DataQualityReport]:
    leagues = leagues or list(LEAGUE_INFO_URLS.keys())
    reports: list[DataQualityReport] = []
    for msg in check_global_csvs(leagues):
        log.error("[global] %s", msg)
        if strict:
            raise RuntimeError(f"Data quality failed: {msg}")
    for league in leagues:
        raw = check_league_raw(league)
        feat = check_league_features(league)
        merged = DataQualityReport(
            league=league,
            issues=raw.issues + feat.issues,
            warnings=raw.warnings + feat.warnings,
            stats={**raw.stats, **feat.stats},
        )
        reports.append(merged)
        for msg in merged.warnings:
            log.warning("[%s] %s", league, msg)
        for msg in merged.issues:
            log.error("[%s] %s", league, msg)
        if strict and merged.issues:
            raise RuntimeError(f"Data quality failed for {league}: {merged.issues}")
    return reports
