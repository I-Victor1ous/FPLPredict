#!/usr/bin/env python3
"""Fill missing xG from Understat (past seasons; football-data only has xG from 2026-27)."""

from __future__ import annotations

from fplpredict.config import active_leagues
from fplpredict.data.validate import run_data_quality_checks
from fplpredict.data.xg_enrich import enrich_csv_files


def main() -> int:
    stats = enrich_csv_files(force_understat=True)
    print(
        f"Understat filled {stats.get('understat_filled', 0)} rows; "
        f"xG missing {stats['xG_missing_rate_before']:.1%} → "
        f"{stats['xG_missing_rate_after']:.1%}."
    )
    reports = run_data_quality_checks(active_leagues(), strict=False)
    failed = [r for r in reports if not r.ok]
    if failed:
        for r in failed:
            print(f"WARN {r.league}: {r.issues}")
        return 1
    print("Data checks OK for all leagues.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
