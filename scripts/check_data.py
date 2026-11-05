#!/usr/bin/env python3
"""Run data quality checks (xG coverage, model feature NaN rates)."""

from __future__ import annotations

import logging
import sys

from fplpredict.config import active_leagues
from fplpredict.data.validate import run_data_quality_checks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    strict = "--strict" in sys.argv
    reports = run_data_quality_checks(active_leagues(), strict=strict)
    failed = [r for r in reports if not r.ok]
    for r in reports:
        print(f"\n{r.league}: {'OK' if r.ok else 'FAIL'}")
        for k, v in sorted(r.stats.items()):
            print(f"  {k}: {v}")
        for w in r.warnings:
            print(f"  WARN: {w}")
        for i in r.issues:
            print(f"  ISSUE: {i}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
