"""Write per-league rolling-origin backtest JSON under artifacts/."""

from __future__ import annotations

import argparse
import json

from fplpredict.config import active_leagues
from fplpredict.model.rolling_backtest import write_rolling_backtest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", action="append", dest="leagues")
    args = parser.parse_args()
    for league in args.leagues or active_leagues():
        rows = write_rolling_backtest(league)
        print(json.dumps({league: rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
