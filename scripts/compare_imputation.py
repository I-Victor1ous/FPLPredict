"""Compare validation log loss with zero-fill vs median imputation (no artifacts saved).

Usage: PYTHONPATH=. python scripts/compare_imputation.py [--league NAME ...]
"""

from __future__ import annotations

import argparse
import json
import warnings

from fplpredict.config import CANDIDATE_FEATURES, active_leagues
from fplpredict.model import preprocess
from fplpredict.model.ensemble import (
    _best_val_row,
    build_eval_split,
    compare_candidates,
    load_past_matches,
    select_features_for_league,
)
from fplpredict.model.splits import completed_past_matches, restrict_last_n_seasons

MODES = ("zero", "median")


def evaluate(league: str, mode: str) -> dict:
    preprocess.IMPUTE_STRATEGY = mode
    modeling = restrict_last_n_seasons(completed_past_matches(load_past_matches(league)))
    candidates = [f for f in CANDIDATE_FEATURES if f in modeling.columns]
    features, _ = select_features_for_league(modeling, candidates, league)
    comparison = compare_candidates(build_eval_split(modeling, features))
    best = _best_val_row(comparison.val_scores)
    return {
        "best_model": best["model"],
        "val_log_loss": round(best["log_loss"], 4),
        "val_accuracy": round(best["accuracy"], 4),
        "n_features": len(features),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", action="append", dest="leagues")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    results = {}
    for league in args.leagues or active_leagues():
        results[league] = {mode: evaluate(league, mode) for mode in MODES}
        print(json.dumps({league: results[league]}), flush=True)
    deltas = [r["median"]["val_log_loss"] - r["zero"]["val_log_loss"] for r in results.values()]
    deltas.sort()
    print(f"Median change in val log loss (median - zero): {deltas[len(deltas) // 2]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
