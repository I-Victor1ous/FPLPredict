"""Pick among model strategies: log loss, then accuracy, then simpler models."""

from __future__ import annotations

from fplpredict.config import WF_LOG_LOSS_TIE_EPS

# Lower rank = simpler (preferred when metrics are tied).
MODEL_COMPLEXITY_RANK: dict[str, int] = {
    "lr": 0,
    "svc": 1,
    "rf": 2,
    "hgb": 3,
    "xgb": 4,
    "soft_vote_fixed_4_4_1_1": 5,
    "soft_vote_val_log_loss": 6,
    "soft_vote_val_accuracy": 7,
    "stacking_lr_meta_oof": 8,
}


def model_complexity_rank(name: str) -> int:
    return MODEL_COMPLEXITY_RANK.get(name, 99)


def pick_best_score_row(
    rows: list[dict],
    *,
    log_loss_tie_eps: float | None = None,
) -> dict:
    """
    Choose one row from walk-forward or validation score dicts.

    1. Minimum log loss (within ``log_loss_tie_eps`` of the best).
    2. Highest accuracy among that band.
    3. Lower ``MODEL_COMPLEXITY_RANK`` if still tied.
    """
    if not rows:
        raise ValueError("No candidate scores to compare")
    eps = log_loss_tie_eps if log_loss_tie_eps is not None else WF_LOG_LOSS_TIE_EPS
    best_ll = min(float(r["log_loss"]) for r in rows)
    band = [r for r in rows if float(r["log_loss"]) <= best_ll + eps]
    return max(
        band,
        key=lambda r: (
            float(r["accuracy"]),
            -model_complexity_rank(str(r["model"])),
        ),
    )
