from fplpredict.model.selection import pick_best_score_row


def test_pick_best_prefers_accuracy_within_log_loss_band():
    rows = [
        {"model": "lr", "log_loss": 1.0387, "accuracy": 0.42},
        {"model": "stacking_lr_meta_oof", "log_loss": 1.0394, "accuracy": 0.45},
    ]
    best = pick_best_score_row(rows, log_loss_tie_eps=0.005)
    assert best["model"] == "stacking_lr_meta_oof"


def test_pick_best_prefers_simpler_when_accuracy_tied():
    rows = [
        {"model": "stacking_lr_meta_oof", "log_loss": 1.03, "accuracy": 0.42},
        {"model": "lr", "log_loss": 1.031, "accuracy": 0.42},
    ]
    best = pick_best_score_row(rows, log_loss_tie_eps=0.005)
    assert best["model"] == "lr"
