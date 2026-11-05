# Repository guide

End-to-end flow: **scrape FBref → enrich xG → train/refit models → predict → export JSON → Next.js site**.

```
FBref scrape ──► data/*.csv ──► features + model ──► outputs/predictions-*.csv
       │                              │
       └── football-data / Understat ─┘          └──► web/public/data/*.json
```

## Root files

| Path | Role |
|------|------|
| `README.md` | Setup, weekly commands, modeling summary |
| `requirements.txt` | Python dependencies |
| `pyproject.toml` | Ruff + pytest configuration |
| `.github/workflows/ci.yml` | CI: ruff, pytest, `next build` |

## `fplpredict/` — Python package

### Entry

| File | Role |
|------|------|
| `__main__.py` | `python -m fplpredict` → weekly pipeline |
| `__init__.py` | Package marker |
| `config.py` | Env-tunable constants: seasons, features, leagues, walk-forward |
| `paths.py` | Paths for data, outputs, artifacts, web JSON slugs |
| `data_io.py` | Read/write CSVs under `data/` |
| `teams.py` | Club name normalization + `TEAM_ALIASES` |
| `export.py` | Build `web/public/data/<league>.json` from predictions + matches |
| `performance_export.py` | Gameweek performance + model strategy timeline for the site |
| `predictions_util.py` | Prediction CSV hygiene: keys, dedupe, `Actual` backfill |
| `predictions_log.py` | Append-only JSONL audit log; merge integrity; restore from log |
| `predictions_store.py` | One-time migration from legacy prediction file names |

### `fplpredict/pipeline/`

| File | Role |
|------|------|
| `weekly.py` | Main orchestrator: scrape (optional), enrich, QA, predict all leagues, export |
| `status.py` | `pipeline_status.json`, weekly lock file |

### `fplpredict/scrape/`

| File | Role |
|------|------|
| `fbref.py` | Selenium FBref scrape: past results → `matches.csv`, next fixture → `next_matches.csv`, cups → `history.csv` |
| `fbref_backfill.py` | CLI to backfill specific season years per league |
| `team_logs.py` | Parse team fixture tables, merge optional match stats |

### `fplpredict/data/`

| File | Role |
|------|------|
| `matches_clean.py` | Filter leagues, dedupe, mirror missing team perspectives, xG propagation |
| `football_data.py` | Download/cache football-data.co.uk CSVs; xG + result backfill |
| `understat.py` | Historical xG from Understat API |
| `xg_enrich.py` | Orchestrate Understat + football-data enrichment on `matches.csv` |
| `validate.py` | Post-scrape data quality checks (coverage, NaNs, fixtures) |

### `fplpredict/model/`

| File | Role |
|------|------|
| `features.py` | Rolling form, H2H, standings join → modeling matrix (past + future rows) |
| `standings.py` | Table position as-of each fixture date |
| `splits.py` | Season windows; train/val/test holdout by season |
| `preprocess.py` | Imputation + scaling pipeline for sklearn |
| `feature_selection.py` | Greedy forward feature selection on validation log loss |
| `ensemble.py` | Candidate models, soft vote, stacking; fit/save artifacts; weekly refit |
| `walkforward_selection.py` | Multi-season walk-forward strategy comparison |
| `selection.py` | Shared winner rule: log loss band → accuracy → simpler model |
| `predict.py` | Per-league prediction pipeline → `outputs/predictions-<slug>.csv` |
| `calibration.py` | Temperature scaling; feature pool without market columns |
| `draw_tuning.py` | Validation-tuned draw probability threshold |
| `metrics.py` | Log loss, RPS, Brier, calibration bins |
| `market.py` | Bookmaker implied probs (evaluation/baselines, export only) |
| `sample_weights.py` | Uniform training weights |
| `rolling_backtest.py` | Optional rolling accuracy JSON for performance UI |

## `data/` — Source tables (git-tracked)

| File | Role |
|------|------|
| `matches.csv` | Completed league fixtures (one row per club per match) |
| `next_matches.csv` | Next scheduled fixture per team (FBref) |
| `history.csv` | Non-league / cup rows used for H2H features |
| `.understat_xg_backfilled` | Flag file: Understat historical pass done |

## `odds/` — Cached downloads

football-data.co.uk season CSVs (`{year}_{div}.csv`), `fixtures.csv`, and optional `market/` odds copies. Used for xG (current season), result backfill, and market baselines.

## `outputs/` — Predictions

| Pattern | Role |
|---------|------|
| `predictions-<slug>.csv` | Live prediction history per league |
| `predictions-log-<slug>.jsonl` | Immutable first prediction per fixture (integrity) |

## `artifacts/<slug>/` — Per-league models

| File | Role |
|------|------|
| `predictor.joblib` | Fitted voter / stacker |
| `scaler.joblib` | Preprocessing pipeline |
| `features.json` | Selected feature list |
| `model_selection.json` | Strategy, metrics, calibration, walk-forward scores |
| `rolling_backtest.json` | Optional backtest series for charts |

Legacy `artifacts/model_selection.json` is Premier League only (pre multi-league).

## `scripts/` — Ops helpers

| Script | Role |
|--------|------|
| `weekly_update.sh` | Cron-friendly wrapper for `python -m fplpredict` |
| `weekly_update.ps1` | Windows Task Scheduler template |
| `install_cron.sh` | Install Sunday cron (user may change schedule) |
| `scrape_one_league.sh` | Single-league FBref scrape + coverage |
| `backfill_seasons.sh` | Historical season backfill + optional model update |
| `wait_backfill_and_update_models.sh` | Poll backfill log then run weekly with reselect |
| `backfill_understat_xg.py` | One-shot Understat xG for old seasons |
| `check_data.py` | Human-readable data/xG/feature NaN report |
| `check_scrape_coverage.py` | Season/club/xG coverage table |
| `compare_imputation.py` | Ad-hoc zero vs median imputation comparison |
| `rolling_backtest.py` | CLI to refresh rolling backtest JSON |
| `push_site_data.sh` | Git commit exported JSON (optional publish) |

## `tests/` — Pytest

Unit and smoke tests mirroring modules above. `tests/contract.py` holds shared export shape checks.

## `web/` — Next.js frontend

| Path | Role |
|------|------|
| `app/page.tsx` | League hub |
| `app/[league]/page.tsx` | Upcoming + awaiting result fixtures |
| `app/[league]/picks/page.tsx` | High-confidence picks |
| `app/[league]/acca/page.tsx` | Acca builder |
| `app/[league]/performance/page.tsx` | Accuracy / strategy charts |
| `lib/data.ts`, `lib/types.ts`, `lib/fixtures.ts` | Load `public/data/*.json` |
| `components/*` | UI: cards, charts, theme, nav |
| `public/data/*.json` | Generated site payloads (commit after export) |

## `logs/` (gitignored)

Weekly run logs and `weekly.lock` from the pipeline.
