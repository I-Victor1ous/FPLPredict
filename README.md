# FPLPredict

Match outcome predictions (Win / Draw / Loss) for the Premier League, La Liga, Bundesliga, Serie A and Ligue 1. The production pipeline auto-selects features and models (base models, soft-vote ensembles, stacking), re-evaluated about monthly.

## Project layout

See **[docs/REPO_GUIDE.md](docs/REPO_GUIDE.md)** for a file-by-file map.

```
fplpredict/          Python package (scrape, features, model, pipeline)
data/                Match CSVs (matches, history, next_matches)
odds/                Cached football-data.co.uk CSVs (xG source for the current season)
outputs/             predictions-<slug>.csv + predictions-log-<slug>.jsonl (integrity)
artifacts/           Per-league model (predictor.joblib, features.json, model_selection.json)
web/                 Next.js site (reads web/public/data/*.json after export)
scripts/             Weekly, scrape and data-check helpers
```

## Setup

```bash
cd FPLPredict
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Weekly update (manual)

```bash
./scripts/weekly_update.sh                    # scrape + predict + export JSON
./scripts/weekly_update.sh --skip-scrape        # use existing data/ CSVs only
./scripts/weekly_update.sh --force-model-reselect  # ignore monthly model cache
```

Each run scrapes, refits the cached strategy on the latest completed matches, predicts, and exports the site JSON. Feature selection and strategy comparison re-run when older than `FPL_MODEL_RESELECT_DAYS` (default 30).

**Leagues:** `FPL_LEAGUES` (comma-separated names, default: all in `LEAGUE_INFO_URLS`). Each league has `outputs/predictions-<slug>.csv`, model artifacts under `artifacts/<slug>/`, and site data at `web/public/data/<slug>.json`. **Commit `web/public/data/*.json` after export** so Vercel serves every league.

**Scraping:** FBref match logs for every squad, saved append-only to `data/matches.csv` with one row per club per fixture. The weekly scrape walks only the newest stored season. To scrape one league and print per-season club counts, fixtures and xG coverage:

```bash
./scripts/scrape_one_league.sh "Premier League"          # newest season only
./scripts/scrape_one_league.sh "Premier League" --full   # every season in the window (~3h per league)
./scripts/backfill_seasons.sh                          # only 2021–2022 (default) + Understat xG
export FPL_MODEL_DATA_SEASONS=7 FPL_YEAR_DIFF=7        # set before backfill so prune keeps 7 seasons
PYTHONPATH=. python scripts/check_scrape_coverage.py     # coverage report only
```

**Standings:** table positions are computed locally per competition and season from completed matches before each fixture date (points, goal difference, goals for).

**Features:** monthly greedy forward selection picks up to `FPL_FEATURE_SELECTION_MAX` (default 15) features from `CANDIDATE_FEATURES` by validation log loss; the result is saved in `artifacts/<slug>/features.json`.

**Modeling window:** last **7** seasons by default (`FPL_MODEL_DATA_SEASONS`), split **5 train / 1 val / 1 test** by full seasons on **completed matches only**. Strategy choice uses **mean walk-forward log loss** over completed test seasons (`FPL_WF_TRAIN_SEASONS`, `FPL_WF_MIN_TEST_ROWS`); within a log-loss band (`FPL_WF_LOG_LOSS_TIE_EPS`) the winner is **higher accuracy**, then the **simpler** model (`fplpredict/model/selection.py`). Classifiers train with **uniform** 1X2 weights; **draw picks** use a validation-tuned threshold (`t_draw`). **Temperature scaling** (`calibration.temperature` in `model_selection.json`, grid `FPL_TEMPERATURE_GRID`) is fit on the validation season and applied to all exported probabilities. Form features use a **5-match** rolling window (`FPL_ROLLING_FORM_WINDOW`). Bookmaker odds remain for **evaluation baselines** only (not model features). **Test** metrics refresh weekly (out-of-sample refit before the test season). Ensemble includes **HistGradientBoosting**.

**xG:** [football-data.co.uk](https://www.football-data.co.uk/) (`HxG` / `AxG`, cached under `odds/`) covers the current season; past seasons come from [Understat](https://understat.com/) (`--full` scrapes fill these automatically, or run `PYTHONPATH=. python scripts/backfill_understat_xg.py`). Matches without xG are kept; their xG features are just blank. `PYTHONPATH=. python scripts/check_data.py` shows coverage and feature NaN rates.

**Prediction integrity:** `outputs/predictions-log-<slug>.jsonl` append-only log; scored or started fixtures keep their original probabilities in `predictions-<slug>.csv`. Played fixtures without a scraped result yet show on the site as "Awaiting result" for up to 14 days.

**Optional publish:** `./scripts/push_site_data.sh` commits `web/public/data/*.json` after export.

**CI:** GitHub Actions runs `ruff`, `pytest`, and `next build` (see `.github/workflows/ci.yml`).

## Local cron (WSL / Linux)

```bash
./scripts/install_cron.sh
```

Default schedule: **Sunday 20:00**. Logs go to `logs/cron.log`.

### Windows + WSL

Use Task Scheduler to run:

```powershell
wsl.exe -d Ubuntu -e bash -lc "cd /home/ivan/projects/FPLPredict && ./scripts/weekly_update.sh"
```

See `scripts/weekly_update.ps1` for a template.

## Website

After a pipeline run:

```bash
cd web && npm install && npm run dev
```

Visit `http://localhost:3000` (redirects to the default league) or `http://localhost:3000/premier-league`, etc. Per league: **confident picks** (`/picks`), **acca builder** (`/acca`), **evaluation** with rolling backtest table (`/performance`). Fixture cards show **model vs bookmaker** probabilities when odds are available. Deploy: set the Vercel project root to `web/`.

**Rolling backtest:** `PYTHONPATH=. python scripts/rolling_backtest.py` writes `artifacts/<slug>/rolling_backtest.json` (also updated during weekly export when possible).

## Development

- **Scrape only:** `PYTHONPATH=. python -m fplpredict.scrape.fbref [--league NAME] [--full] [--skip-post-process]`
- **Pipeline:** `PYTHONPATH=. python -m fplpredict.pipeline.weekly --skip-scrape`
- **Ensemble comparison:** `PYTHONPATH=. python -c "from fplpredict.model.ensemble import run_comparison_report; run_comparison_report()"`
- **Tests:** `PYTHONPATH=. python -m pytest tests -q`

## Disclaimer

Predictions are for analysis only, not betting advice.
