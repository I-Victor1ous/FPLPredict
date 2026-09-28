# FPLPredict

Premier League match outcome predictions (Win / Draw / Loss). The production pipeline auto-selects among base models, soft-vote ensembles, and stacking (re-evaluated about monthly).

## Project layout

```
fplpredict/          Python package (scrape, features, model, pipeline)
data/                Match CSVs (matches, history, next_matches)
outputs/             predictions-<league-slug>.csv (one per league)
artifacts/           Selected model (predictor.joblib, model_selection.json)
web/                 Next.js site (reads web/public/data/*.json after export)
scripts/             Weekly cron helpers
model.ipynb          Experiments and ensemble comparison notes
```

## Setup

```bash
cd FPLPredict
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.json config.json   # optional; FBref scrape does not require API key
```

## Weekly update (manual)

```bash
./scripts/weekly_update.sh                    # scrape + predict + export JSON
./scripts/weekly_update.sh --skip-scrape        # use existing data/ CSVs only
./scripts/weekly_update.sh --force-model-reselect  # ignore monthly model cache
```

Each run scrapes, predicts, and exports the site JSON. Model strategy is re-compared when older than `FPL_MODEL_RESELECT_DAYS` (default 30).

**Leagues:** `FPL_LEAGUES` (comma-separated names, default: all in `LEAGUE_INFO_URLS` — Premier League, La Liga, Bundesliga, Serie A, Ligue 1). Each league has `outputs/predictions-<slug>.csv`, model artifacts under `artifacts/<slug>/`, and site data at `web/public/data/<slug>.json`. **Commit `web/public/data/*.json` after export** so Vercel serves every league.

**Modeling window:** last 5 seasons (`FPL_MODEL_DATA_SEASONS`), split **3 train / 1 val / 1 test** by full seasons on **completed matches only** (`FPL_TRAIN_SEASONS`, `FPL_VAL_SEASONS`, `FPL_TEST_SEASONS`). Scrape depth follows `FPL_YEAR_DIFF` (default 5).

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

Visit `http://localhost:3000` (redirects to the default league) or `http://localhost:3000/premier-league`, etc. League routes feed Vercel Web Analytics automatically via `@vercel/analytics/react`. Enable Web Analytics in the Vercel project dashboard. Deploy: set the Vercel project root to `web/`. The header tagline shows the active model from exported JSON.

## Development

- **Scrape only:** `PYTHONPATH=. python -c "from fplpredict.scrape.fbref import run_scrape; run_scrape()"`
- **Pipeline:** `PYTHONPATH=. python -m fplpredict.pipeline.weekly --skip-scrape`
- **Ensemble comparison:** `PYTHONPATH=. python -c "from fplpredict.model.ensemble import run_comparison_report; run_comparison_report()"`
- **Season splits:** `python -m pytest tests/test_season_splits.py`
- **Notebook:** `model.ipynb` for exploration

## Disclaimer

Predictions are for analysis only, not betting advice.
