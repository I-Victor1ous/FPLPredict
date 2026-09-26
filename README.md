# FPLPredict

Premier League match outcome predictions (Win / Draw / Loss). The production pipeline auto-selects among base models, soft-vote ensembles, and stacking (re-evaluated about monthly).

## Project layout

```
fplpredict/          Python package (scrape, features, model, pipeline)
data/                Match CSVs (matches, history, next_matches)
outputs/             predictions.csv + predictions.json
artifacts/           Selected model (predictor.joblib, model_selection.json)
web/                 Static site (reads public/data/predictions.json)
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
cd web && python3 -m http.server 8080
```

Visit `http://localhost:8080`. The header tagline shows the active model from `artifacts/model_selection.json` (via exported JSON).

## Development

- **Scrape only:** `PYTHONPATH=. python -c "from fplpredict.scrape.fbref import run_scrape; run_scrape()"`
- **Pipeline:** `PYTHONPATH=. python -m fplpredict.pipeline.weekly --skip-scrape`
- **Ensemble comparison:** `PYTHONPATH=. python tests/test_ensemble_stacking.py --val-cutoffs 2024-03-14`
- **Notebook:** `model.ipynb` for exploration

## Disclaimer

Predictions are for analysis only, not betting advice.
