from pathlib import Path

import pandas as pd

from fplpredict.paths import DATA_DIR


def data_path(filename: str) -> Path:
    """Resolve CSV paths under ``data/`` (canonical location for scrape output)."""
    return DATA_DIR / filename


def read_csv(filename: str) -> pd.DataFrame:
    path = data_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path, index_col=0)


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    df.to_csv(path)
    return path
