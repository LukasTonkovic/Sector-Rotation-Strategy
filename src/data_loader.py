
from pathlib import Path
import pandas as pd

# project root: src/ -> ..
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = BASE_DIR / "data" / "raw"


def load_prices():
    path = DATA_RAW_DIR / "asset_prices_adj_close_2005_2025.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def load_returns():
    path = DATA_RAW_DIR / "returns_2005_2025.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def load_excess_returns():
    path = DATA_RAW_DIR / "excess_returns_2005_2025.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df
