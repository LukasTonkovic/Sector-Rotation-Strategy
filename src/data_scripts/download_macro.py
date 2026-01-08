from pathlib import Path
import pandas as pd
from fredapi import Fred
import os



BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

API_KEY = None

if ENV_PATH.exists():
    with open(ENV_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("FRED_API_KEY="):
                API_KEY = line.split("=", 1)[1].strip()
                break

if not API_KEY:
    raise ValueError(
        f"FRED_API_KEY not found in {ENV_PATH}. "
        "Make sure your .env contains: FRED_API_KEY=yourkey"
    )

fred = Fred(api_key=API_KEY)

# -------------------------------------------------------
# Continue with rest of your script (FRED_SERIES, functions...)
# -------------------------------------------------------

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

START = "2004-01-01"
END = None

FRED_SERIES = {
    "CPIAUCSL": "cpi_index",
    "FEDFUNDS": "fed_funds_rate",
    "DGS10": "yield_10y",
    "DGS2": "yield_2y",
    "UNRATE": "unemployment_rate",
    "INDPRO": "industrial_production",
    "VIXCLS": "vix",
}

def download_series():
    frames = []
    for code, new_name in FRED_SERIES.items():
        print(f"Downloading {code} → {new_name} ...")
        s = fred.get_series(code)
        s = s.rename(new_name)
        frames.append(s)
    df = pd.concat(frames, axis=1)
    
    # Monthly frequency (end-of-month)
    df = df.resample("M").last()
    df.index.name = "Date"
    return df


def add_transforms(df):
    df = df.copy()
    df["cpi_yoy"] = df["cpi_index"].pct_change(12)
    df["term_spread_10y_2y"] = df["yield_10y"] - df["yield_2y"]
    df = df.dropna()
    return df


def main():
    print("=" * 60)
    print("Downloading macro data from FRED")
    print("=" * 60)

    df = download_series()
    df = add_transforms(df)

    output = DATA_RAW_DIR / "macro_2005_2025.csv"
    df.to_csv(output)

    print(f"\nSaved macro data to: {output}")
    print("✓ Done.")


if __name__ == "__main__":
    main()
