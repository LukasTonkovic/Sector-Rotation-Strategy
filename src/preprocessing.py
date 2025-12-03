"""
Preprocessing utilities for building the ML dataset.

- Load macro data (monthly)
- Load excess returns (SPY-relative)
- Align by month
- Create next-month targets
- Save final dataset to data/processed/rotation_dataset.csv
"""

from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]  # src -> project root
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROC_DIR = BASE_DIR / "data" / "processed"
DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_macro():
    """Load macro data from CSV and set monthly PeriodIndex."""
    path = DATA_RAW_DIR / "macro_2005_2025.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    # convert to monthly period index (e.g. 2005-01, 2005-02, ...)
    df.index = df.index.to_period("M")
    return df


def load_excess_returns():
    """Load SPY-relative excess returns and set monthly PeriodIndex."""
    path = DATA_RAW_DIR / "excess_returns_2005_2025.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    # convert to monthly period index
    df.index = df.index.to_period("M")
    return df


def build_dataset():
    """
    Build the ML dataset with:
    - features X_t: macro variables at month t
    - targets y_t: next-month excess returns (t+1) for each asset

    Returns:
        dataset (DataFrame): combined features + target columns
    """
    macro = load_macro()
    excess = load_excess_returns()

    # 1) Keep only months that exist in BOTH macro and excess returns
    common_periods = macro.index.intersection(excess.index)
    macro = macro.loc[common_periods].copy()
    excess = excess.loc[common_periods].copy()

    # 2) Create next-month targets:
    #    row for month t will have target = excess return at month t+1
    targets_next = excess.shift(-1)

    # 3) Add prefix to target columns (to separate them from features)
    targets_next = targets_next.add_prefix("target_")

    # 4) Combine features (macro) and targets (next-month excess)
    dataset = pd.concat([macro, targets_next], axis=1)

    # 5) Drop rows with missing values (last month has NaN targets)
    dataset = dataset.dropna()

    # 6) Save to processed folder
    output_path = DATA_PROC_DIR / "rotation_dataset.csv"
    dataset.to_csv(output_path, index_label="DatePeriod")

    return dataset


def load_features_targets():
    """
    Convenience function:
    - builds dataset
    - returns X (features) and y (targets) separately.
    """
    dataset = build_dataset()

    # Features = all non-target columns (macro variables)
    feature_cols = [c for c in dataset.columns if not c.startswith("target_")]
    target_cols = [c for c in dataset.columns if c.startswith("target_")]

    X = dataset[feature_cols].copy()
    y = dataset[target_cols].copy()

    return X, y


if __name__ == "__main__":
    # Quick manual test
    ds = build_dataset()
    print("Dataset shape:", ds.shape)
    print("Columns:", ds.columns.tolist()[:10], "...")
    print(ds.head())
