

from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]  # src
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROC_DIR = BASE_DIR / "data" / "processed"
DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_macro():
    
    path = DATA_RAW_DIR / "macro_2005_2025.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    # convert to monthly period index 
    df.index = df.index.to_period("M")
    return df


def load_excess_returns():
    
    path = DATA_RAW_DIR / "excess_returns_2005_2025.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    # convert to monthly period index
    df.index = df.index.to_period("M")
    return df


def build_dataset():
    
    macro = load_macro()
    excess = load_excess_returns()

    common_periods = macro.index.intersection(excess.index)
    macro = macro.loc[common_periods].copy()
    excess = excess.loc[common_periods].copy()

    
    targets_next = excess.shift(-1)

    
    targets_next = targets_next.add_prefix("target_")

   
    dataset = pd.concat([macro, targets_next], axis=1)

    
    dataset = dataset.dropna()

    
    output_path = DATA_PROC_DIR / "rotation_dataset.csv"
    dataset.to_csv(output_path, index_label="DatePeriod")

    return dataset


def load_features_targets():

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
