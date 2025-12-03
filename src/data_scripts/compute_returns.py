"""
Compute monthly returns and SPY-relative excess returns
from monthly adjusted close prices.

Input:
    data/raw/asset_prices_adj_close_2005_2025.csv
        - columns: XLK, XLE, XLF, GLD, VNQ
        - index: Date (monthly)

Output:
    data/raw/returns_2005_2025.csv
        - columns: SPY, XLK, XLE, XLF, GLD, VNQ
        - index: Date (monthly returns)

    data/raw/excess_returns_2005_2025.csv
        - columns: SPY, XLK, XLE, XLF, GLD, VNQ
        - index: Date
        - SPY column is 0 (benchmark), others are excess vs SPY
"""

from pathlib import Path
import pandas as pd
import yfinance as yf


# Paths
BASE_DIR = Path(__file__).resolve().parents[2]  # src/data_scripts -> src -> project root
DATA_RAW_DIR = BASE_DIR / "data" / "raw"


PRICES_FILE = DATA_RAW_DIR / "asset_prices_adj_close_2005_2025.csv"
RETURNS_FILE = DATA_RAW_DIR / "returns_2005_2025.csv"
EXCESS_FILE = DATA_RAW_DIR / "excess_returns_2005_2025.csv"


def load_prices():
    """Load ETF adjusted close prices from CSV."""
    df = pd.read_csv(PRICES_FILE, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def download_spy_monthly(start_date, end_date=None):
    """
    Download monthly adjusted close prices for SPY
    for a given date range.
    """
    data = yf.download(
        tickers=["SPY"],
        start=start_date,
        end=end_date,
        interval="1mo",
        auto_adjust=False,
        progress=True,
    )

    spy = data["Adj Close"].copy()
    spy.name = "SPY"
    spy.index.name = "Date"
    spy = spy.sort_index()
    return spy


def compute_returns(prices, spy_prices):
    """
    Given ETF prices and SPY prices (both with Date index),
    compute monthly returns and excess returns vs SPY.
    """
    # Combine ETFs and SPY into same DataFrame
    df = prices.copy()
    df["SPY"] = spy_prices

    # Sort and align by dates, keep only dates where we have all columns
    df = df.dropna().sort_index()

    # Simple monthly returns: (P_t / P_{t-1}) - 1
    returns = df.pct_change().dropna()

    # Extract SPY returns
    spy_ret = returns["SPY"]

    # Excess returns: asset_ret - spy_ret
    excess = returns.drop(columns=["SPY"]).sub(spy_ret, axis=0)

    # Optionally add SPY excess = 0
    excess["SPY"] = 0.0

    # Reorder columns so SPY is first
    cols = ["SPY"] + [c for c in excess.columns if c != "SPY"]
    excess = excess[cols]

    return returns, excess


def main():
    print("=" * 70)
    print("Computing monthly returns and SPY-relative excess returns")
    print("=" * 70)

    # 1) Load ETF prices
    prices = load_prices()
    print(f"Loaded ETF prices: {prices.shape}")
    print(f"Columns: {list(prices.columns)}")
    print(f"Date range: {prices.index.min().date()} → {prices.index.max().date()}")

    # 2) Download SPY prices for the same period
    start = prices.index.min().strftime("%Y-%m-%d")
    # add one extra month to ensure coverage
    end = (prices.index.max() + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")

    print(f"\nDownloading SPY prices from {start} to {end}...")
    spy_prices = download_spy_monthly(start, end)

    print(f"SPY prices: {spy_prices.shape}")
    print(f"SPY date range: {spy_prices.index.min().date()} → {spy_prices.index.max().date()}")

    # 3) Compute returns and excess returns
    returns, excess = compute_returns(prices, spy_prices)

    print("\nReturns (first few rows):")
    print(returns.head())
    print("\nExcess returns (first few rows):")
    print(excess.head())

    # 4) Save to CSV
    returns.to_csv(RETURNS_FILE)
    excess.to_csv(EXCESS_FILE)

    print(f"\nSaved returns to: {RETURNS_FILE}")
    print(f"Saved excess returns to: {EXCESS_FILE}")
    print("\n✓ Done.")


if __name__ == "__main__":
    main()
