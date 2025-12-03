"""
Download monthly adjusted close prices for selected ETFs
and save them to data/raw/asset_prices_adj_close_2005_2025.csv
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

# Project root path (go up 2 levels: data_scripts -> src -> project)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

# ETFs you want to download
TICKERS = ["XLK", "XLE", "XLF", "GLD", "VNQ"]
START_DATE = "2005-01-01"
END_DATE = None  # None = up to today

def download_monthly_adj_close(tickers, start, end=None):
    """
    Downloads monthly adjusted close prices for the given tickers.
    Returns a DataFrame where columns = tickers, index = Date.
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=False,
        progress=True,
    )

    # If yfinance returns a MultiIndex (Adj Close, XLK), keep only Adj Close
    if isinstance(data.columns, pd.MultiIndex):
        df = data["Adj Close"].copy()
    else:
        df = data.rename(columns={"Adj Close": tickers[0]})

    df.index.name = "Date"
    df = df.sort_index()
    return df

def main():
    print("=" * 60)
    print("Downloading monthly adjusted close prices")
    print("=" * 60)
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Start date: {START_DATE}")
    print(f"End date: {END_DATE or 'today'}\n")

    df_prices = download_monthly_adj_close(TICKERS, START_DATE, END_DATE)

    output_path = DATA_RAW_DIR / "asset_prices_adj_close_2005_2025.csv"
    df_prices.to_csv(output_path)

    print(f"Saved file: {output_path}")
    print(f"Shape: {df_prices.shape}")
    print("\nHead:")
    print(df_prices.head())

if __name__ == "__main__":
    main()
