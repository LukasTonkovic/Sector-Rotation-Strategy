import yfinance as yf
import pandas as pd

# Tvoji konačni tickersi
TICKERS = ['SPY', 'XLK', 'XLE', 'XLF', 'GLD']

# Skidamo mjesečne podatke od 2005 do 2025
raw = yf.download(
    TICKERS,
    start='2005-01-01',
    end='2025-01-01',
    interval='1mo',
    auto_adjust=False,   # želimo original + Adj Close
    progress=False
)

# Ako je MultiIndex (tipično za yfinance), uzmi samo "Adj Close"
if isinstance(raw.columns, pd.MultiIndex):
    prices = raw['Adj Close']
else:
    prices = raw  # fallback, ali u pravilu ne treba

# Makni redove gdje su SVE vrijednosti NaN (npr. prije GLD početka)
prices = prices.dropna(how='all')

# Filtriraj još jednom period za svaki slučaj
prices = prices.loc['2005-01-01':'2025-01-01', TICKERS]

# Ispiši prvih par redova da vidiš da je ok
print("First rows of adjusted prices:")
print(prices.head())

# Spremi u CSV
output_file = 'asset_prices_adj_close_2005_2025.csv'
prices.to_csv(output_file)

print(f"\n✔ Done! Saved adjusted monthly prices to '{output_file}'")
