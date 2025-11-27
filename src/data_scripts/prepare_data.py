import pandas as pd

# 1. Učitaj cijene
prices = pd.read_csv('asset_prices_adj_close_2005_2025.csv',
                     index_col=0, parse_dates=True)

print("Raw prices:")
print(prices.head())

# 2. Izračunaj mjesečne prinose
returns = prices.pct_change().dropna()
print("\nMonthly returns:")
print(returns.head())

# 3. Izračunaj excess returne VS SPY
excess_returns = returns.sub(returns['SPY'], axis=0)
print("\nExcess returns vs SPY:")
print(excess_returns.head())

# 4. Spremi rezultate u CSV
returns.to_csv('returns_2005_2025.csv')
excess_returns.to_csv('excess_returns_2005_2025.csv')

print("\n✔ Saved 'returns_2005_2025.csv' and 'excess_returns_2005_2025.csv'")
