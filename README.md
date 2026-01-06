# Sector Rotation Strategy: Model Comparison & Rolling Forecasting

## Research Question
Which machine learning model can, and to what extent, correctly predict the monthly stock returns of different sectors based on historical data: Linear Regression, Ridge Regression, or Random Forest?

## Usage
Run the main entry point using Python 3: 
```bash
python3 main.py
```

Expected output:
- Model comparison for Linear Regression, Ridge Regression, Random Forest
- Rolling forecasts: expanding window and 60-month window
- CSV files saved to results/model_comparison.csv and results/rolling_summary.csv

## Data
Raw macroeconomic variables and sector excess returns (2005–2025) are included
in `data/raw/`. The processed dataset is automatically generated and saved to
`data/processed/` when running `main.py`.

## Project Structure
```text
Sector-Rotation-Strategy/
├── main.py                # Main entry point
├── src/                   # Source code
│   ├── data_loader.py     # Data loading
│   ├── preprocessing.py  # Feature engineering & transformations
│   ├── models.py          # Model definitions and training
│   ├── evaluation.py     # Evaluation metrics & backtesting
│   └── data_scripts/      # Data download and preprocessing scripts
├── data/                  # Raw and processed datasets
├── results/               # Output CSV files
├── notebooks/             # Optional exploratory notebooks
├── figures/               # Figures for report
├── requirements.txt       # Project dependencies
└── README.md              # Setup and usage instructions
```
## Results 
```text
Model comparison on the test period (2019–2025) shows:

- Linear Regression  
  - Rotation hit rate: 17.7%  
  - Cumulative excess return vs SPY: −0.08%  
  - Annualized Sharpe (excess): 0.07  

- Ridge Regression
  - Rotation hit rate: 17.7%  
  - Cumulative excess return vs SPY: −3.78%  
  - Annualized Sharpe (excess): 0.03  

- Random Forest  
  - Rotation hit rate: 17.7%  
  - Cumulative excess return vs SPY: −38.63%  
  - Annualized Sharpe (excess): −0.28 

```
Overall, none of the models consistently outperform the SPY benchmark, indicating limited exploitable predictability in sector returns, in line with the market efficiency hypothesis.

## Requirements
- Python 3.10+
- pandas, numpy, scikit-learn, yfinance
