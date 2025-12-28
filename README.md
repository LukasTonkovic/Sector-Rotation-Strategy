# Sector Rotation Strategy: Model Comparison & Rolling Forecasting

## Research Question
Which machine learning model can, and to what extent, correctly predict the monthly stock returns of different sectors based on historical data: Linear Regression, Ridge Regression, or Random Forest?

## Setup

# Create environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
pip install -r requirements.txt

## Usage
python3 main.py

Expected output:
- Model comparison for Linear Regression, Ridge Regression, Random Forest
- Rolling forecasts: expanding window and 60-month window
- CSV files saved to results/model_comparison.csv and results/rolling_summary.csv

## Project Structure
Sector-Rotation-Strategy/
├── main.py              # Main entry point
├── src/                 # Source code
│   ├── data_loader.py   # Data loading and preprocessing
│   ├── models.py        # Model training and rolling forecasts
│   └── evaluation.py    # Evaluation metrics and rotation strategy
├── data/                # Raw data or download instructions
├── results/             # Output CSV files
├── notebooks/           # Optional exploratory notebooks
├── figures/             # Optional figures for the report
├── requirements.txt     # Project dependencies
└── README.md            # Setup and usage instructions