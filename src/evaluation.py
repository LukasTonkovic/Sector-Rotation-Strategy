"""
Evaluation utilities for the Sector Rotation project.

- Hit rate: how often the model picks the best-performing asset
- Rotation strategy backtest: monthly rotation into the asset with highest predicted excess return
- Sharpe ratio and cumulative excess return of the strategy
"""

from typing import Dict

import numpy as np
import pandas as pd


def _get_trade_assets(columns, exclude_spy: bool = True):
    """
    Helper: which target columns we trade on.
    Usually we exclude SPY (it is the benchmark).
    """
    if exclude_spy:
        return [c for c in columns if c != "target_SPY"]
    return list(columns)


def compute_hit_rate(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    exclude_spy: bool = True,
) -> float:
    """
    Compute the fraction of months where the model correctly
    predicts the best-performing asset (among traded assets).

    y_true: DataFrame with columns = target_* and index = months
    y_pred: ndarray of same shape as y_true.values
    """
    trade_cols = _get_trade_assets(list(y_true.columns), exclude_spy=exclude_spy)

    # Restrict to traded assets
    true_vals = y_true[trade_cols].to_numpy()
    pred_vals = y_pred[:, [y_true.columns.get_loc(c) for c in trade_cols]]

    # For each month: index of best predicted asset and best actual asset
    pred_winner_idx = np.argmax(pred_vals, axis=1)
    true_winner_idx = np.argmax(true_vals, axis=1)

    hits = (pred_winner_idx == true_winner_idx)
    hit_rate = hits.mean()

    return float(hit_rate)


def backtest_rotation_strategy(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    exclude_spy: bool = True,
    freq: str = "M",
) -> Dict[str, object]:
    """
    Backtest a simple rotation strategy:

    - Each month, pick the asset with the highest predicted excess return
      (among traded assets, usually GLD, VNQ, XLE, XLF, XLK).
    - Realized strategy return for that month = actual excess return of that asset.

    Returns:
        dict with:
            - "strategy_returns": Series of monthly excess returns
            - "cumulative_excess_return": float
            - "annualized_sharpe": float
            - "hit_rate": float
            - "selection": DataFrame with chosen assets per month
    """
    trade_cols = _get_trade_assets(list(y_true.columns), exclude_spy=exclude_spy)

    # Subset arrays: only traded assets (without SPY)
    true_vals = y_true[trade_cols].to_numpy()
    pred_vals = y_pred[:, [y_true.columns.get_loc(c) for c in trade_cols]]

    # Index of predicted winner each month
    winner_idx = np.argmax(pred_vals, axis=1)

    n_periods = true_vals.shape[0]
    realized = np.empty(n_periods)
    chosen_assets = []

    for t in range(n_periods):
        j = winner_idx[t]
        realized[t] = true_vals[t, j]  # actual excess return of chosen asset
        chosen_assets.append(trade_cols[j])

    strategy_returns = pd.Series(
        realized,
        index=y_true.index,
        name="strategy_excess_return",
    )

    # Cumulative excess return vs SPY
    cumulative_excess_return = float((1 + strategy_returns).prod() - 1)

    # Annualized Sharpe (benchmark excess = 0)
    mean_r = strategy_returns.mean()
    std_r = strategy_returns.std(ddof=1)

    if freq.upper().startswith("M"):
        scale = np.sqrt(12)
    else:
        scale = 1.0

    if std_r > 0:
        annualized_sharpe = float(scale * mean_r / std_r)
    else:
        annualized_sharpe = np.nan

    # Hit-rate (using helper)
    hit_rate = compute_hit_rate(y_true, y_pred, exclude_spy=exclude_spy)

    # Selection DataFrame: which asset chosen each month
    selection = pd.DataFrame(
        {
            "chosen_asset": chosen_assets,
            "strategy_excess_return": strategy_returns,
        },
        index=y_true.index,
    )

    return {
        "strategy_returns": strategy_returns,
        "cumulative_excess_return": cumulative_excess_return,
        "annualized_sharpe": annualized_sharpe,
        "hit_rate": hit_rate,
        "selection": selection,
    }
def compute_directional_accuracy(y_true: pd.DataFrame, y_pred: np.ndarray):
    """
    Compute % of times the model predicts the correct sign
    (positive/negative) for each asset.
    """
    results = []
    true_vals = y_true.to_numpy()

    for idx, col in enumerate(y_true.columns):
        t = true_vals[:, idx]
        p = y_pred[:, idx]

        # sign() returns -1, 0, 1 but returns of exactly 0 are rare
        acc = (np.sign(t) == np.sign(p)).mean()
        results.append({"asset": col, "directional_accuracy": acc})

    return pd.DataFrame(results)

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from .preprocessing import load_features_targets
from .models import run_rolling_forecast


# -------------------------
# Report helpers (figures)
# Saves outputs to ./results (per course structure)
# -------------------------

BASE_DIR = Path(__file__).resolve().parents[1]  # src -> project root
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_results_csvs():
    model_path = RESULTS_DIR / "model_comparison.csv"
    rolling_path = RESULTS_DIR / "rolling_summary.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Run python3 main.py first.")
    if not rolling_path.exists():
        raise FileNotFoundError(f"Missing {rolling_path}. Run python3 main.py first.")

    model_df = pd.read_csv(model_path)
    rolling_df = pd.read_csv(rolling_path)

    required_model = {"model", "hit_rate", "cumulative_excess_return", "annualized_sharpe"}
    required_rolling = {"setup", "hit_rate", "cumulative_excess_return", "annualized_sharpe"}

    if not required_model.issubset(model_df.columns):
        raise ValueError(f"model_comparison.csv missing: {required_model - set(model_df.columns)}")
    if not required_rolling.issubset(rolling_df.columns):
        raise ValueError(f"rolling_summary.csv missing: {required_rolling - set(rolling_df.columns)}")

    return model_df, rolling_df


def plot_figure1_cum_excess_by_model(save_path: Path | None = None) -> Path:
    """
    Figure 1: Cumulative excess return vs SPY by model (bar chart).
    """
    model_df, _ = _load_results_csvs()
    df = model_df.sort_values("cumulative_excess_return", ascending=False)

    plt.figure()
    plt.bar(df["model"], df["cumulative_excess_return"])
    plt.axhline(0)
    plt.ylabel("Cumulative excess return vs SPY")
    plt.title("Model Comparison: Cumulative Excess Return (Test Period)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out = save_path or (RESULTS_DIR / "figure1_cum_excess_by_model.png")
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def _monthly_excess_series(Y_true: pd.DataFrame, Y_pred: pd.DataFrame) -> pd.Series:
    """
    Monthly excess return series of the rotation strategy vs SPY:
    - pick the asset with highest predicted return each month (from Y_pred)
    - take its realized return (from Y_true)
    - subtract SPY realized return
    """
    if "target_SPY" not in Y_true.columns:
        raise ValueError("Expected 'target_SPY' in Y_true columns for benchmark.")

    # winner (column name) each month based on predictions
    winner = Y_pred.idxmax(axis=1)

    # realized return of chosen asset each month
    chosen_realized = [Y_true.loc[dt, winner.loc[dt]] for dt in Y_true.index]

    # subtract SPY realized return
    excess = pd.Series(chosen_realized, index=Y_true.index) - Y_true["target_SPY"].values
    excess.name = "monthly_excess_return"
    return excess


def plot_figure2_cum_excess_over_time(save_path: Path | None = None) -> Path:
    """
    Figure 2: Cumulative excess return vs SPY OVER TIME (line plot)
    for expanding vs 60-month rolling window forecasts.

    This does NOT modify main.py; it recomputes the rolling forecasts inside evaluation.py.
    """
    # Load the full dataset (features + targets)
    X, Y = load_features_targets()

    # Model setup (same idea as in main.py)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    # Expanding window forecast
    Y_test_exp, Y_pred_exp = run_rolling_forecast(model, X, Y, train_window=None)

    # 60-month rolling window forecast
    Y_test_60, Y_pred_60 = run_rolling_forecast(model, X, Y, train_window=60)

    # Monthly excess return series for each setup
    excess_exp = _monthly_excess_series(Y_test_exp, Y_pred_exp)
    excess_60 = _monthly_excess_series(Y_test_60, Y_pred_60)

    # Plot cumulative excess returns over time
    plt.figure(figsize=(9, 4))
    plt.plot(excess_exp.index.to_timestamp(), excess_exp.cumsum(), label="Expanding window", linewidth=2)
    plt.plot(excess_60.index.to_timestamp(), excess_60.cumsum(), label="Rolling 60 months", linewidth=2)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.legend()
    plt.ylabel("Cumulative excess return vs SPY")
    plt.title("Rolling Forecast Setups: Cumulative Excess Return Over Time")
    plt.tight_layout()

    out = save_path or (RESULTS_DIR / "figure2_cum_excess_over_time.png")
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_figure3_sharpe_by_model(save_path: Path | None = None) -> Path:
    """
    Figure 3: Annualized excess Sharpe ratio by model (bar chart).
    """
    model_df, _ = _load_results_csvs()
    df = model_df.sort_values("annualized_sharpe", ascending=False)

    plt.figure()
    plt.bar(df["model"], df["annualized_sharpe"])
    plt.axhline(0)
    plt.ylabel("Annualized Sharpe (excess)")
    plt.title("Model Comparison: Annualized Excess Sharpe")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out = save_path or (RESULTS_DIR / "figure3_sharpe_by_model.png")
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def generate_report_artifacts() -> dict:
    """
    Convenience function: generates all figures into ./results.
    Returns paths.
    """
    fig1 = plot_figure1_cum_excess_by_model()
    fig2 = plot_figure2_cum_excess_over_time()
    fig3 = plot_figure3_sharpe_by_model()

    return {
        "figure1": str(fig1),
        "figure2": str(fig2),
        "figure3": str(fig3),
    }
