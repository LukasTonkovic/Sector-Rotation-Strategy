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
