from src.models import run_all_models, run_rolling_forecast
from src.evaluation import (
    backtest_rotation_strategy,
    compute_directional_accuracy,
)

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# The entry point of the project, model comparison and rolling-window evaluation
def main():
    print("=" * 70)
    print("Sector Rotation Project – Model Comparison")
    print("=" * 70)

    # Training 3 models, using a time-based split
    # Until 12/2018 = train, after = test 
    all_results = run_all_models(split_period="2018-12")

    summary_rows = []
    # Loop trough the models, retrieve splits and predictions, and prints metrics
    for name, res in all_results.items():
        print("\n" + "-" * 70)
        print(f"Model: {name}")

        X_train = res["X_train"]
        X_test = res["X_test"]
        Y_train = res["Y_train"]
        Y_test = res["Y_test"]
        Y_pred = res["Y_pred"]
        metrics = res["metrics"]

        print(f"Train period: {X_train.index.min()} → {X_train.index.max()}")
        print(f"Test period : {X_test.index.min()} → {X_test.index.max()}")

        print("\nPer-asset test metrics (MSE, R^2):")
        print(metrics.to_string(index=False))

        # Computing directional accuracy (so +/-)
        dir_acc = compute_directional_accuracy(Y_test, Y_pred)
        print("\nPer-asset directional accuracy:")
        print(dir_acc.to_string(index=False))

        # Rotation strategyt backtest evaluation: practical usefulness of the models
        eval_res = backtest_rotation_strategy(Y_test, Y_pred, exclude_spy=True)

        hit_rate = eval_res["hit_rate"]
        cum_excess = eval_res["cumulative_excess_return"]
        sharpe = eval_res["annualized_sharpe"]

        print("\nRotation strategy evaluation (test period, vs SPY):")
        print(f"Hit rate (correct winner %): {hit_rate:.2%}")
        print(f"Cumulative excess return:    {cum_excess:.2%}")
        print(f"Annualized Sharpe (excess): {sharpe:.2f}")

        summary_rows.append(
            {
                "model": name,
                "hit_rate": hit_rate,
                "cumulative_excess_return": cum_excess,
                "annualized_sharpe": sharpe,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("Model comparison summary (test period)")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Save static model comparison results to CSV
    summary_df.to_csv("results/model_comparison.csv", index=False)

    # 2) Rolling / expanding window forecast (example with Linear Regression)
    print("\n" + "=" * 70)
    print("Rolling Window Forecast – Linear Regression (expanding)")
    print("=" * 70)

    # Uzmemo bilo koji rezultat (svi modeli imaju isti X/Y split)
    any_result = next(iter(all_results.values()))
    X_train = any_result["X_train"]
    X_test = any_result["X_test"]
    Y_train = any_result["Y_train"]
    Y_test = any_result["Y_test"]

    # Full sample (train + test)
    X_full = pd.concat([X_train, X_test])
    Y_full = pd.concat([Y_train, Y_test])

    # Linear Regression with scaling (isti setup kao u models.py)
    rolling_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    # --- Expanding window ---
    Y_test_exp, Y_pred_exp = run_rolling_forecast(
        rolling_model,
        X_full,
        Y_full,
        train_window=None,  # expanding window
    )

    print("\nPer-asset directional accuracy (expanding):")
    print(
        compute_directional_accuracy(
            Y_test_exp, Y_pred_exp.to_numpy()
        ).to_string(index=False)
    )

    roll_eval_exp = backtest_rotation_strategy(
        Y_test_exp, Y_pred_exp.to_numpy(), exclude_spy=True
    )

    print("\nExpanding-window rotation strategy evaluation (vs SPY):")
    print(f"Hit rate (correct winner %): {roll_eval_exp['hit_rate']:.2%}")
    print(
        f"Cumulative excess return:    {roll_eval_exp['cumulative_excess_return']:.2%}"
    )
    print(f"Annualized Sharpe (excess): {roll_eval_exp['annualized_sharpe']:.2f}")

    # --- 60-month rolling window ---
    print("\n" + "=" * 70)
    print("Rolling Window Forecast – Linear Regression (60-month window)")
    print("=" * 70)

    Y_test_60, Y_pred_60 = run_rolling_forecast(
        rolling_model,
        X_full,
        Y_full,
        train_window=60,  # zadnjih 60 mjeseci ≈ 5 godina
    )

    print("\nPer-asset directional accuracy (60-month):")
    print(
        compute_directional_accuracy(
            Y_test_60, Y_pred_60.to_numpy()
        ).to_string(index=False)
    )

    roll_eval_60 = backtest_rotation_strategy(
        Y_test_60, Y_pred_60.to_numpy(), exclude_spy=True
    )

    print("\n60-month rolling rotation strategy evaluation (vs SPY):")
    print(f"Hit rate (correct winner %): {roll_eval_60['hit_rate']:.2%}")
    print(
        f"Cumulative excess return:    {roll_eval_60['cumulative_excess_return']:.2%}"
    )
    print(f"Annualized Sharpe (excess): {roll_eval_60['annualized_sharpe']:.2f}")

    # Summary table for rolling setups (expanding vs 60-month)
    rolling_summary = pd.DataFrame(
        [
            {
                "setup": "expanding_window",
                "hit_rate": roll_eval_exp["hit_rate"],
                "cumulative_excess_return": roll_eval_exp["cumulative_excess_return"],
                "annualized_sharpe": roll_eval_exp["annualized_sharpe"],
            },
            {
                "setup": "rolling_60_months",
                "hit_rate": roll_eval_60["hit_rate"],
                "cumulative_excess_return": roll_eval_60["cumulative_excess_return"],
                "annualized_sharpe": roll_eval_60["annualized_sharpe"],
            },
        ]
    )

    print("\nRolling setups summary:")
    print(rolling_summary.to_string(index=False))

    # Save rolling performance comparison to CSV
    rolling_summary.to_csv("results/rolling_summary.csv", index=False)


if __name__ == "__main__":
    main()
