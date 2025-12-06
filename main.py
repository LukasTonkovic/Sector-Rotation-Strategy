from src.models import run_all_models, run_rolling_forecast
from src.evaluation import (
    backtest_rotation_strategy,
    compute_directional_accuracy,
)

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def main():
    print("=" * 70)
    print("Sector Rotation Project – Model Comparison")
    print("=" * 70)

    # 1) Static train/test comparison for 3 models
    all_results = run_all_models(split_period="2018-12")

    summary_rows = []

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

        # Directional accuracy (static model)
        dir_acc = compute_directional_accuracy(Y_test, Y_pred)
        print("\nPer-asset directional accuracy:")
        print(dir_acc.to_string(index=False))

        # Portfolio-style evaluation: rotation strategy
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

    # 2) Rolling / expanding window forecast (example with Linear Regression)
    print("\n" + "=" * 70)
    print("Rolling Window Forecast – Linear Regression (expanding)")
    print("=" * 70)

    # uzmi bilo koji rezultat (svi imaju isti X/Y split)
    any_result = next(iter(all_results.values()))
    X_train = any_result["X_train"]
    X_test = any_result["X_test"]
    Y_train = any_result["Y_train"]
    Y_test = any_result["Y_test"]

    # full sample (train + test)
    X_full = pd.concat([X_train, X_test])
    Y_full = pd.concat([Y_train, Y_test])

    # Linear Regression with scaling (isti kao gore)
    rolling_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    Y_test_roll, Y_pred_roll = run_rolling_forecast(
    rolling_model,
    X_full,
    Y_full,
    train_window=60,  # 5-year rolling window
    )


    print("\nPer-asset directional accuracy (rolling):")
    print(
        compute_directional_accuracy(
            Y_test_roll, Y_pred_roll.to_numpy()
        ).to_string(index=False)
    )

    roll_eval = backtest_rotation_strategy(
        Y_test_roll, Y_pred_roll.to_numpy(), exclude_spy=True
    )

    print("\nRolling rotation strategy evaluation (vs SPY):")
    print(f"Hit rate (correct winner %): {roll_eval['hit_rate']:.2%}")
    print(
        f"Cumulative excess return:    {roll_eval['cumulative_excess_return']:.2%}"
    )
    print(f"Annualized Sharpe (excess): {roll_eval['annualized_sharpe']:.2f}")


if __name__ == "__main__":
    main()
