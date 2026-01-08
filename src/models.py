
from typing import Dict, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from .preprocessing import load_features_targets


def time_series_train_test_split(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    split_period: str = "2018-12",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    train_mask = X.index <= split_period
    test_mask = X.index > split_period

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    Y_train = Y.loc[train_mask].copy()
    Y_test = Y.loc[test_mask].copy()

    return X_train, X_test, Y_train, Y_test


def evaluate_predictions(
    Y_true: pd.DataFrame,
    Y_pred,
) -> pd.DataFrame:
   
    asset_names = list(Y_true.columns)

    mse_values = mean_squared_error(
        Y_true, Y_pred, multioutput="raw_values"
    )
    r2_values = r2_score(
        Y_true, Y_pred, multioutput="raw_values"
    )

    metrics = pd.DataFrame(
        {
            "asset": asset_names,
            "mse": mse_values,
            "r2": r2_values,
        }
    )

    return metrics


def run_single_model(
    model,
    model_name: str,
    split_period: str = "2018-12",
) -> Dict[str, object]:
   
    X, Y = load_features_targets()
    X_train, X_test, Y_train, Y_test = time_series_train_test_split(
        X, Y, split_period=split_period
    )

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    metrics = evaluate_predictions(Y_test, Y_pred)

    return {
        "name": model_name,
        "model": model,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "Y_pred": Y_pred,
    }


def run_all_models(split_period: str = "2018-12") -> Dict[str, Dict[str, object]]:
    
    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "RidgeRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
    }

    

    results = {}
    for name, estimator in models.items():
        print(f"\nFitting model: {name}")
        res = run_single_model(estimator, name, split_period=split_period)
        results[name] = res

    return results

def run_rolling_forecast(
    model,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    train_window: int = None
):
    

    dates = X.index
    preds = []
    actuals = []
    pred_dates = []

    for i in range(len(X)):
        # Can't predict the first month
        if i == 0:
            continue

        # Stop before last (needs next-month target)
        if i+1 >= len(X):
            break

        # Expanding window
        if train_window is None:
            X_train = X.iloc[:i]
            Y_train = Y.iloc[:i]

        # Fixed-size rolling window
        else:
            if i < train_window:
                continue
            X_train = X.iloc[i-train_window:i]
            Y_train = Y.iloc[i-train_window:i]

        # Fit model on rolling window
        model.fit(X_train, Y_train)

        # 1-step-ahead prediction
        y_pred = model.predict(X.iloc[i:i+1])
        preds.append(y_pred[0])

        # Actual next-month value
        actuals.append(Y.iloc[i])
        pred_dates.append(dates[i])

    Y_pred_roll = pd.DataFrame(
        preds,
        index=pred_dates,
        columns=Y.columns
    )

    Y_test_roll = pd.DataFrame(
        actuals,
        index=pred_dates,
        columns=Y.columns
    )

    return Y_test_roll, Y_pred_roll


def run_baseline_linear_regression(split_period: str = "2018-12"):
    
    res = run_single_model(LinearRegression(), "LinearRegression", split_period)
    return (
        res["model"],
        res["metrics"],
        res["X_train"],
        res["X_test"],
        res["Y_train"],
        res["Y_test"],
        res["Y_pred"],
    )


if __name__ == "__main__":
    
    all_results = run_all_models()
    for name, res in all_results.items():
        print(f"\nModel: {name}")
        print(res["metrics"].to_string(index=False))
