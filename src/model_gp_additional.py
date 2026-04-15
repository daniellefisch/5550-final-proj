from __future__ import annotations

import pandas as pd
from pathlib import Path
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "clean" / "final_model_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# regions
GREAT_PLAINS_STATES = {
    "KANSAS",
    "NEBRASKA",
    "NORTH DAKOTA",
    "OKLAHOMA",
    "SOUTH DAKOTA",
    "TEXAS",
}


# eval function
def evaluate_cv(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rmse_list, mae_list, r2_list = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse_list.append(np.sqrt(mean_squared_error(y_test, preds)))
        mae_list.append(mean_absolute_error(y_test, preds))
        r2_list.append(r2_score(y_test, preds))

    return {
        "RMSE": np.mean(rmse_list),
        "MAE": np.mean(mae_list),
        "R2": np.mean(r2_list),
    }


# main
def main():

    df = pd.read_csv(DATA_PATH)

    df_gp = df[df["state"].isin(GREAT_PLAINS_STATES)].copy()
    y_gp = df_gp["yield_bu_acre"]

    print("Great Plains rows:", df_gp.shape[0])

    features_climate = [
        "prism_tmax_may_sep_mean",
        "prism_ppt_may_sep_total",
    ]

    features_full = [
        "prism_tmax_may_sep_mean",
        "prism_ppt_may_sep_total",
        "drought_freq_d2plus",
        "drought_intensity_d2plus",
    ]

    results = []

    # linear regression
    print("\n=== LINEAR REGRESSION (GP CV) ===")

    lr = LinearRegression()

    for feature_set_name, features in [
        ("climate_only", features_climate),
        ("climate_plus_drought", features_full),
    ]:
        metrics = evaluate_cv(lr, df_gp[features], y_gp)

        print(f"\n{feature_set_name}:")
        print(metrics)

        results.append({
            "model": "linear_regression",
            "evaluation_set": "great_plains_cv",
            "feature_set": feature_set_name,
            "rmse": metrics["RMSE"],
            "mae": metrics["MAE"],
            "r2": metrics["R2"],
        })

    # gradient boosting
    print("\n=== GRADIENT BOOSTING (GP CV) ===")

    gb = GradientBoostingRegressor(random_state=42)

    for feature_set_name, features in [
        ("climate_only", features_climate),
        ("climate_plus_drought", features_full),
    ]:
        metrics = evaluate_cv(gb, df_gp[features], y_gp)

        print(f"\n{feature_set_name}:")
        print(metrics)

        results.append({
            "model": "gradient_boost",
            "evaluation_set": "great_plains_cv",
            "feature_set": feature_set_name,
            "rmse": metrics["RMSE"],
            "mae": metrics["MAE"],
            "r2": metrics["R2"],
        })

    # save
    results_df = pd.DataFrame(results)

    out_path = OUTPUT_DIR / "great_plains_additional_metrics.csv"
    results_df.to_csv(out_path, index=False)

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()