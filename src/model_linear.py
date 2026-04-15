'''
train and evaluate a linear regression model for county-level corn yield prediction
compares two feature sets: climate only, climate plus drought
evaluated with 5 fold cross validation within the corn belt region, transfer from corn belt to great plains, transfer from great plains to corn belt
outputs: csv file of evaluation metrics, csv file of model coefficients, csv files of transfer predictions
'''

from __future__ import annotations

import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import numpy as np


DATA_PATH = Path("data/clean/final_model_data.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORN_BELT_STATES = {
    "ILLINOIS",
    "INDIANA",
    "IOWA",
    "MINNESOTA",
    "MISSOURI",
    "OHIO",
    "WISCONSIN",
}

GREAT_PLAINS_STATES = {
    "KANSAS",
    "NEBRASKA",
    "NORTH DAKOTA",
    "OKLAHOMA",
    "SOUTH DAKOTA",
    "TEXAS",
}


def evaluate_cv(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rmse_list, mae_list, r2_list = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = LinearRegression()
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


def evaluate_transfer(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "model": model,
        "predictions": preds,
    }


def main():
    '''
    run the full linear regression workflow
    load cleaned dataset, split into corn belt and great plains, define climate only and climate plus drought feature sets,
        evaluate within region using corn belt cross validation, evaluate cross region transfer in both directions,
        save metrics, coeficients, and prediction outputs to csv files
    '''

    df = pd.read_csv(DATA_PATH)

    # split by region
    df_cb = df[df["state"].isin(CORN_BELT_STATES)].copy()
    df_gp = df[df["state"].isin(GREAT_PLAINS_STATES)].copy()

    print("Corn Belt rows:", df_cb.shape[0])
    print("Great Plains rows:", df_gp.shape[0])

    y_cb = df_cb["yield_bu_acre"]
    y_gp = df_gp["yield_bu_acre"]

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

    # within-region CV (Corn Belt)
    print("\n=== CORN BELT CV ===")

    cb_cv_climate = evaluate_cv(df_cb[features_climate], y_cb)
    print("\nClimate only:")
    for k, v in cb_cv_climate.items():
        print(f"{k}: {v:.4f}")

    cb_cv_full = evaluate_cv(df_cb[features_full], y_cb)
    print("\nClimate + drought:")
    for k, v in cb_cv_full.items():
        print(f"{k}: {v:.4f}")

    # transfer test (CB to GP)
    print("\n=== TRANSFER (CB → GP) ===")

    gp_transfer_climate = evaluate_transfer(
        df_cb[features_climate], y_cb,
        df_gp[features_climate], y_gp
    )
    print("\nClimate only:")
    print(f"RMSE: {gp_transfer_climate['RMSE']:.4f}")
    print(f"MAE: {gp_transfer_climate['MAE']:.4f}")
    print(f"R2: {gp_transfer_climate['R2']:.4f}")

    gp_transfer_full = evaluate_transfer(
        df_cb[features_full], y_cb,
        df_gp[features_full], y_gp
    )
    print("\nClimate + drought:")
    print(f"RMSE: {gp_transfer_full['RMSE']:.4f}")
    print(f"MAE: {gp_transfer_full['MAE']:.4f}")
    print(f"R2: {gp_transfer_full['R2']:.4f}")

    # transfer: GP to CB
    print("\n=== TRANSFER (GP to CB) ===")

    cb_transfer_climate = evaluate_transfer(
        df_gp[features_climate], y_gp,
        df_cb[features_climate], y_cb
    )

    print("\nClimate only:")
    print(f"RMSE: {cb_transfer_climate['RMSE']:.4f}")
    print(f"MAE: {cb_transfer_climate['MAE']:.4f}")
    print(f"R2: {cb_transfer_climate['R2']:.4f}")

    cb_transfer_full = evaluate_transfer(
        df_gp[features_full], y_gp,
        df_cb[features_full], y_cb
    )

    print("\nClimate + drought:")
    print(f"RMSE: {cb_transfer_full['RMSE']:.4f}")
    print(f"MAE: {cb_transfer_full['MAE']:.4f}")
    print(f"R2: {cb_transfer_full['R2']:.4f}")


    # save
    metrics_df = pd.DataFrame([
        {
            "model": "linear_regression",
            "evaluation_set": "corn_belt_cv",
            "feature_set": "climate_only",
            "rmse": cb_cv_climate["RMSE"],
            "mae": cb_cv_climate["MAE"],
            "r2": cb_cv_climate["R2"],
        },
        {
            "model": "linear_regression",
            "evaluation_set": "corn_belt_cv",
            "feature_set": "climate_plus_drought",
            "rmse": cb_cv_full["RMSE"],
            "mae": cb_cv_full["MAE"],
            "r2": cb_cv_full["R2"],
        },
        {
            "model": "linear_regression",
            "evaluation_set": "great_plains_transfer",
            "feature_set": "climate_only",
            "rmse": gp_transfer_climate["RMSE"],
            "mae": gp_transfer_climate["MAE"],
            "r2": gp_transfer_climate["R2"],
        },
        {
            "model": "linear_regression",
            "evaluation_set": "great_plains_transfer",
            "feature_set": "climate_plus_drought",
            "rmse": gp_transfer_full["RMSE"],
            "mae": gp_transfer_full["MAE"],
            "r2": gp_transfer_full["R2"],
        },
        {
            "model": "linear_regression",
            "evaluation_set": "gp_to_cb_transfer",
            "feature_set": "climate_only",
            "rmse": cb_transfer_climate["RMSE"],
            "mae": cb_transfer_climate["MAE"],
            "r2": cb_transfer_climate["R2"],
        },
        {
            "model": "linear_regression",
            "evaluation_set": "gp_to_cb_transfer",
            "feature_set": "climate_plus_drought",
            "rmse": cb_transfer_full["RMSE"],
            "mae": cb_transfer_full["MAE"],
            "r2": cb_transfer_full["R2"],
        },
        
    ])

    metrics_path = OUTPUT_DIR / "linear_regression_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    coef_model = gp_transfer_full["model"]
    coef_df = pd.DataFrame({
        "feature": features_full,
        "coefficient": coef_model.coef_,
    })

    coef_path = OUTPUT_DIR / "linear_regression_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    preds_df = df_gp[["fips", "state", "county", "year", "yield_bu_acre"]].copy()
    preds_df["predicted_yield_climate_only"] = gp_transfer_climate["predictions"]
    preds_df["predicted_yield_climate_plus_drought"] = gp_transfer_full["predictions"]

    preds_path = OUTPUT_DIR / "linear_regression_transfer_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
   
    cb_preds_df = df_cb[["fips", "state", "county", "year", "yield_bu_acre"]].copy()
    cb_preds_df["predicted_yield_climate_only"] = cb_transfer_climate["predictions"]
    cb_preds_df["predicted_yield_climate_plus_drought"] = cb_transfer_full["predictions"]

    cb_preds_path = OUTPUT_DIR / "linear_regression_gp_to_cb_transfer_predictions.csv"
    cb_preds_df.to_csv(cb_preds_path, index=False)

    print("\nSaved files:")
    print(metrics_path)
    print(coef_path)
    print(preds_path)
    print(cb_preds_path)


if __name__ == "__main__":
    main()