'''
train and evaluate a random forest model for county-level corn yield prediction.

this script compares two feature sets:
1. climate only:
   - prism_tmax_may_sep_mean
   - prism_ppt_may_sep_total
2. climate plus drought:
   - prism_tmax_may_sep_mean
   - prism_ppt_may_sep_total
   - drought_freq_d2plus
   - drought_intensity_d2plus

model is evaluated across three settings:
- 5-fold cross-validation within the Corn Belt
- 5-fold cross-validation within the Great Plains
- cross-region transfer:
    train on Corn Belt, test on Great Plains
    train on Great Plains,  test on Corn Belt

in addition to performance metrics (RMSE, MAE, R2), the script computes
and saves feature importances for each evaluation setting.

outputs:
- random_forest_metrics.csv: model performance across all settings
- random_forest_feature_importances.csv: feature importance values
- random_forest_cb_to_gp_transfer_predictions.csv: pred for CB → GP transfer
- random_forest_gp_to_cb_transfer_predictions.csv: pred for GP → CB transfer
'''

from __future__ import annotations

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
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


def evaluate_cv(X, y, feature_names):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rmse_list, mae_list, r2_list = [], [], []
    feature_importances = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse_list.append(np.sqrt(mean_squared_error(y_test, preds)))
        mae_list.append(mean_absolute_error(y_test, preds))
        r2_list.append(r2_score(y_test, preds))
        feature_importances.append(model.feature_importances_)

    mean_importances = np.mean(feature_importances, axis=0)

    return {
        "RMSE": np.mean(rmse_list),
        "MAE": np.mean(mae_list),
        "R2": np.mean(r2_list),
        "feature_importances": dict(zip(feature_names, mean_importances)),
    }


def evaluate_transfer(X_train, y_train, X_test, y_test, feature_names):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "model": model,
        "predictions": preds,
        "feature_importances": dict(zip(feature_names, model.feature_importances_)),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    df_cb = df[df["state"].isin(CORN_BELT_STATES)].copy()
    df_gp = df[df["state"].isin(GREAT_PLAINS_STATES)].copy()

    print("Corn Belt rows:", df_cb.shape[0])
    print("Great Plains rows:", df_gp.shape[0])

    y_cb = df_cb["yield_bu_acre"]
    y_gp = df_gp["yield_bu_acre"]

    regions = {
        "corn_belt": (df_cb, y_cb),
        "great_plains": (df_gp, y_gp),
    }

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

    # Corn Belt CV
    print("\n=== CORN BELT CV ===")

    cb_cv_climate = evaluate_cv(df_cb[features_climate], y_cb, features_climate)
    print("\nClimate only:")
    print(f"RMSE: {cb_cv_climate['RMSE']:.4f}")
    print(f"MAE: {cb_cv_climate['MAE']:.4f}")
    print(f"R2: {cb_cv_climate['R2']:.4f}")

    cb_cv_full = evaluate_cv(df_cb[features_full], y_cb, features_full)
    print("\nClimate + drought:")
    print(f"RMSE: {cb_cv_full['RMSE']:.4f}")
    print(f"MAE: {cb_cv_full['MAE']:.4f}")
    print(f"R2: {cb_cv_full['R2']:.4f}")

    # Great Plains CV
    print("\n=== GREAT PLAINS CV ===")

    gp_cv_climate = evaluate_cv(df_gp[features_climate], y_gp, features_climate)
    print("\nClimate only:")
    print(f"RMSE: {gp_cv_climate['RMSE']:.4f}")
    print(f"MAE: {gp_cv_climate['MAE']:.4f}")
    print(f"R2: {gp_cv_climate['R2']:.4f}")

    gp_cv_full = evaluate_cv(df_gp[features_full], y_gp, features_full)
    print("\nClimate + drought:")
    print(f"RMSE: {gp_cv_full['RMSE']:.4f}")
    print(f"MAE: {gp_cv_full['MAE']:.4f}")
    print(f"R2: {gp_cv_full['R2']:.4f}")

    #transfer: CB to GP
    print("\n=== TRANSFER (CB to GP) ===")

    gp_transfer_climate = evaluate_transfer(
        df_cb[features_climate], y_cb,
        df_gp[features_climate], y_gp,
        features_climate
    )
    print("\nClimate only:")
    print(f"RMSE: {gp_transfer_climate['RMSE']:.4f}")
    print(f"MAE: {gp_transfer_climate['MAE']:.4f}")
    print(f"R2: {gp_transfer_climate['R2']:.4f}")

    gp_transfer_full = evaluate_transfer(
        df_cb[features_full], y_cb,
        df_gp[features_full], y_gp,
        features_full
    )
    print("\nClimate + drought:")
    print(f"RMSE: {gp_transfer_full['RMSE']:.4f}")
    print(f"MAE: {gp_transfer_full['MAE']:.4f}")
    print(f"R2: {gp_transfer_full['R2']:.4f}")

    # transfer: GP to CB
    print("\n=== TRANSFER (GP to CB) ===")

    cb_transfer_climate = evaluate_transfer(
        df_gp[features_climate], y_gp,
        df_cb[features_climate], y_cb,
        features_climate
    )
    print("\nClimate only:")
    print(f"RMSE: {cb_transfer_climate['RMSE']:.4f}")
    print(f"MAE: {cb_transfer_climate['MAE']:.4f}")
    print(f"R2: {cb_transfer_climate['R2']:.4f}")

    cb_transfer_full = evaluate_transfer(
        df_gp[features_full], y_gp,
        df_cb[features_full], y_cb,
        features_full
    )
    print("\nClimate + drought:")
    print(f"RMSE: {cb_transfer_full['RMSE']:.4f}")
    print(f"MAE: {cb_transfer_full['MAE']:.4f}")
    print(f"R2: {cb_transfer_full['R2']:.4f}")

    # save metrics
    metrics_df = pd.DataFrame([
        {
            "model": "random_forest",
            "evaluation_set": "corn_belt_cv",
            "feature_set": "climate_only",
            "rmse": cb_cv_climate["RMSE"],
            "mae": cb_cv_climate["MAE"],
            "r2": cb_cv_climate["R2"],
        },
        {
            "model": "random_forest",
            "evaluation_set": "corn_belt_cv",
            "feature_set": "climate_plus_drought",
            "rmse": cb_cv_full["RMSE"],
            "mae": cb_cv_full["MAE"],
            "r2": cb_cv_full["R2"],
        },
        {
            "model": "random_forest",
            "evaluation_set": "great_plains_cv",
            "feature_set": "climate_only",
            "rmse": gp_cv_climate["RMSE"],
            "mae": gp_cv_climate["MAE"],
            "r2": gp_cv_climate["R2"],
        },
        {
            "model": "random_forest",
            "evaluation_set": "great_plains_cv",
            "feature_set": "climate_plus_drought",
            "rmse": gp_cv_full["RMSE"],
            "mae": gp_cv_full["MAE"],
            "r2": gp_cv_full["R2"],
        },
        {
            "model": "random_forest",
            "evaluation_set": "cb_to_gp_transfer",
            "feature_set": "climate_only",
            "rmse": gp_transfer_climate["RMSE"],
            "mae": gp_transfer_climate["MAE"],
            "r2": gp_transfer_climate["R2"],
        },
        {
            "model": "random_forest",
            "evaluation_set": "cb_to_gp_transfer",
            "feature_set": "climate_plus_drought",
            "rmse": gp_transfer_full["RMSE"],
            "mae": gp_transfer_full["MAE"],
            "r2": gp_transfer_full["R2"],
        },
        {
            "model": "random_forest",
            "evaluation_set": "gp_to_cb_transfer",
            "feature_set": "climate_only",
            "rmse": cb_transfer_climate["RMSE"],
            "mae": cb_transfer_climate["MAE"],
            "r2": cb_transfer_climate["R2"],
        },
        {
            "model": "random_forest",
            "evaluation_set": "gp_to_cb_transfer",
            "feature_set": "climate_plus_drought",
            "rmse": cb_transfer_full["RMSE"],
            "mae": cb_transfer_full["MAE"],
            "r2": cb_transfer_full["R2"],
        },
    ])

    metrics_path = OUTPUT_DIR / "random_forest_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    importance_rows = []

    # Corn Belt CV importances
    for feature, importance in cb_cv_climate["feature_importances"].items():
        importance_rows.append({
            "model": "random_forest",
            "evaluation_set": "corn_belt_cv",
            "feature_set": "climate_only",
            "feature": feature,
            "importance": importance,
        })

    for feature, importance in cb_cv_full["feature_importances"].items():
        importance_rows.append({
            "model": "random_forest",
            "evaluation_set": "corn_belt_cv",
            "feature_set": "climate_plus_drought",
            "feature": feature,
            "importance": importance,
        })

    # Great Plains CV importances
    for feature, importance in gp_cv_climate["feature_importances"].items():
        importance_rows.append({
            "model": "random_forest",
            "evaluation_set": "great_plains_cv",
            "feature_set": "climate_only",
            "feature": feature,
            "importance": importance,
        })

    for feature, importance in gp_cv_full["feature_importances"].items():
        importance_rows.append({
            "model": "random_forest",
            "evaluation_set": "great_plains_cv",
            "feature_set": "climate_plus_drought",
            "feature": feature,
            "importance": importance,
        })

    # CB to GP transfer importances
    for feature, importance in gp_transfer_climate["feature_importances"].items():
        importance_rows.append({
            "model": "random_forest",
            "evaluation_set": "cb_to_gp_transfer",
            "feature_set": "climate_only",
            "feature": feature,
            "importance": importance,
        })

    for feature, importance in gp_transfer_full["feature_importances"].items():
        importance_rows.append({
            "model": "random_forest",
            "evaluation_set": "cb_to_gp_transfer",
            "feature_set": "climate_plus_drought",
            "feature": feature,
            "importance": importance,
        })

    # GP to CB transfer importances
    for feature, importance in cb_transfer_climate["feature_importances"].items():
        importance_rows.append({
            "model": "random_forest",
            "evaluation_set": "gp_to_cb_transfer",
            "feature_set": "climate_only",
            "feature": feature,
            "importance": importance,
        })

    for feature, importance in cb_transfer_full["feature_importances"].items():
        importance_rows.append({
            "model": "random_forest",
            "evaluation_set": "gp_to_cb_transfer",
            "feature_set": "climate_plus_drought",
            "feature": feature,
            "importance": importance,
        })

    importance_df = pd.DataFrame(importance_rows)
    importance_path = OUTPUT_DIR / "random_forest_feature_importances.csv"
    importance_df.to_csv(importance_path, index=False)

    # save transfer predictions: CB to GP
    gp_preds_df = df_gp[["fips", "state", "county", "year", "yield_bu_acre"]].copy()
    gp_preds_df["predicted_yield_climate_only"] = gp_transfer_climate["predictions"]
    gp_preds_df["predicted_yield_climate_plus_drought"] = gp_transfer_full["predictions"]

    gp_preds_path = OUTPUT_DIR / "random_forest_cb_to_gp_transfer_predictions.csv"
    gp_preds_df.to_csv(gp_preds_path, index=False)

    # save transfer predictions: GP to CB
    cb_preds_df = df_cb[["fips", "state", "county", "year", "yield_bu_acre"]].copy()
    cb_preds_df["predicted_yield_climate_only"] = cb_transfer_climate["predictions"]
    cb_preds_df["predicted_yield_climate_plus_drought"] = cb_transfer_full["predictions"]

    cb_preds_path = OUTPUT_DIR / "random_forest_gp_to_cb_transfer_predictions.csv"
    cb_preds_df.to_csv(cb_preds_path, index=False)

    print("\nSaved files:")
    print(metrics_path)
    print(importance_path)
    print(gp_preds_path)
    print(cb_preds_path)


if __name__ == "__main__":
    main()