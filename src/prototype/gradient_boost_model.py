from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


CORN_BELT_STATES = ["ILLINOIS", "IOWA", "INDIANA"]
OUT_REGION_STATES = ["KANSAS"]

def evaluate_regression(y_true, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

def main(
    data_path: str | Path = "data/clean/prototype/merged_model_data.csv",
    output_dir: str | Path = "outputs/prototype",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load data
    df = pd.read_csv(data_path)

    # split into CB v Kansas
    corn_belt_df = df[df["state"].isin(CORN_BELT_STATES)].copy()
    out_region_df = df[df["state"].isin(OUT_REGION_STATES)].copy()

    # features + target
    feature_cols = ["avg_temp"]
    target_col = "yield_bu_acre"

    X_corn = corn_belt_df[feature_cols]
    y_corn = corn_belt_df[target_col]

    X_out = out_region_df[feature_cols]
    y_out = out_region_df[target_col]

    # w/in reg eval, CV on corn belt
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )

    cornbelt_cv_preds = cross_val_predict(cv_model, X_corn, y_corn, cv=cv, n_jobs=-1)
    cornbelt_metrics = evaluate_regression(y_corn, cornbelt_cv_preds)

    cornbelt_pred_df = corn_belt_df[["state", "county", "year", target_col]].copy()
    cornbelt_pred_df["predicted_yield"] = cornbelt_cv_preds
    cornbelt_pred_df["evaluation_set"] = "corn_belt_cv"

    # final model on all cb data
    final_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    final_model.fit(X_corn, y_corn)

    # out of region eval on kansas
    out_preds = final_model.predict(X_out)
    out_region_metrics = evaluate_regression(y_out, out_preds)

    out_region_pred_df = out_region_df[["state", "county", "year", target_col]].copy()
    out_region_pred_df["predicted_yield"] = out_preds
    out_region_pred_df["evaluation_set"] = "kansas_transfer"

    # save
    all_preds = pd.concat([cornbelt_pred_df, out_region_pred_df], ignore_index=True)
    all_preds.to_csv(output_dir / "gradient_boost_predictions.csv", index=False)

    metrics_df = pd.DataFrame([
        {
            "evaluation_set": "corn_belt_cv",
            "rmse": cornbelt_metrics["rmse"],
            "mae": cornbelt_metrics["mae"],
            "r2": cornbelt_metrics["r2"],
        },
        {
            "evaluation_set": "kansas_transfer",
            "rmse": out_region_metrics["rmse"],
            "mae": out_region_metrics["mae"],
            "r2": out_region_metrics["r2"],
        },
    ])
    metrics_df.to_csv(output_dir / "gradient_boost_metrics.csv", index=False)

    # print
    print("Gradient Boosting Results")
    print("------------------------")
    print("Within-region (Corn Belt CV):")
    print(f"RMSE: {cornbelt_metrics['rmse']:.2f}")
    print(f"MAE:  {cornbelt_metrics['mae']:.2f}")
    print(f"R^2:  {cornbelt_metrics['r2']:.3f}")

    print("\nOut-of-region transfer (Kansas):")
    print(f"RMSE: {out_region_metrics['rmse']:.2f}")
    print(f"MAE:  {out_region_metrics['mae']:.2f}")
    print(f"R^2:  {out_region_metrics['r2']:.3f}")

    print("\nFeature importance:")
    for feature, importance in zip(feature_cols, final_model.feature_importances_):
        print(f"{feature}: {importance:.4f}")

    print("\nSaved files:")
    print(output_dir / "gradient_boost_predictions.csv")
    print(output_dir / "gradient_boost_metrics.csv")


if __name__ == "__main__":
    main()
