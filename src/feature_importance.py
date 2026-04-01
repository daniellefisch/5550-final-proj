from __future__ import annotations

from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


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

FEATURES = [
    "prism_tmax_may_sep_mean",
    "prism_ppt_may_sep_total",
    "drought_freq_d2plus",
    "drought_intensity_d2plus",
]

TARGET = "yield_bu_acre"


def make_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )


def compute_permutation_importance(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    trained_on: str,
    evaluated_on: str,
    n_repeats: int = 10,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring="r2",
    )

    out = pd.DataFrame({
        "trained_on": trained_on,
        "evaluated_on": evaluated_on,
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    out = out.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    df_cb = df[df["state"].isin(CORN_BELT_STATES)].copy()
    df_gp = df[df["state"].isin(GREAT_PLAINS_STATES)].copy()

    X_cb = df_cb[FEATURES]
    y_cb = df_cb[TARGET]

    X_gp = df_gp[FEATURES]
    y_gp = df_gp[TARGET]

    print("Corn Belt rows:", df_cb.shape[0])
    print("Great Plains rows:", df_gp.shape[0])

    # model 1: train on Corn Belt
    rf_cb = make_rf()
    rf_cb.fit(X_cb, y_cb)

    cb_on_cb = compute_permutation_importance(
        rf_cb, X_cb, y_cb,
        trained_on="corn_belt",
        evaluated_on="corn_belt"
    )

    cb_on_gp = compute_permutation_importance(
        rf_cb, X_gp, y_gp,
        trained_on="corn_belt",
        evaluated_on="great_plains"
    )

    # model 2: train on Great Plains
    rf_gp = make_rf()
    rf_gp.fit(X_gp, y_gp)

    gp_on_gp = compute_permutation_importance(
        rf_gp, X_gp, y_gp,
        trained_on="great_plains",
        evaluated_on="great_plains"
    )

    gp_on_cb = compute_permutation_importance(
        rf_gp, X_cb, y_cb,
        trained_on="great_plains",
        evaluated_on="corn_belt"
    )

    # save
    importance_df = pd.concat(
        [cb_on_cb, cb_on_gp, gp_on_gp, gp_on_cb],
        ignore_index=True
    )

    importance_path = OUTPUT_DIR / "feature_importance_comparison.csv"
    importance_df.to_csv(importance_path, index=False)

    print("\nSaved:")
    print(importance_path)

    print("\nSample output:")
    print(importance_df.head(12))

    print("\nGrouped summary:")
    for (trained_on, evaluated_on), group in importance_df.groupby(["trained_on", "evaluated_on"]):
        print(f"\nTrained on: {trained_on} | Evaluated on: {evaluated_on}")
        print(group[["feature", "importance_mean", "importance_std"]].sort_values(
            "importance_mean", ascending=False
        ).to_string(index=False))


if __name__ == "__main__":
    main()