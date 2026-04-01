from __future__ import annotations

from pathlib import Path
import pandas as pd


USDA_PATH = Path("data/clean/usda_cornbelt_greatplains_combined.csv")
PRISM_PATH = Path("data/clean/prism_county_year.csv")
DROUGHT_PATH = Path("data/clean/drought_county_year.csv")
OUTPUT_PATH = Path("data/clean/final_model_data.csv")


def load_usda() -> pd.DataFrame:
    df = pd.read_csv(USDA_PATH)

    # keep only columns needed for final merge
    keep_cols = [
        "fips",
        "state",
        "county",
        "county_clean",
        "year",
        "yield_bu_acre",
    ]
    df = df[keep_cols].copy()

    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df


def load_prism() -> pd.DataFrame:
    df = pd.read_csv(PRISM_PATH)

    keep_cols = [
        "fips",
        "state",
        "county",
        "year",
        "prism_tmax_may_sep_mean",
        "prism_ppt_may_sep_total",
    ]
    df = df[keep_cols].copy()

    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df


def load_drought() -> pd.DataFrame:
    df = pd.read_csv(DROUGHT_PATH)

    keep_cols = [
        "fips",
        "state",
        "county",
        "year",
        "drought_freq_d2plus",
        "drought_intensity_d2plus",
        "n_growing_season_weeks",
    ]
    df = df[keep_cols].copy()

    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df


def main() -> None:
    usda = load_usda()
    prism = load_prism()
    drought = load_drought()

    print("Input shapes:")
    print("USDA:", usda.shape)
    print("PRISM:", prism.shape)
    print("Drought:", drought.shape)

    # merge USDA + PRISM
    merged = usda.merge(
        prism.drop(columns=["state", "county"]),
        on=["fips", "year"],
        how="inner"
    )

    print("\nAfter merging USDA + PRISM:", merged.shape)

    # merge drought
    merged = merged.merge(
        drought.drop(columns=["state", "county"]),
        on=["fips", "year"],
        how="inner"
    )

    print("After merging USDA + PRISM + Drought:", merged.shape)

    # final column order
    final_cols = [
        "fips",
        "state",
        "county",
        "county_clean",
        "year",
        "yield_bu_acre",
        "prism_tmax_may_sep_mean",
        "prism_ppt_may_sep_total",
        "drought_freq_d2plus",
        "drought_intensity_d2plus",
        "n_growing_season_weeks",
    ]
    merged = merged[final_cols].copy()

    # sort nicely
    merged = merged.sort_values(["state", "county", "year"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("\nFinal shape:")
    print(merged.shape)

    print("\nYears covered:")
    print(merged["year"].min(), "to", merged["year"].max())

    print("\nUnique counties:")
    print(merged["fips"].nunique())

    print("\nMissing values:")
    print(merged.isna().sum())

    print("\nSample:")
    print(merged.head())


if __name__ == "__main__":
    main()