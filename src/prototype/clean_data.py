from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

# change for full state list
VALID_STATES = {"ILLINOIS", "IOWA", "INDIANA", "KANSAS"}

# minimal temp data starts in 2013 not 2000
START_YEAR = 2013
END_YEAR = 2022

def standardize_county_name(name: str) -> str:
    """
    standardize county names so USDA and temperature data merge more reliably
    """

    if pd.isna(name):
        return name
    
    name = str(name).strip().upper()

    replacements = {
        " COUNTY": "",
        " PARISH": "",
        "'": "",
        "-": "",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)

    name = " ".join(name.split())
    return name

# USDA yield cleaning
def clean_yield_data(yield_path: str | Path) -> pd.DataFrame:
    """
    clean USDA county level corn yield data
    """

    df = pd.read_csv(yield_path)

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    required_cols = ["year", "state", "county", "state_ansi", "county_ansi", "value"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Yield data is missing required columns: {missing}")
    
    # relevant states and years, change for bigger data
    df["state"] = df["state"].astype(str).str.strip().str.upper()
    df["county"] = df["county"].astype(str).str.strip().str.upper()
    df["year"] = pd.to_numeric(df["year"], errors = "coerce")

    df = df[df["state"].isin(VALID_STATES)].copy()
    df = df[df["year"].between(START_YEAR, END_YEAR)].copy()

    # remove non county aggregate rows
    df = df[~df["county"].str.contains("OTHER COUNTIES", na=False)].copy()

    df["value"] = (
        df["value"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["yield_bu_acre"] = pd.to_numeric(df["value"], errors = "coerce")
    df = df.dropna(subset = ["yield_bu_acre"]).copy()

    # county key
    df["county_clean"] = df["county"].apply(standardize_county_name)

    # FIPS code
    df["state_ansi"] = pd.to_numeric(df["state_ansi"], errors="coerce")
    df["county_ansi"] = pd.to_numeric(df["county_ansi"], errors="coerce")
    df["fips"] = (
        df["state_ansi"].astype("Int64").astype(str).str.zfill(2)
        + df["county_ansi"].astype("Int64").astype(str).str.zfill(3)
    )

    keep_cols = [
        "year",
        "state",
        "county",
        "county_clean",
        "state_ansi",
        "county_ansi",
        "fips",
        "yield_bu_acre",
    ]

    df = df[keep_cols].drop_duplicates().reset_index(drop=True)
    return df


# temperature cleaning (won't use for final)
def clean_temperature_data(temp_path: str | Path) -> pd.DataFrame:
    """
    clean weekly county level temp data and aggregate to yearly averages
    """
    df = pd.read_csv(temp_path)

    original_cols = df.columns.tolist()
    df.columns = [col.strip() for col in df.columns]

    county_col = "county_name"
    state_col = "state_id"

    state_id_to_name = {
        17: "ILLINOIS",
        18: "INDIANA",
        19: "IOWA",
        20: "KANSAS",
    }

    if state_col == "state_id" or "id" in state_col.lower():
        df["state"] = pd.to_numeric(df[state_col], errors="coerce").map(state_id_to_name)
    else:
        df["state"] = df[state_col].astype(str).str.strip().str.upper()

    df["county"] = df[county_col].astype(str).str.strip().str.upper()
    df["county_clean"] = df["county"].apply(standardize_county_name)

    # only keeping target states, for building
    df = df[df["state"].isin(VALID_STATES)].copy()

    # weekly columns, columns are titled like "Week 1 (2013)"
    week_cols = [col for col in df.columns if col.startswith("Week ") and "(" in col and ")" in col]
    if not week_cols:
        raise ValueError("No weekly temperature columns were found.")

    df_long = df.melt(
        id_vars=["state", "county", "county_clean"],
        value_vars=week_cols,
        var_name="week_label",
        value_name="temperature",
    )

    df_long["year"] = df_long["week_label"].str.extract(r"\((\d{4})\)")
    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce")

    df_long = df_long[df_long["year"].between(START_YEAR, END_YEAR)].copy()

    df_long["temperature"] = pd.to_numeric(df_long["temperature"], errors="coerce")
    df_long = df_long.dropna(subset=["temperature"]).copy()

    # aggregate weekly to yearly average by county
    df_yearly = (
        df_long.groupby(["state", "county", "county_clean", "year"], as_index=False)["temperature"]
        .mean()
        .rename(columns={"temperature": "avg_temp"})
    )

    # changing into celsius
    df_yearly["avg_temp"] = df_yearly["avg_temp"] - 273.15


    return df_yearly

# merge
def merge_cleaned_data(yield_df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
    """
    merge cleaned yield and yearly temperature data.
    """
    merged = yield_df.merge(
        temp_df,
        on=["state", "county_clean", "year"],
        how="inner",
        suffixes=("_yield", "_temp"),
    )

    if "county_yield" in merged.columns:
        merged = merged.rename(columns={"county_yield": "county"})
    elif "county" not in merged.columns and "county_temp" in merged.columns:
        merged = merged.rename(columns={"county_temp": "county"})

    final_cols = [col for col in [
        "state",
        "county",
        "county_clean",
        "year",
        "fips",
        "yield_bu_acre",
        "avg_temp",
    ] if col in merged.columns]

    merged = merged[final_cols].drop_duplicates().reset_index(drop=True)
    return merged



# run cleaning
def main(
    yield_path: str | Path = "data/raw/usda_yield.csv",
    temp_path: str | Path = "data/raw/county_temperature.csv",
    processed_dir: str | Path = "data/clean",
) -> None:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    yield_df = clean_yield_data(yield_path)
    temp_df = clean_temperature_data(temp_path)
    merged_df = merge_cleaned_data(yield_df, temp_df)

    yield_df.to_csv(processed_dir / "yield_cleaned.csv", index=False)
    temp_df.to_csv(processed_dir / "temperature_yearly.csv", index=False)
    merged_df.to_csv(processed_dir / "merged_model_data.csv", index=False)

    print("Saved:")
    print(f"- {processed_dir / 'yield_cleaned.csv'}")
    print(f"- {processed_dir / 'temperature_yearly.csv'}")
    print(f"- {processed_dir / 'merged_model_data.csv'}")
    print()
    print("Shapes:")
    print(f"Yield cleaned: {yield_df.shape}")
    print(f"Temperature yearly: {temp_df.shape}")
    print(f"Merged modeling data: {merged_df.shape}")
    print()
    print("Merged sample:")
    print(merged_df.head())


if __name__ == "__main__":
    main()
