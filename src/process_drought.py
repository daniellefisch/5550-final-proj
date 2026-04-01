from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_DROUGHT_PATH = Path("data/raw/drought.csv")
OUTPUT_DIR = Path("data/clean")

CORN_BELT_STATE_ABBRS = {
    "IL",
    "IN",
    "IA",
    "MN",
    "MO",
    "OH",
    "WI",
}

GREAT_PLAINS_STATE_ABBRS = {
    "KS",
    "NE",
    "ND",
    "OK",
    "SD",
    "TX",
}

TARGET_STATE_ABBRS = CORN_BELT_STATE_ABBRS | GREAT_PLAINS_STATE_ABBRS

STATE_ABBR_TO_NAME = {
    "IL": "ILLINOIS",
    "IN": "INDIANA",
    "IA": "IOWA",
    "MN": "MINNESOTA",
    "MO": "MISSOURI",
    "OH": "OHIO",
    "WI": "WISCONSIN",
    "KS": "KANSAS",
    "NE": "NEBRASKA",
    "ND": "NORTH DAKOTA",
    "OK": "OKLAHOMA",
    "SD": "SOUTH DAKOTA",
    "TX": "TEXAS",
}

START_YEAR = 2000
END_YEAR = 2022
GROWING_SEASON_MONTHS = {5, 6, 7, 8, 9}


def standardize_county_name(name: str) -> str:
    if pd.isna(name):
        return name

    name = str(name).strip().upper()

    replacements = {
        " COUNTY": "",
        " PARISH": "",
        " BOROUGH": "",
        " CITY AND BOROUGH": "",
        " CENSUS AREA": "",
        ".": "",
        "'": "",
        "-": " ",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    name = " ".join(name.split())
    return name


def load_and_clean_drought(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]

    required_cols = [
        "MapDate",
        "FIPS",
        "County",
        "State",
        "D2",
        "D3",
        "D4",
        "ValidStart",
        "ValidEnd",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in drought file: {missing}")

    # identifiers
    df["State"] = df["State"].astype(str).str.strip().str.upper()
    df["County"] = df["County"].astype(str).str.strip()
    df["county_clean"] = df["County"].apply(standardize_county_name)

    # dates
    df["MapDate"] = pd.to_datetime(df["MapDate"].astype(str), format="%Y%m%d", errors="coerce")
    df["ValidStart"] = pd.to_datetime(df["ValidStart"], errors="coerce")
    df["ValidEnd"] = pd.to_datetime(df["ValidEnd"], errors="coerce")

    df["year"] = df["MapDate"].dt.year
    df["month"] = df["MapDate"].dt.month

    # restrict to proj states, years, months (growing szn)
    df = df[df["State"].isin(TARGET_STATE_ABBRS)].copy()
    df = df[df["year"].between(START_YEAR, END_YEAR)].copy()
    df = df[df["month"].isin(GROWING_SEASON_MONTHS)].copy()

    # FIPS
    df["FIPS"] = pd.to_numeric(df["FIPS"], errors="coerce")
    df = df.dropna(subset=["FIPS"]).copy()
    df["fips"] = df["FIPS"].astype(int).astype(str).str.zfill(5)

    # drought percentages
    drought_cols = ["None", "D0", "D1", "D2", "D3", "D4"]
    for col in drought_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # D2+ features
    df["d2plus_pct"] = df[["D2", "D3", "D4"]].sum(axis=1)
    df["d2plus_present"] = (df["d2plus_pct"] > 0).astype(int)

    # full state name so that consistent w/ other files
    df["state"] = df["State"].map(STATE_ABBR_TO_NAME)

    keep_cols = [
        "MapDate",
        "ValidStart",
        "ValidEnd",
        "year",
        "month",
        "fips",
        "State",
        "state",
        "County",
        "county_clean",
        "D2",
        "D3",
        "D4",
        "d2plus_pct",
        "d2plus_present",
    ]

    df = df[keep_cols].drop_duplicates().reset_index(drop=True)
    return df


def aggregate_drought_features(df: pd.DataFrame) -> pd.DataFrame:
    drought_yearly = (
        df.groupby(["fips", "state", "County", "county_clean", "year"], as_index=False)
        .agg(
            drought_freq_d2plus=("d2plus_present", "mean"),
            drought_intensity_d2plus=("d2plus_pct", "mean"),
            n_growing_season_weeks=("MapDate", "count"),
        )
        .rename(columns={"County": "county"})
    )

    return drought_yearly


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    weekly_df = load_and_clean_drought(RAW_DROUGHT_PATH)
    yearly_df = aggregate_drought_features(weekly_df)

    weekly_out = OUTPUT_DIR / "drought_weekly_filtered.csv"
    yearly_out = OUTPUT_DIR / "drought_county_year.csv"

    weekly_df.to_csv(weekly_out, index=False)
    yearly_df.to_csv(yearly_out, index=False)

    print("Saved files:")
    print(weekly_out)
    print(yearly_out)

    print("\nShapes:")
    print("Weekly filtered:", weekly_df.shape)
    print("County-year:", yearly_df.shape)

    print("\nWeekly sample:")
    print(weekly_df.head())

    print("\nCounty-year sample:")
    print(yearly_df.head())

    print("\nYears covered:")
    print(yearly_df["year"].min(), "to", yearly_df["year"].max())

    print("\nUnique counties:")
    print(yearly_df["fips"].nunique())


if __name__ == "__main__":
    main()