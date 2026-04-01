from __future__ import annotations

from pathlib import Path

import pandas as pd


# state lists
CORN_BELT_STATES = {
    "ILLINOIS",
    "INDIANA",
    "IOWA",
    "MINNESOTA",
    "MISSOURI",
    "OHIO",
    "WISCONSIN"
}

GREAT_PLAINS_STATES = {
    "KANSAS",
    "NEBRASKA",
    "NORTH DAKOTA",
    "OKLAHOMA",
    "SOUTH DAKOTA",
    "TEXAS"
}

VALID_STATES = CORN_BELT_STATES.union(GREAT_PLAINS_STATES)

START_YEAR = 2000
END_YEAR = 2022


def standardize_county_name(name: str) -> str:
    """
    standardize county names so USDA county names are more consistent
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


def assign_region(state: str) -> str:
    """
    assign each state to corn belt, great plains, or both
    """
    in_corn_belt = state in CORN_BELT_STATES
    in_great_plains = state in GREAT_PLAINS_STATES

    if in_corn_belt and in_great_plains:
        return "BOTH"
    if in_corn_belt:
        return "CORN_BELT"
    if in_great_plains:
        return "GREAT_PLAINS"
    return "OTHER"


def clean_yield_data(yield_path: str | Path) -> pd.DataFrame:
    """
    clean USDA county level corn yield data
    """

    df = pd.read_csv(yield_path)

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    required_cols = ["year", "state", "county", "state_ansi", "county_ansi", "commodity", "data_item", "value"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Yield data is missing required columns: {missing}")

    df["state"] = df["state"].astype(str).str.strip().str.upper()
    df["county"] = df["county"].astype(str).str.strip().str.upper()
    df["commodity"] = df["commodity"].astype(str).str.strip().str.upper()
    df["data_item"] = df["data_item"].astype(str).str.strip().str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # keep only target states and years
    df = df[df["state"].isin(VALID_STATES)].copy()
    df = df[df["year"].between(START_YEAR, END_YEAR)].copy()

    # keep only corn grain yield rows
    df = df[df["commodity"] == "CORN"].copy()
    df = df[df["data_item"].str.contains("YIELD", na=False)].copy()
    df = df[df["data_item"].str.contains("BU / ACRE", na=False)].copy()

    # remove agg rows that are not actual counties
    bad_county_patterns = [
        "OTHER COUNTIES",
        "COMBINED COUNTIES",
    ]
    for pattern in bad_county_patterns:
        df = df[~df["county"].str.contains(pattern, na=False)].copy()

    # numeric yield
    df["value"] = (
        df["value"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["yield_bu_acre"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["yield_bu_acre"]).copy()

    # county cleaning
    df["county_clean"] = df["county"].apply(standardize_county_name)

    # FIPS code
    df["state_ansi"] = pd.to_numeric(df["state_ansi"], errors="coerce")
    df["county_ansi"] = pd.to_numeric(df["county_ansi"], errors="coerce")
    df = df.dropna(subset=["state_ansi", "county_ansi"]).copy()

    df["fips"] = (
        df["state_ansi"].astype(int).astype(str).str.zfill(2)
        + df["county_ansi"].astype(int).astype(str).str.zfill(3)
    )

    # region label
    df["region_group"] = df["state"].apply(assign_region)

    keep_cols = [
        "year",
        "state",
        "county",
        "county_clean",
        "state_ansi",
        "county_ansi",
        "fips",
        "commodity",
        "data_item",
        "yield_bu_acre",
        "region_group",
    ]

    df = df[keep_cols].drop_duplicates().reset_index(drop=True)
    return df


def main(
    yield_path: str | Path = "data/raw/usda_full_yield.csv",
    processed_dir: str | Path = "data/clean",
) -> None:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    yield_df = clean_yield_data(yield_path)

    corn_belt_df = yield_df[yield_df["state"].isin(CORN_BELT_STATES)].copy()
    great_plains_df = yield_df[yield_df["state"].isin(GREAT_PLAINS_STATES)].copy()

    yield_df.to_csv(processed_dir / "usda_cornbelt_greatplains_combined.csv", index=False)
    corn_belt_df.to_csv(processed_dir / "usda_cornbelt_cleaned.csv", index=False)
    great_plains_df.to_csv(processed_dir / "usda_greatplains_cleaned.csv", index=False)

    print("Saved:")
    print(f"- {processed_dir / 'usda_cornbelt_greatplains_combined.csv'}")
    print(f"- {processed_dir / 'usda_cornbelt_cleaned.csv'}")
    print(f"- {processed_dir / 'usda_greatplains_cleaned.csv'}")
    print()
    print("Shapes:")
    print(f"Combined: {yield_df.shape}")
    print(f"Corn Belt: {corn_belt_df.shape}")
    print(f"Great Plains: {great_plains_df.shape}")
    print()
    print("States in combined dataset:")
    print(sorted(yield_df['state'].unique()))
    print()
    print("Sample:")
    print(yield_df.head())


if __name__ == "__main__":
    main()