from __future__ import annotations

from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask


COUNTY_SHP = Path("data/shapefiles/tl_2022_us_county/tl_2022_us_county.shp")
TMAX_DIR = Path("data/raw/prism_unzipped/tmax")
PPT_DIR = Path("data/raw/prism_unzipped/ppt")
OUTPUT_DIR = Path("data/clean")

STATEFP_TO_NAME = {
    "17": "ILLINOIS",
    "18": "INDIANA",
    "19": "IOWA",
    "27": "MINNESOTA",
    "29": "MISSOURI",
    "39": "OHIO",
    "55": "WISCONSIN",
    "20": "KANSAS",
    "31": "NEBRASKA",
    "38": "NORTH DAKOTA",
    "40": "OKLAHOMA",
    "46": "SOUTH DAKOTA",
    "48": "TEXAS",
}
TARGET_STATES = set(STATEFP_TO_NAME.values())
VALID_MONTHS = [5, 6, 7, 8, 9]
START_YEAR = 2000
END_YEAR = 2022


def load_counties() -> gpd.GeoDataFrame:
    counties = gpd.read_file(COUNTY_SHP)

    counties["STATEFP"] = counties["STATEFP"].astype(str)
    counties["state"] = counties["STATEFP"].map(STATEFP_TO_NAME)

    counties = counties[counties["state"].isin(TARGET_STATES)].copy()

    counties["fips"] = counties["STATEFP"] + counties["COUNTYFP"]

    counties = counties[["fips", "NAME", "STATEFP", "state", "geometry"]].copy()
    return counties


def parse_year_month(filename: str) -> tuple[int, int]:
    match = re.search(r"(\d{4})(\d{2})", filename)
    if not match:
        raise ValueError(f"Could not parse year/month from filename: {filename}")
    year = int(match.group(1))
    month = int(match.group(2))
    return year, month


def get_tif_files(base_dir: Path) -> list[Path]:
    return sorted(base_dir.rglob("*.tif"))


def summarize_raster_for_counties(
    raster_path: Path,
    counties: gpd.GeoDataFrame,
    value_col: str,
) -> pd.DataFrame:
    year, month = parse_year_month(raster_path.name)

    with rasterio.open(raster_path) as src:
        counties_proj = counties.to_crs(src.crs)

        rows = []
        for _, row in counties_proj.iterrows():
            geom = [row.geometry.__geo_interface__]

            out_image, _ = mask(src, geom, crop=True, filled=False)
            data = out_image[0]

            mean_val = float(data.mean()) if data.count() > 0 else np.nan

            rows.append({
                "fips": row["fips"],
                "state": row["state"],
                "county": row["NAME"],
                "year": year,
                "month": month,
                value_col: mean_val,
            })

    return pd.DataFrame(rows)

def process_variable(
    variable_dir: Path,
    variable_name: str,
    counties: gpd.GeoDataFrame,
) -> pd.DataFrame:
    tif_files = get_tif_files(variable_dir)

    all_rows = []
    for i, tif in enumerate(tif_files):
        year, month = parse_year_month(tif.name)

        if year < START_YEAR or year > END_YEAR:
            continue
        if month not in VALID_MONTHS:
            continue

        print(f"Processing {variable_name}: {tif.name}")
        df = summarize_raster_for_counties(tif, counties, variable_name)
        all_rows.append(df)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} {variable_name} files")

    if not all_rows:
        raise ValueError(f"No valid files processed for {variable_name}")

    return pd.concat(all_rows, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counties = load_counties()
    print(f"Loaded {len(counties)} counties")

    # monthly county-level tmax and ppt
    tmax_monthly = process_variable(TMAX_DIR, "tmax", counties)
    ppt_monthly = process_variable(PPT_DIR, "ppt", counties)

    # monthly intermediates
    tmax_monthly.to_csv(OUTPUT_DIR / "prism_tmax_county_month.csv", index=False)
    ppt_monthly.to_csv(OUTPUT_DIR / "prism_ppt_county_month.csv", index=False)

    # merge monthly variables
    prism_monthly = tmax_monthly.merge(
        ppt_monthly[["fips", "state", "county", "year", "month", "ppt"]],
        on=["fips", "state", "county", "year", "month"],
        how="inner",
    )

    prism_monthly.to_csv(OUTPUT_DIR / "prism_county_month.csv", index=False)

    # agg to county-year growing-season features
    prism_yearly = (
        prism_monthly
        .groupby(["fips", "state", "county", "year"], as_index=False)
        .agg(
            prism_tmax_may_sep_mean=("tmax", "mean"),
            prism_ppt_may_sep_total=("ppt", "sum"),
        )
    )

    prism_yearly.to_csv(OUTPUT_DIR / "prism_county_year.csv", index=False)

    print("\nSaved files:")
    print(OUTPUT_DIR / "prism_tmax_county_month.csv")
    print(OUTPUT_DIR / "prism_ppt_county_month.csv")
    print(OUTPUT_DIR / "prism_county_month.csv")
    print(OUTPUT_DIR / "prism_county_year.csv")

    print("\nShapes:")
    print("tmax monthly:", tmax_monthly.shape)
    print("ppt monthly:", ppt_monthly.shape)
    print("prism monthly:", prism_monthly.shape)
    print("prism yearly:", prism_yearly.shape)

    print("\nSample yearly output:")
    print(prism_yearly.head())


if __name__ == "__main__":
    main()