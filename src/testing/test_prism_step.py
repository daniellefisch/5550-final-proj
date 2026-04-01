from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask

COUNTY_SHP = "data/shapefiles/tl_2022_us_county/tl_2022_us_county.shp"
PRISM_DIR = Path("data/raw/prism_unzipped/tmax")

# Load counties
counties = gpd.read_file(COUNTY_SHP)
counties = counties[counties["STATEFP"].isin(["17", "18", "19", "20"])].copy()
counties["fips"] = counties["STATEFP"] + counties["COUNTYFP"]

print(f"Loaded {len(counties)} counties")

# Find one PRISM tif
tif_files = sorted(PRISM_DIR.rglob("*.tif"))
if not tif_files:
    raise ValueError("No .tif files found.")

prism_file = tif_files[0]
print(f"Using PRISM file: {prism_file}")

# Use just 5 counties for testing
counties_small = counties.head(5).copy()

with rasterio.open(prism_file) as src:
    counties_small = counties_small.to_crs(src.crs)

    results = []
    for _, row in counties_small.iterrows():
        geom = [row.geometry.__geo_interface__]

        out_image, _ = mask(src, geom, crop=True, filled=False)
        data = out_image[0]

        # mean of unmasked values
        mean_val = float(data.mean()) if data.count() > 0 else np.nan

        results.append({
            "fips": row["fips"],
            "county": row["NAME"],
            "tmax": mean_val
        })

df = gpd.pd.DataFrame(results)

print("\nSample output:")
print(df)

print("\nSummary stats:")
print(df["tmax"].describe())