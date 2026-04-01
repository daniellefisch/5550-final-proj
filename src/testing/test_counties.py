import geopandas as gpd
import rasterio

COUNTY_SHP = "data/shapefiles/tl_2022_us_county/tl_2022_us_county.shp"
RASTER_PATH = "data/raw/prism_unzipped/tmax/prism_tmax_us_25m_200005/prism_tmax_us_25m_200005.tif"

counties = gpd.read_file(COUNTY_SHP)
counties = counties[counties["STATEFP"].isin(["17", "18", "19", "20"])].copy()
counties["fips"] = counties["STATEFP"] + counties["COUNTYFP"]

with rasterio.open(RASTER_PATH) as src:
    raster_crs = src.crs

counties = counties.to_crs(raster_crs)

print("Loaded and reprojected counties successfully")
print(counties[["fips", "NAME"]].head())
print("CRS:", counties.crs)