import rasterio

raster_path = "data/raw/prism_unzipped/tmax/prism_tmax_us_25m_200005/prism_tmax_us_25m_200005.tif"

with rasterio.open(raster_path) as src:
    print("Opened raster successfully")
    print("CRS:", src.crs)
    print("Shape:", src.shape)
    print("Bounds:", src.bounds)
    print("Count:", src.count)

    