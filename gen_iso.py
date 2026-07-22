import os
import numpy as np
import rasterio
from shapely.geometry import LineString
import geopandas as gpd

data_dir = "./data/climate/Annual mean temperature/"
ras_file = data_dir + "mean.tif"
RAIN_INTER = 200
TEMP_INTER = 5

def test_raster(tif_path):
    with rasterio.open(tif_path) as src:
        arr_raw = src.read(1)
        nodata = src.nodata
        arr = arr_raw.astype(np.float32)
        if nodata is not None:
            arr[arr_raw == nodata] = np.nan
        print(f"File: {os.path.basename(tif_path)}")
        print(f"  Shape:    {src.shape}")
        print(f"  Coordinates:  {src.crs.to_string()}")
        print(f"  Transform:\n{src.transform}")
        print(f"  nodata:  {nodata}")
        print(f"  Data Type: {src.dtypes[0]}")
        if nodata is not None:
            arr[arr_raw == nodata] = np.nan
            valid_mask = ~np.isnan(arr)
            print(f"  Valid pixels: {valid_mask.sum()} / {arr.size} ({valid_mask.sum()/arr.size*100:.1f}%)")
            print(f"  Valid value range: {arr[valid_mask].min():.2f} ~ {arr[valid_mask].max():.2f}")
        else:
            print(f"  Value Range:  {arr.min():.2f} ~ {arr.max():.2f}")
    print("-" * 50)
    
def raster_to_contour(tif_path, interval, out_dir=None):
    if out_dir is None:
        out_dir = os.path.dirname(tif_path)
    name = os.path.splitext(os.path.basename(tif_path))[0]

    fname = tif_path.lower()
    if "precip" in fname:
        dtype = "precip"
        unit = "mm"
        value_field = "value_mm"
        name += "rain"
    else:
        dtype = "temp"
        unit = "℃"
        value_field = "value_c"
        name += "temp"
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
    mask = ~np.isnan(arr)
    arr_filled = np.where(mask, arr, 0)
    arr_clean = median_filter(arr_filled, size=20)
    arr_coarse = uniform_filter(arr_clean, size=35)
    arr_smooth = gaussian_filter(arr_coarse, sigma=75)
    arr = np.where(mask, arr_smooth, np.nan)
    data_min, data_max = np.nanmin(arr), np.nanmax(arr)

    if dtype == "temp" and interval < 1:
        inner = np.arange(
            np.ceil(data_min / interval) * interval,
            np.floor(data_max / interval) * interval + interval / 2,
            interval,
        )
    else:
        inner = np.arange(
            np.ceil(data_min / interval) * interval,
            np.floor(data_max / interval) * interval + interval,
            interval,
        )
    print(f"Valid values range: {data_min:.1f} ~ {data_max:.1f} {unit}")
    print(f"Contours: {inner.tolist()}")
    from matplotlib import pyplot as plt
    fig = plt.figure()
    cs = plt.contour(arr, levels=inner)
    plt.close(fig)
    lines = []
    for level, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) < 2:
                continue
            geo_xy = [transform * (col, row) for col, row in seg]
            line = LineString(geo_xy)
            if not line.is_empty and line.is_valid:
                line = line.simplify(tolerance=0.1)
                if line.length < 0.5:
                    continue
                v_display = round(level, 1) if dtype == "temp" else int(level)
                lines.append({
                    value_field: v_display,
                    "label": f"{v_display} {unit}",
                    "geometry": line,
                })
    gdf = gpd.GeoDataFrame(lines, crs=crs)
    out_path = os.path.join(out_dir, f"{name}_contour_{int(interval)}.shp")
    gdf.to_file(out_path, encoding="utf-8")
    print(f"\nFinished: {out_path}  ({len(gdf)} Contours)")
    return out_path
    
def main():
    test_raster(ras_file)
    raster_to_contour(ras_file, RAIN_INTER)
    
if __name__ == "__main__":
    main()