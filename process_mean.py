import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import os
from tqdm import tqdm

def main():
    data_dir = r"./data/climate/Annual mean temperature"
    tifs = sorted(glob.glob(data_dir + r"\*.tif"))
    SCALE = 0.1

    with rasterio.open(tifs[0]) as ref:
        profile = ref.profile
        h, w = ref.height, ref.width
        nodata = ref.nodata
        bounds = ref.bounds
    sum_arr = np.zeros((h, w), dtype=np.float64)
    count_arr = np.zeros((h, w), dtype=np.int16)
    for f in tqdm(tifs, desc="Progress", unit="Years"):
        with rasterio.open(f) as src:
            file_nodata = src.nodata
            if src.shape == (h, w):
                arr_raw = src.read(1).astype(np.float64)
            else:
                arr_raw_src = src.read(1).astype(np.float64)
                arr_raw = np.empty((h, w), dtype=np.float64)
                reproject(
                    source=arr_raw_src,
                    destination=arr_raw,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=Resampling.nearest,
                )

        if file_nodata is not None:
            mask = arr_raw != file_nodata
        else:
            mask = np.ones_like(arr_raw, dtype=bool)

        arr = arr_raw * SCALE
        sum_arr[mask] += arr[mask]
        count_arr[mask] += 1
    mean_arr = np.full((h, w), nodata, dtype=np.float32)
    valid = count_arr > 0
    mean_arr[valid] = (sum_arr[valid] / count_arr[valid]).astype(np.float32)
    profile.update(dtype=rasterio.float32, nodata=nodata, count=1)
    out_path = data_dir + "/mean.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mean_arr, 1)
    print(f"Finished: {out_path}")
    
if __name__ == "__main__":
    main()