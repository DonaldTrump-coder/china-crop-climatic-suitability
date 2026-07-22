import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from glob import glob

folder = "./data/climate/Annual GDD/"
def main():
    tif_files = sorted(glob(os.path.join(folder, "*.tif")))
    if not tif_files:
        print("tif files not found.")
        return
    
    with rasterio.open(tif_files[0]) as src:
        profile = src.profile
    total = None
    count = None
    first = True
    total = None
    count = None
    first = True
    for f in tif_files:
        with rasterio.open(f) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            invalid = (data == nodata) | np.isnan(data) | np.isinf(data)
        else:
            invalid = np.isnan(data) | np.isinf(data)
        if first:
            total = np.where(invalid, 0.0, data)
            count = np.where(invalid, 0, 1)
            first = False
        else:
            total += np.where(invalid, 0.0, data)
            count += np.where(invalid, 0, 1)
    mean_data = np.where(count > 0, total / count, np.nan)
    profile.update(dtype=rasterio.float32, nodata=np.nan)
    out_path = os.path.join(folder, "mean_GDD.tif")
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(mean_data.astype(np.float32), 1)
    print(f"Saved: {out_path}")
    print(f"Data of {len(tif_files)} years processed.")
    
if __name__ == "__main__":
    main()