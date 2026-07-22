import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestRegressor
from rasterio.enums import Resampling
from rasterio.warp import reproject
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")
# ============================================================
# 1. Config
# ============================================================
richness_tif = "./output/crop_richness.tif"
factor_tifs = {
    "Temp":      "./data/climate/Total mean/mean_temperature.tif",
    "Precip":    "./data/climate/Total mean/mean_precipitation.tif",
    "AccTemp":   "./data/climate/Total mean/mean_GDD.tif",
    "DEM":       "./data/terrain/dem.tif",
    "Slope":     "./data/terrain/slope.tif",
    "Roughness": "./data/terrain/rough.tif",
    "DistRiver": "./data/terrain/rivers_dir.tif",
    "TWI":   "./data/terrain/TWI.tif",
}
n_samples = 15000
random_seed = 42
output_csv = "factor_importance.csv"
# ============================================================
# 2. Read reference raster (richness) → determine target grid
# ============================================================
with rasterio.open(richness_tif) as ref:
    ref_arr    = ref.read(1)
    ref_meta   = ref.meta.copy()
    ref_nodata = ref.nodata
    ref_shape  = ref.shape
    ref_transform = ref.transform
    ref_crs   = ref.crs
print(f"Reference raster: {ref_shape[1]}×{ref_shape[0]}, CRS={ref_crs}")
# Valid pixels (exclude NoData / NaN)
valid = (ref_arr != ref_nodata) & (~np.isnan(ref_arr))
rows_valid, cols_valid = np.where(valid)
n_valid = len(rows_valid)
print(f"Valid pixels: {n_valid}")
# ============================================================
# 3. Align each factor raster to reference grid
# ============================================================
factor_arrays = {}
for name, path in factor_tifs.items():
    with rasterio.open(path) as src:
        src_arr    = src.read(1).astype(np.float32)
        src_nodata = src.nodata
        # Missing CRS with lon 70–140, lat 0–60 → force EPSG:4326
        if src.crs is None:
            src_crs = "EPSG:4326"
        else:
            src_crs = src.crs
        same_shape = src.shape == ref_shape
        same_res   = (abs(src.transform.a - ref_transform.a) < 1e-8 and
                      abs(src.transform.e - ref_transform.e) < 1e-8)
        same_crs   = str(src_crs) == str(ref_crs)
        if same_shape and same_res and same_crs:
            print(f"  {name}: aligned")
            aligned = src_arr
        else:
            print(f"  {name}: {src.shape[1]}×{src.shape[0]}, CRS={src_crs} → {ref_crs}")
            aligned = np.full(ref_shape, np.nan, dtype=np.float32)
            reproject(
                source        = src_arr,
                destination   = aligned,
                src_transform = src.transform,
                src_crs       = src_crs,
                src_nodata    = src_nodata,
                dst_transform = ref_transform,
                dst_crs       = ref_crs,
                dst_nodata    = np.nan,
                resampling    = Resampling.bilinear,
            )
            n_good = (~np.isnan(aligned)).sum()
            print(f"    Valid after resampling: {n_good}")
        factor_arrays[name] = aligned
# ============================================================
# 4. Random sampling + extract values
# ============================================================
np.random.seed(random_seed)
n_sample = min(n_samples, n_valid)
idx = np.random.choice(n_valid, n_sample, replace=False)
sr = rows_valid[idx]
sc = cols_valid[idx]
data = {}
data["Richness"] = ref_arr[sr, sc]
for name in factor_tifs.keys():
    vals = factor_arrays[name][sr, sc].astype(np.float64)
    vals = np.where(np.isnan(vals) | np.isinf(vals), np.nan, vals)
    data[name] = vals
df = pd.DataFrame(data).dropna()
factor_names = list(factor_tifs.keys())
print(f"\nValid samples: {len(df)} / {n_sample}")
# ============================================================
# 5. Pearson r
# ============================================================
corr = df.corr()["Richness"].drop("Richness")
# ============================================================
# 6. RF modeling
# ============================================================
X = df[factor_names].values
y = df["Richness"].values
rf = RandomForestRegressor(
    n_estimators=300, max_depth=10, min_samples_leaf=5,
    random_state=random_seed, n_jobs=1
)
rf.fit(X, y)
r2 = rf.score(X, y)
# ============================================================
# 7. Permutation importance
# ============================================================
perm = permutation_importance(rf, X, y, n_repeats=10, random_state=random_seed, scoring="r2")
# ============================================================
# 8. Output
# ============================================================
result = pd.DataFrame({
    "Factor":            factor_names,
    "Permutation_Imp":   perm.importances_mean.round(4),
    "Std":               perm.importances_std.round(4),
    "Pearson_|r|":      np.abs(corr.values).round(3),
    "Gini_Imp":         rf.feature_importances_.round(4),
}).sort_values("Permutation_Imp", ascending=False).reset_index(drop=True)
result.insert(0, "Rank", range(1, len(result) + 1))
print(f"\nRF R² = {r2:.3f}\n")
print(f"{'Rank':<5s} {'Factor':10s} {'PermImp':>10s} {'±Std':>10s} {'|r|':>8s} {'Gini':>8s}")
print("-" * 60)
for _, row in result.iterrows():
    print(f"#{int(row['Rank']):<4d} {row['Factor']:10s} {row['Permutation_Imp']:>10.4f} "
          f"±{row['Std']:<9.4f} {row['Pearson_|r|']:>8.3f} {row['Gini_Imp']:>8.3f}")
result.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\nDone → {output_csv}")