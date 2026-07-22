import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from elapid import MaxentModel
from sklearn.inspection import permutation_importance
# ============================================================
# 0. Config — modify paths as needed
# ============================================================
POLYGON_SHP = "./output/crop_polygon.shp"
POINT_SHP   = "./output/crop_point.shp"
CROP_FIELD  = "name"
ENV_RASTERS = {
    "Temp":      "./data/climate/Total mean/mean_temperature.tif",
    "Precip":    "./data/climate/Total mean/mean_precipitation.tif",
    "AccTemp":   "./data/climate/Total mean/mean_GDD.tif",
    "DEM":       "./data/terrain/dem.tif",
    "Slope":     "./data/terrain/slope.tif",
    "Roughness": "./data/terrain/rough.tif",
    "DistRiver": "./data/terrain/rivers_dir.tif",
    "TWI":       "./data/terrain/TWI.tif",
}
OUTPUT_DIR  = "./crop_factors"
N_BG_POINTS = 15000
POLYGON_SAMPLE_DENSITY = 0.0001   # sampling density within polygons
TARGET_RES  = 1000
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================================================
# 1. Load environmental factor rasters
# ============================================================
print("1. Loading environmental rasters...")
BOUNDARY_SHP = "./data/boundaries/national_border.shp"
env_arrays = {}
env_names = list(ENV_RASTERS.keys())
# 1.1 Read all rasters
for i, (name, path) in enumerate(ENV_RASTERS.items()):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        else:
            arr_min = np.nanmin(arr)
            if arr_min < -999 or arr_min == -1000:
                arr[arr == arr_min] = np.nan
        env_arrays[name] = arr
        if i == 0:      # use first raster as reference
            ref_transform = src.transform
            ref_crs = src.crs if src.crs is not None else rasterio.crs.CRS.from_epsg(4326)
            ref_height, ref_width = arr.shape
print(f"   Reference raster (Temp): {ref_width} × {ref_height}, CRS: {ref_crs}")
# 1.2 Align mismatched rasters
for name in env_names:
    h, w = env_arrays[name].shape
    if h != ref_height or w != ref_width:
        print(f"   [WARN] {name}: {w}×{h} != {ref_width}×{ref_height}, aligning...")
        aligned = np.empty((ref_height, ref_width), dtype=np.float32)
        with rasterio.open(ENV_RASTERS[name]) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=aligned,
                src_transform=src.transform,
                src_crs=src.crs if src.crs else ref_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
            )
        aligned[(aligned < -999) | (aligned == -1000)] = np.nan
        env_arrays[name] = aligned
        print(f"   -> aligned")
# 1.3 Rasterize boundary mask (much faster than rio_mask)
from rasterio import features
china_gdf = gpd.read_file(BOUNDARY_SHP)
if china_gdf.crs is None:
    china_gdf = china_gdf.set_crs(ref_crs)
if china_gdf.crs != ref_crs:
    china_gdf = china_gdf.to_crs(ref_crs)
china_mask = features.rasterize(
    [(g, 1) for g in china_gdf.geometry],
    out_shape=(ref_height, ref_width),
    transform=ref_transform,
    fill=0,
    dtype='uint8',
    all_touched=True,
)
# 1.4 Apply mask + summary stats
for name in env_names:
    env_arrays[name][china_mask == 0] = np.nan
    nan_pct = np.isnan(env_arrays[name]).mean() * 100
    print(f"   {name}: {np.nanmin(env_arrays[name]):.2f} ~ {np.nanmax(env_arrays[name]):.2f}, NaN={nan_pct:.1f}%")
# 1.5 Valid pixel mask
mask = np.ones((ref_height, ref_width), dtype=bool)
for arr in env_arrays.values():
    mask &= ~np.isnan(arr)
dst_crs = ref_crs
raster_transform = ref_transform
print(f"   Valid pixels: {mask.sum():,} / {ref_width * ref_height:,}")
print(f"   Loaded: {ref_width} × {ref_height}")
# ============================================================
# 2. Prepare presence points per crop
# ============================================================
print("2. Preparing presence points...")
poly_gdf = gpd.read_file(POLYGON_SHP)
point_gdf = gpd.read_file(POINT_SHP)
# Project to match raster CRS
if poly_gdf.crs is None:
    poly_gdf = poly_gdf.set_crs(ref_crs)
if point_gdf.crs is None:
    point_gdf = point_gdf.set_crs(ref_crs)
if poly_gdf.crs != ref_crs:
    poly_gdf = poly_gdf.to_crs(ref_crs)
if point_gdf.crs != ref_crs:
    point_gdf = point_gdf.to_crs(ref_crs)
poly_gdf = poly_gdf.explode(index_parts=False).reset_index(drop=True)
point_gdf = point_gdf.explode(index_parts=False).reset_index(drop=True)
crop_names = sorted(set(poly_gdf[CROP_FIELD].unique()) | set(point_gdf[CROP_FIELD].unique()))
print(f"   Crop types: {crop_names}")
# Precompute mask geo-coords for fast filtering
mask_rows, mask_cols = np.where(mask)
mask_xs = raster_transform.c + (mask_cols + 0.5) * raster_transform.a
mask_ys = raster_transform.f + (mask_rows + 0.5) * raster_transform.e
# Mask bounding box for approximate clipping
mask_xmin, mask_xmax = mask_xs.min(), mask_xs.max()
mask_ymin, mask_ymax = mask_ys.min(), mask_ys.max()
def is_valid_point(x, y):
    """Check if a point falls within a valid raster cell"""
    col = int((x - raster_transform.c) / raster_transform.a)
    row = int((y - raster_transform.f) / raster_transform.e)
    if row < 0 or row >= ref_height or col < 0 or col >= ref_width:
        return False
    return mask[row, col]
MAX_SAMPLES_PER_POLY = 1000
MIN_SAMPLES_PER_POLY = 400
crop_presence_points = {}
for crop in crop_names:
    pts = []
    # Random sampling within polygons
    crop_polys = poly_gdf[poly_gdf[CROP_FIELD] == crop]
    for _, row in crop_polys.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # Clip polygon to mask bounding box first
        from shapely.geometry import box
        bbox = box(mask_xmin, mask_ymin, mask_xmax, mask_ymax)
        try:
            geom_clipped = geom.intersection(bbox)
        except Exception:
            continue
        if geom_clipped.is_empty:
            continue
        n_samples = min(MAX_SAMPLES_PER_POLY, max(MIN_SAMPLES_PER_POLY,
                         int(geom_clipped.area / 1e9)))
        samples = []
        minx, miny, maxx, maxy = geom_clipped.bounds
        attempts = 0
        while len(samples) < n_samples and attempts < n_samples * 30:
            xs = np.random.uniform(minx, maxx, n_samples * 2)
            ys = np.random.uniform(miny, maxy, n_samples * 2)
            for x, y in zip(xs, ys):
                p = Point(x, y)
                if geom_clipped.contains(p) and is_valid_point(x, y):
                    samples.append((x, y))
                if len(samples) >= n_samples:
                    break
            attempts += n_samples * 2
        pts.extend(samples)
    # Point features (keep only those within valid cells)
    crop_points = point_gdf[point_gdf[CROP_FIELD] == crop]
    for _, row in crop_points.iterrows():
        x, y = row.geometry.x, row.geometry.y
        if is_valid_point(x, y):
            pts.append((x, y))
    if pts:
        crop_presence_points[crop] = np.array(pts)
        print(f"   {crop}: {len(pts)} presence points")
    else:
        print(f"   {crop}: no valid presence points")
# ============================================================
# 3. Run MaxEnt per crop
# ============================================================
print("3. Running models...")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
results = {}
for crop, pres_pts in crop_presence_points.items():
    print(f"\n--- {crop} ---")
    # 3.1 Extract environmental values at presence points
    presence_vals = []
    for x, y in pres_pts:
        col = int((x - raster_transform.c) / raster_transform.a)
        row = int((y - raster_transform.f) / raster_transform.e)
        if row < 0 or row >= ref_height or col < 0 or col >= ref_width:
            continue
        vals = [env_arrays[name][row, col] for name in env_names]
        if not any(np.isnan(v) for v in vals):
            presence_vals.append(vals)
    presence_vals = np.array(presence_vals)
    if len(presence_vals) < 10:
        print(f"   Too few valid presence points ({len(presence_vals)}), skipping")
        continue
    print(f"   Valid presence points: {len(presence_vals)}")
    # 3.2 Background points (vectorized)
    valid_rows, valid_cols = np.where(mask)
    bg_idx = np.random.choice(len(valid_rows), N_BG_POINTS, replace=True)
    bg_rows = valid_rows[bg_idx]
    bg_cols = valid_cols[bg_idx]
    background_vals = np.column_stack([
        env_arrays[name][bg_rows, bg_cols] for name in env_names
    ])
    # 3.3 Build training data + standardize
    X = np.vstack([presence_vals, background_vals])
    y = np.hstack([np.ones(len(presence_vals)), np.zeros(N_BG_POINTS)])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_pres = len(presence_vals)
    n_bg = N_BG_POINTS
    # 3.4 Train LogisticRegression (L1 penalty = MaxEnt equivalent)
    model = LogisticRegression(
        penalty='l1',
        C=1.0,                  # regularization strength (smaller = stronger)
        solver='saga',          # saga supports L1 + large data
        max_iter=2000,
        class_weight='balanced',
        random_state=42,
    )
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)
    model.fit(X_poly, y)
    # 3.5 Self-check
    base_preds = model.predict_proba(poly.transform(X_scaled))[:, 1]
    pres_mean = base_preds[y == 1].mean()
    bg_mean   = base_preds[y == 0].mean()
    acc = ((base_preds > 0.5).astype(int) == y).mean()
    print(f"   Presence prob mean: {pres_mean:.4f}, Background: {bg_mean:.4f}, Accuracy: {acc:.4f}")
    # 3.6 Contribution rate (permutation → drop in presence probability)
    base_pres_mean = pres_mean
    importances = []
    for fi in range(len(env_names)):
        drops = []
        for _ in range(5):
            X_perm = X_scaled.copy()
            np.random.shuffle(X_perm[:, fi])
            X_perm_poly = poly.transform(X_perm)
            perm_preds = model.predict_proba(X_perm_poly)[:, 1]
            drops.append(base_pres_mean - perm_preds[y == 1].mean())
        importances.append(max(np.mean(drops), 0))
    importances = np.array(importances)
    total = importances.sum()
    if total > 0:
        importances = importances / total * 100
    else:
        importances = np.full(len(env_names), 100.0 / len(env_names))
    contrib_df = pd.DataFrame({
        "factor": env_names,
        "contribution": importances
    }).sort_values("contribution", ascending=False)
    top_factor = contrib_df.iloc[0]["factor"]
    print(f"   Top factor: {top_factor} ({contrib_df.iloc[0]['contribution']:.1f}%)")
    print(f"   Contributions: " + ", ".join(f"{r['factor']}={r['contribution']:.0f}%" for _, r in contrib_df.iterrows()))
    results[crop] = {
        "model": model,
        "scaler": scaler,
        "poly": poly,
        "contribution": contrib_df,
        "top_factor": top_factor,
    }
# ============================================================
# 4. Contribution plots
# ============================================================
print("\n4. Plotting contributions...")
# Figure 1: Line chart per crop
fig, ax = plt.subplots(figsize=(14, 6))
x = range(len(env_names))
for crop, res in results.items():
    df = res["contribution"].set_index("factor").reindex(env_names)
    ax.plot(x, df["contribution"].values, marker="o", label=crop, linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(env_names, rotation=45, ha="right")
ax.set_title("Factor Contribution by Crop", fontsize=14, fontweight="bold")
ax.set_ylabel("Contribution (%)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "contribution_line.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   -> contribution_line.png")
# Figure 2: Mean contribution bar chart
fig, ax = plt.subplots(figsize=(10, 5))
avg_contrib = {}
for name in env_names:
    vals = [results[c]["contribution"].set_index("factor").loc[name, "contribution"]
            for c in results if name in results[c]["contribution"]["factor"].values]
    avg_contrib[name] = np.mean(vals) if vals else 0
sorted_f = sorted(avg_contrib.items(), key=lambda x: x[1], reverse=True)
factors, values = zip(*sorted_f)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(factors)))
ax.barh(range(len(factors)), values, color=colors, edgecolor="white")
ax.set_yticks(range(len(factors)))
ax.set_yticklabels(factors)
ax.set_xlabel("Mean Contribution (%)")
ax.set_title("Mean Factor Contribution", fontsize=14, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "contribution_bar.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   -> contribution_bar.png")
# ============================================================
# 5. Response curves for top factor
# ============================================================
print("5. Plotting response curves...")
for crop, res in results.items():
    fig, ax = plt.subplots(figsize=(7, 4))
    model = res["model"]
    factor_idx = env_names.index(res["top_factor"])
    factor_vals = np.linspace(
        np.nanpercentile(env_arrays[res["top_factor"]], 1),
        np.nanpercentile(env_arrays[res["top_factor"]], 99),
        100
    )
    median_bg = np.array([np.nanmedian(env_arrays[n]) for n in env_names])
    X_fixed = np.tile(median_bg, (100, 1))
    X_fixed[:, factor_idx] = factor_vals
    X_fixed_scaled = res["scaler"].transform(X_fixed)
    X_poly_fixed = poly.transform(X_fixed_scaled)
    preds = res["model"].predict_proba(X_poly_fixed)[:, 1]
    ax.plot(factor_vals, preds, color="#2c7bb6", linewidth=2)
    ax.fill_between(factor_vals, 0, preds, alpha=0.15, color="#2c7bb6")
    ax.set_xlabel(res["top_factor"], fontsize=11)
    ax.set_ylabel("Suitability Probability", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_name = crop.replace("/", "_").replace("\\", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, f"response_{safe_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   -> response_{safe_name}.png")
# ============================================================
# 6. Suitability prediction maps
# ============================================================
print("6. Generating suitability maps...")
full_data = np.column_stack([env_arrays[n].ravel() for n in env_names])
mask_flat = mask.ravel()
valid_mask_global = ~np.isnan(full_data).any(axis=1) & mask_flat
out_profile = {
    "driver": "GTiff",
    "height": ref_height,
    "width": ref_width,
    "count": 1,
    "dtype": "float32",
    "crs": dst_crs,
    "transform": raster_transform,
    "nodata": -9999,
}
for crop, res in results.items():
    print(f"   Predicting {crop}...")
    model = res["model"]
    scaler = res["scaler"]
    poly = res["poly"]
    predictions = np.full(full_data.shape[0], np.nan, dtype=np.float32)
    batch_size = 50000
    valid_idx = np.where(valid_mask_global)[0]
    for start in range(0, len(valid_idx), batch_size):
        end = min(start + batch_size, len(valid_idx))
        batch_idx = valid_idx[start:end]
        batch_scaled = scaler.transform(full_data[batch_idx])
        batch_poly = poly.transform(batch_scaled)
        predictions[batch_idx] = model.predict_proba(batch_poly)[:, 1]
    pred_2d = predictions.reshape(ref_height, ref_width)
    out_path = os.path.join(OUTPUT_DIR, f"suitability_{crop}.tif")
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(pred_2d, 1)
print("\nDone!")