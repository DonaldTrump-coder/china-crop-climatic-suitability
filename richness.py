import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.mask import mask
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Microsoft YaHei"
# ===== Load data =====
poly = gpd.read_file("./output/crop_polygon.shp")
pts  = gpd.read_file("./output/crop_point.shp")
china = gpd.read_file("./output/china_boundary.shp")
pts = pts.explode(index_parts=False).reset_index(drop=True)
crs_proj = "ESRI:102025"  # Asia North Albers Equal Area Conic
poly  = poly.to_crs(crs_proj)
pts   = pts.to_crs(crs_proj)
china = china.to_crs(crs_proj)
crops = ["Wheat", "Rice", "Cotton", "Peanut", "Rapeseed", "SugarBeet", "Sugarcane"]
cell_size = 1000  # meters
# Use China boundary as grid extent
bounds = china.total_bounds  # [xmin, ymin, xmax, ymax]
width  = int((bounds[2] - bounds[0]) / cell_size)
height = int((bounds[3] - bounds[1]) / cell_size)
transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
print(f"Grid: {width}×{height}")
POLY_SIGMA = 5
PT_SIGMA   = 35
TOTAL_SIGMA = 15
n_cols = 4
n_rows = int(np.ceil(len(crops) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
axes = axes.flatten()
china_geom = china.geometry.union_all()
# ===== Per-crop binarization + point counting =====
all_crop = []
for i, crop in enumerate(crops):
    print(f"Processing {crop} ...")
    sub_poly = poly[poly["name"] == crop]
    sub_pts  = pts[pts["name"] == crop]
    # Polygon → 0/1
    if len(sub_poly) > 0:
        poly_arr = rasterize([(g, 1) for g in sub_poly.geometry],
                             out_shape=(height, width), transform=transform,
                             fill=0, dtype=np.float32)
        poly_arr = gaussian_filter(poly_arr, sigma=POLY_SIGMA)
        if poly_arr.max() > 0:
            poly_arr /= poly_arr.max()
    else:
        poly_arr = np.zeros((height, width), dtype=np.float32)
    # Point → count + Gaussian smoothing
    pts_arr = np.zeros((height, width), dtype=np.float32)
    if len(sub_pts) > 0:
        xs, ys = sub_pts.geometry.x.values, sub_pts.geometry.y.values
        cols = ((xs - bounds[0]) / cell_size).astype(int)
        rows = ((bounds[3] - ys) / cell_size).astype(int)
        valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        for c, r in zip(cols[valid], rows[valid]):
            pts_arr[r, c] += 1
        pts_arr = gaussian_filter(pts_arr, sigma=PT_SIGMA)
        if pts_arr.max() > 0:
            pts_arr /= pts_arr.max()
    # Fusion
    fused = np.maximum(poly_arr, pts_arr)
    all_crop.append(fused)
    profile = {
        "driver": "GTiff", "height": height, "width": width,
        "count": 1, "dtype": np.float32, "crs": poly.crs, "transform": transform
    }
    safe_name = crop.replace("/", "_")
    with rasterio.open(f"./output/fused_{safe_name}.tif", "w", **profile) as dst:
        dst.write(fused, 1)
    with rasterio.open(f"./output/fused_{safe_name}.tif") as src:
        clipped, clip_transform = mask(src, [china_geom], crop=True, nodata=np.nan)
        clipped_profile = src.profile.copy()
        clipped_profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=clip_transform,
            nodata=np.nan
        )
        with rasterio.open(f"./output/fused_{safe_name}_clip.tif", "w", **clipped_profile) as dst:
            dst.write(clipped)
    # ===== Plot each crop =====
    ax = axes[i]
    masked = np.ma.masked_where(fused < 0.01, fused)
    im = ax.imshow(masked, cmap="YlOrRd", extent=bounds[[0,2,1,3]], vmin=0, vmax=1)
    china.boundary.plot(ax=ax, color="black", linewidth=0.5)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_title(crop)
    ax.axis("off")
# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.colorbar(im, ax=axes, fraction=0.02, label="Distribution intensity", pad=0.02)
plt.suptitle("Fused crop distribution rasters (yellow = distribution area / dense points, black = border)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
# ===== Richness map =====
richness = np.sum(all_crop, axis=0)
richness = median_filter(richness, size=3)     # Remove isolated speckles
richness = gaussian_filter(richness, sigma=TOTAL_SIGMA)  # Overall smoothing
with rasterio.open("./output/crop_richness_full.tif", "w", **profile) as dst:
    dst.write(richness, 1)
with rasterio.open("./output/crop_richness_full.tif") as src:
    clipped, clip_transform = mask(src, [china_geom], crop=True, nodata=np.nan)
    cp = src.profile.copy()
    cp.update(
        height=clipped.shape[1],
        width=clipped.shape[2],
        transform=clip_transform,
        nodata=np.nan
    )
    with rasterio.open("./output/crop_richness.tif", "w", **cp) as dst:
        dst.write(clipped)
fig2, ax2 = plt.subplots(figsize=(10, 10))
masked_r = np.ma.masked_where(richness < 0.05, richness)
im2 = ax2.imshow(masked_r, cmap="YlOrRd", extent=bounds[[0,2,1,3]])
china.boundary.plot(ax=ax2, color="black", linewidth=0.8)
ax2.set_xlim(bounds[0], bounds[2])
ax2.set_ylim(bounds[1], bounds[3])
ax2.set_title("Crop richness", fontsize=14, fontweight="bold")
ax2.axis("off")
plt.colorbar(im2, ax=ax2, fraction=0.03, label="Species richness")
plt.tight_layout()
plt.show()
print("Done → output/crop_richness.tif")