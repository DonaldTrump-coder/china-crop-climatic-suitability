import os, numpy as np, pandas as pd, rasterio, geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from rasterio import features
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ---- 0. Config ----
AGRO_SHP      = "./output/agro_provinces.shp"
PASTORAL_SHP  = "./output/pastoral_provinces.shp"
POLYGON_SHP   = "./output/crop_polygon.shp"
POINT_SHP     = "./output/crop_point.shp"
CROP_FIELD    = "name"
ENV_RASTERS = {
    "Temp":"./data/climate/Total mean/mean_temperature.tif","AccTemp":"./data/climate/Total mean/mean_GDD.tif","Precip":"./data/climate/Total mean/mean_precipitation.tif",
    "DEM":"./data/terrain/dem.tif","Slope":"./data/terrain/slope.tif","DistRiver":"./data/terrain/rivers_dir.tif",
    "Roughness":"./data/terrain/rough.tif","TWI":"./data/terrain/TWI.tif",
}
MAX_SAMPLES = 1600
MIN_SAMPLES = 500
# ========== 1. Load env rasters + rasterize agro-pastoral zones ==========
print("1. Loading data...")
ref_path = ENV_RASTERS["Temp"]
with rasterio.open(ref_path) as src:
    ref_t = src.transform; ref_crs = src.crs or rasterio.crs.CRS.from_epsg(4326)
    ref_h, ref_w = src.shape; ref_shape = (ref_h, ref_w)
env_n = list(ENV_RASTERS.keys())
env = {}
for k, p in ENV_RASTERS.items():
    with rasterio.open(p) as s:
        a = s.read(1).astype(np.float32)
        if a.shape != ref_shape:
            from rasterio.warp import reproject
            al = np.empty(ref_shape, dtype=np.float32)
            reproject(source=rasterio.band(s,1), destination=al,
                      src_transform=s.transform, src_crs=s.crs or ref_crs,
                      dst_transform=ref_t, dst_crs=ref_crs, resampling=0)
            a = al
        nd = s.nodata
        a[(a<-999)|(a==-1000)] = np.nan
        if nd is not None: a[a==nd] = np.nan
    env[k] = a
env_mask = np.ones(ref_shape, dtype=bool)
for a in env.values(): env_mask &= ~np.isnan(a)
# Rasterize agro-pastoral zones
zone_label = np.zeros(ref_shape, dtype=np.int8)
for shp_path, val in [(AGRO_SHP, 1), (PASTORAL_SHP, 2)]:
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None: gdf = gdf.set_crs(ref_crs)
    if gdf.crs != ref_crs: gdf = gdf.to_crs(ref_crs)
    rast = features.rasterize([(g, 1) for g in gdf.geometry],
                              out_shape=ref_shape, transform=ref_t, fill=0,
                              dtype='uint8', all_touched=True)
    zone_label[rast == 1] = val
agro_mask  = zone_label == 1
pastoral_mask = zone_label == 2
mask = env_mask & (agro_mask | pastoral_mask)
# Determine agro/pastoral attribute for each valid pixel
vr, vc = np.where(mask)
zy = agro_mask[vr, vc].astype(np.int8)
pr = (zy == 0).mean()
print(f"   Agro:{(zy==1).sum():,}  Pastoral:{(zy==0).sum():,}  Pastoral baseline={pr:.1%}")
# Helper: determine which zone a coordinate falls in
def get_zone(x, y):
    c = int((x - ref_t.c) / ref_t.a)
    r = int((y - ref_t.f) / ref_t.e)
    if r < 0 or r >= ref_h or c < 0 or c >= ref_w: return -1
    if not mask[r, c]: return -1
    return int(agro_mask[r, c])  # 1=agro, 0=pastoral
# ========== 2. Agro-pastoral zone drivers ==========
print("\n2. Zone drivers...")
XF = np.column_stack([env[n][vr, vc] for n in env_n])
n_s = min(50000, len(zy)); si = np.random.choice(len(zy), n_s, replace=False)
Xs, ys = XF[si], zy[si]
Xs = StandardScaler().fit_transform(Xs)
mz = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, random_state=42)
mz.fit(Xs, ys)
bp = mz.predict_proba(Xs)[:, 1]
print(f"   Agro prob:{bp[ys==1].mean():.4f}  Pastoral:{bp[ys==0].mean():.4f}  AUC:{roc_auc_score(ys,bp):.4f}")
am = bp[ys==1].mean()
imps = []
for fi in range(len(env_n)):
    drops = []
    for _ in range(5):
        Xp = Xs.copy(); np.random.shuffle(Xp[:, fi])
        drops.append(am - mz.predict_proba(Xp)[:, 1][ys==1].mean())
    imps.append(max(np.mean(drops), 0))
imps = np.array(imps)
imps = imps / imps.sum() * 100
zone_c = pd.DataFrame({"Factor":env_n, "Contribution":imps}).sort_values("Contribution", ascending=False)
print("   Zone driver ranking:")
for _, r in zone_c.iterrows(): print(f"      {r['Factor']:10s}  {r['Contribution']:5.1f}%")
# ========== 3. Crop distribution stats across zones ==========
print("\n3. Crop agro-pastoral distribution...")
poly_gdf = gpd.read_file(POLYGON_SHP)
point_gdf = gpd.read_file(POINT_SHP)
if poly_gdf.crs is None:
    poly_gdf = poly_gdf.set_crs(ref_crs)
if poly_gdf.crs != ref_crs:
    poly_gdf = poly_gdf.to_crs(ref_crs)
if point_gdf.crs is None:
    point_gdf = point_gdf.set_crs(ref_crs)
if point_gdf.crs != ref_crs:
    point_gdf = point_gdf.to_crs(ref_crs)
poly_gdf = poly_gdf.explode(index_parts=False).reset_index(drop=True)
point_gdf = point_gdf.explode(index_parts=False).reset_index(drop=True)
crop_names = sorted(set(poly_gdf[CROP_FIELD].unique()) | set(point_gdf[CROP_FIELD].unique()))
print(f"   Crops: {crop_names}")
# Mask bounding box (for polygon sampling)
mask_r, mask_c = np.where(mask)
mxmin = ref_t.c + mask_c.min() * ref_t.a
mxmax = ref_t.c + mask_c.max() * ref_t.a
mymin = ref_t.f + mask_r.max() * ref_t.e
mymax = ref_t.f + mask_r.min() * ref_t.e
TOTAL_PER_CROP = 2000
crop_stats = {}
for crop in crop_names:
    agro_pts = 0; pastoral_pts = 0
    valid_polys = []
    for _, row in poly_gdf[poly_gdf[CROP_FIELD] == crop].iterrows():
        g = row.geometry
        if g is None or g.is_empty: continue
        gc = g.intersection(box(mxmin, mymin, mxmax, mymax))
        if gc.is_empty: continue
        valid_polys.append(gc)
    if valid_polys:
        areas = [p.area for p in valid_polys]
        total_area = sum(areas)
        n_per_poly = [max(1, int(TOTAL_PER_CROP * a / total_area)) for a in areas]
        diff = TOTAL_PER_CROP - sum(n_per_poly)
        n_per_poly[n_per_poly.index(max(n_per_poly))] += diff
        for gc, n in zip(valid_polys, n_per_poly):
            collected = 0; att = 0; b = gc.bounds
            while collected < n and att < n * 30:
                xs = np.random.uniform(b[0], b[2], n * 2)
                ys = np.random.uniform(b[1], b[3], n * 2)
                for x, y in zip(xs, ys):
                    if gc.contains(Point(x, y)):
                        z = get_zone(x, y)
                        if z == 1: agro_pts += 1
                        elif z == 0: pastoral_pts += 1
                        collected += 1
                    if collected >= n: break
                att += n * 2
    for _, row in point_gdf[point_gdf[CROP_FIELD] == crop].iterrows():
        x, y = row.geometry.x, row.geometry.y
        z = get_zone(x, y)
        if z == 1: agro_pts += 1
        elif z == 0: pastoral_pts += 1
    total = agro_pts + pastoral_pts
    if total > 0:
        r_k = pastoral_pts / total
        crop_stats[crop] = {"AgroPts": agro_pts, "PastoralPts": pastoral_pts,
                           "TotalPts": total, "PastoralRatio": r_k}
        print(f"   {crop}: total={total}, agro={agro_pts} pastoral={pastoral_pts}, pastoral ratio={r_k:.1%}")
# Relative to baseline
r_median = np.median([s["PastoralRatio"] for s in crop_stats.values()])
print(f"   Pastoral area baseline={pr:.1%}, crop pastoral ratio median={r_median:.1%}")
br = []; cs = []
for crop, s in crop_stats.items():
    r_k = s["PastoralRatio"]
    C_abs = (r_k - pr) / pr * 100
    C_rel = (r_k - r_median) / r_median * 100
    cs.append({"Crop": crop, "AgroPts": s["AgroPts"], "PastoralPts": s["PastoralPts"],
               "TotalPts": s["TotalPts"], "PastoralRatio": r_k, "CrossIndex": C_rel})
    br.append({"Crop": crop, "PastoralRatio": r_k, "AbsCross": C_abs, "RelCross": C_rel})
    print(f"   {crop}: abs_cross={C_abs:+.0f}%, rel_cross={C_rel:+.0f}%")
bd = pd.DataFrame(br)
# ========== 4. Plots ==========
print("\n4. Generating plots...")
# Fig 1: Zone drivers
fig, ax = plt.subplots(figsize=(10, 5))
zp = zone_c.sort_values("Contribution", ascending=True)
ax.barh(zp["Factor"], zp["Contribution"], color=plt.cm.RdBu_r(np.linspace(0.15, 0.85, len(zp))), edgecolor="white")
for i, (_, r) in enumerate(zp.iterrows()):
    ax.text(r["Contribution"] + 0.5, i, f'{r["Contribution"]:.1f}%', va="center")
ax.set_xlabel("Contribution (%)"); ax.set_title("Environmental Drivers of Agro-Pastoral Zoning", fontsize=14, fontweight="bold")
ax.set_xlim(0, zp["Contribution"].max() * 1.2); ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "zone_drivers.png"), dpi=300); plt.close()
# Fig 2: Cross-boundary index (relative)
pv = bd.set_index("Crop").sort_values("RelCross", ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
cl = ["#e74c3c" if v > 0 else "#3498db" for v in pv["RelCross"]]
ax.barh(pv.index, pv["RelCross"], color=cl, edgecolor="white"); ax.axvline(x=0, color="black", lw=1.2)
for i, (_, r) in enumerate(pv.iterrows()):
    s = "+" if r["RelCross"] > 0 else ""
    ax.text(r["RelCross"] + (2 if r["RelCross"] >= 0 else -8), i, f'{s}{r["RelCross"]:.0f}%', va="center")
ax.set_xlabel("Cross-boundary Index (%)"); ax.set_title("Positive=Pastoral-biased · Negative=Agro-biased", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "cross_boundary_index.png"), dpi=300); plt.close()
# Fig 2.5: Absolute cross-boundary index (vs pastoral area baseline)
pv_abs = bd.set_index("Crop").sort_values("AbsCross", ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
cl_abs = ["#e67e22" if v > -80 else "#c0392b" for v in pv_abs["AbsCross"]]
ax.barh(pv_abs.index, pv_abs["AbsCross"], color=cl_abs, edgecolor="white")
ax.axvline(x=0, color="black", lw=1.2)
for i, (_, r) in enumerate(pv_abs.iterrows()):
    ax.text(r["AbsCross"] + 1, i, f'{r["AbsCross"]:.0f}%', va="center")
ax.set_xlabel("Absolute Cross-boundary Index (%)")
ax.set_title(f"Absolute Cross-boundary Index (vs pastoral baseline {pr:.1%})\nAll negative = all crops biased toward agro-zone", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "cross_boundary_absolute.png"), dpi=300); plt.close()
# Fig 3: Agro vs pastoral point counts
crl = [c["Crop"] for c in cs]
agro_counts = [c["AgroPts"] for c in cs]
pastoral_counts = [c["PastoralPts"] for c in cs]
fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(len(crl)); w = 0.35
ax.bar(x - w/2, agro_counts, w, label="Agro-zone", color="#27ae60", edgecolor="white")
ax.bar(x + w/2, pastoral_counts, w, label="Pastoral-zone", color="#e67e22", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(crl); ax.set_ylabel("Sample Point Count")
ax.set_title("Agro- vs Pastoral-zone Point Counts by Crop", fontsize=14, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "agro_vs_pastoral_counts.png"), dpi=300); plt.close()
# Summary
print("\n" + "=" * 60)
print("1. Agro-pastoral zone drivers:")
for _, r in zone_c.iterrows(): print(f"    {r['Factor']:10s} → {r['Contribution']:5.1f}%")
print("\n2. Cross-boundary index (relative vs crop median):")
for _, r in pv.iterrows():
    d = ">> Pastoral-type" if r["RelCross"] > 20 else "> Pastoral-leaning" if r["RelCross"] > 0 else "> Agro-leaning" if r["RelCross"] > -20 else ">> Agro-type"
    print(f"    {r.name:6s}  {r['RelCross']:+.1f}%  {d}")
print("\nDone!")