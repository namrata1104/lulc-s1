"""Project-wide settings: GEE project, study areas, classes, periods, grid."""
import os
from pathlib import Path

GEE_PROJECT = "lulc-sentinel-1"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_global"          # GeoTIFF tiles downloaded from GEE
PATCH_DIR = Path(os.environ.get("LULC_PATCH_DIR", DATA_DIR / "patches_global"))   # 256x256 numpy patches per split

# --- Sentinel-1 -------------------------------------------------------------
# One pass per study area: mixing passes changes the incidence angle -> fake change.
# Since Sentinel-1B failed (Dec 2021) many regions are only well covered by one pass,
# so the pass is chosen per area (see AOIS, chosen by scripts/choose_orbits.py).
BORDER_DB = -30.0           # pixels below this (VV, dB) are treated as no-data / border noise
LEE_KERNEL = 5              # Lee speckle filter window (pixels)
ENL = 4.4                   # equivalent number of looks, Sentinel-1 IW GRD-HR
DB_SCALE = 100              # dB stored on disk as int16 = round(dB * 100)
S1_NODATA = -32768
S1_BANDS = ["VV", "VH", "VV_VH"]
CLIP_DB = [(-25.0, 0.0), (-32.0, -5.0), (-5.0, 20.0)]   # per-channel clip before normalisation

# --- Grid ---------------------------------------------------------------------
PIXEL_SIZE = 10      # metres
AOI_PX = 2048        # 20.48 km square study area
TILE_PX = 1024       # size of one computePixels request (keeps each request well under 48 MB)
PATCH_PX = 256

# --- Classes ------------------------------------------------------------------
# Same classes as the Esri 10 m Annual LULC map (minus "clouds"), which serves as the
# independent reference in the case study.
CLASSES = ["water", "trees", "flooded_vegetation", "crops", "rangeland", "built", "bare", "snow_ice"]
CLASS_COLORS = ["#419bdf", "#397d49", "#7a87c6", "#e49635", "#e3e2c3", "#c4281b", "#a59b8f", "#a8ebff"]
IGNORE = 255

DW_BANDS = ["water", "trees", "grass", "flooded_vegetation", "crops",
            "shrub_and_scrub", "built", "bare", "snow_and_ice"]
# Dynamic World class index -> project class index. Grass and shrub/scrub become "rangeland"
# (as in the Esri map): their C-band backscatter is too similar to separate reliably.
DW_TO_CLASS = [0, 1, 4, 2, 3, 4, 5, 6, 7]

CONF_MIN = 0.3       # pixels whose mean DW top-class probability is below this become IGNORE
MIN_LABELED = 0.5    # patches with fewer labelled pixels than this are dropped

# --- Time ---------------------------------------------------------------------
def _months(year, months):
    return [f"{year}-{m:02d}" for m in months]

# Every 2nd month, offset per year so that all 12 calendar months are covered.
TRAIN_MONTHS = _months(2023, [1, 3, 5, 7, 9, 11]) + _months(2024, [2, 4, 6, 8, 10, 12])
TEST_MONTHS = _months(2025, [1, 3, 5, 7, 9, 11]) + _months(2026, [2, 4, 6, 8, 10])
TEST_TEMPORAL_MONTHS = _months(2025, [3, 7, 11]) + _months(2026, [1, 5, 9])

# split name -> (study-area split, months)
#   test          : unseen regions in the future period (main test)
#   test_temporal : training regions in the future period (only measures change over time)
SPLITS = {
    "train": ("train", TRAIN_MONTHS),
    "val": ("val", TRAIN_MONTHS),
    "test": ("test", TEST_MONTHS),
    "test_temporal": ("train", TEST_TEMPORAL_MONTHS),
}

# --- Study areas (centre lat/lon, 20.48 km square) --------------------------------
# Split is by region, never by patch. "orbit" is the Sentinel-1 pass with more scenes
# over the area's period (scripts/choose_orbits.py, 2023-24 for train/val, 2025-26 for test).
# Sahara, Okavango, Gobi and Kazakh steppe were replaced: no Sentinel-1 IW scenes in 2023-24.
AOIS = {
    # --- train: Europe
    "paris_fr":          dict(lat=48.857, lon=2.352,    kind="urban",       split="train", orbit="DESCENDING"),
    "po_valley_it":      dict(lat=45.100, lon=10.000,   kind="crops",       split="train", orbit="ASCENDING"),
    "black_forest_de":   dict(lat=47.900, lon=8.100,    kind="forest",      split="train", orbit="ASCENDING"),
    "danube_delta_ro":   dict(lat=45.150, lon=29.300,   kind="wetland",     split="train", orbit="ASCENDING"),
    "aletsch_ch":        dict(lat=46.480, lon=8.050,    kind="snow_ice",    split="train", orbit="ASCENDING"),
    # --- train: North America
    "chicago_us":        dict(lat=41.880, lon=-87.630,  kind="urban",       split="train", orbit="ASCENDING"),
    "iowa_us":           dict(lat=42.000, lon=-93.500,  kind="crops",       split="train", orbit="ASCENDING"),
    "everglades_us":     dict(lat=25.850, lon=-80.850,  kind="wetland",     split="train", orbit="ASCENDING"),
    "black_rock_us":     dict(lat=40.850, lon=-119.100, kind="bare",        split="train", orbit="DESCENDING"),
    # --- train: South America
    "sao_paulo_br":      dict(lat=-23.550, lon=-46.630, kind="urban",       split="train", orbit="DESCENDING"),
    "amazon_br":         dict(lat=-3.150, lon=-60.400,  kind="forest",      split="train", orbit="DESCENDING"),
    "pampas_ar":         dict(lat=-34.600, lon=-61.000, kind="crops",       split="train", orbit="DESCENDING"),
    # --- train: Africa
    "lagos_ng":          dict(lat=6.520, lon=3.380,     kind="urban",       split="train", orbit="DESCENDING"),
    "nile_delta_eg":     dict(lat=30.900, lon=31.100,   kind="crops",       split="train", orbit="DESCENDING"),
    "western_desert_eg": dict(lat=29.300, lon=30.000,   kind="bare",        split="train", orbit="DESCENDING"),
    "inner_niger_ml":    dict(lat=14.500, lon=-4.500,   kind="wetland",     split="train", orbit="ASCENDING"),
    "maasai_mara_ke":    dict(lat=-1.450, lon=35.150,   kind="rangeland",   split="train", orbit="ASCENDING"),
    "karoo_za":          dict(lat=-32.300, lon=22.500,  kind="rangeland",   split="train", orbit="ASCENDING"),
    # --- train: Asia
    "delhi_in":          dict(lat=28.610, lon=77.210,   kind="urban",       split="train", orbit="DESCENDING"),
    "punjab_in":         dict(lat=30.600, lon=75.400,   kind="crops",       split="train", orbit="ASCENDING"),
    "mekong_delta_vn":   dict(lat=10.000, lon=105.800,  kind="crops",       split="train", orbit="ASCENDING"),
    "tokyo_jp":          dict(lat=35.680, lon=139.760,  kind="urban",       split="train", orbit="DESCENDING"),
    "borneo_id":         dict(lat=0.500, lon=113.500,   kind="forest",      split="train", orbit="DESCENDING"),
    "everest_np":        dict(lat=27.950, lon=86.920,   kind="snow_ice",    split="train", orbit="ASCENDING"),
    "dasht_e_kavir_ir":  dict(lat=34.500, lon=54.500,   kind="bare",        split="train", orbit="DESCENDING"),
    # --- train: Oceania
    "sydney_au":         dict(lat=-33.870, lon=151.210, kind="urban",       split="train", orbit="DESCENDING"),
    "riverina_au":       dict(lat=-34.600, lon=146.000, kind="crops",       split="train", orbit="DESCENDING"),

    # --- validation
    "madrid_es":         dict(lat=40.420, lon=-3.700,   kind="urban",       split="val", orbit="DESCENDING"),
    "louisiana_us":      dict(lat=29.400, lon=-89.600,  kind="wetland",     split="val", orbit="ASCENDING"),
    "ghana_forest_gh":   dict(lat=6.200, lon=-2.000,    kind="forest",      split="val", orbit="ASCENDING"),
    "anatolia_tr":       dict(lat=39.000, lon=33.000,   kind="rangeland",   split="val", orbit="DESCENDING"),

    # --- test (never used for training or model selection)
    "dubai_ae":          dict(lat=25.200, lon=55.270,   kind="urban",       split="test", orbit="ASCENDING"),
    "mexico_city_mx":    dict(lat=19.430, lon=-99.130,  kind="urban",       split="test", orbit="ASCENDING"),
    "dhaka_bd":          dict(lat=23.810, lon=90.410,   kind="urban",       split="test", orbit="DESCENDING"),
    "kansas_us":         dict(lat=38.000, lon=-98.000,  kind="crops",       split="test", orbit="ASCENDING"),
    "canterbury_nz":     dict(lat=-43.600, lon=172.000, kind="crops",       split="test", orbit="DESCENDING"),
    "gabon_forest_ga":   dict(lat=-0.500, lon=12.000,   kind="forest",      split="test", orbit="DESCENDING"),
    "pantanal_br":       dict(lat=-18.000, lon=-57.000, kind="wetland",     split="test", orbit="DESCENDING"),
    "atacama_cl":        dict(lat=-23.500, lon=-68.200, kind="bare",        split="test", orbit="DESCENDING"),
    "vatnajokull_is":    dict(lat=64.400, lon=-16.700,  kind="snow_ice",    split="test", orbit="ASCENDING"),
}


def export_jobs(include_temporal=True):
    """(aoi name, aoi dict, month) for every study-area month that has to be downloaded."""
    jobs = []
    for split, (aoi_split, months) in SPLITS.items():
        if split == "test_temporal" and not include_temporal:
            continue
        jobs += [(n, a, m) for n, a in AOIS.items() if a["split"] == aoi_split for m in months]
    return jobs
