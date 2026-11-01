"""Project-wide settings: GEE project, study areas, classes, months, grid."""
from pathlib import Path

GEE_PROJECT = "lulc-sentinel-1"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # GeoTIFF tiles downloaded from GEE
PATCH_DIR = DATA_DIR / "patches"    # 256x256 numpy patches per split

# --- Sentinel-1 -------------------------------------------------------------
ORBIT_PASS = "DESCENDING"   # one pass only: mixing passes changes incidence angle -> fake change
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
CLASSES = ["water", "trees", "crops", "grass_shrub", "built", "bare"]
CLASS_COLORS = ["#419bdf", "#397d49", "#e49635", "#88b053", "#c4281b", "#a59b8f"]
IGNORE = 255

DW_BANDS = ["water", "trees", "grass", "flooded_vegetation", "crops",
            "shrub_and_scrub", "built", "bare", "snow_and_ice"]
# Dynamic World class index -> project class index.
# grass, flooded vegetation and shrub are merged: their backscatter is too similar in C-band.
DW_TO_CLASS = [0, 1, 3, 3, 2, 3, 4, 5, IGNORE]

CONF_MIN = 0.3       # pixels whose mean DW top-class probability is below this become IGNORE (0.5 removed almost all grass and bare)
MIN_LABELED = 0.5    # patches with fewer labelled pixels than this are dropped

# --- Time ---------------------------------------------------------------------
def _months(year, months):
    return [f"{year}-{m:02d}" for m in months]

# Every 2nd month, offset per year so that all 12 calendar months are covered.
EXPORT_MONTHS = (_months(2024, [1, 3, 5, 7, 9, 11])
                 + _months(2025, [2, 4, 6, 8, 10, 12])
                 + _months(2026, [1, 3, 5, 7, 9]))

# --- Study areas (centre lat/lon, 20.48 km square) --------------------------------
# Split is by region, never by patch, so test regions are completely unseen.
# Case-study cities (Munich, Leipzig, Heidelberg) are deliberately not included.
AOIS = {
    # urban
    "berlin":            dict(lat=52.520, lon=13.405, kind="urban",       split="train"),
    "hamburg":           dict(lat=53.551, lon=9.994,  kind="urban",       split="train"),
    "cologne":           dict(lat=50.938, lon=6.960,  kind="urban",       split="val"),
    "frankfurt":         dict(lat=50.111, lon=8.682,  kind="urban",       split="test"),
    # agriculture
    "magdeburger_boerde": dict(lat=52.060, lon=11.400, kind="agriculture", split="train"),
    "hildesheim_boerde": dict(lat=52.100, lon=10.050, kind="agriculture", split="train"),
    "mecklenburg":       dict(lat=53.800, lon=12.200, kind="agriculture", split="train"),
    "lower_bavaria":     dict(lat=48.850, lon=12.600, kind="agriculture", split="test"),
    # forest
    "black_forest":      dict(lat=47.900, lon=8.100,  kind="forest",      split="train"),
    "bavarian_forest":   dict(lat=48.950, lon=13.400, kind="forest",      split="train"),
    "harz":              dict(lat=51.790, lon=10.610, kind="forest",      split="val"),
    "thuringian_forest": dict(lat=50.700, lon=10.730, kind="forest",      split="test"),
    # water / wetland
    "lake_constance":    dict(lat=47.680, lon=9.150,  kind="water",       split="train"),
    "mueritz":           dict(lat=53.420, lon=12.700, kind="water",       split="train"),
    "jade_bay":          dict(lat=53.480, lon=8.200,  kind="water",       split="val"),
    # bare / open-pit mining
    "garzweiler":        dict(lat=51.050, lon=6.500,  kind="mining",      split="train"),
    "lusatia":           dict(lat=51.550, lon=14.350, kind="mining",      split="test"),
}
