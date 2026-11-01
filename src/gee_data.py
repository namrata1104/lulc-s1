"""Sentinel-1 monthly composites and Dynamic World monthly labels from Google Earth Engine.

The same functions are used for building the training set and by the prototype,
so training and inference see identically preprocessed data.
"""
import os
import threading
import time

import ee
import rasterio
from rasterio.io import MemoryFile
from pyproj import Transformer

from . import config as C

TILE_BANDS = C.S1_BANDS + ["label", "conf", "dw_n"]


def init(project=C.GEE_PROJECT):
    ee.Initialize(project=project)


# --- Grid helpers -----------------------------------------------------------------

def utm_epsg(lat, lon):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def aoi_grid(lat, lon, size_px=C.AOI_PX, px=C.PIXEL_SIZE):
    """Local UTM EPSG and top-left corner (snapped to the 10 m grid) of a square centred on lat/lon."""
    epsg = utm_epsg(lat, lon)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    half = size_px * px / 2
    return epsg, round((x - half) / px) * px, round((y + half) / px) * px


def grid_geometry(epsg, x0, y0, w_px, h_px, px=C.PIXEL_SIZE):
    return ee.Geometry.Rectangle([x0, y0 - h_px * px, x0 + w_px * px, y0],
                                 proj=f"EPSG:{epsg}", geodesic=False)


def month_range(month):
    """'2024-03' -> (start, end) as ee.Date, end exclusive."""
    start = ee.Date(f"{month}-01")
    return start, start.advance(1, "month")


# --- Sentinel-1 -------------------------------------------------------------------

def s1_collection(geom, start, end, orbit=C.ORBIT_PASS):
    """GRD scenes from GEE: already thermal-noise removed, calibrated to sigma0,
    terrain corrected (SRTM 30 m) and converted to dB."""
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geom)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .filter(ee.Filter.eq("orbitProperties_pass", orbit)))


def lee_filter(img, kernel=C.LEE_KERNEL, enl=C.ENL):
    """Lee (1980) speckle filter, applied to linear-power bands."""
    k = ee.Kernel.square(kernel // 2)
    mean = img.reduceNeighborhood(ee.Reducer.mean(), k)
    var = img.reduceNeighborhood(ee.Reducer.variance(), k)
    cu2 = 1.0 / enl
    ci2 = var.divide(mean.pow(2))
    w = ee.Image(1).subtract(ee.Image(cu2).divide(ci2)).divide(1 + cu2).clamp(0, 1)
    return mean.add(w.multiply(img.subtract(mean))).rename(img.bandNames())


def _prep_scene(img):
    db = img.select(["VV", "VH"])
    db = db.updateMask(db.select("VV").gt(C.BORDER_DB))
    lin = ee.Image(10).pow(db.divide(10)).rename(["VV", "VH"])
    return lee_filter(lin).copyProperties(img, ["system:time_start"])


def s1_monthly(geom, month, orbit=C.ORBIT_PASS):
    """Monthly median of speckle-filtered scenes, in dB, with the VV-VH ratio band."""
    start, end = month_range(month)
    lin = s1_collection(geom, start, end, orbit).map(_prep_scene).median()
    db = lin.log10().multiply(10).rename(["VV", "VH"])
    return db.addBands(db.select("VV").subtract(db.select("VH")).rename("VV_VH"))


# --- Dynamic World labels ---------------------------------------------------------------

def dw_collection(geom, start, end):
    return ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(geom).filterDate(start, end)


def dw_monthly(geom, month):
    """Monthly label = argmax of the mean class probabilities over cloud-free DW observations.

    Returns bands: label (project classes), conf (mean top-class probability), dw_n (observations).
    """
    start, end = month_range(month)
    col = dw_collection(geom, start, end)
    probs = col.select(C.DW_BANDS).mean()
    dw_idx = probs.toArray().arrayArgmax().arrayGet([0])
    label = dw_idx.remap(list(range(len(C.DW_BANDS))), C.DW_TO_CLASS).rename("label")
    conf = probs.reduce(ee.Reducer.max()).rename("conf")
    n = col.select("label").count().rename("dw_n")
    return label.addBands(conf).addBands(n)


def training_image(geom, month, orbit=C.ORBIT_PASS):
    """S1 (dB x 100) + label + confidence (x 100) + DW observation count, as int16."""
    s1 = s1_monthly(geom, month, orbit).multiply(C.DB_SCALE).round().unmask(C.S1_NODATA)
    dw = dw_monthly(geom, month)
    return (s1.addBands(dw.select("label").unmask(C.IGNORE))
              .addBands(dw.select("conf").multiply(100).round().unmask(0))
              .addBands(dw.select("dw_n").unmask(0))
              .toInt16())


def month_counts(geom, month, orbit=C.ORBIT_PASS):
    start, end = month_range(month)
    return ee.Dictionary({
        "s1": s1_collection(geom, start, end, orbit).size(),
        "dw": dw_collection(geom, start, end).size(),
    }).getInfo()


# --- Download ---------------------------------------------------------------------

def download_tile(image, epsg, x0, y0, out_path, size_px=C.TILE_PX, band_names=None, retries=4):
    """Fetch one tile on an exact UTM grid with computePixels and save it as compressed GeoTIFF."""
    request = {
        "expression": image,
        "fileFormat": "GEO_TIFF",
        "grid": {
            "dimensions": {"width": size_px, "height": size_px},
            "affineTransform": {"scaleX": C.PIXEL_SIZE, "shearX": 0, "translateX": x0,
                                "shearY": 0, "scaleY": -C.PIXEL_SIZE, "translateY": y0},
            "crsCode": f"EPSG:{epsg}",
        },
    }
    for attempt in range(retries):
        try:
            data = ee.data.computePixels(request)
            break
        except ee.EEException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * 2 ** attempt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # unique temp name, so two exports running at once can never write into the same file
    tmp = out_path.with_name(f"{out_path.stem}.{os.getpid()}.{threading.get_ident()}.part")
    with MemoryFile(data) as mem, mem.open() as src:
        profile = src.profile | dict(driver="GTiff", compress="deflate", predictor=2,
                                     tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(src.read())
            if band_names:
                dst.descriptions = tuple(band_names)
    tmp.rename(out_path)


def export_aoi_month(name, aoi, month, raw_dir=C.RAW_DIR, orbit=C.ORBIT_PASS):
    """Download all tiles of one study area for one month. Skips tiles that already exist."""
    epsg, x0, y0 = aoi_grid(aoi["lat"], aoi["lon"])
    geom = grid_geometry(epsg, x0, y0, C.AOI_PX, C.AOI_PX)
    log = dict(aoi=name, month=month, **month_counts(geom, month, orbit), tiles=0, status="ok")
    if log["s1"] == 0 or log["dw"] == 0:
        log["status"] = "skipped: no S1 or DW images"
        return log

    image = training_image(geom, month, orbit)
    n_tiles = C.AOI_PX // C.TILE_PX
    t0 = time.time()
    for r in range(n_tiles):
        for c in range(n_tiles):
            path = raw_dir / name / f"{name}_{month}_t{r}{c}.tif"
            if not path.exists():
                download_tile(image, epsg, x0 + c * C.TILE_PX * C.PIXEL_SIZE,
                              y0 - r * C.TILE_PX * C.PIXEL_SIZE, path, band_names=TILE_BANDS)
            log["tiles"] += 1
    log["seconds"] = round(time.time() - t0, 1)
    return log
