"""Prototype pipeline: location + start month -> 12 monthly LULC maps, class percentages and changes.

    run(lat, lon, start_month)  ->  dict with the monthly S1 composites, maps and statistics

Used by the Streamlit app (app/streamlit_app.py) and the case-study notebook.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from tensorflow import keras

from . import config as C, gee_data as G, model as M

WEIGHTS_DIR = C.ROOT / "weights"

# Esri 10 m Annual LULC (Impact Observatory), independent of Dynamic World; never used for training.
ESRI_LULC = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
ESRI_CODES = [1, 2, 4, 5, 7, 8, 9, 11]                # water, trees, flooded veg, crops, built, bare, snow, rangeland
ESRI_TO_CLASS = [0, 1, 2, 3, 5, 6, 7, 4]              # -> project class index; 10 (clouds) becomes IGNORE


def month_list(start, n=12):
    """'2025-03' -> ['2025-03', ..., '2026-02']"""
    y, m = map(int, start.split("-"))
    return [f"{y + (m - 1 + i) // 12}-{(m - 1 + i) % 12 + 1:02d}" for i in range(n)]


MODEL_PATH = Path(os.environ.get("LULC_MODEL", WEIGHTS_DIR / "best_model.keras"))


def load_model(path=MODEL_PATH):
    return keras.models.load_model(path, compile=False)


def load_stats(path=WEIGHTS_DIR / "stats.json"):
    return json.loads(path.read_text())


# --- Data retrieval -------------------------------------------------------------------

def area(lat, lon, size_px):
    epsg, x0, y0 = G.aoi_grid(lat, lon, size_px=size_px)
    return epsg, x0, y0, G.grid_geometry(epsg, x0, y0, size_px, size_px)


def choose_orbit(geom, months):
    """The pass with more scenes over the whole 12 months, used for every month (consistent geometry)."""
    start = ee.Date(f"{months[0]}-01")
    end = ee.Date(f"{months[-1]}-01").advance(1, "month")
    counts = ee.Dictionary({o: G.s1_collection(geom, start, end, o).size()
                            for o in ["ASCENDING", "DESCENDING"]}).getInfo()
    return max(counts, key=counts.get), counts


def fetch_s1(lat, lon, months, size_px, workers=3):
    """Monthly Sentinel-1 composites (H, W, 3) in dB with exactly the training preprocessing.
    A month without any scene gets None."""
    epsg, x0, y0, geom = area(lat, lon, size_px)
    orbit, orbit_counts = choose_orbit(geom, months)

    def one(month):
        start, end = G.month_range(month)
        n = G.s1_collection(geom, start, end, orbit).size().getInfo()
        if n == 0:
            return n, None
        image = G.s1_monthly(geom, month, orbit).multiply(C.DB_SCALE).round().unmask(C.S1_NODATA).toInt16()
        a = G.fetch_array(image, epsg, x0, y0, size_px).astype(np.float32)
        a[a == C.S1_NODATA] = np.nan
        return n, (a / C.DB_SCALE).transpose(1, 2, 0)

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, months))
    return dict(epsg=epsg, x0=x0, y0=y0, orbit=orbit, orbit_counts=orbit_counts,
                scenes={m: n for m, (n, _) in zip(months, results)},
                s1={m: a for m, (_, a) in zip(months, results)})


def fetch_esri(lat, lon, size_px, year):
    """Esri annual LULC remapped to the project classes; falls back to the latest earlier year available."""
    epsg, x0, y0, geom = area(lat, lon, size_px)
    col = ee.ImageCollection(ESRI_LULC).filterBounds(geom)
    for y in range(year, 2016, -1):
        yearly = col.filterDate(f"{y}-01-01", f"{y + 1}-01-01")
        if yearly.size().getInfo() > 0:
            image = yearly.mosaic().remap(ESRI_CODES, ESRI_TO_CLASS, C.IGNORE).unmask(C.IGNORE).toInt16()
            return y, G.fetch_array(image, epsg, x0, y0, size_px)[0].astype(np.uint8)
    return None, None


# --- Inference --------------------------------------------------------------------------

def predict_map(model, s1, stats, patch=C.PATCH_PX, stride=C.PATCH_PX // 2, batch=16):
    """Sliding-window prediction with 50 % overlap; class probabilities of overlapping windows are averaged.
    Pixels without Sentinel-1 data get IGNORE."""
    h, w, _ = s1.shape
    valid = ~np.isnan(s1).any(-1)
    x = M.normalise(np.nan_to_num(s1, nan=0.0), stats)
    x[~valid] = 0.0                                          # = training mean after z-scoring

    H = max(patch, int(np.ceil((h - patch) / stride)) * stride + patch)
    W = max(patch, int(np.ceil((w - patch) / stride)) * stride + patch)
    x = np.pad(x, ((0, H - h), (0, W - w), (0, 0)), mode="reflect")
    probs = np.zeros((H, W, M.NUM_CLASSES), np.float32)
    hits = np.zeros((H, W, 1), np.float32)
    windows = [(r, c) for r in range(0, H - patch + 1, stride) for c in range(0, W - patch + 1, stride)]
    for i in range(0, len(windows), batch):
        part = windows[i:i + batch]
        out = model.predict_on_batch(np.stack([x[r:r + patch, c:c + patch] for r, c in part]))
        for (r, c), p in zip(part, np.asarray(out)):
            probs[r:r + patch, c:c + patch] += p
            hits[r:r + patch, c:c + patch] += 1
    label = np.argmax(probs[:h, :w] / hits[:h, :w], -1).astype(np.uint8)
    label[~valid] = C.IGNORE
    return label


def temporal_mode(maps, months, window=3):
    """Per-pixel majority class over a centred window of months. Suppresses one-month flicker caused by
    classification noise; real persistent change survives."""
    available = [m for m in months if maps[m] is not None]
    stack = np.stack([maps[m] for m in available])
    smoothed = {m: None for m in months}
    for i, m in enumerate(available):
        part = stack[max(0, i - window // 2): i + window // 2 + 1]
        votes = np.stack([(part == c).sum(0) for c in range(M.NUM_CLASSES)], -1)
        out = np.argmax(votes, -1).astype(np.uint8)
        out[stack[i] == C.IGNORE] = C.IGNORE
        smoothed[m] = out
    return smoothed


# --- Statistics ---------------------------------------------------------------------------

def class_percentages(maps, months):
    """% of the mapped area per class and month (pixels without data are excluded). Missing month -> NaN."""
    rows = {}
    for m in months:
        lab = maps[m]
        if lab is None:
            rows[m] = [np.nan] * M.NUM_CLASSES
            continue
        counts = np.bincount(lab[lab != C.IGNORE].ravel(), minlength=M.NUM_CLASSES)[:M.NUM_CLASSES]
        rows[m] = 100 * counts / max(counts.sum(), 1)
    return pd.DataFrame(rows, index=C.CLASSES).T.rename_axis("month")


def changes(pct):
    """Month-to-month change and total change (last minus first available month), in percentage points."""
    month_to_month = pct.diff()
    available = pct.dropna()
    total = available.iloc[-1] - available.iloc[0] if len(available) > 1 else pct.iloc[0] * np.nan
    total.name = f"{available.index[0]} → {available.index[-1]}" if len(available) > 1 else "total"
    return month_to_month, total


def annual_majority(maps, months):
    """Most frequent class per pixel over the 12 months (for comparison with an annual reference map)."""
    stack = np.stack([maps[m] for m in months if maps[m] is not None])
    votes = np.stack([(stack == c).sum(0) for c in range(M.NUM_CLASSES)], -1)
    out = np.argmax(votes, -1).astype(np.uint8)
    out[votes.sum(-1) == 0] = C.IGNORE
    return out


def compare_with_reference(pred, ref):
    """Pixel agreement and class shares of a prediction vs. a reference map (both in project classes)."""
    keep = (pred != C.IGNORE) & (ref != C.IGNORE)
    share = lambda a: 100 * np.bincount(a.ravel(), minlength=M.NUM_CLASSES)[:M.NUM_CLASSES] / max(a.size, 1)
    table = pd.DataFrame({"model_%": share(pred[keep]), "reference_%": share(ref[keep])}, index=C.CLASSES)
    table["difference_pp"] = table["model_%"] - table["reference_%"]
    agreement = float((pred[keep] == ref[keep]).mean()) if keep.any() else float("nan")
    return agreement, table


# --- Full run -------------------------------------------------------------------------------

def run(lat, lon, start_month, size_px=512, model=None, stats=None, smooth=False, reference=True, workers=3):
    months = month_list(start_month)
    model = model or load_model()
    stats = stats or load_stats()
    data = fetch_s1(lat, lon, months, size_px, workers=workers)
    maps = {m: (predict_map(model, a, stats) if a is not None else None) for m, a in data["s1"].items()}
    if smooth:
        maps = temporal_mode(maps, months)
    pct = class_percentages(maps, months)
    month_to_month, total = changes(pct)
    result = dict(lat=lat, lon=lon, months=months, size_km=size_px * C.PIXEL_SIZE / 1000, smooth=smooth,
                  maps=maps, pct=pct, month_to_month=month_to_month, total=total, **data)
    if reference:
        year = int(np.median([int(m[:4]) for m in months]))
        ref_year, ref = fetch_esri(lat, lon, size_px, year)
        if ref is not None:
            agreement, table = compare_with_reference(annual_majority(maps, months), ref)
            result.update(ref_year=ref_year, ref_map=ref, ref_agreement=agreement, ref_table=table)
    return result
