"""Cut downloaded GeoTIFF tiles into 256x256 patches and write one .npy set per split."""
import json

import numpy as np
import pandas as pd
import rasterio

from . import config as C


def load_tile(path):
    """Return S1 (H, W, 3) float32 in dB (NaN = no-data) and label (H, W) uint8 with IGNORE applied."""
    with rasterio.open(path) as src:
        a = src.read()
    s1 = a[:3].transpose(1, 2, 0).astype(np.float32)
    s1[s1 == C.S1_NODATA] = np.nan
    s1 /= C.DB_SCALE
    label, conf, dw_n = a[3], a[4], a[5]
    unreliable = (conf < C.CONF_MIN * 100) | (dw_n < 1)
    label = np.where(unreliable, C.IGNORE, label).astype(np.uint8)
    return s1, label


def patch_windows(s1, label, p=C.PATCH_PX):
    """Yield (row, col) of patches with complete S1 coverage and enough labelled pixels."""
    h, w = label.shape
    for r in range(0, h - p + 1, p):
        for c in range(0, w - p + 1, p):
            if np.isnan(s1[r:r + p, c:c + p]).any():
                continue
            if (label[r:r + p, c:c + p] != C.IGNORE).mean() < C.MIN_LABELED:
                continue
            yield r, c


def build_split(split, raw_dir=C.RAW_DIR, out_dir=C.PATCH_DIR, months=None):
    """Write X_<split>.npy (N, 256, 256, 3) float16 dB, y_<split>.npy (N, 256, 256) uint8, meta_<split>.csv.

    Two passes so the arrays are written straight to disk (memmap) instead of held in RAM.
    """
    names = [n for n, a in C.AOIS.items() if a["split"] == split]
    tiles = sorted(p for n in names for p in (raw_dir / n).glob("*.tif"))
    if months is not None:
        tiles = [t for t in tiles if t.stem.split("_")[-2] in months]

    meta = []
    for path in tiles:
        s1, label = load_tile(path)
        aoi, month = path.stem.rsplit("_", 2)[:2]
        meta += [dict(aoi=aoi, month=month, tile=path.name, row=r, col=c)
                 for r, c in patch_windows(s1, label)]
    meta = pd.DataFrame(meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    n, p = len(meta), C.PATCH_PX
    X = np.lib.format.open_memmap(out_dir / f"X_{split}.npy", "w+", np.float16, (n, p, p, 3))
    y = np.lib.format.open_memmap(out_dir / f"y_{split}.npy", "w+", np.uint8, (n, p, p))
    i = 0
    for tile, rows in meta.groupby("tile", sort=False):
        s1, label = load_tile(raw_dir / rows.aoi.iloc[0] / tile)
        for r, c in zip(rows.row, rows.col):
            X[i] = s1[r:r + p, c:c + p]
            y[i] = label[r:r + p, c:c + p]
            i += 1
    X.flush(); y.flush()
    meta.to_csv(out_dir / f"meta_{split}.csv", index=False)
    return meta


def class_distribution(split, out_dir=C.PATCH_DIR):
    """Pixel count and share per class (labelled pixels only)."""
    y = np.load(out_dir / f"y_{split}.npy", mmap_mode="r")
    counts = np.zeros(len(C.CLASSES), dtype=np.int64)
    for i in range(0, len(y), 256):
        counts += np.bincount(y[i:i + 256].ravel(), minlength=256)[:len(C.CLASSES)]
    return pd.DataFrame({"class": C.CLASSES, "pixels": counts,
                         "percent": 100 * counts / max(counts.sum(), 1)})


def normalisation_stats(out_dir=C.PATCH_DIR):
    """Per-channel mean/std of the clipped training data, saved to stats.json for training and the prototype."""
    X = np.load(out_dir / "X_train.npy", mmap_mode="r")
    lo, hi = np.array(C.CLIP_DB).T
    s = np.zeros(3); s2 = np.zeros(3); n = 0
    for i in range(0, len(X), 256):
        x = np.clip(X[i:i + 256].astype(np.float32), lo, hi).reshape(-1, 3)
        s += x.sum(0); s2 += (x.astype(np.float64) ** 2).sum(0); n += len(x)
    mean = s / n
    stats = dict(bands=C.S1_BANDS, clip=C.CLIP_DB, mean=mean.tolist(),
                 std=np.sqrt(s2 / n - mean ** 2).tolist())
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
