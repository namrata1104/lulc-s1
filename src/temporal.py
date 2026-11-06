"""Multi-temporal input: the target month plus the two previous months, stacked as channels.

Training data were exported every second month, so for a target month the inputs are the two previous exported
months, each 2 (at the turn of the year 3) months apart: e.g. Mar + May + Jul for July. The prototype uses exactly
two months apart (m-4, m-2, m). The arrays are built from the existing patches and the downloaded tiles; nothing
is downloaded again.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from . import config as C

N_TIMES = 3
STEP = 2            # months between stacked inputs in the prototype
GAPS = (2, 3)       # accepted gaps in the training data (every 2nd month, 3 at the turn of the year)


def month_index(month):
    y, m = map(int, month.split("-"))
    return 12 * y + m - 1


def index_month(i):
    return f"{i // 12}-{i % 12 + 1:02d}"


def shift_month(month, k):
    return index_month(month_index(month) + k)


def predecessors(month, available, n=N_TIMES - 1, gaps=GAPS):
    """The n previous months from `available`, oldest first, each 2-3 months before the next; None if missing."""
    have = {month_index(a) for a in available}
    chain, cur = [], month_index(month)
    for _ in range(n):
        prev = next((cur - g for g in gaps if cur - g in have), None)
        if prev is None:
            return None
        chain.append(prev)
        cur = prev
    return [index_month(i) for i in reversed(chain)]


def read_window(path, row, col, p=C.PATCH_PX):
    """S1 bands (p, p, 3) in dB of one patch window of a downloaded tile; NaN where there is no data."""
    with rasterio.open(path) as src:
        a = src.read([1, 2, 3], window=Window(col, row, p, p)).astype(np.float32)
    a[a == C.S1_NODATA] = np.nan
    return (a / C.DB_SCALE).transpose(1, 2, 0)


def build_split(split, raw_dir=C.RAW_DIR, patch_dir=C.PATCH_DIR, n_times=N_TIMES):
    """Write X_<split>_mt<n>.npy (N, 256, 256, 3n) float16, oldest month first and the target month last, plus the
    matching y and meta (with the input months). Patches whose earlier months are missing or incomplete are dropped.
    Two passes so the arrays go straight to disk."""
    meta = pd.read_csv(patch_dir / f"meta_{split}.csv")
    X = np.load(patch_dir / f"X_{split}.npy", mmap_mode="r")
    y = np.load(patch_dir / f"y_{split}.npy", mmap_mode="r")
    months_of = {a: sorted({p.stem.split("_")[-2] for p in (raw_dir / a).glob("*.tif")}) for a in meta.aoi.unique()}

    def prev_paths(r):
        prev = predecessors(r.month, months_of[r.aoi], n_times - 1)
        if prev is None:
            return None, None
        suffix = r.tile.rsplit("_", 1)[1]                        # e.g. "t01.tif"
        return prev, [raw_dir / r.aoi / f"{r.aoi}_{m}_{suffix}" for m in prev]

    keep = []
    for i, r in enumerate(meta.itertuples()):
        prev, paths = prev_paths(r)
        if prev and all(p.exists() for p in paths) and \
                not any(np.isnan(read_window(p, r.row, r.col)).any() for p in paths):
            keep.append((i, prev, paths))

    n, p, b = len(keep), C.PATCH_PX, len(C.S1_BANDS)
    tag = f"{split}_mt{n_times}"
    Xm = np.lib.format.open_memmap(patch_dir / f"X_{tag}.npy", "w+", np.float16, (n, p, p, b * n_times))
    ym = np.lib.format.open_memmap(patch_dir / f"y_{tag}.npy", "w+", np.uint8, (n, p, p))
    rows = []
    for k, (i, prev, paths) in enumerate(keep):
        r = meta.iloc[i]
        frames = [read_window(path, r.row, r.col) for path in paths] + [np.asarray(X[i], np.float32)]
        Xm[k] = np.concatenate(frames, -1)
        ym[k] = y[i]
        rows.append({**r.to_dict(), "source_index": i, "input_months": " ".join(prev + [r.month])})
    Xm.flush(); ym.flush()
    out = pd.DataFrame(rows)
    out.to_csv(patch_dir / f"meta_{tag}.csv", index=False)
    return out


def stack_months(s1, month, n_times=N_TIMES, step=STEP):
    """Prototype: (H, W, 3n) input for `month` from monthly composites s1[month] (None = no scene). Each earlier
    month is taken `step` months apart, or the neighbouring month if that one has no scene; pixels missing in an
    earlier month are filled with the target month, so the map always covers the target month's footprint."""
    cur = s1[month]
    frames = []
    for k in range(n_times - 1, 0, -1):
        target = month_index(month) - step * k
        found = next((s1[index_month(c)] for c in (target, target - 1, target + 1)
                      if c < month_index(month) and s1.get(index_month(c)) is not None), None)
        frame = cur if found is None else np.where(np.isnan(found), cur, found)
        frames.append(frame)
    return np.concatenate(frames + [cur], -1)
