"""Count Sentinel-1 scenes per pass for every study area over its period and print the better pass.

    python scripts/choose_orbits.py

Copy the result into the "orbit" field of config.AOIS.
"""
import sys
from pathlib import Path

import ee
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, gee_data as G  # noqa: E402

G.init()
rows = []
for name, a in C.AOIS.items():
    months = C.TEST_MONTHS if a["split"] == "test" else C.TRAIN_MONTHS
    start, end = ee.Date(f"{months[0]}-01"), ee.Date(f"{months[-1]}-01").advance(1, "month")
    epsg, x0, y0 = G.aoi_grid(a["lat"], a["lon"])
    geom = G.grid_geometry(epsg, x0, y0, C.AOI_PX, C.AOI_PX)
    counts = {o: G.s1_collection(geom, start, end, o).size() for o in ["ASCENDING", "DESCENDING"]}
    counts["dw"] = G.dw_collection(geom, start, end).size()
    counts = ee.Dictionary(counts).getInfo()
    best = max(["ASCENDING", "DESCENDING"], key=counts.get)
    rows.append(dict(aoi=name, split=a["split"], **counts, orbit=best))
    print(rows[-1], flush=True)

print(pd.DataFrame(rows).to_string(index=False))
