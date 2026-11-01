"""Find downloaded tiles that cannot be read completely and delete them (and leftover .part files).

    python scripts/check_tiles.py            # report only
    python scripts/check_tiles.py --delete   # delete, then re-run scripts/run_export.py to fetch them again
"""
import sys
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

delete = "--delete" in sys.argv
tiles = sorted(C.RAW_DIR.rglob("*.tif"))
bad = []
for i, path in enumerate(tiles, 1):
    try:
        with rasterio.open(path) as src:
            src.read()
    except Exception as e:
        bad.append(path)
        print(f"BAD  {path.relative_to(C.RAW_DIR)}  ({type(e).__name__})", flush=True)
    if i % 100 == 0:
        print(f"checked {i}/{len(tiles)}", flush=True)

parts = sorted(C.RAW_DIR.rglob("*.part"))
print(f"\n{len(tiles)} tiles checked, {len(bad)} unreadable, {len(parts)} leftover .part files")
if delete:
    for p in bad + parts:
        p.unlink()
    print("Deleted. Now run: python scripts/run_export.py")
elif bad or parts:
    print("Run again with --delete to remove them.")
