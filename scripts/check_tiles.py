"""Find downloaded tiles that cannot be read completely and delete them (and leftover .part files).

    python scripts/check_tiles.py            # report only
    python scripts/check_tiles.py --delete   # delete, then re-run scripts/run_export.py to fetch them again
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, patching as P  # noqa: E402

bad = P.find_unreadable()
parts = sorted(C.RAW_DIR.rglob("*.part"))
for p in bad:
    print("BAD ", p.relative_to(C.RAW_DIR))
print(f"{len(list(C.RAW_DIR.rglob('*.tif')))} tiles, {len(bad)} unreadable, {len(parts)} leftover .part files")
if "--delete" in sys.argv:
    for p in bad + parts:
        p.unlink()
    print("Deleted. Now run: python scripts/run_export.py")
elif bad or parts:
    print("Run again with --delete to remove them.")
