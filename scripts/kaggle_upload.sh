# Upload the dataset to Kaggle. Run on the lab server from the repo folder, after `source scripts/server_setup.sh`:
#
#     nohup bash scripts/kaggle_upload.sh > kaggle_upload.log 2>&1 &
#
# Needs the Kaggle API key in ~/vault/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token).
# Packs data/raw_global (Earth Engine tiles) and data/patches_global (256 x 256 patches, without the derived
# multi-temporal arrays) into two zip files in ~/vault/kaggle_upload and creates a public Kaggle dataset.
# Re-running skips zip files that already exist.

set -e
VAULT="$HOME/vault"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KAGGLE_CONFIG_DIR="$VAULT/.kaggle"
STAGE="$VAULT/kaggle_upload"
SLUG="sentinel1-lulc-global"

if [ ! -f "$KAGGLE_CONFIG_DIR/kaggle.json" ]; then
    echo "Missing $KAGGLE_CONFIG_DIR/kaggle.json - download it from kaggle.com -> Settings -> API -> Create New Token"
    exit 1
fi
chmod 600 "$KAGGLE_CONFIG_DIR/kaggle.json"
python -m pip install -q kaggle
KUSER=$(python -c "import json, os; print(json.load(open(os.path.join(os.environ['KAGGLE_CONFIG_DIR'], 'kaggle.json')))['username'])")
mkdir -p "$STAGE"

echo "$(date +%H:%M) packing ..."
python - "$REPO/data" "$STAGE" << 'PY'
import sys, zipfile
from pathlib import Path
data, stage = Path(sys.argv[1]), Path(sys.argv[2])
for folder, skip in [("raw_global", (".part",)), ("patches_global", ("_mt3",))]:
    out = stage / f"{folder}.zip"
    if out.exists():
        print("exists:", out.name); continue
    files = [p for p in sorted((data / folder).rglob("*")) if p.is_file() and not any(s in p.name for s in skip)]
    tmp = out.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:   # tiles/arrays: no recompression
        for p in files:
            z.write(p, p.relative_to(data))
    tmp.rename(out)
    print(f"{out.name}: {len(files)} files, {out.stat().st_size / 1e9:.1f} GB")
PY
cp -f "$REPO/data/export_log.csv" "$STAGE/" 2> /dev/null || true

cat > "$STAGE/dataset-metadata.json" << EOF
{
  "title": "Sentinel-1 LULC global monthly composites",
  "subtitle": "Sentinel-1 SAR monthly composites with Dynamic World labels, 40 areas, 2023-2026",
  "id": "$KUSER/$SLUG",
  "licenses": [{"name": "CC-BY-4.0"}],
  "description": "Dataset of the project https://github.com/namrata1104/lulc-s1 (land use / land cover segmentation from Sentinel-1 only).\n\n**raw_global.zip** - GeoTIFF tiles exported from Google Earth Engine: 40 study areas (20.48 km squares on six continents), every second month from 2023-01 to 2026-10, 1024 x 1024 px at 10 m in local UTM. Bands: VV, VH, VV-VH (Sentinel-1 monthly median after Lee speckle filter, dB x 100, int16, no data = -32768), dw_class (Dynamic World class index 0-8, 255 = none), conf (mean top-class probability x 100), dw_n (number of cloud-free Dynamic World observations).\n\n**patches_global.zip** - 256 x 256 patches used for training: X_<split>.npy (N, 256, 256, 3) float16 dB, y_<split>.npy (N, 256, 256) uint8 with 8 classes (water, trees, flooded_vegetation, crops, rangeland, built, bare, snow_ice; 255 = ignored), meta_<split>.csv, stats.json. Splits: train (26 areas, 2023-24), val (4 areas, 2023-24), test (9 unseen areas, 2025-26), test_temporal (train areas, 2025-26).\n\n**export_log.csv** - scenes and label observations per area-month.\n\nSources: Copernicus Sentinel-1 (ESA), Google Dynamic World V1 (CC-BY 4.0)."
}
EOF

echo "$(date +%H:%M) uploading to kaggle.com/datasets/$KUSER/$SLUG ..."
kaggle datasets create -p "$STAGE" --public
echo "$(date +%H:%M) DONE: https://www.kaggle.com/datasets/$KUSER/$SLUG"
