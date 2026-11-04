# LULC from Sentinel-1

Global land use / land cover segmentation (8 classes) using Sentinel-1 SAR as the only input (TensorFlow), plus a prototype
that maps a location over 12 months and reports class percentages and changes.

## Setup

```bash
conda create -n lulc python=3.11 -y
conda activate lulc
pip install -r requirements.txt
earthengine authenticate --auth_mode=localhost
```

Requires a Google Earth Engine account with a Cloud project registered for noncommercial use
(this project uses `lulc-sentinel-1`, set in `src/config.py`).

## Structure

| Path | Contents |
|---|---|
| `src/config.py` | Study areas, classes, months, grid and preprocessing settings |
| `src/gee_data.py` | Sentinel-1 monthly composites and Dynamic World labels from Earth Engine |
| `src/patching.py` | Tiles → 256 × 256 patches, class distribution, normalisation stats |
| `notebooks/01_data_export.ipynb` | Dataset export and preprocessing |
| `app/` | Prototype (to come) |
| `weights/` | Trained model weights (to come) |
