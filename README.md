# SAR Land Cover Monitor

Global land use / land cover (LULC) semantic segmentation from **Sentinel-1 radar only**, plus a prototype
that maps any location over 12 months and reports class percentages and changes.

* **Input:** Sentinel-1 VV and VH backscatter (and their ratio), monthly composites
* **Output:** 8 classes: water, trees, flooded vegetation, crops, rangeland, built, bare, snow/ice
* **Labels:** Google Dynamic World (training); Esri 10 m Annual LULC as an independent check
* **Framework:** TensorFlow / Keras

## Repository

| Path | Contents |
|---|---|
| `notebooks/01_data_export.ipynb` | Dataset: 40 study areas on six continents, Earth Engine export, preprocessing, patching |
| `notebooks/02_training.ipynb` | Baseline CNN, U-Net and DeepLabV3+: training curves, test metrics, model selection |
| `app/streamlit_app.py` | Prototype (web app) |
| `src/config.py` | Study areas, classes, periods, grid and preprocessing settings |
| `src/gee_data.py` | Sentinel-1 monthly composites and Dynamic World labels from Earth Engine |
| `src/patching.py` | Tiles → 256 × 256 patches, class distribution, normalisation statistics |
| `src/model.py` | Model architectures, loss, normalisation |
| `src/evaluation.py` | Overall accuracy, mIoU, per-class IoU / F1 |
| `src/prototype.py` | Prototype pipeline: retrieval, inference, percentages, changes, reference comparison |
| `src/plots.py` | Figures for the prototype |
| `weights/` | Trained models (`best_model.keras` is used by the prototype), `stats.json`, training logs |
| `scripts/` | Background export, tile check, orbit selection, lab-server setup |

## Required account

**Google Earth Engine** with a Google Cloud project registered for noncommercial use (free):

1. Sign up at <https://earthengine.google.com> and create a Cloud project.
2. Register the project for noncommercial use (Community tier is sufficient for the prototype).
3. Put your project ID in `src/config.py` (`GEE_PROJECT`).

No other API keys are needed.

## Setup

Python 3.11.

```bash
conda create -n lulc python=3.11 -y
conda activate lulc
pip install -r requirements.txt
earthengine authenticate          # on a machine without a browser: --auth_mode=notebook
```

## Running the prototype

```bash
streamlit run app/streamlit_app.py
```

1. Choose a preset city or enter latitude / longitude.
2. Choose the start month; the 12 months must end by the last complete month.
3. Choose the area size (2.56, 5.12 or 10.24 km square) and press **Run**.

Optional switches: **temporal smoothing** (3-month majority, to see how much of the change is noise) and
**6 radar-separable classes** (crops + rangeland and bare + snow/ice merged, the pairs C-band radar cannot
separate; same prediction, no new run).

A run takes about 30–90 seconds. The app shows the 12 monthly maps, the class percentages per month, the
month-to-month and total change in percentage points, a comparison with the Esri map, and the Sentinel-1
input. Percentages and changes can be downloaded as CSV.

### How it works

1. **Retrieval:** for each of the 12 months, Earth Engine builds a Sentinel-1 composite exactly as for training:
   IW GRD scenes (calibrated, terrain-corrected, dB), border-noise mask, Lee speckle filter, per-pixel median.
   One orbit pass is used for all 12 months (the one with more scenes), so that viewing geometry does not
   change between months.
2. **Inference:** the composite is normalised with the training statistics (`weights/stats.json`) and classified
   with `weights/best_model.keras` in overlapping 256 × 256 windows (50 % overlap, probabilities averaged).
3. **Calculation:** class percentage = class pixels / pixels with radar data × 100 for each month;
   month-to-month change = share(month) − share(previous month); total change = share(last month) −
   share(first month); all changes in percentage points.
4. **Check:** the majority class per pixel over the 12 months is compared with the Esri annual map.

## Reproducing the dataset and training

Run `notebooks/01_data_export.ipynb`, then `notebooks/02_training.ipynb` (GPU recommended). The full export is
about 15 GB and needs more than one month of the Community-tier Earth Engine quota.
The dataset download link is given in the report.
