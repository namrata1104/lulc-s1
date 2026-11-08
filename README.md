# SAR Land Cover Monitor

Land use / land cover (LULC) semantic segmentation from **Sentinel-1 radar only**, trained on 40 study areas on six
continents, plus a prototype that maps any location over 12 months and reports class percentages and changes.

* **Input:** Sentinel-1 IW GRD, VV + VH (+ VV−VH ratio), monthly composites at 10 m
* **Labels:** Google Dynamic World (Sentinel-2 based), 8 classes: water, trees, flooded vegetation, crops, rangeland,
  built, bare, snow/ice
* **Independent check:** Esri 10 m Annual LULC (never used for training)
* **Framework:** TensorFlow / Keras

## Key results

| Evaluation | Overall accuracy | mIoU |
|---|---|---|
| Test set (9 unseen regions, 2025–26), 8 classes | 0.51 | 0.30 |
| **Same predictions, 6 radar-separable classes** | **0.72** | **0.46** |
| 6 classes, without the two extreme regions (Atacama, Vatnajökull) | **0.77** | 0.46 |
| Training regions in 2025–26 (images never seen), 8 classes | 0.76–0.78 | 0.56–0.58 |
| Case study Munich 2025 vs. the independent Esri map | **0.88** (pixel agreement) | – |

Model: DeepLabV3+ with ResNet50 (`weights/best_model.keras`). Full report: [`report/report.pdf`](report/report.pdf).

## What was done

| Notebook | Content |
|---|---|
| [`01_data_export`](notebooks/01_data_export.ipynb) | Study areas (27 train, 26 with data · 4 val · 9 test; split by region), Earth Engine export, preprocessing, patching, class distribution |
| [`02_training`](notebooks/02_training.ipynb) | Baseline CNN, U-Net, DeepLabV3+ (ResNet50): training curves, test metrics, model selection |
| [`03_experiments`](notebooks/03_experiments.ipynb) | Six DeepLabV3+ / AE-DeepLabV3+ variants with ResNet101, ResNet152 and Xception encoders |
| [`04_case_study`](notebooks/04_case_study.ipynb) | Prototype pipeline for Munich 2025, comparison with Esri, seasonal effects, error inflation of changes |
| [`05_multitemporal`](notebooks/05_multitemporal.ipynb) | 3-month input model; evaluation with 6 radar-separable classes |

**Dataset:** https://www.kaggle.com/datasets/namratabhoyar/sentinel1-lulc-global (unzip into `data/` to reuse it without a new Earth Engine export).

**Data and preprocessing.** Train / validation on 2023–24, test on 9 unseen regions in 2025–26. Sentinel-1 scenes
(calibrated, terrain-corrected, dB by Earth Engine) use one orbit pass per area, border-noise masking, a Lee speckle
filter and a monthly median. Labels are the monthly mean Dynamic World probabilities (argmax) on the same 10 m grid;
uncertain pixels (probability < 0.3) are ignored. 256 × 256 patches, z-score normalisation with training statistics,
flips / rotations / brightness / speckle-noise augmentation. 14,952 training, 1,808 validation and 4,536 test patches.

**Models.** Ten models were trained with the same procedure (weighted cross-entropy + Dice, AdamW, learning-rate
halving and early stopping on validation mIoU):

| Models | Test OA | Test mIoU |
|---|---|---|
| Baseline CNN | 0.42 | 0.20 |
| U-Net | 0.50 | 0.27 |
| **DeepLabV3+ · ResNet50** (used in the prototype) | **0.51** | **0.30** |
| 6 DeepLabV3+ / AE-DeepLabV3+ variants (ResNet101, ResNet152, Xception) | 0.43–0.52 | 0.24–0.30 |
| DeepLabV3+ · ResNet50 with 3 months as input | 0.48 | 0.28 |

## Findings (short)

1. **Context and pretraining help** (baseline → U-Net → DeepLabV3+), **larger encoders and temporal input do not**:
   all deep models end at about 0.30 test mIoU on unseen regions, while on the training regions in 2025–26 the same
   models reach 0.56–0.58 mIoU and 0.76–0.78 OA. The limit is the diversity of the training landscapes, not the network.
2. **Most errors fall into two class pairs that C-band radar cannot separate physically:** bare ↔ snow/ice (dry snow
   is nearly transparent to radar) and crops ↔ rangeland. Two test regions (Atacama salt flats, Vatnajökull glacier)
   fail almost completely.
3. **From 8 to 6 classes:** merging these pairs into *low vegetation* and *bare / snow* raises the test OA of the same
   predictions from **0.51 to 0.72** (mIoU 0.30 → 0.46), and to **0.77** without the two extreme regions. The prototype
   offers both views.
4. **Case study Munich 2025:** 88 % pixel agreement with the Esri map. Part of the monthly change is seasonal (trees
   10.6 % in January vs. 17.1 % in July, leaf-on/off), so the January → December total change is inflated; most
   classification errors are systematic and partly cancel in the changes. The Oktoberfest on the Theresienwiese is
   detected as a real temporary change.
5. **Few radar scenes per month make maps noisier**; the prototype flags months with fewer than 4 scenes.

Details, numbers and the critical discussion are in the notebooks and the project report.

## Prototype

```bash
streamlit run app/streamlit_app.py
```

1. Search a place by name (any city, country or landmark, OpenStreetMap) or enter latitude / longitude.
2. Choose the start month; the 12 months must end by the last complete month.
3. Choose the area (2.56, 5.12 or 10.24 km square) and press **Run** (about 1–3 min for 10.24 km, less for smaller areas; depends on Earth Engine load).

The app shows the 12 monthly maps, class percentages per month, month-to-month and total change in percentage points,
the comparison with the Esri map and the Sentinel-1 input; results can be downloaded as CSV. Optional switches:
**temporal smoothing** (3-month majority) and **6 radar-separable classes** (same prediction, no new run).

**How it works.** For each month, Earth Engine builds a Sentinel-1 composite exactly as for training (one orbit pass
for all 12 months). The composite is normalised with `weights/stats.json` and classified by `weights/best_model.keras`
in overlapping 256 × 256 windows. Class percentage = class pixels / pixels with radar data × 100; month-to-month change =
share(month) − share(previous month); total change = share(last month) − share(first month), in percentage points.

## Setup

Python 3.11.

```bash
conda create -n lulc python=3.11 -y
conda activate lulc
pip install -r requirements.txt
earthengine authenticate          # without a browser: --auth_mode=notebook
```

**Required account: Google Earth Engine** with a Google Cloud project registered for noncommercial use (free,
Community tier is enough for the prototype). Put the project ID in `src/config.py` (`GEE_PROJECT`). No other keys
are needed; the place search uses the public OpenStreetMap service.

## Repository

| Path | Contents |
|---|---|
| `notebooks/` | The five notebooks above, with outputs |
| `report/` | Project report (IEEE format, Cambria/Caladea 12 pt): PDF, LaTeX source, figures |
| `app/streamlit_app.py` | Prototype |
| `src/` | `config` (areas, classes, settings), `gee_data` (Earth Engine), `patching`, `temporal`, `model`, `training`, `evaluation`, `prototype`, `plots` |
| `weights/` | `best_model.keras` (prototype model), all notebook-02 models, `stats.json`, training logs and figures; `experiments/` and `multitemporal/` logs and results |
| `scripts/` | Background export, tile check, orbit selection, lab-server setup |

## Submission

| Requirement | Where |
|---|---|
| Dataset | **https://www.kaggle.com/datasets/namratabhoyar/sentinel1-lulc-global** (≈ 25 GB: `raw_global.zip` Earth Engine tiles, `patches_global.zip` training patches, `export_log.csv`) |
| Notebooks with results and training curves | `notebooks/01`–`05` |
| Prototype source, README, requirements | `app/`, `src/`, this file, `requirements.txt` |
| Trained model weights | `weights/best_model.keras` (+ all compared models in `weights/`) |
| Project report (IEEE format) | [`report/report.pdf`](report/report.pdf), LaTeX source `report/report.tex` |
| Screen recording | `Screen recording - Munich full run .mp4` (3:59 min, Munich 2025; Earth Engine waiting time shortened), submitted separately |
