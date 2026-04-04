# Urban Resilience Engine — Project Scope & Build Guide


> Each phase produces a clean output that feeds the next.

---

## Project Goal

Build a multi-stage ML pipeline that predicts urban infrastructure risk from extreme weather
events using open geospatial data for Kenya's major cities (Nairobi focus).
Output: a working FastAPI backend + Streamlit dashboard + 1-page bias audit.

---

## Scope (lean)

| Phase | What we build | What we skip |
|---|---|---|
| 1 – ETL | Fetch + clean weather, OSM, soil data. Output: `data/processed/features.parquet` | IoT/sensor feeds, US Census API |
| 2 – Vision | NDVI from Sentinel-2 via GEE script. Append to parquet. | Training CNN from scratch |
| 3 – Model | XGBoost risk classifier + SHAP explainability | Bayesian NNs, hyperparameter grid search |
| 4 – API + UI | FastAPI serving predictions + Streamlit dashboard with risk map | Docker, Kubernetes, auth |

**Target area:** Nairobi + food basket counties (Nakuru, Kisumu, Eldoret corridors)
**Time horizon:** Historical 2013–2023, forecast to 2040

---

## Tooling

```
Language        Python 3.11+
ETL             pandas, geopandas, osmnx, requests, pyproj
Spatial         rasterio, shapely, fiona
Climate data    CHIRPS via API, NOAA GSOD
Satellite       Google Earth Engine Python API (ee) — NDVI only
ML              xgboost, scikit-learn, shap
Validation      time-series cross-validation (TimeSeriesSplit)
API             fastapi, uvicorn, pydantic
Dashboard       streamlit, folium, plotly
Tracking        mlflow (one autolog() call)
Serialization   joblib (model artifacts)
Testing         pytest (smoke tests only)
Notebooks       jupyter (exploration only, not submission)
```

### Environment Setup (Anaconda)

> Use conda — not pip — for this project. The geospatial stack (gdal, fiona, rasterio, geopandas)
> has C dependencies that conda resolves cleanly. pip will likely fail or conflict on Mac.

**Step 1 — Create the environment from scratch:**

```bash
conda create -n urban-resilience python=3.11 -y
conda activate urban-resilience
```

**Step 2 — Install geospatial stack via conda-forge (critical — do this before anything else):**

```bash
conda install -c conda-forge geopandas rasterio fiona pyproj shapely osmnx -y
```

**Step 3 — Install ML + API stack (conda-forge where available, pip for the rest):**

```bash
conda install -c conda-forge xgboost scikit-learn shap mlflow joblib pandas requests -y
conda install -c conda-forge fastapi uvicorn pydantic -y
conda install -c conda-forge streamlit folium plotly -y
conda install -c conda-forge pytest jupyter notebook -y
pip install streamlit-folium earthengine-api
```

> `streamlit-folium` and `earthengine-api` are pip-only — safe to pip install after
> conda has handled all the C-dependency packages above.

**Step 4 — Export environment for reproducibility:**

```bash
conda env export > environment.yml
# commit this file — teammate or marker can recreate with:
# conda env create -f environment.yml
```

**To reactivate after closing terminal:**

```bash
conda activate urban-resilience
```

**To deactivate:**

```bash
conda deactivate
```

### GEE Auth (one-time, manual step)

```bash
# with urban-resilience env active:
earthengine authenticate
# opens browser — sign in with Google, paste token back into terminal
```

---

## Codebase Structure

```
urban-resilience-engine/
│
├── data/
│   ├── raw/                    # never edit — downloaded files land here
│   │   ├── osm/
│   │   ├── chirps/
│   │   └── soil/
│   └── processed/
│       ├── features.parquet    # ETL output — single source of truth
│       └── ndvi.parquet        # Phase 2 output, merged in Phase 3
│
├── src/
│   ├── etl.py                  # Phase 1: fetch + clean + merge → features.parquet
│   ├── vision.py               # Phase 2: GEE NDVI extraction → ndvi.parquet
│   ├── model.py                # Phase 3: train XGBoost, SHAP, save artifact
│   ├── predict.py              # load model, run inference on new GeoJSON input
│   └── audit.py                # bias analysis by neighborhood income proxy
│
├── api/
│   └── main.py                 # FastAPI app — /predict endpoint accepts GeoJSON
│
├── dashboard/
│   └── app.py                  # Streamlit app — map + risk chart + SHAP plot
│
├── models/
│   └── xgb_risk_model.joblib   # serialized model artifact
│
├── notebooks/
│   ├── 01_etl_exploration.ipynb
│   ├── 02_ndvi_exploration.ipynb
│   └── 03_model_exploration.ipynb
│
├── tests/
│   └── test_smoke.py           # import checks + predict() returns expected shape
│
├── environment.yml             # conda env — recreate with: conda env create -f environment.yml
├── README.md
└── PROJECT_SCOPE.md            # this file
```

---

## Phase 1 — ETL (`src/etl.py`)

**Inputs:**
- OSM via `osmnx`: building footprints, road networks, hospital locations for Nairobi
- CHIRPS rainfall: daily precipitation 2013–2023 (API: `https://data.chc.ucsb.edu/products/CHIRPS-2.0/`)
- NOAA GSOD: temperature min/max (download via `requests` + parse CSV)
- iSDA soil: soil organic carbon (download GeoTIFF for Kenya bounding box)

**Steps:**
1. Pull OSM admin boundaries for target counties → `data/raw/osm/`
2. Fetch CHIRPS monthly aggregates clipped to bounding box
3. Fetch GSOD station data, spatial join to county polygons via kriging interpolation
4. Load iSDA soil raster, extract mean SOC per county polygon
5. Compute composite risk proxy: `risk = f(rainfall_anomaly, temp_max, soc)`
6. Merge all on `(county_id, year_month)` → export `data/processed/features.parquet`

**Key rule:** write to `data/raw/` first, never overwrite raw. Processed outputs go to `data/processed/`.

---

## Phase 2 — NDVI (`src/vision.py`)

Use the GEE Python API. Do not download satellite tiles locally.

**Steps:**
1. Define AOI as Kenya food basket bounding box
2. Filter Sentinel-2 SR collection (2013–2023), cloud mask < 20%
3. Compute annual mean NDVI: `(NIR - RED) / (NIR + RED)`
4. Export annual NDVI mean per county polygon as CSV → `data/processed/ndvi.parquet`

**Scope limit:** one NDVI value per county per year. No pixel-level CNN inference.

```python
# sketch — agent fills in full implementation
import ee
ee.Initialize()

s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterDate('2013-01-01', '2023-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI'))
```

---

## Phase 3 — Model (`src/model.py`)

**Target variable:** binary `high_risk` (1 = risk score above threshold, 0 = normal)
Threshold defined as top quartile of composite risk index from Phase 1.

**Features:** rainfall_anomaly, temp_max, temp_min, soc, ndvi, population_density (from WorldPop or OSM proxy), road_density

**Training:**
```python
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import mlflow

mlflow.sklearn.autolog()

tscv = TimeSeriesSplit(n_splits=5)
model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
# fit with tscv splits, log metrics per fold
```

**SHAP:**
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)  # save as PNG for report
```

**Output:** `models/xgb_risk_model.joblib` + `models/shap_summary.png`

---

## Phase 4 — API (`api/main.py`)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json

app = FastAPI(title="Urban Resilience Engine")
model = joblib.load("models/xgb_risk_model.joblib")

class FarmInput(BaseModel):
    geojson: dict
    year: int

@app.post("/predict")
def predict(payload: FarmInput):
    # extract features from geojson, run model.predict_proba
    # return risk score + decline year estimate
    ...
```

---

## Phase 4 — Dashboard (`dashboard/app.py`)

Three sections only:
1. **Risk map** — `folium` choropleth of county risk scores
2. **Forecast chart** — `plotly` line chart of risk trend 2013–2040
3. **SHAP plot** — `st.image()` rendering `models/shap_summary.png`

Run: `streamlit run dashboard/app.py`

---

## Audit Report (`src/audit.py`)

One script, one output. Compare model performance metrics (precision, recall) split by:
- High vs low population density counties
- Urban core vs peri-urban counties

Output: `audit_report.md` — 3 paragraphs + 1 table. This goes directly into the final PDF.

---

## What "done" looks like

- [ ] `features.parquet` exists with no nulls
- [ ] `ndvi.parquet` merged into features
- [ ] Model trained, R² or AUC logged in MLflow
- [ ] `xgb_risk_model.joblib` saved
- [ ] `shap_summary.png` saved
- [ ] `/predict` endpoint returns JSON with `risk_score` and `decline_year`
- [ ] Streamlit dashboard runs without errors
- [ ] `audit_report.md` written
- [ ] `pytest tests/` passes

---

## Notes for Agent

- Always activate `conda activate urban-resilience` before running any script
- Always read from `data/processed/features.parquet` — never re-fetch raw data in model scripts
- Use `geopandas` for all spatial joins — not manual coordinate math
- Keep each `src/*.py` runnable standalone: `python src/etl.py` should work
- Log everything to MLflow: `mlflow ui` should show experiment runs
- No Jupyter notebooks in `src/` — notebooks are scratch only
- Never use `pip install` for geospatial packages — always `conda install -c conda-forge`
