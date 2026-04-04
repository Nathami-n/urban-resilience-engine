# Urban Resilience Engine

Minimal starter scaffold for the project defined in [docs/PROJECT_SCOPE.md](/Users/mac/Desktop/clients/strath-assignment/docs/PROJECT_SCOPE.md).

## What is in place

- Project directory structure for data, source code, API, dashboard, models, notebooks, and tests.
- Conda environment definition in [environment.yml](/Users/mac/Desktop/clients/strath-assignment/environment.yml).
- Starter Python entry points for ETL, NDVI, modeling, prediction, audit, API, and dashboard.
- A `.gitignore` that keeps raw data, processed parquet files, model artifacts, and local caches out of git.

## Recommended start order

1. Create the environment: `conda env create -f environment.yml`
2. Activate it: `conda activate urban-resilience`
3. Authenticate Earth Engine once: `earthengine authenticate`
4. Build the ETL pipeline in `src/etl.py`
5. Add NDVI extraction in `src/vision.py`
6. Train the model in `src/model.py`
7. Wire the API and dashboard

## Starter commands

- Run ETL scaffold: `python src/etl.py`
- Run NDVI scaffold: `python src/vision.py`
- Run model scaffold: `python src/model.py`
- Start API: `uvicorn api.main:app --reload`
- Start dashboard: `streamlit run dashboard/app.py`
- Run tests: `pytest tests/`

## Notes

- Raw downloads belong under `data/raw/` and should never be edited manually.
- `data/processed/features.parquet` should become the single source of truth for modeling.
- `src/predict.py` is intended to hold shared inference logic used by the API and tests.
