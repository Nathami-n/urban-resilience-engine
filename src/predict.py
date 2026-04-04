"""Prediction module - shared inference logic for API and tests."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_risk_model.joblib"

# Load model once at module import
try:
    _model = joblib.load(MODEL_PATH)
    print(f"+ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"WARNING: Model not found at {MODEL_PATH}, predictions will be stubs")
    _model = None


def extract_features_from_geojson(
    geojson: dict[str, Any], year: int
) -> dict[str, float]:
    """
    Extract features from GeoJSON for prediction.
    In production, this would compute spatial aggregates.
    For demo, returns typical Nairobi values.
    """
    # Default feature values (Nairobi averages)
    features = {
        "rainfall_mm": 73.0,
        "rainfall_anomaly": 5.0,
        "temp_max_c": 31.0,
        "temp_min_c": 15.0,
        "soil_organic_carbon": 1.6,
        "road_density_km_per_km2": 1.2,
        "population_density": 4800.0,
        "ndvi_mean": 0.37,
    }

    # Could extract from GeoJSON properties if provided
    if "features" in geojson and len(geojson["features"]) > 0:
        props = geojson["features"][0].get("properties", {})
        for key in features.keys():
            if key in props:
                features[key] = float(props[key])

    return features


def predict_from_geojson(geojson: dict[str, Any], year: int) -> dict[str, Any]:
    """Main prediction function used by API."""
    if not isinstance(geojson, dict):
        raise TypeError("geojson must be a dictionary")
    if not isinstance(year, int):
        raise TypeError("year must be an integer")

    if _model is None:
        return {
            "risk_score": 0.0,
            "decline_year": year,
            "status": "model_not_loaded",
            "risk_label": "unknown",
        }

    try:
        # Extract features
        features = extract_features_from_geojson(geojson, year)

        # Create DataFrame (model expects specific column order)
        feature_cols = [
            "rainfall_mm",
            "rainfall_anomaly",
            "temp_max_c",
            "temp_min_c",
            "soil_organic_carbon",
            "road_density_km_per_km2",
            "population_density",
            "ndvi_mean",
        ]

        X = pd.DataFrame([features], columns=feature_cols)

        # Predict
        risk_proba = _model.predict_proba(X)[0, 1]  # Probability of high risk
        risk_class = _model.predict(X)[0]

        # Simple decline year forecast (in reality would use full time-series model)
        # Assume peak around 2030 based on current trends
        decline_year = 2030 if risk_proba > 0.5 else 2035

        # MSE placeholder (would come from validation set)
        mse = 0.13

        return {
            "risk_score": float(risk_proba),
            "risk_label": "high" if risk_class == 1 else "low",
            "decline_year": int(decline_year),
            "mse": float(mse),
            "status": "success",
        }

    except Exception as e:
        return {
            "risk_score": 0.0,
            "decline_year": year,
            "status": f"error: {str(e)}",
            "risk_label": "error",
        }
