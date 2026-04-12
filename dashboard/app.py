"""Streamlit Dashboard for Urban Resilience Engine"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict import predict_from_geojson

st.set_page_config(
    page_title="Urban Resilience Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
METRICS_PATH = MODELS_DIR / "metrics.json"
SHAP_PLOT_PATH = MODELS_DIR / "shap_summary.png"
FORECAST_PLOT_PATH = MODELS_DIR / "forecast_chart.png"

# Header
st.title("Urban Resilience Engine")
st.markdown("**Climate Risk Assessment for Kenyan Infrastructure**")
st.markdown("---")

# Load metrics
if METRICS_PATH.exists():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
else:
    metrics = {}
    st.warning("Model metrics not found. Run `python src/model.py` first.")

# Load features
if FEATURES_PATH.exists():
    df = pd.read_parquet(FEATURES_PATH)
else:
    st.error("Features not found. Run `python src/etl.py` first.")
    st.stop()

# Navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Risk Map",
        "Forecast Projection",
        "SHAP Analysis",
        "Model Performance",
        "Live Prediction",
    ]
)

with tab1:
    st.header("County Risk Assessment")
    st.caption("Average risk index by county (2013-2023)")

    # Aggregate risk by county
    county_risk = (
        df.groupby("county_id")
        .agg({"risk_index": "mean", "high_risk": "mean"})
        .reset_index()
    )

    county_risk["risk_label"] = county_risk["high_risk"].apply(
        lambda x: "High Risk" if x > 0.3 else "Low Risk"
    )

    # Bar chart
    fig = px.bar(
        county_risk,
        x="county_id",
        y="risk_index",
        color="risk_label",
        color_discrete_map={"High Risk": "#d62728", "Low Risk": "#2ca02c"},
        title="Average Risk Index by County",
        labels={"county_id": "County", "risk_index": "Risk Index"},
    )
    fig.update_layout(height=450, showlegend=True)
    st.plotly_chart(fig, width="stretch")

    # Data table
    st.subheader("Summary Statistics")
    st.dataframe(
        county_risk.rename(
            columns={
                "county_id": "County",
                "risk_index": "Avg Risk Index",
                "high_risk": "Proportion High Risk",
                "risk_label": "Overall Status",
            }
        ),
        hide_index=True,
        width="stretch",
    )

with tab2:
    st.header("Infrastructure Risk Forecast")
    st.caption("Projected risk trend (2013-2040)")

    if FORECAST_PLOT_PATH.exists():
        st.image(str(FORECAST_PLOT_PATH), width="stretch")

        if "peak_year" in metrics and "trough_year" in metrics:
            col1, col2 = st.columns(2)
            col1.metric("Forecasted Peak Risk Year", metrics["peak_year"])
            col2.metric("Forecasted Trough Year", metrics["trough_year"])
    else:
        st.info("Forecast chart not available. Run `python src/model.py` to generate.")

    # Historical trend
    st.subheader("Historical Risk Trend")
    yearly_risk = df.groupby("year")["risk_index"].mean().reset_index()

    fig2 = px.line(
        yearly_risk,
        x="year",
        y="risk_index",
        markers=True,
        title="Historical Mean Risk Index (2013-2023)",
    )
    fig2.update_traces(line=dict(width=3, color="#1f77b4"))
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, width="stretch")

with tab3:
    st.header("SHAP Feature Importance Analysis")
    st.caption("Identifying key drivers of infrastructure risk")

    if SHAP_PLOT_PATH.exists():
        st.image(str(SHAP_PLOT_PATH), width="stretch")

        st.markdown(
            """
        **Interpretation Guide:**
        - Features are ranked by average absolute SHAP value (global importance)
        - Color indicates feature value (red = high, blue = low)
        - Horizontal position shows impact on risk prediction
        - Rainfall anomaly and soil organic carbon are primary drivers
        """
        )
    else:
        st.info("SHAP plot not available. Run `python src/model.py` to generate.")

with tab4:
    st.header("Model Performance Metrics")

    if metrics:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Test AUC",
            f"{metrics.get('test_auc', 0):.4f}",
            help="Area under ROC curve on test set (2021-2023)",
        )

        col2.metric(
            "Cross-Validation AUC",
            f"{metrics.get('cv_auc_mean', 0):.4f}",
            help="5-fold time-series cross-validation AUC",
        )

        col3.metric(
            "Test MSE",
            f"{metrics.get('test_mse', 0):.4f}",
            help="Mean squared error for risk score regression",
        )

        st.markdown("---")

        st.subheader("Complete Metrics Summary")
        metrics_df = pd.DataFrame(
            [
                {
                    "Metric": k.replace("_", " ").title(),
                    "Value": f"{v:.4f}" if isinstance(v, float) else str(v),
                }
                for k, v in metrics.items()
                if not isinstance(v, list)
            ]
        )
        st.dataframe(metrics_df, hide_index=True, width="stretch")
    else:
        st.warning("No metrics available.")

with tab5:
    st.header("Live Risk Prediction")
    st.caption("Enter field values to get an instant infrastructure risk prediction.")

    col1, col2 = st.columns(2)
    with col1:
        rainfall_mm = st.number_input(
            "Rainfall (mm)", min_value=0.0, max_value=500.0, value=73.0
        )
        rainfall_anomaly = st.number_input(
            "Rainfall Anomaly (mm)", min_value=-100.0, max_value=200.0, value=5.0
        )
        temp_max_c = st.number_input(
            "Max Temperature (°C)", min_value=10.0, max_value=50.0, value=31.0
        )
        temp_min_c = st.number_input(
            "Min Temperature (°C)", min_value=0.0, max_value=35.0, value=15.0
        )
    with col2:
        soil_organic_carbon = st.number_input(
            "Soil Organic Carbon (g/kg)", min_value=0.1, max_value=5.0, value=1.6
        )
        road_density = st.number_input(
            "Road Density (km/km²)", min_value=0.0, max_value=5.0, value=1.2
        )
        population_density = st.number_input(
            "Population Density (people/km²)",
            min_value=0.0,
            max_value=10000.0,
            value=4800.0,
        )
        ndvi_mean = st.number_input(
            "NDVI Mean", min_value=0.0, max_value=1.0, value=0.37
        )
    year = st.slider("Year", min_value=2024, max_value=2040, value=2026)

    if st.button("Predict Risk", type="primary"):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [36.817223, -1.286389],
                    },
                    "properties": {
                        "rainfall_mm": rainfall_mm,
                        "rainfall_anomaly": rainfall_anomaly,
                        "temp_max_c": temp_max_c,
                        "temp_min_c": temp_min_c,
                        "soil_organic_carbon": soil_organic_carbon,
                        "road_density_km_per_km2": road_density,
                        "population_density": population_density,
                        "ndvi_mean": ndvi_mean,
                    },
                }
            ],
        }
        result = predict_from_geojson(geojson, year)

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        risk_score = result.get("risk_score", 0)
        risk_label = result.get("risk_label", "unknown")
        decline_year = result.get("decline_year", year)

        color = "🔴" if risk_label == "high" else "🟢"
        r1.metric("Risk Score", f"{risk_score:.2f}")
        r2.metric("Risk Label", f"{color} {risk_label.upper()}")
        r3.metric("Projected Decline Year", decline_year)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                title={"text": "Risk Score (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#d62728" if risk_label == "high" else "#2ca02c"},
                    "steps": [
                        {"range": [0, 40], "color": "#eafaea"},
                        {"range": [40, 70], "color": "#fff3cd"},
                        {"range": [70, 100], "color": "#fde8e8"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 70,
                    },
                },
            )
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.caption(
    "Data Sources: CHIRPS, NOAA GSOD, OpenStreetMap, Sentinel-2, iSDA Soil | Model: XGBoost + SHAP"
)
