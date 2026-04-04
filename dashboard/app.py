"""Streamlit Dashboard for Urban Resilience Engine"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Urban Resilience Engine", layout="wide")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
METRICS_PATH = MODELS_DIR / "metrics.json"
SHAP_PLOT_PATH = MODELS_DIR / "shap_summary.png"
FORECAST_PLOT_PATH = MODELS_DIR / "forecast_chart.png"

# Title and header
st.title("🌍 Urban Resilience Engine")
st.caption("Predicting climate impact on urban infrastructure in Kenya")
st.divider()

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

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Risk Map", "📈 Forecast", "🔍 SHAP Explainability", "📋 Model Metrics"])

with tab1:
    st.header("County Risk Map")
    st.caption("Average risk index by county (2013-2023)")
    
    # Aggregate risk by county
    county_risk = df.groupby('county_id').agg({
        'risk_index': 'mean',
        'high_risk': 'mean'
    }).reset_index()
    
    county_risk['risk_label'] = county_risk['high_risk'].apply(
        lambda x: 'High Risk' if x > 0.3 else 'Low Risk'
    )
    
    # Bar chart
    fig = px.bar(
        county_risk,
        x='county_id',
        y='risk_index',
        color='risk_label',
        color_discrete_map={'High Risk': '#d62728', 'Low Risk': '#2ca02c'},
        title='Average Risk Index by County',
        labels={'county_id': 'County', 'risk_index': 'Risk Index'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("County Risk Summary")
    st.dataframe(
        county_risk.rename(columns={
            'county_id': 'County',
            'risk_index': 'Avg Risk Index',
            'high_risk': '% High Risk Months',
            'risk_label': 'Overall Status'
        }),
        hide_index=True,
        use_container_width=True
    )

with tab2:
    st.header("Risk Forecast (2013-2040)")
    st.caption("Projected infrastructure risk trend")
    
    if FORECAST_PLOT_PATH.exists():
        st.image(str(FORECAST_PLOT_PATH), use_container_width=True)
        
        if 'peak_year' in metrics and 'trough_year' in metrics:
            col1, col2 = st.columns(2)
            col1.metric("Peak Risk Year", metrics['peak_year'])
            col2.metric("Trough Year", metrics['trough_year'])
    else:
        st.info("Forecast chart not available. Run `python src/model.py` to generate.")
    
    # Historical trend
    st.subheader("Historical Risk Trend")
    yearly_risk = df.groupby('year')['risk_index'].mean().reset_index()
    
    fig2 = px.line(
        yearly_risk,
        x='year',
        y='risk_index',
        markers=True,
        title='Historical Risk Index (2013-2023)'
    )
    fig2.update_traces(line=dict(width=3))
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("Feature Importance (SHAP)")
    st.caption("Which factors drive infrastructure risk?")
    
    if SHAP_PLOT_PATH.exists():
        st.image(str(SHAP_PLOT_PATH), use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - Features sorted by average absolute SHAP value (global importance)
        - Color indicates feature value (red = high, blue = low)
        - X-axis shows impact on risk prediction
        - Rainfall anomaly and temperature are key drivers
        """)
    else:
        st.info("SHAP plot not available. Run `python src/model.py` to generate.")

with tab4:
    st.header("Model Performance Metrics")
    
    if metrics:
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "Test AUC",
            f"{metrics.get('test_auc', 0):.4f}",
            help="Area under ROC curve on test set (2021-2023)"
        )
        
        col2.metric(
            "CV AUC (mean)",
            f"{metrics.get('cv_auc_mean', 0):.4f}",
            help="5-fold time-series cross-validation AUC"
        )
        
        col3.metric(
            "Test MSE",
            f"{metrics.get('test_mse', 0):.4f}",
            help="Mean squared error for risk score regression"
        )
        
        st.divider()
        
        st.subheader("Full Metrics")
        metrics_df = pd.DataFrame([
            {"Metric": k.replace('_', ' ').title(), "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
            for k, v in metrics.items()
            if not isinstance(v, list)
        ])
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    else:
        st.warning("No metrics available.")

st.divider()
st.caption("Built with Streamlit | Data: CHIRPS, NOAA, OSM, Sentinel-2 | Model: XGBoost + SHAP")
