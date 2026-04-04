# Bias Audit Report

## Urban Resilience Engine - Model Performance Analysis

**Date:** 2026-04-04  
**Model:** XGBoost Risk Classifier  
**Audit Scope:** Performance comparison across urban density strata

---

## Methodology

Counties were stratified into two groups based on population density:

- **High Density (Urban Core):** Population density > 158.4 people/km²  
  Proxy for established urban areas with higher infrastructure concentration.
  
- **Low Density (Peri-urban):** Population density ≤ 158.4 people/km²  
  Proxy for peri-urban and emerging development zones.

**Justification:** Direct socioeconomic data (ward-level income, wealth indices) was not available from KNBS at the time of this analysis. Population density from OpenStreetMap building footprints serves as a reasonable proxy for infrastructure maturity and urbanization level.

---

## Results

### Performance Metrics by Group

| Group | Sample Count | Precision | Recall | F1-Score |
|-------|--------------|-----------|--------|----------|
| High Density (Urban Core) | 264 | 0.951 | 0.947 | 0.946 |
| Low Density (Peri-urban) | 264 | 0.985 | 0.985 | 0.985 |

---

## Findings

**Performance Disparity:** The F1-score difference between high-density and low-density groups is **0.039**.

**Assessment:** Model performance is **relatively consistent** across density groups, suggesting no significant bias in favor of urban or peri-urban areas. The classifier generalizes well across different infrastructure contexts.

---

## Limitations

1. **Proxy Metric:** Population density is used as a proxy for socioeconomic status. Future work should integrate:  
   - KNBS ward-level income data  
   - Household wealth indices  
   - Infrastructure service coverage metrics

2. **Sample Size:** The analysis is based on 528 monthly records across 4 counties. Expanding to all 47 Kenyan counties would provide more robust statistical power.

3. **Temporal Coverage:** Historical data spans 2013-2023. Post-pandemic urban dynamics (2020+) may introduce confounding factors.

---

## Recommendations

1. **Data Enhancement:**  
   Integrate KNBS socioeconomic datasets for more precise stratification beyond population density.

2. **Threshold Adjustment:**  
   Consider county-specific risk thresholds rather than a uniform 75th percentile cutoff to account for local context.

3. **Longitudinal Monitoring:**  
   Re-run this audit annually as new data becomes available to track performance equity over time.

---

**Report Generated:** 2026-04-04 21:44:36