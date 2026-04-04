"""
Bias Audit: Performance comparison across urban density groups
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_fscore_support, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_risk_model.joblib"
AUDIT_REPORT_PATH = PROJECT_ROOT / "audit_report.md"


def main() -> None:
    """Run bias audit comparing model performance across density groups."""
    print("=" * 70)
    print("BIAS AUDIT")
    print("=" * 70)

    # Load data and model
    print("\n→ Loading data and model...")
    df = pd.read_parquet(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)

    # Drop duplicate columns
    df = df[[c for c in df.columns if not c.endswith("_x") and not c.endswith("_y")]]

    # Split into density groups using population_density as proxy
    # High density = urban central (Nairobi), Low density = peri-urban
    density_threshold = df["population_density"].median()
    df["density_group"] = df["population_density"].apply(
        lambda x: (
            "High Density (Urban Core)"
            if x > density_threshold
            else "Low Density (Peri-urban)"
        )
    )

    print(f"  ✓ Loaded {len(df)} records")
    print(f"  ✓ Density threshold: {density_threshold:.1f} people/km²")
    print(
        f"  ✓ High density: {(df['density_group'] == 'High Density (Urban Core)').sum()} records"
    )
    print(
        f"  ✓ Low density: {(df['density_group'] == 'Low Density (Peri-urban)').sum()} records"
    )

    # Prepare features
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

    X = df[feature_cols]
    y_true = df["high_risk"]
    y_pred = model.predict(X)

    # Overall performance
    print("\n→ Overall Model Performance:")
    print(classification_report(y_true, y_pred, target_names=["Low Risk", "High Risk"]))

    # Group-wise performance
    print("\n→ Performance by Density Group:\n")

    results = []
    for group in df["density_group"].unique():
        mask = df["density_group"] == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_group, y_pred_group, average="weighted", zero_division=0
        )

        sample_count = len(y_true_group)

        results.append(
            {
                "Group": group,
                "Sample Count": sample_count,
                "Precision": f"{precision:.3f}",
                "Recall": f"{recall:.3f}",
                "F1-Score": f"{f1:.3f}",
            }
        )

        print(f"**{group}**")
        print(f"  Sample Count: {sample_count}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print()

    # Generate markdown report
    print("→ Generating audit report...")

    report_lines = [
        "# Bias Audit Report",
        "",
        "## Urban Resilience Engine - Model Performance Analysis",
        "",
        "**Date:** 2026-04-04  ",
        "**Model:** XGBoost Risk Classifier  ",
        "**Audit Scope:** Performance comparison across urban density strata",
        "",
        "---",
        "",
        "## Methodology",
        "",
        "Counties were stratified into two groups based on population density:",
        "",
        f"- **High Density (Urban Core):** Population density > {density_threshold:.1f} people/km²  ",
        "  Proxy for established urban areas with higher infrastructure concentration.",
        "  ",
        f"- **Low Density (Peri-urban):** Population density ≤ {density_threshold:.1f} people/km²  ",
        "  Proxy for peri-urban and emerging development zones.",
        "",
        "**Justification:** Direct socioeconomic data (ward-level income, wealth indices) was not available from KNBS at the time of this analysis. Population density from OpenStreetMap building footprints serves as a reasonable proxy for infrastructure maturity and urbanization level.",
        "",
        "---",
        "",
        "## Results",
        "",
        "### Performance Metrics by Group",
        "",
        "| Group | Sample Count | Precision | Recall | F1-Score |",
        "|-------|--------------|-----------|--------|----------|",
    ]

    for r in results:
        report_lines.append(
            f"| {r['Group']} | {r['Sample Count']} | {r['Precision']} | {r['Recall']} | {r['F1-Score']} |"
        )

    # Calculate disparity
    if len(results) == 2:
        f1_diff = abs(float(results[0]["F1-Score"]) - float(results[1]["F1-Score"]))

        report_lines.extend(
            [
                "",
                "---",
                "",
                "## Findings",
                "",
                f"**Performance Disparity:** The F1-score difference between high-density and low-density groups is **{f1_diff:.3f}**.",
                "",
            ]
        )

        if f1_diff < 0.05:
            report_lines.append(
                "**Assessment:** Model performance is **relatively consistent** across density groups, suggesting no significant bias in favor of urban or peri-urban areas. The classifier generalizes well across different infrastructure contexts."
            )
        elif f1_diff < 0.10:
            report_lines.append(
                "**Assessment:** Model shows **minor performance variation** between density groups. This is within acceptable limits for real-world ML systems and may reflect genuine differences in data quality or feature distributions rather than systematic bias."
            )
        else:
            report_lines.append(
                "**Assessment:** Model exhibits **noticeable performance disparity** between density groups. Further investigation recommended to ensure equitable predictive accuracy across urban and peri-urban counties."
            )

    report_lines.extend(
        [
            "",
            "---",
            "",
            "## Limitations",
            "",
            "1. **Proxy Metric:** Population density is used as a proxy for socioeconomic status. Future work should integrate:  ",
            "   - KNBS ward-level income data  ",
            "   - Household wealth indices  ",
            "   - Infrastructure service coverage metrics",
            "",
            f"2. **Sample Size:** The analysis is based on {len(df)} monthly records across 4 counties. Expanding to all 47 Kenyan counties would provide more robust statistical power.",
            "",
            "3. **Temporal Coverage:** Historical data spans 2013-2023. Post-pandemic urban dynamics (2020+) may introduce confounding factors.",
            "",
            "---",
            "",
            "## Recommendations",
            "",
            "1. **Data Enhancement:**  ",
            "   Integrate KNBS socioeconomic datasets for more precise stratification beyond population density.",
            "",
            "2. **Threshold Adjustment:**  ",
            "   Consider county-specific risk thresholds rather than a uniform 75th percentile cutoff to account for local context.",
            "",
            "3. **Longitudinal Monitoring:**  ",
            "   Re-run this audit annually as new data becomes available to track performance equity over time.",
            "",
            "---",
            "",
            f"**Report Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
    )

    report_md = "\n".join(report_lines)

    # Save report
    with open(AUDIT_REPORT_PATH, "w") as f:
        f.write(report_md)

    print(f"  ✓ Audit report saved to {AUDIT_REPORT_PATH}")

    print("\n" + "=" * 70)
    print("✅ BIAS AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
