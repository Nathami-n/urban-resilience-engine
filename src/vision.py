"""
Phase 2: NDVI Extraction (Simulated Sentinel-2)
In production, this would use Google Earth Engine API.
For demonstration, generates realistic NDVI values based on climate patterns.
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
NDVI_PATH = PROCESSED_DIR / "ndvi.parquet"


def simulate_ndvi_from_climate(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate NDVI values based on rainfall and temperature patterns.
    In production, this would query Google Earth Engine Sentinel-2 collection.

    NDVI (Normalized Difference Vegetation Index):
    - Range: -1 to 1
    - Higher values = more vegetation
    - Positively correlated with rainfall
    - Negatively correlated with temperature extremes
    """
    print(">> Simulating NDVI extraction (Sentinel-2 proxy)...")

    np.random.seed(45)

    # Annual aggregation (as per project scope: one NDVI value per county per year)
    annual_data = (
        features_df.groupby(["county_id", "year"])
        .agg(
            {
                "rainfall_mm": "mean",
                "temp_max_c": "mean",
                "soil_organic_carbon": "first",  # Static per year
            }
        )
        .reset_index()
    )

    # NDVI model: higher rainfall + moderate temp + higher SOC = higher NDVI
    # Normalize inputs
    rain_norm = (annual_data["rainfall_mm"] - annual_data["rainfall_mm"].min()) / (
        annual_data["rainfall_mm"].max() - annual_data["rainfall_mm"].min()
    )

    temp_norm = (annual_data["temp_max_c"] - annual_data["temp_max_c"].min()) / (
        annual_data["temp_max_c"].max() - annual_data["temp_max_c"].min()
    )

    soc_norm = (
        annual_data["soil_organic_carbon"] - annual_data["soil_organic_carbon"].min()
    ) / (
        annual_data["soil_organic_carbon"].max()
        - annual_data["soil_organic_carbon"].min()
    )

    # NDVI formula: weighted combination
    # Base range: 0.2 (bare soil/sparse vegetation) to 0.8 (dense vegetation)
    ndvi_base = 0.2 + 0.6 * (0.5 * rain_norm + 0.2 * soc_norm + 0.3 * (1 - temp_norm))

    # Add realistic noise
    noise = np.random.normal(0, 0.05, size=len(ndvi_base))
    annual_data["ndvi_mean"] = np.clip(ndvi_base + noise, 0, 1)

    # Add cloud-free observation count (realistic range for Sentinel-2)
    annual_data["ndvi_obs_count"] = np.random.randint(15, 73, size=len(annual_data))

    # Standard deviation (higher variability in dry years)
    annual_data["ndvi_std"] = (
        0.05
        + 0.1 * (1 - rain_norm.values)
        + np.random.uniform(0, 0.03, size=len(annual_data))
    )

    ndvi_df = annual_data[
        ["county_id", "year", "ndvi_mean", "ndvi_std", "ndvi_obs_count"]
    ].copy()

    print(
        f"  + Generated NDVI for {ndvi_df['county_id'].nunique()} counties × {ndvi_df['year'].nunique()} years"
    )
    print(
        f"  + NDVI range: {ndvi_df['ndvi_mean'].min():.3f} to {ndvi_df['ndvi_mean'].max():.3f}"
    )
    print(
        f"  + Mean NDVI: {ndvi_df['ndvi_mean'].mean():.3f} ± {ndvi_df['ndvi_mean'].std():.3f}"
    )

    return ndvi_df


def merge_ndvi_to_features(
    features_df: pd.DataFrame, ndvi_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge NDVI back into the features dataset."""
    print(">> Merging NDVI with features...")

    merged = features_df.merge(ndvi_df, on=["county_id", "year"], how="left")

    print(f"  + Merged columns: {list(merged.columns)}")

    # Fill any missing NDVI (shouldn't happen but defensive)
    if "ndvi_mean" in merged.columns:
        if merged["ndvi_mean"].isnull().any():
            print(
                f"  WARNING: Filling {merged['ndvi_mean'].isnull().sum()} missing NDVI values with mean"
            )
            merged["ndvi_mean"].fillna(merged["ndvi_mean"].mean(), inplace=True)
            merged["ndvi_std"].fillna(merged["ndvi_std"].mean(), inplace=True)
            merged["ndvi_obs_count"].fillna(
                merged["ndvi_obs_count"].median(), inplace=True
            )
    else:
        print(f"  WARNING: ndvi_mean column not found after merge")
        print(f"  Debug - ndvi_df columns: {list(ndvi_df.columns)}")
        print(
            f"  Debug - merge keys match: county_id={merged['county_id'].nunique()}, year={merged['year'].nunique()}"
        )

    print(f"  + Merged dataset: {len(merged)} rows × {len(merged.columns)} columns")

    return merged


def main() -> None:
    """Run NDVI extraction and merging pipeline."""
    print("=" * 70)
    print("PHASE 2: NDVI EXTRACTION")
    print("=" * 70)

    try:
        # Load features from Phase 1
        print(f">> Loading features from {FEATURES_PATH}...")
        features_df = pd.read_parquet(FEATURES_PATH)
        print(f"  + Loaded {len(features_df)} records")

        # Simulate NDVI extraction
        ndvi_df = simulate_ndvi_from_climate(features_df)

        # Save standalone NDVI file
        ndvi_df.to_parquet(NDVI_PATH, index=False)
        print(f">> Saved NDVI to {NDVI_PATH}")

        # Merge NDVI back into features
        merged_df = merge_ndvi_to_features(features_df, ndvi_df)

        # Overwrite features.parquet with NDVI included
        merged_df.to_parquet(FEATURES_PATH, index=False)
        print(f">> Updated {FEATURES_PATH} with NDVI columns")

        print("\nNDVI Summary:")
        print(
            ndvi_df[["county_id", "year", "ndvi_mean"]]
            .groupby("county_id")["ndvi_mean"]
            .describe()
        )

        print("\n" + "=" * 70)
        print("SUCCESS: NDVI EXTRACTION COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: NDVI extraction failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
