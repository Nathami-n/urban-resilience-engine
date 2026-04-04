"""
Phase 1: ETL Pipeline for Urban Resilience Engine
Integrates climate, soil, OSM infrastructure, and population data for Kenyan counties.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import json

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"


def fetch_county_boundaries() -> pd.DataFrame:
    """Create Kenya county data with simplified boundaries."""
    print("→ Creating Kenya county boundaries...")
    
    # Target counties: Nairobi, Nakuru, Kisumu, Uasin Gishu (Eldoret)
    counties = {
        'Nairobi': {'lat': -1.286389, 'lon': 36.817223, 'area_km2': 696},
        'Nakuru': {'lat': -0.303099, 'lon': 36.080026, 'area_km2': 7496},
        'Kisumu': {'lat': -0.091702, 'lon': 34.767956, 'area_km2': 2096},
        'Uasin Gishu': {'lat': 0.520556, 'lon': 35.269722, 'area_km2': 3345},
    }
    
    data = []
    for county_name, info in counties.items():
        lon, lat, area = info['lon'], info['lat'], info['area_km2']
        # Approximate bounding box
        delta = np.sqrt(area) / 111 / 2
        
        data.append({
            'county_id': county_name.lower().replace(' ', '_'),
            'county_name': county_name,
            'centroid_lat': lat,
            'centroid_lon': lon,
            'area_km2': area,
            'bbox_south': lat - delta,
            'bbox_north': lat + delta,
            'bbox_west': lon - delta,
            'bbox_east': lon + delta
        })
    
    df = pd.DataFrame(data)
    
    # Save to raw
    osm_dir = RAW_DIR / "osm"
    osm_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(osm_dir / "counties.csv", index=False)
    print(f"  ✓ Created {len(df)} county records → {osm_dir}/counties.csv")
    
    return df


def generate_climate_data(counties_df: pd.DataFrame) -> pd.DataFrame:
    """Generate realistic climate time series for 2013-2023."""
    print("→ Generating climate data (rainfall, temperature)...")
    
    # Time range: 2013-2023, monthly
    start_date = datetime(2013, 1, 1)
    end_date = datetime(2023, 12, 31)
    months = pd.date_range(start_date, end_date, freq='MS')
    
    records = []
    np.random.seed(42)
    
    for _, county in counties_df.iterrows():
        county_id = county['county_id']
        lat = county['centroid_lat']
        
        # Rainfall: bimodal pattern typical of Kenya (long rains Mar-May, short rains Oct-Dec)
        # Higher in western counties (Kisumu, Nakuru), lower in Nairobi
        base_rainfall = 80 if 'kisumu' in county_id or 'nakuru' in county_id else 50
        
        for month in months:
            month_num = month.month
            
            # Seasonal rainfall pattern
            if month_num in [3, 4, 5]:  # Long rains
                rainfall = base_rainfall * np.random.uniform(1.5, 2.5)
            elif month_num in [10, 11, 12]:  # Short rains
                rainfall = base_rainfall * np.random.uniform(1.2, 1.8)
            else:  # Dry seasons
                rainfall = base_rainfall * np.random.uniform(0.3, 0.7)
            
            # Temperature: inversely related to latitude (higher temp near equator)
            # Nairobi (higher altitude) is cooler
            base_temp = 20 if county_id == 'nairobi' else 24
            temp_max = base_temp + np.random.normal(8, 2)
            temp_min = base_temp + np.random.normal(-2, 1.5)
            
            records.append({
                'county_id': county_id,
                'year': month.year,
                'month': month.month,
                'year_month': month.strftime('%Y-%m'),
                'rainfall_mm': max(0, rainfall + np.random.normal(0, 15)),
                'temp_max_c': temp_max,
                'temp_min_c': temp_min,
            })
    
    climate_df = pd.DataFrame(records)
    
    # Compute rainfall anomaly (deviation from county 10-year mean)
    county_means = climate_df.groupby('county_id')['rainfall_mm'].transform('mean')
    climate_df['rainfall_anomaly'] = climate_df['rainfall_mm'] - county_means
    
    # Save to raw
    chirps_dir = RAW_DIR / "chirps"
    chirps_dir.mkdir(parents=True, exist_ok=True)
    climate_df.to_csv(chirps_dir / "climate_monthly.csv", index=False)
    print(f"  ✓ Generated {len(climate_df)} climate records → {chirps_dir}/climate_monthly.csv")
    
    return climate_df


def generate_soil_data(counties_df: pd.DataFrame) -> pd.DataFrame:
    """Generate soil organic carbon data."""
    print("→ Generating soil data...")
    
    np.random.seed(43)
    soil_records = []
    
    for _, county in counties_df.iterrows():
        # SOC typically higher in western highlands (Kisumu, Nakuru, Uasin Gishu)
        base_soc = 1.8 if county['county_id'] in ['kisumu', 'nakuru', 'uasin_gishu'] else 1.2
        
        for year in range(2013, 2024):
            # Slight decline over time due to land degradation
            soc = base_soc * (1 - 0.01 * (year - 2013)) + np.random.normal(0, 0.15)
            
            soil_records.append({
                'county_id': county['county_id'],
                'year': year,
                'soil_organic_carbon': max(0.5, soc)  # g/kg
            })
    
    soil_df = pd.DataFrame(soil_records)
    
    soil_dir = RAW_DIR / "soil"
    soil_dir.mkdir(parents=True, exist_ok=True)
    soil_df.to_csv(soil_dir / "soil_annual.csv", index=False)
    print(f"  ✓ Generated {len(soil_df)} soil records → {soil_dir}/soil_annual.csv")
    
    return soil_df


def generate_infrastructure_data(counties_df: pd.DataFrame) -> pd.DataFrame:
    """Generate OSM-based infrastructure metrics."""
    print("→ Generating infrastructure data (road density)...")
    
    np.random.seed(44)
    infra_records = []
    
    for _, county in counties_df.iterrows():
        # Road density: higher in Nairobi (urban), lower in rural counties
        if county['county_id'] == 'nairobi':
            road_km = np.random.uniform(800, 1000)
        elif county['county_id'] == 'uasin_gishu':
            road_km = np.random.uniform(400, 500)
        else:
            road_km = np.random.uniform(300, 600)
        
        road_density = road_km / county['area_km2']  # km/km²
        
        # Population density proxy (higher in Nairobi)
        if county['county_id'] == 'nairobi':
            pop_density = np.random.uniform(4500, 5000)  # people/km²
        elif county['county_id'] == 'nakuru':
            pop_density = np.random.uniform(150, 200)
        else:
            pop_density = np.random.uniform(100, 180)
        
        infra_records.append({
            'county_id': county['county_id'],
            'road_density_km_per_km2': road_density,
            'population_density': pop_density,
            'building_footprint_density': pop_density / 100  # proxy
        })
    
    infra_df = pd.DataFrame(infra_records)
    
    osm_dir = RAW_DIR / "osm"
    infra_df.to_csv(osm_dir / "infrastructure.csv", index=False)
    print(f"  ✓ Generated infrastructure metrics → {osm_dir}/infrastructure.csv")
    
    return infra_df


def merge_and_engineer_features(climate_df: pd.DataFrame, soil_df: pd.DataFrame, 
                                 infra_df: pd.DataFrame) -> pd.DataFrame:
    """Merge all datasets and compute composite risk index."""
    print("→ Merging datasets and engineering features...")
    
    # Merge climate with soil (by county and year)
    merged = climate_df.copy()
    merged = merged.merge(soil_df, on=['county_id', 'year'], how='left')
    
    # Merge with infrastructure (by county only - static data)
    merged = merged.merge(infra_df, on='county_id', how='left')
    
    # Normalize features for composite risk index (manual min-max scaling)
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-10)
    
    risk_features = ['rainfall_anomaly', 'temp_max_c', 'soil_organic_carbon', 
                     'road_density_km_per_km2', 'population_density']
    
    # Fill any missing values
    merged[risk_features] = merged[risk_features].fillna(merged[risk_features].mean())
    
    # Compute composite risk index (normalized weighted average)
    # High risk = high temp + low SOC + high rainfall anomaly (drought or flood)
    merged['rainfall_anomaly_abs'] = merged['rainfall_anomaly'].abs()
    
    rain_risk = normalize(merged['rainfall_anomaly_abs'])
    temp_risk = normalize(merged['temp_max_c'])
    soc_risk = normalize(1 / (merged['soil_organic_carbon'] + 0.1))
    
    # Composite risk: weighted average
    merged['risk_index'] = 0.4 * rain_risk + 0.35 * temp_risk + 0.25 * soc_risk
    
    # Binary target: high_risk = 1 if risk_index > 75th percentile
    threshold = merged['risk_index'].quantile(0.75)
    merged['high_risk'] = (merged['risk_index'] > threshold).astype(int)
    
    print(f"  ✓ Merged {len(merged)} records")
    print(f"  ✓ Risk threshold (75th percentile): {threshold:.3f}")
    print(f"  ✓ High risk instances: {merged['high_risk'].sum()} ({merged['high_risk'].mean()*100:.1f}%)")
    
    return merged


def export_features(df: pd.DataFrame) -> None:
    """Export final feature dataset to parquet."""
    print(f"→ Exporting features to {FEATURES_PATH}...")
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False, engine='pyarrow')
    
    print(f"  ✓ Exported {len(df)} rows × {len(df.columns)} columns")
    print(f"  ✓ Schema: {dict(df.dtypes)}")
    print(f"  ✓ Null counts: {dict(df.isnull().sum())}")
    
    # Summary stats
    print("\n📊 Feature Summary:")
    print(df[['county_id', 'year', 'rainfall_mm', 'temp_max_c', 
              'soil_organic_carbon', 'risk_index', 'high_risk']].describe())


def main() -> None:
    """Run complete ETL pipeline."""
    print("="*70)
    print("PHASE 1: ETL PIPELINE")
    print("="*70)
    
    try:
        # Step 1: Create county data
        counties_df = fetch_county_boundaries()
        
        # Step 2: Generate climate data
        climate_df = generate_climate_data(counties_df)
        
        # Step 3: Generate soil data
        soil_df = generate_soil_data(counties_df)
        
        # Step 4: Generate infrastructure data
        infra_df = generate_infrastructure_data(counties_df)
        
        # Step 5: Merge and engineer features
        features_df = merge_and_engineer_features(climate_df, soil_df, infra_df)
        
        # Step 6: Export to parquet
        export_features(features_df)
        
        print("\n" + "="*70)
        print("✅ ETL PIPELINE COMPLETE")
        print("="*70)
        print(f"Output: {FEATURES_PATH}")
        
    except Exception as e:
        print(f"\n❌ ETL failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
