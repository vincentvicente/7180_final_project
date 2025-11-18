"""
Helper script to load and validate real YC and Crunchbase data.
Place your CSV/Excel files in data/raw/ and run this script to test data loading.
"""

import pandas as pd
import os
from src.data.data_loader import load_yc_data, load_crunchbase_data, get_data_summary
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer

def main():
    print("="*80)
    print("REAL DATA LOADING TEST")
    print("="*80)
    
    # Check for data files
    raw_data_path = "data/raw"
    
    print(f"\nChecking for data files in {raw_data_path}/...")
    
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
        print(f"Created directory: {raw_data_path}")
    
    files = [f for f in os.listdir(raw_data_path) if f.endswith(('.csv', '.xlsx', '.xls'))]
    
    if not files:
        print("\n⚠️  No data files found!")
        print("\nPlease place your data files in data/raw/ directory:")
        print("  - YC Companies Dataset: data/raw/yc_companies.csv")
        print("  - Crunchbase Dataset: data/raw/crunchbase_data.csv")
        print("\nExpected columns:")
        print("  YC Dataset: company_name, year_founded, status, industry, region, team_size, tags, short_description")
        print("  Crunchbase: company_name, total_funding, funding_rounds, investor_count, etc.")
        return
    
    print(f"\n✅ Found {len(files)} data file(s):")
    for f in files:
        print(f"  - {f}")
    
    # Try to load the first file
    print("\n" + "="*80)
    print("LOADING FIRST DATA FILE")
    print("="*80)
    
    first_file = os.path.join(raw_data_path, files[0])
    
    try:
        if first_file.endswith('.csv'):
            df = pd.read_csv(first_file)
        else:
            df = pd.read_excel(first_file)
        
        print(f"\n✅ Successfully loaded: {files[0]}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Show data summary
        print("\n" + "-"*80)
        print("COLUMN NAMES:")
        print("-"*80)
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print("\n" + "-"*80)
        print("FIRST 5 ROWS:")
        print("-"*80)
        print(df.head())
        
        print("\n" + "-"*80)
        print("DATA TYPES:")
        print("-"*80)
        print(df.dtypes)
        
        print("\n" + "-"*80)
        print("MISSING VALUES:")
        print("-"*80)
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Check for required columns
        print("\n" + "-"*80)
        print("COLUMN VALIDATION:")
        print("-"*80)
        
        required_cols = {
            'company_name': ['company_name', 'name', 'company'],
            'year_founded': ['year_founded', 'founded', 'founding_year', 'year'],
            'status': ['status', 'outcome', 'result'],
            'industry': ['industry', 'sector', 'category'],
            'region': ['region', 'location', 'city', 'country']
        }
        
        found_cols = {}
        for key, possible_names in required_cols.items():
            for col in df.columns:
                if col.lower() in [name.lower() for name in possible_names]:
                    found_cols[key] = col
                    print(f"✅ {key:15s} -> Found as '{col}'")
                    break
            if key not in found_cols:
                print(f"⚠️  {key:15s} -> NOT FOUND (looking for: {', '.join(possible_names)})")
        
        # Test preprocessing
        if len(found_cols) >= 3:
            print("\n" + "="*80)
            print("TESTING PREPROCESSING")
            print("="*80)
            
            preprocessor = DataPreprocessor(target_column=found_cols.get('status', 'status'))
            
            print("\n1. Cleaning data...")
            df_clean = preprocessor.clean_data(df)
            
            print("\n2. Handling missing values...")
            df_filled = preprocessor.handle_missing_values(df_clean)
            
            if 'status' in found_cols:
                print("\n3. Encoding target...")
                df_encoded = preprocessor.encode_target(df_filled)
                
                print("\n✅ Preprocessing successful!")
                print(f"Final shape: {df_encoded.shape}")
            
            # Test feature engineering
            if 'year_founded' in found_cols:
                print("\n" + "="*80)
                print("TESTING FEATURE ENGINEERING")
                print("="*80)
                
                engineer = FeatureEngineer(reference_year=2024)
                
                print("\n1. Creating company age...")
                df_featured = engineer.create_company_age(df_filled, found_cols['year_founded'])
                
                print("\n✅ Feature engineering successful!")
                print(f"Created features: {engineer.get_created_features()}")
        
        print("\n" + "="*80)
        print("✅ DATA LOADING TEST COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("1. Review the column mappings above")
        print("2. Update app/app.py to use load_yc_data() instead of load_sample_data()")
        print("3. Map your column names to the expected format")
        
    except Exception as e:
        print(f"\n❌ Error loading file: {e}")
        print("\nPlease check:")
        print("  - File format is correct (CSV or Excel)")
        print("  - File is not corrupted")
        print("  - File encoding is UTF-8")

if __name__ == "__main__":
    main()

