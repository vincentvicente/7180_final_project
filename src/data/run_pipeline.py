"""
Data Pipeline Execution Script.
Loads raw data, executes preprocessing steps (cleaning, merging, encoding),
and saves the final dataset for model training and application use.
The script now includes a missing value report printout before saving.

USAGE: python run_pipeline.py
"""
import pandas as pd
import os
import sys
import numpy as np # 需要 np 来做缺失值填充 (虽然目前只做报告)

# Assume preprocessing.py is accessible in the Python path
from preprocessing import DataPreprocessor 


def generate_missing_value_report(df: pd.DataFrame) -> None:
    """Calculates and prints the missing value report for the DataFrame."""
    missing_data = df.isnull().sum()
    total_rows = len(df)
    
    # Create a DataFrame for count and percentage
    missing_info = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percent': (missing_data / total_rows) * 100
    })
    
    # Filter out columns with 0 missing values and sort
    missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(
        by='Missing Count', ascending=False
    )

    print("\n--- 📊 Final Dataset Missing Value Report (Only Columns with NaN) ---")
    
    if missing_info.empty:
        print("🎉 Congratulations! No missing values found in the final dataset.")
    else:
        # Format the percentage column
        missing_info['Missing Percent'] = missing_info['Missing Percent'].map('{:.2f}%'.format)
        print(missing_info)
        
        # Highlight the funding column specifically
        if 'funding_total_usd' in missing_info.index:
            funding_nan_count = missing_info.loc['funding_total_usd', 'Missing Count']
            funding_nan_percent = missing_info.loc['funding_total_usd', 'Missing Percent']
            print(f"\n📢 Key Finding: 'funding_total_usd' has {funding_nan_count} missing values ({funding_nan_percent}).")
            print("💡 Reminder: You must handle these remaining NaNs in your model training script (e.g., fill with 0 or mean).")

def execute_data_pipeline(crunchbase_path: str, yc_path: str, save_path: str) -> None:
    """
    Executes the full data processing pipeline, including a missing value report printout.
    """
    print("--- 1. Initializing and Loading Raw Data ---")
    preprocessor = DataPreprocessor(target_column='status')

    try:
        df_cb_raw = pd.read_csv(crunchbase_path)
        df_yc_raw = pd.read_csv(yc_path)
    except FileNotFoundError as e:
        print(f"Error: Raw data file not found: {e}")
        return

    # --- 2. Cleaning Stage ---
    print("\n--- 2. Cleaning Separate Datasets ---")
    df_yc_cleaned = preprocessor.clean_yc_data(df_yc_raw.copy())
    df_cb_cleaned = preprocessor.clean_cb_data(df_cb_raw.copy())

    # --- 3. Merging and Imputation (No 'source' created) ---
    print("\n--- 3. Merging YC and Crunchbase Data & Imputing Funding ---")
    df_merged = preprocessor.merge_yc_cb(df_yc_cleaned, df_cb_cleaned)
    df_merged_filled = preprocessor.fill_missing_funding_total(df_merged.copy(), df_cb_cleaned)
    df_cleaned = preprocessor.clean_data(df_merged_filled)
    
    # --- 4. Feature Engineering ---
    print("\n--- 4. Feature Engineering: Handling Rare Categories ---")
    df_rare_merged = preprocessor.merge_rare_categories(df_cleaned, 
                                                        column='industry', 
                                                        threshold=0.01,
                                                        new_category='Other_Industry')
    df_rare_merged = preprocessor.merge_rare_categories(df_rare_merged, 
                                                        column='region', 
                                                        threshold=0.02,
                                                        new_category='Other_Region')
    
    # --- 5. Encoding Stage ---
    print("\n--- 5. Encoding Target and Features ---")
    
    # Encode target variable 'status' -> 'target'
    df_encoded = preprocessor.encode_target(df_rare_merged)
    
    # Performing One-Hot Encoding for categorical features 'industry' and 'region'
    df_final = preprocessor.one_hot_encode(df_encoded, 
                                        columns=['industry', 'region'],
                                        drop_first=True) 

    # --- 5.5. Missing Value Report ---
    generate_missing_value_report(df_final)

    # --- 6. Save Results ---
    print(f"\n--- 6. Saving Final Preprocessed Dataset ({len(df_final)} rows) ---")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_final.to_csv(save_path, index=False)
    print(f"✅ Preprocessed dataset successfully saved to {save_path}")


if __name__ == "__main__":
    CRUNCHBASE_FILE_PATH = 'data/raw/crunchbase_data.csv' 
    YC_FILE_PATH = 'data/raw/yc_companies.csv'
    SAVE_FILE_PATH = 'data/processed/final_dataset.csv'

    print("Starting full data preprocessing pipeline...")
    execute_data_pipeline(CRUNCHBASE_FILE_PATH, YC_FILE_PATH, SAVE_FILE_PATH)
    print("\nPipeline execution complete. The final dataset is ready for model training and app use.")