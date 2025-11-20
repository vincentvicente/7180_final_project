"""
CheckCleanData — Diagnostic script for data cleaning + merging

This script loads data using your actual DataConfig.load_data(),
and prints detailed diagnostics so you can inspect:
- Before/after cleaning row counts
- Status → Target mapping
- Missing values
- Final merged dataset info
"""

from app.data_config import load_real_data_fast, YC_DATA_FILE, CRUNCHBASE_DATA_FILE
import pandas as pd
import numpy as np
from src.data.preprocessing import run_full_preprocessing_pipeline



def check_raw_data():
    """Check raw files BEFORE cleaning"""
    print("\n================ RAW DATA OVERVIEW ================")

    print(f"\n📂 Loading raw YC from: {YC_DATA_FILE}")
    df_yc_raw = pd.read_csv(YC_DATA_FILE)
    print(f"YC raw rows: {len(df_yc_raw)}")
    print("Columns:", list(df_yc_raw.columns))

    print(f"\n📂 Loading raw Crunchbase from: {CRUNCHBASE_DATA_FILE}")
    df_cb_raw = pd.read_csv(CRUNCHBASE_DATA_FILE)
    print(f"Crunchbase raw rows: {len(df_cb_raw)}")
    print("Columns:", list(df_cb_raw.columns))

    return df_yc_raw, df_cb_raw


def check_cleaned():
    """Check cleaned & merged data AFTER DataConfig processing"""
    print("\n================ CLEANED DATA OVERVIEW ================")


    run_full_preprocessing_pipeline(
    crunchbase_path="data/raw/crunchbase_data.csv",
    yc_path="data/raw/yc_companies.csv",
    save_path="data/processed/merged_with_funding_missing.csv"
)
    df_cleaned = pd.read_csv("data/processed/final_merged.csv")
    print("\n[INFO] Cleaned DataFrame shape:", df_cleaned.shape)
    print("[INFO] Columns:", list(df_cleaned.columns))
    print("\n[INFO] Missing values per column:\n", df_cleaned.isnull().sum())
    print("\n[INFO] Data types:\n", df_cleaned.dtypes)
    print("\n[INFO] Sample rows:\n", df_cleaned.head())
    print("\n================ END CHECK =================")
    



if __name__ == "__main__":
    print("\n########################################################")
    print("#                 CHECK CLEAN DATA                     #")
    print("########################################################")

    # Step  — inspect cleaned output
    check_cleaned()