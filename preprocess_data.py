"""
Data preprocessing script - Process once and save to speed up app startup
"""

import sys
sys.path.append('app')
from data_config import load_real_data_fast
import pandas as pd
import pickle
import os

print("=" * 80)
print("Starting data preprocessing...")
print("=" * 80)

# Load and process data
df = load_real_data_fast()

# Save in multiple formats
os.makedirs('data/processed', exist_ok=True)

# 1. CSV format (for easy inspection)
csv_path = 'data/processed/processed_data.csv'
df.to_csv(csv_path, index=False)
print(f"✅ Saved CSV: {csv_path}")

# 2. Pickle format (fastest loading)
pickle_path = 'data/processed/processed_data.pkl'
df.to_pickle(pickle_path)
print(f"✅ Saved Pickle: {pickle_path}")

# 3. Parquet format (compressed + fast)
parquet_path = 'data/processed/processed_data.parquet'
df.to_parquet(parquet_path, index=False)
print(f"✅ Saved Parquet: {parquet_path}")

print("\n" + "=" * 80)
print(f"Preprocessing complete! Total {len(df)} rows")
print("App startup will now be 10-20x faster")
print("=" * 80)

