"""
Data loading configuration for the Streamlit App.
Loads already processed data for fast application startup.
"""

import pandas as pd
import numpy as np
import os
# The import of run_full_preprocessing_pipeline is REMOVED
# as this file should only READ processed data, not CREATE it.

# ============================================================================
# CONFIGURATION
# ============================================================================

USE_REAL_DATA = True
# Must match the output path in run_pipeline.py
PROCESSED_DATA_FILE = "data/processed/final_dataset.csv" 


# ============================================================================
# FUNCTIONS
# ============================================================================

def load_data():
    """Load data based on configuration (real processed data or sample data)"""
    if USE_REAL_DATA:
        return load_processed_data()
    else:
        return load_sample_data()


def load_processed_data():
    """Load the final processed and encoded dataset."""
    print(f"Attempting to load processed data from: {PROCESSED_DATA_FILE}")
    if not os.path.exists(PROCESSED_DATA_FILE):
        print("❌ Error: Processed data file not found.")
        print("Please run 'python run_pipeline.py' script first to generate the file.")
        # Fallback to sample data for app demonstration
        return load_sample_data() 
    
    try:
        df = pd.read_csv(PROCESSED_DATA_FILE)
        
        # Rename columns if necessary to match app.py expectations
        if 'funding_total_usd' in df.columns and 'total_funding' not in df.columns:
            df = df.rename(columns={'funding_total_usd': 'total_funding'})

        # If app.py uses columns not explicitly created/preserved (like 'funding_rounds' or 'team_size'),
        # we add placeholders here to prevent app crash if they are missing.
        if 'funding_rounds' not in df.columns:
             df['funding_rounds'] = np.random.randint(1, 5, len(df)) 
        if 'team_size' not in df.columns:
             df['team_size'] = np.random.randint(5, 20, len(df)) 
        
        print(f"✅ Successfully loaded {len(df)} rows of processed data.")
        return df

    except Exception as e:
        print(f"❌ Error loading processed data from CSV: {e}")
        return load_sample_data()


def load_sample_data():
    """Generate sample data for development or fallback use."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'company_name': [f'Company_{i}' for i in range(n_samples)],
        'company_age': np.random.randint(1, 20, n_samples),
        'total_funding': np.random.lognormal(10, 2, n_samples),
        'funding_rounds': np.random.randint(1, 10, n_samples),
        'team_size': np.random.randint(1, 50, n_samples),
        'industry': np.random.choice(['Tech', 'Healthcare', 'Finance', 'E-commerce', 'Other'], n_samples),
        'region': np.random.choice(['San Francisco', 'New York', 'Boston', 'Seattle', 'London', 'Other'], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    }
    
    return pd.DataFrame(data)