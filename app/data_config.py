"""
Data loading configuration - EDIT THIS FILE to use your real data
"""

import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================================

USE_REAL_DATA = True  # Set to True when you have real data files

# File paths for your real data (edit these)
YC_DATA_FILE = "data/raw/yc_companies.csv"
CRUNCHBASE_DATA_FILE = "data/raw/crunchbase_data.csv"

# Column mappings - map your actual column names to expected names
COLUMN_MAPPING = {
    # Expected name: Your actual column name
    'company_name': 'name',
    'year_founded': 'founded',
    'status': 'status',
    'industry': 'industry',
    'region': 'region',
    'team_size': 'team_size',
    'total_funding': 'funding_total_usd',  # from crunchbase
    'funding_rounds': 'funding_rounds',     # from crunchbase
    'tags': 'tags',
    'short_description': 'short_description',
}

# Status value mappings - map your status values to success/failure
STATUS_SUCCESS = ['Active', 'Acquired', 'Public']  # Edit based on your data
STATUS_FAILURE = ['Inactive', 'Closed', 'Dead']

# ============================================================================
# DATA LOADING FUNCTIONS - DO NOT EDIT BELOW THIS LINE
# ============================================================================

def load_data():
    """Load data based on configuration"""
    if USE_REAL_DATA:
        return load_real_data()
    else:
        return load_sample_data()


def load_real_data():
    """Load real YC/Crunchbase data"""
    try:
        # Load main dataset
        if YC_DATA_FILE.endswith('.csv'):
            df = pd.read_csv(YC_DATA_FILE)
        elif YC_DATA_FILE.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(YC_DATA_FILE)
        else:
            raise ValueError(f"Unsupported file format: {YC_DATA_FILE}")
        
        # Rename columns to standard names
        df = df.rename(columns={v: k for k, v in COLUMN_MAPPING.items() if v in df.columns})
        
        # Ensure required columns exist
        required = ['company_name', 'status']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Add default values for optional columns
        if 'company_age' not in df.columns and 'year_founded' in df.columns:
            df['company_age'] = 2024 - df['year_founded']
        
        if 'company_age' not in df.columns:
            df['company_age'] = np.random.randint(1, 20, len(df))
        
        if 'total_funding' not in df.columns:
            df['total_funding'] = np.random.lognormal(14, 2, len(df))
        
        if 'funding_rounds' not in df.columns:
            df['funding_rounds'] = np.random.randint(1, 10, len(df))
        
        if 'team_size' not in df.columns:
            df['team_size'] = np.random.randint(1, 50, len(df))
        
        if 'industry' not in df.columns:
            df['industry'] = 'Other'
        
        if 'region' not in df.columns:
            df['region'] = 'Other'
        
        # Create target variable
        df['target'] = df['status'].apply(
            lambda x: 1 if x in STATUS_SUCCESS else 0
        )
        
        print(f"Loaded real data: {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Falling back to sample data")
        return load_sample_data()


def load_sample_data():
    """Generate sample data for demonstration"""
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

