"""
Data loading configuration - SIMPLIFIED FOR FAST LOADING
"""

import pandas as pd
import numpy as np
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

USE_REAL_DATA = True
YC_DATA_FILE = "data/raw/yc_companies.csv"
CRUNCHBASE_DATA_FILE = "data/raw/crunchbase_data.csv"

COLUMN_MAPPING = {
    'company_name': 'name',
    'year_founded': 'founded',
    'status': 'status',
    'industry': 'industry',
    'region': 'region',
    'team_size': 'team_size',
    'tags': 'tags',
    'short_description': 'short_description',
}

STATUS_SUCCESS = ['Active', 'Acquired', 'Public']
STATUS_FAILURE = ['Inactive', 'Closed', 'Dead']

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_data():
    """Load data based on configuration"""
    if USE_REAL_DATA:
        return load_real_data_fast()
    else:
        return load_sample_data()


def load_real_data_fast():
    """Load and merge data quickly (optimized version)"""
    try:
        # Load YC data
        print("Loading YC dataset...")
        df_yc = pd.read_csv(YC_DATA_FILE)
        print(f"Loaded: {len(df_yc)} YC companies")
        
        # Rename columns
        df_yc = df_yc.rename(columns={v: k for k, v in COLUMN_MAPPING.items() if v in df_yc.columns})
        
        # Create company_age
        if 'year_founded' in df_yc.columns:
            df_yc['company_age'] = 2024 - df_yc['year_founded']
            df_yc['company_age'] = df_yc['company_age'].clip(lower=0)
        
        # Load Crunchbase for funding (fast matching only)
        try:
            print("Loading Crunchbase (fast mode)...")
            df_cb = pd.read_csv(CRUNCHBASE_DATA_FILE)
            
            # Normalize names for matching
            def norm(name):
                if pd.isna(name):
                    return ""
                name = str(name).lower().strip()
                for suf in [' inc', ' llc', ' ltd', ' corp', ' co', ' technologies', ' tech', ' labs', ' ai']:
                    if name.endswith(suf):
                        name = name[:-len(suf)].strip()
                name = re.sub(r'[^\w\s]', '', name)
                name = re.sub(r'\s+', ' ', name)
                return name
            
            df_yc['name_norm'] = df_yc['company_name'].apply(norm)
            df_cb['name_norm'] = df_cb['name'].apply(norm)
            
            # Prepare CB funding data
            df_cb['total_funding'] = pd.to_numeric(df_cb['funding_total_usd'], errors='coerce')
            df_cb['funding_rounds'] = pd.to_numeric(df_cb['funding_rounds'], errors='coerce')
            df_cb_clean = df_cb[['name_norm', 'total_funding', 'funding_rounds']].dropna(subset=['name_norm'])
            df_cb_clean = df_cb_clean.drop_duplicates(subset=['name_norm'], keep='first')
            
            # Fast merge
            df = pd.merge(df_yc, df_cb_clean, on='name_norm', how='left')
            df = df.drop('name_norm', axis=1)
            
            matched = df['total_funding'].notna().sum()
            print(f"Matched: {matched}/{len(df)} companies ({matched/len(df)*100:.1f}%)")
            
            # Fill missing with industry median
            for industry in df['industry'].unique():
                mask = (df['industry'] == industry) & df['total_funding'].isna()
                median_val = df.loc[df['industry'] == industry, 'total_funding'].median()
                if pd.notna(median_val):
                    df.loc[mask, 'total_funding'] = median_val
            
            # Global median for remaining
            df['total_funding'] = df['total_funding'].fillna(df['total_funding'].median())
            df['funding_rounds'] = df['funding_rounds'].fillna(1)
            
        except Exception as e:
            print(f"Crunchbase merge failed: {e}, using defaults")
            df = df_yc
            df['total_funding'] = np.random.lognormal(14, 2, len(df))
            df['funding_rounds'] = np.random.randint(1, 10, len(df))
        
        # Create target
        df['target'] = df['status'].apply(lambda x: 1 if x in STATUS_SUCCESS else 0)
        
        print(f"Final: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        print(f"Error: {e}, falling back to sample data")
        return load_sample_data()


def load_sample_data():
    """Generate sample data"""
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

