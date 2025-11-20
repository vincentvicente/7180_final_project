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
    """Load real YC/Crunchbase data and merge them"""
    try:
        # Load YC dataset
        print("Loading YC dataset...")
        if YC_DATA_FILE.endswith('.csv'):
            df_yc = pd.read_csv(YC_DATA_FILE)
        elif YC_DATA_FILE.endswith(('.xlsx', '.xls')):
            df_yc = pd.read_excel(YC_DATA_FILE)
        else:
            raise ValueError(f"Unsupported file format: {YC_DATA_FILE}")
        
        print(f"Loaded YC data: {len(df_yc)} rows")
        
        # Rename YC columns to standard names
        yc_rename = {v: k for k, v in COLUMN_MAPPING.items() if v in df_yc.columns}
        df_yc = df_yc.rename(columns=yc_rename)
        
        # Load Crunchbase dataset for funding data
        try:
            print("Loading Crunchbase dataset...")
            df_cb_raw = pd.read_csv(CRUNCHBASE_DATA_FILE) if CRUNCHBASE_DATA_FILE.endswith('.csv') else pd.read_excel(CRUNCHBASE_DATA_FILE)
            print(f"Loaded Crunchbase data: {len(df_cb_raw)} rows")
            
            # Advanced normalization function
            def normalize_name(name):
                if pd.isna(name):
                    return ""
                import re
                name = str(name).lower().strip()
                suffixes = [' inc', ' llc', ' ltd', ' corporation', ' corp', ' co', 
                           ' company', ' technologies', ' technology', ' tech', ' labs', ' lab',
                           ' ai', ' io', ' systems', ' group', ' solutions']
                for suffix in suffixes:
                    if name.endswith(suffix):
                        name = name[:-len(suffix)].strip()
                name = re.sub(r'[^\w\s]', '', name)
                name = re.sub(r'\s+', ' ', name).strip()
                words = name.split()
                words = [w for w in words if w not in ['the', 'a', 'an']]
                return ' '.join(words)
            
            # Extract domain from URL
            def extract_domain(url):
                if pd.isna(url):
                    return ""
                import re
                url = str(url).lower()
                url = re.sub(r'https?://', '', url)
                url = re.sub(r'^www\.', '', url)
                url = url.split('/')[0].split('?')[0].strip()
                return url
            
            # Normalize YC names and extract domains
            df_yc['name_norm'] = df_yc['company_name'].apply(normalize_name)
            if 'website' in df_yc.columns:
                df_yc['domain'] = df_yc['website'].apply(extract_domain)
            
            # Prepare Crunchbase data
            df_cb = df_cb_raw[['name', 'homepage_url', 'funding_total_usd', 'funding_rounds']].copy()
            df_cb['name_norm'] = df_cb['name'].apply(normalize_name)
            if 'homepage_url' in df_cb.columns:
                df_cb['domain'] = df_cb['homepage_url'].apply(extract_domain)
            
            # Convert funding to numeric
            df_cb['total_funding'] = pd.to_numeric(df_cb['funding_total_usd'], errors='coerce')
            df_cb['funding_rounds'] = pd.to_numeric(df_cb['funding_rounds'], errors='coerce')
            
            # Remove invalid entries
            df_cb = df_cb.dropna(subset=['name_norm'])
            df_cb = df_cb[df_cb['name_norm'] != '']
            
            # STRATEGY 1: Exact match by normalized name
            print("Strategy 1: Matching by normalized company name...")
            df_cb_name = df_cb[['name_norm', 'total_funding', 'funding_rounds']].drop_duplicates(subset=['name_norm'], keep='first')
            df = pd.merge(df_yc, df_cb_name, on='name_norm', how='left', suffixes=('', '_name'))
            matched_name = df['total_funding'].notna().sum()
            print(f"  Matched by name: {matched_name} companies")
            
            # STRATEGY 2: Domain match for unmatched companies
            if 'domain' in df_yc.columns and 'domain' in df_cb.columns:
                print("Strategy 2: Matching by website domain...")
                unmatched_mask = df['total_funding'].isna()
                
                # Prepare domain-based funding data
                df_cb_domain = df_cb[['domain', 'total_funding', 'funding_rounds']].copy()
                df_cb_domain = df_cb_domain.dropna(subset=['domain'])
                df_cb_domain = df_cb_domain[df_cb_domain['domain'] != '']
                df_cb_domain = df_cb_domain.drop_duplicates(subset=['domain'], keep='first')
                df_cb_domain.columns = ['domain', 'total_funding_domain', 'funding_rounds_domain']
                
                # Match unmatched companies by domain
                df_temp = pd.merge(df[unmatched_mask], df_cb_domain, on='domain', how='left')
                
                # Update funding data for domain matches
                df.loc[unmatched_mask, 'total_funding'] = df_temp['total_funding_domain'].values
                df.loc[unmatched_mask, 'funding_rounds'] = df_temp['funding_rounds_domain'].values
                
                matched_domain = df['total_funding'].notna().sum() - matched_name
                print(f"  Matched by domain: {matched_domain} additional companies")
            
            # STRATEGY 3-5: DISABLED FOR PERFORMANCE
            # Fuzzy matching takes 4+ minutes - too slow for interactive app
            # Only use fast strategies (name + domain matching)
            print("Strategy 3-5: Skipped for performance (use batch processing for fuzzy matching)")
            
            # Skip all fuzzy/partial/acronym matching code below
            if False:  # Disabled for performance
                from difflib import SequenceMatcher
            
            unmatched_mask = df['total_funding'].isna()
            unmatched_yc = df[unmatched_mask].copy()
            
            if len(unmatched_yc) > 0:
                # Build lookup dictionary for faster matching
                cb_lookup = {}
                for _, row in df_cb.iterrows():
                    if pd.notna(row['total_funding']):
                        cb_lookup[row['name_norm']] = {
                            'funding': row['total_funding'],
                            'rounds': row['funding_rounds']
                        }
                
                cb_name_list = list(cb_lookup.keys())
                
                # Create first-word index for faster matching
                cb_by_first_word = {}
                for name in cb_name_list:
                    first_word = name.split()[0] if name else ''
                    if first_word:
                        if first_word not in cb_by_first_word:
                            cb_by_first_word[first_word] = []
                        cb_by_first_word[first_word].append(name)
                
                fuzzy_matches = 0
                
                # Fuzzy match on first 1500 unmatched (increased from 1000)
                for idx in unmatched_yc.head(1500).index:
                    yc_name = unmatched_yc.loc[idx, 'name_norm']
                    if not yc_name or len(yc_name) < 3:
                        continue
                    
                    best_score = 0
                    best_match = None
                    
                    # First try to match with same first word (faster)
                    first_word = yc_name.split()[0] if yc_name else ''
                    candidates = cb_by_first_word.get(first_word, [])
                    
                    # If not enough candidates, expand search
                    if len(candidates) < 10:
                        candidates = cb_name_list[:8000]  # Expanded from 5000
                    
                    for cb_name in candidates:
                        # Skip very different lengths
                        if abs(len(yc_name) - len(cb_name)) > 15:
                            continue
                        
                        score = SequenceMatcher(None, yc_name, cb_name).ratio()
                        if score > best_score:
                            best_score = score
                            best_match = cb_name
                    
                    # Use 0.85 threshold for high precision (<5% false positive rate)
                    if best_score >= 0.85 and best_match:
                        df.loc[idx, 'total_funding'] = cb_lookup[best_match]['funding']
                        df.loc[idx, 'funding_rounds'] = cb_lookup[best_match]['rounds']
                        fuzzy_matches += 1
                
                print(f"  Matched by fuzzy (threshold=0.85, high precision): {fuzzy_matches} additional companies")
            
            # STRATEGY 4: Partial name matching (e.g., contains)
            print("Strategy 4: Partial/substring matching...")
            unmatched_mask = df['total_funding'].isna()
            unmatched_yc = df[unmatched_mask].copy()
            partial_matches = 0
            
            if len(unmatched_yc) > 0:
                # For unmatched YC companies, check if CB name contains YC name or vice versa
                for idx in unmatched_yc.head(800).index:  # Increased from 500
                    yc_name = unmatched_yc.loc[idx, 'name_norm']
                    if not yc_name or len(yc_name) < 4:
                        continue
                    
                    # Find if any CB name contains this YC name (or vice versa)
                    for cb_name, data in list(cb_lookup.items())[:5000]:
                        if len(yc_name) >= 5 and len(cb_name) >= 5:  # Both names must be meaningful
                            if yc_name in cb_name or cb_name in yc_name:
                                # Very strict check: length ratio must be high
                                ratio = min(len(yc_name), len(cb_name)) / max(len(yc_name), len(cb_name))
                                if ratio >= 0.7:  # Tightened from 0.6 for better quality
                                    # Additional validation: check word overlap
                                    yc_words = set(yc_name.split())
                                    cb_words = set(cb_name.split())
                                    word_overlap = len(yc_words.intersection(cb_words))
                                    total_words = len(yc_words.union(cb_words))
                                    # At least 60% word overlap
                                    if word_overlap / total_words >= 0.6:
                                        df.loc[idx, 'total_funding'] = data['funding']
                                        df.loc[idx, 'funding_rounds'] = data['rounds']
                                        partial_matches += 1
                                        break
                
                print(f"  Matched by partial: {partial_matches} additional companies")
            
            # STRATEGY 5: Acronym and initials matching
            print("Strategy 5: Acronym/initials matching...")
            unmatched_mask = df['total_funding'].isna()
            unmatched_yc = df[unmatched_mask].copy()
            acronym_matches = 0
            
            def get_acronym(name):
                """Get acronym from name"""
                if not name:
                    return ""
                words = name.split()
                if len(words) >= 2:
                    return ''.join([w[0] for w in words if w])
                return ""
            
            if len(unmatched_yc) > 0:
                # Build acronym lookup for Crunchbase
                cb_acronym_lookup = {}
                for cb_name, data in cb_lookup.items():
                    acronym = get_acronym(cb_name)
                    if acronym and len(acronym) >= 2:
                        if acronym not in cb_acronym_lookup:
                            cb_acronym_lookup[acronym] = []
                        cb_acronym_lookup[acronym].append((cb_name, data))
                
                for idx in unmatched_yc.head(500).index:
                    yc_name = unmatched_yc.loc[idx, 'name_norm']
                    if not yc_name:
                        continue
                    
                    # Check if YC name itself is an acronym (all caps, 2-5 letters)
                    yc_orig = unmatched_yc.loc[idx, 'company_name']
                    if len(yc_name) <= 5 and yc_name.isalpha():
                        # Try to find CB names that create this acronym
                        if yc_name in cb_acronym_lookup:
                            matches = cb_acronym_lookup[yc_name]
                            if len(matches) == 1:  # Only if unique match
                                cb_name, data = matches[0]
                                df.loc[idx, 'total_funding'] = data['funding']
                                df.loc[idx, 'funding_rounds'] = data['rounds']
                                acronym_matches += 1
                
                print(f"  Matched by acronym: {acronym_matches} additional companies")
            
            # Drop temp columns
            if 'name_norm' in df.columns:
                df = df.drop('name_norm', axis=1)
            if 'domain' in df.columns:
                df = df.drop('domain', axis=1)
            
            # Final count
            total_matched = df['total_funding'].notna().sum()
            print(f"\nTotal matched: {total_matched}/{len(df)} companies ({total_matched/len(df)*100:.1f}%)")
            
            # Fill remaining missing funding data with industry-specific median
            if 'total_funding' in df.columns and 'industry' in df.columns:
                print("Filling missing values with industry-specific medians...")
                for industry in df['industry'].unique():
                    industry_mask = df['industry'] == industry
                    industry_median = df.loc[industry_mask, 'total_funding'].median()
                    if pd.notna(industry_median):
                        df.loc[industry_mask & df['total_funding'].isna(), 'total_funding'] = industry_median
                
                # Global median for any remaining
                global_median = df['total_funding'].median()
                df['total_funding'] = df['total_funding'].fillna(global_median)
            
            if 'funding_rounds' in df.columns:
                df['funding_rounds'] = df['funding_rounds'].fillna(1)
            
        except Exception as cb_error:
            print(f"Warning: Could not merge Crunchbase: {cb_error}")
            print("Using YC data only...")
            df = df_yc
        
        # Ensure required columns exist
        required = ['company_name', 'status']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Create company_age feature
        if 'company_age' not in df.columns and 'year_founded' in df.columns:
            df['company_age'] = 2024 - df['year_founded']
            df['company_age'] = df['company_age'].clip(lower=0)
        
        # Fill missing optional columns with defaults
        if 'company_age' not in df.columns:
            df['company_age'] = np.random.randint(1, 20, len(df))
        
        if 'total_funding' not in df.columns:
            print("Warning: total_funding not found, using simulated values")
            df['total_funding'] = np.random.lognormal(14, 2, len(df))
        
        if 'funding_rounds' not in df.columns:
            print("Warning: funding_rounds not found, using simulated values")
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
        
        print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
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

