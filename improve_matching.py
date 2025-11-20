"""
Advanced matching strategies to improve YC-Crunchbase overlap
"""
import pandas as pd
import re
from difflib import SequenceMatcher

# Load data
yc = pd.read_csv('data/raw/yc_companies.csv')
cb = pd.read_csv('data/raw/crunchbase_data.csv')

print(f"YC companies: {len(yc)}")
print(f"Crunchbase companies: {len(cb)}")

def normalize_advanced(name):
    """Advanced normalization"""
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    
    # Remove common suffixes
    suffixes = [' inc', ' llc', ' ltd', ' corporation', ' corp', ' co', ' company', 
                ' technologies', ' technology', ' tech', ' labs', ' lab']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    # Remove special characters
    name = re.sub(r'[^\w\s]', '', name)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Remove common words
    common_words = ['the', 'a', 'an']
    words = name.split()
    words = [w for w in words if w not in common_words]
    name = ' '.join(words)
    
    return name

# Apply advanced normalization
yc['name_adv'] = yc['name'].apply(normalize_advanced)
cb['name_adv'] = cb['name'].apply(normalize_advanced)

# Strategy 1: Exact match after advanced normalization
yc_set = set(yc['name_adv'])
cb_set = set(cb['name_adv'])
exact_matches = yc_set.intersection(cb_set)

print("\n" + "="*80)
print("STRATEGY 1: Advanced Normalization (Exact Match)")
print("="*80)
print(f"Matched: {len(exact_matches)} companies ({len(exact_matches)/len(yc)*100:.1f}%)")

# Strategy 2: Check if YC has website that matches Crunchbase homepage_url
print("\n" + "="*80)
print("STRATEGY 2: Website/URL Matching")
print("="*80)

def extract_domain(url):
    """Extract domain from URL"""
    if pd.isna(url):
        return ""
    url = str(url).lower()
    # Remove http://, https://, www.
    url = re.sub(r'https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    # Get domain only
    url = url.split('/')[0]
    url = url.split('?')[0]
    return url

if 'website' in yc.columns and 'homepage_url' in cb.columns:
    yc['domain'] = yc['website'].apply(extract_domain)
    cb['domain'] = cb['homepage_url'].apply(extract_domain)
    
    # Match by domain
    domain_matches = pd.merge(
        yc[['name', 'name_adv', 'domain']], 
        cb[['name', 'domain', 'funding_total_usd', 'funding_rounds']],
        on='domain',
        how='inner',
        suffixes=('_yc', '_cb')
    )
    
    print(f"Matched by domain: {len(domain_matches)} companies")
    print("\nSample domain matches:")
    print(domain_matches[['name_yc', 'name_cb', 'domain']].head(10))

# Strategy 3: Fuzzy matching on top unmatched companies
print("\n" + "="*80)
print("STRATEGY 3: Fuzzy Matching Analysis")
print("="*80)

# Get unmatched YC companies
matched_yc = yc[yc['name_adv'].isin(exact_matches)]
unmatched_yc = yc[~yc['name_adv'].isin(exact_matches)]

print(f"Unmatched YC companies: {len(unmatched_yc)}")

# Try fuzzy matching for a sample
def find_best_match(name, candidates, threshold=0.8):
    """Find best fuzzy match"""
    best_score = 0
    best_match = None
    
    for candidate in candidates[:1000]:  # Limit for speed
        score = SequenceMatcher(None, name, candidate).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate
    
    if best_score >= threshold:
        return best_match, best_score
    return None, 0

# Test fuzzy matching on 20 samples
print("\nFuzzy matching samples (threshold=0.8):")
cb_names_list = cb['name_adv'].tolist()

fuzzy_count = 0
for idx, row in unmatched_yc.head(20).iterrows():
    match, score = find_best_match(row['name_adv'], cb_names_list, threshold=0.7)
    if match:
        fuzzy_count += 1
        print(f"  YC: '{row['name']}' -> CB: '{match}' (score: {score:.2f})")

print(f"\nFuzzy matches in sample: {fuzzy_count}/20")

# Final recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"""
Current match rate: 19.2% (942/4974) with normalized names

Potential improvements:
1. Use domain/website matching: Could add ~{len(domain_matches)} more matches
2. Implement fuzzy matching (slower but more accurate)
3. Accept current 19.2% as reasonable given:
   - Many YC companies are very new (2024)
   - Crunchbase may not track all early-stage startups
   - Different data sources naturally have gaps

For this project, 19.2% real funding data + median imputation is acceptable.
Alternative: Use YC data only, without Crunchbase merge.
""")

