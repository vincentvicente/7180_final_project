"""Check why YC and Crunchbase matching rate is low"""
import pandas as pd
import re

# Load data
yc = pd.read_csv('data/raw/yc_companies.csv')
cb = pd.read_csv('data/raw/crunchbase_data.csv')

print(f"YC companies: {len(yc)}")
print(f"Crunchbase companies: {len(cb)}")

# Check company name formats
print("\n" + "="*80)
print("COMPANY NAME SAMPLES")
print("="*80)
print("\nYC names (first 10):")
for name in yc['name'].head(10):
    print(f"  - {name}")

print("\nCrunchbase names (first 10):")
for name in cb['name'].head(10):
    print(f"  - {name}")

# Normalize and check overlap
def normalize_name(name):
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name

yc['name_norm'] = yc['name'].apply(normalize_name)
cb['name_norm'] = cb['name'].apply(normalize_name)

# Find overlaps
yc_set = set(yc['name_norm'])
cb_set = set(cb['name_norm'])
overlap = yc_set.intersection(cb_set)

print("\n" + "="*80)
print("MATCHING ANALYSIS")
print("="*80)
print(f"Unique YC names: {len(yc_set)}")
print(f"Unique CB names: {len(cb_set)}")
print(f"Overlap: {len(overlap)} ({len(overlap)/len(yc_set)*100:.1f}%)")

print("\n10 matched company names:")
for name in list(overlap)[:10]:
    print(f"  - {name}")

# Check why so low
print("\n" + "="*80)
print("REASONS FOR LOW MATCH RATE")
print("="*80)

# Sample non-matching YC companies
non_match_yc = yc_set - cb_set
print(f"\nNon-matching YC companies (sample 10):")
for name in list(non_match_yc)[:10]:
    print(f"  - {name}")

# Check if Crunchbase has different naming convention
print("\nPossible reasons:")
print("1. YC companies are newer (2024) - Crunchbase may not have them yet")
print("2. Different naming conventions (legal names vs common names)")
print("3. Crunchbase focuses on funded companies only")
print("4. YC dataset may use abbreviated names")

# Check YC status distribution in matched vs non-matched
yc_matched = yc[yc['name_norm'].isin(overlap)]
yc_non_matched = yc[~yc['name_norm'].isin(overlap)]

print("\n" + "="*80)
print("STATUS DISTRIBUTION")
print("="*80)
print("\nMatched companies status:")
print(yc_matched['status'].value_counts())
print("\nNon-matched companies status:")
print(yc_non_matched['status'].value_counts())

