
import sys
import os
import pandas as pd
import numpy as np

# Add app directory to path to import data_config
sys.path.append('app')
from data_config import load_real_data_fast

def generate_table():
    print("Loading data...")
    try:
        df = load_real_data_fast()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Add computed features if they don't exist (simulating feature engineering)
    if 'is_major_hub' not in df.columns and 'region' in df.columns:
        major_hubs = ['San Francisco', 'New York', 'Boston', 'London']
        df['is_major_hub'] = df['region'].apply(lambda x: 1 if any(hub in str(x) for hub in major_hubs) else 0)

    if 'time_to_first_funding' not in df.columns:
        # Dummy calculation if not present, or skip
        pass

    # Define features to summarize
    features_config = [
        ('company_age', 'Age of the company (years)'),
        ('total_funding', 'Total funding amount (USD)'),
        ('funding_rounds', 'Number of funding rounds'),
        ('team_size', 'Size of the team'),
        ('industry', 'Industry sector'),
        ('region', 'Geographic region'),
        ('target', 'Success status (1=Success, 0=Failure)'),
        ('is_major_hub', 'Located in major startup hub'),
        # ('time_to_first_funding', 'Years to first funding') # Might not be in base df
    ]

    print("\n" + "="*80)
    print("### Table 1: Feature Summary After Cleaning")
    print("="*80 + "\n")
    
    print('| Feature | Description | Unique Values | Min | Max | Median |')
    print('| :--- | :--- | :---: | :---: | :---: | :---: |')

    for col, desc in features_config:
        if col in df.columns:
            unique_val = df[col].nunique()
            
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()
                median_val = df[col].median()
                
                # Formatting
                if col == 'total_funding':
                    min_str = f"${min_val:,.0f}"
                    max_str = f"${max_val:,.0f}"
                    med_str = f"${median_val:,.0f}"
                elif col in ['funding_rounds', 'team_size', 'company_age', 'target', 'is_major_hub']:
                    min_str = f"{min_val:.0f}"
                    max_str = f"{max_val:.0f}"
                    med_str = f"{median_val:.0f}"
                else:
                    min_str = f"{min_val:.2f}"
                    max_str = f"{max_val:.2f}"
                    med_str = f"{median_val:.2f}"
            else:
                min_str = "-"
                max_str = "-"
                med_str = "-"
                
            print(f'| {col} | {desc} | {unique_val} | {min_str} | {max_str} | {med_str} |')
    
    print("\n")

if __name__ == "__main__":
    generate_table()

