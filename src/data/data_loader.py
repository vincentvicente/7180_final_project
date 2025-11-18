"""
Data loading module for startup success prediction project.
Handles loading of Y Combinator and Crunchbase datasets.
"""

import pandas as pd
import os
from typing import Optional, Tuple


def load_yc_data(filepath: str) -> pd.DataFrame:
    """
    Load Y Combinator companies dataset.
    
    Args:
        filepath: Path to the YC dataset file
        
    Returns:
        DataFrame containing YC companies data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load data based on file extension
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    elif filepath.endswith('.json'):
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    print(f"Loaded YC data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_crunchbase_data(filepath: str) -> pd.DataFrame:
    """
    Load Crunchbase startup dataset.
    
    Args:
        filepath: Path to the Crunchbase dataset file
        
    Returns:
        DataFrame containing Crunchbase companies data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load data based on file extension
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    elif filepath.endswith('.json'):
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    print(f"Loaded Crunchbase data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def merge_datasets(yc_df: pd.DataFrame, 
                   crunchbase_df: pd.DataFrame,
                   on_column: str = 'company_name',
                   how: str = 'outer') -> pd.DataFrame:
    """
    Merge Y Combinator and Crunchbase datasets.
    
    Args:
        yc_df: Y Combinator dataset
        crunchbase_df: Crunchbase dataset
        on_column: Column name to merge on
        how: Type of merge ('left', 'right', 'outer', 'inner')
        
    Returns:
        Merged DataFrame
    """
    merged_df = pd.merge(yc_df, crunchbase_df, on=on_column, how=how, 
                         suffixes=('_yc', '_cb'))
    
    print(f"Merged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    return merged_df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    return summary

