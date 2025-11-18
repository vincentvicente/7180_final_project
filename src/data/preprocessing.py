"""
Data preprocessing module for startup success prediction.
Handles data cleaning, missing value imputation, and encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing import List, Optional, Dict, Any


class DataPreprocessor:
    """
    Data preprocessing class for handling missing values,
    encoding categorical variables, and data cleaning.
    """
    
    def __init__(self, target_column: str = 'status'):
        """
        Initialize the DataPreprocessor.
        
        Args:
            target_column: Name of the target column
        """
        self.target_column = target_column
        self.label_encoders = {}
        self.feature_names = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by removing duplicates and handling basic issues.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove duplicate rows
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Remove rows where target is missing
        if self.target_column in df_clean.columns:
            df_clean = df_clean.dropna(subset=[self.target_column])
            print(f"Dataset after removing missing targets: {len(df_clean)} rows")
        
        return df_clean
    
    def handle_missing_values(self, 
                             df: pd.DataFrame,
                             numeric_strategy: str = 'median',
                             categorical_strategy: str = 'unknown') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            numeric_strategy: Strategy for numeric columns ('mean', 'median', 'drop')
            categorical_strategy: Strategy for categorical columns ('mode', 'unknown', 'drop')
            
        Returns:
            DataFrame with handled missing values
        """
        df_filled = df.copy()
        
        # Handle numeric columns
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_filled[col].isnull().any():
                if numeric_strategy == 'mean':
                    df_filled[col].fillna(df_filled[col].mean(), inplace=True)
                elif numeric_strategy == 'median':
                    df_filled[col].fillna(df_filled[col].median(), inplace=True)
                print(f"Filled {col} with {numeric_strategy}")
        
        # Handle categorical columns
        categorical_cols = df_filled.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != self.target_column and df_filled[col].isnull().any():
                if categorical_strategy == 'mode':
                    mode_value = df_filled[col].mode()[0] if len(df_filled[col].mode()) > 0 else 'unknown'
                    df_filled[col].fillna(mode_value, inplace=True)
                elif categorical_strategy == 'unknown':
                    df_filled[col].fillna('unknown', inplace=True)
                print(f"Filled {col} with {categorical_strategy}")
        
        return df_filled
    
    def merge_rare_categories(self, 
                             df: pd.DataFrame,
                             column: str,
                             threshold: float = 0.02,
                             new_category: str = 'Other') -> pd.DataFrame:
        """
        Merge rare categories in a categorical column.
        
        Args:
            df: Input DataFrame
            column: Column name to process
            threshold: Minimum proportion threshold for keeping a category
            new_category: Name for merged rare categories
            
        Returns:
            DataFrame with merged rare categories
        """
        df_merged = df.copy()
        
        if column in df_merged.columns:
            # Calculate category proportions
            value_counts = df_merged[column].value_counts(normalize=True)
            rare_categories = value_counts[value_counts < threshold].index
            
            # Merge rare categories
            df_merged[column] = df_merged[column].apply(
                lambda x: new_category if x in rare_categories else x
            )
            
            print(f"Merged {len(rare_categories)} rare categories in {column}")
        
        return df_merged
    
    def encode_target(self, df: pd.DataFrame, 
                     success_labels: List[str] = ['Acquired', 'Public'],
                     failure_labels: List[str] = ['Inactive', 'Closed']) -> pd.DataFrame:
        """
        Encode target variable.
        Success (Acquired/Public) = 1, Failure (Inactive/Closed) = 0
        
        Args:
            df: Input DataFrame
            success_labels: List of labels indicating success
            failure_labels: List of labels indicating failure
            
        Returns:
            DataFrame with encoded target
        """
        df_encoded = df.copy()
        
        if self.target_column in df_encoded.columns:
            df_encoded['target'] = df_encoded[self.target_column].apply(
                lambda x: 1 if x in success_labels else (0 if x in failure_labels else -1)
            )
            
            # Remove rows with unknown labels (-1)
            unknown_count = (df_encoded['target'] == -1).sum()
            if unknown_count > 0:
                print(f"Removing {unknown_count} rows with unknown target labels")
                df_encoded = df_encoded[df_encoded['target'] != -1]
            
            print(f"Target distribution:\n{df_encoded['target'].value_counts()}")
            print(f"Target proportions:\n{df_encoded['target'].value_counts(normalize=True)}")
        
        return df_encoded
    
    def one_hot_encode(self, 
                      df: pd.DataFrame,
                      columns: List[str],
                      drop_first: bool = False) -> pd.DataFrame:
        """
        One-hot encode categorical columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to encode
            drop_first: Whether to drop first category to avoid multicollinearity
            
        Returns:
            DataFrame with one-hot encoded columns
        """
        df_encoded = df.copy()
        
        for col in columns:
            if col in df_encoded.columns:
                # Create one-hot encoded columns
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=drop_first)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(col, axis=1)
                print(f"One-hot encoded {col}: {len(dummies.columns)} new columns")
        
        return df_encoded
    
    def get_feature_importance_ready_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for model training by separating features and target.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        if 'target' not in df.columns:
            raise ValueError("Target column not found. Please encode target first.")
        
        # Separate features and target
        X = df.drop(['target', self.target_column], axis=1, errors='ignore')
        y = df['target']
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        
        return X, y

