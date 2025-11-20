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
    

    def clean_yc_data(self, df_yc: pd.DataFrame) -> pd.DataFrame:
        # delete unnecessary columns
        columns_to_drop = [
            "active_founders", "batch", "jobs", "logo",
            "long_description", "short_description", "website", "team_size", "tags"
        ]
        df_yc.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        # handle 'founded' column: convert to company age
        current_year = 2025
        if "founded" in df_yc.columns:
            df_yc["company_age"] = current_year - pd.to_numeric(df_yc["founded"], errors="coerce")
            df_yc.drop(columns=["founded"], inplace=True, errors='ignore')

            # Combine location and region into a unified location_combined column
            df_yc["location_combined"] = df_yc["location"]
            df_yc["location_combined"] = df_yc["location_combined"].fillna(df_yc["region"])
            df_yc["location_combined"] = df_yc["location_combined"].fillna("Unknown")

            # Drop original location columns
            df_yc.drop(columns=["location", "region"], inplace=True)

        # Fill missing company_age with median
        df_yc["company_age"] = df_yc["company_age"].fillna(df_yc["company_age"].median())

        return df_yc
    
    def clean_cb_data(self, df_cb: pd.DataFrame) -> pd.DataFrame:
        # Drop unnecessary columns
        columns_to_drop = [
            "permalink", "homepage_url", "funding_rounds",
            "last_funding_at"
        ]
        df_cb.drop(columns=columns_to_drop, inplace=True, errors="ignore")

        # Combine location columns into a new column called 'location_combined'
        df_cb["location_combined"] = df_cb["city"]
        df_cb["location_combined"] = df_cb["location_combined"].fillna(df_cb["region"])
        df_cb["location_combined"] = df_cb["location_combined"].fillna(df_cb["state_code"])
        df_cb["location_combined"] = df_cb["location_combined"].fillna(df_cb["country_code"])
        df_cb["location_combined"] = df_cb["location_combined"].fillna("Unknown")

        # Drop original geographic columns
        df_cb.drop(columns=["country_code", "state_code", "region", "city"], inplace=True)

        # Handle founding date
        if "founded_at" not in df_cb.columns and "first_funding_at" not in df_cb.columns:
            return df_cb  # nothing to process

        # Fill 'founded_at' with 'first_funding_at' if missing
        if "founded_at" in df_cb.columns and "first_funding_at" in df_cb.columns:
            df_cb["founded_at"] = df_cb["founded_at"].fillna(df_cb["first_funding_at"])
        elif "founded_at" not in df_cb.columns and "first_funding_at" in df_cb.columns:
            df_cb["founded_at"] = df_cb["first_funding_at"]

        # Extract year and compute company age
        current_year = 2025
        df_cb["founded_year"] = pd.to_datetime(df_cb["founded_at"], errors="coerce").dt.year
        df_cb["company_age"] = current_year - df_cb["founded_year"]

        df_cb.drop(columns=["founded_at", "first_funding_at", "founded_year"], inplace=True, errors="ignore")

        # Fill missing values in 'category_list' with 'Other'
        if "category_list" in df_cb.columns:
            df_cb["category_list"] = df_cb["category_list"].fillna("Other")

        # Fill missing values in 'company_age' with median
        if "company_age" in df_cb.columns:
            df_cb["company_age"] = df_cb["company_age"].fillna(df_cb["company_age"].median())

        # Drop rows where 'name' is missing
        df_cb.dropna(subset=["name"], inplace=True)

        # Replace dash-like values in FoundingTotalUSD with NaN early
        if "funding_total_usd" in df_cb.columns:
            df_cb["funding_total_usd"] = (
                df_cb["funding_total_usd"]
                    .astype(str)
                    .str.strip()
                    .replace(
                        to_replace=[
                            r'^\s*[-–—−]+\s*$',   # All dash types
                            r'^\s*$',            # Empty or whitespace
                            r'None',             # Literal None
                            r'N/A'               # Literal N/A
                        ],
                        value=np.nan,
                        regex=True
                    )
            )

        return df_cb
    
    
    def fill_missing_funding_total(self, df: pd.DataFrame, cb_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing 'funding_total_usd' values in df using Crunchbase data.
        Priority:
        1. If company name matches in cb_df, use funding_total_usd from cb_df.
        2. Otherwise, fill with 'unknown'.

        Parameters:
        - df: Merged dataframe (YC + CB).
        - cb_df: Cleaned Crunchbase dataframe with funding info.

        Returns:
        - DataFrame with 'funding_total_usd' filled.
        """
        print("Filling funding_total_usd using Crunchbase where possible...")
        df['funding_total_usd'] = df.apply(
            lambda row: cb_df.loc[cb_df['name'] == row['name'], 'funding_total_usd'].values[0]
            if pd.isna(row['funding_total_usd']) and row['name'] in cb_df['name'].values
            else row['funding_total_usd'],
            axis=1
        )
        # Replace any remaining NaNs with np.nan
        df['funding_total_usd'] = df['funding_total_usd'].fillna(np.nan)
        print("Filled funding_total_usd with unknown")
        return df

    
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
    
    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode target variable.
        Supports a set of status labels mapped to binary outcomes.
        Unknown labels mapped to -1 and removed.
        
        Args:
            df: Input DataFrame   
        Returns:
            DataFrame with encoded target
        """
        df_encoded = df.copy()

        print(df[self.target_column].value_counts())
        
        label_map = {
            # YC statuses
            'active': 1,
            'inactive': 0,
            'acquired': 1,
            'public': 1,

            # Crunchbase statuses
            'operating': 1,
            'closed': 0,
            'acquired': 1,
            'ipo': 1
        }
        
        df_encoded[self.target_column] = df_encoded[self.target_column].astype(str).str.lower()
        
        if self.target_column in df_encoded.columns:
            df_encoded['target'] = df_encoded[self.target_column].apply(
                lambda x: label_map.get(x, -1)
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
    

    def merge_yc_cb(self, cleaned_yc: pd.DataFrame, cleaned_cb: pd.DataFrame) -> pd.DataFrame:
        # Rename for consistency
        cleaned_cb_renamed = cleaned_cb.rename(columns={
            "category_list": "industry",
            "location_combined": "region",
            "funding_total_usd": "FoundingTotalUSD"
        })
        cleaned_yc_renamed = cleaned_yc.rename(columns={
            "location_combined": "region"
        })

        # Merge on 'name'
        merged_df = pd.merge(
            cleaned_yc_renamed,
            cleaned_cb_renamed[["name", "industry", "FoundingTotalUSD", "status", "company_age", "region"]],
            on="name",
            how="outer",
            suffixes=("_yc", "_cb")
        )

        merged_df["funding_total_usd"] = (
            merged_df["FoundingTotalUSD"]
                .astype(str)
                .str.strip()
                .replace(
                    to_replace=[
                        r'^\s*[-–—−]+\s*$',   # All dash types
                        r'^\s*$',            # Empty strings
                        r'None',             # Literal "None"
                        r'N/A',              # Literal "N/A"
                    ],
                    value=np.nan,
                    regex=True
                )
                .str.replace(r'[\$,]', '', regex=True)  # Remove $ and commas only
        )
        merged_df["funding_total_usd"] = pd.to_numeric(merged_df["funding_total_usd"], errors="coerce")

        # Add indicator column for funding availability
        merged_df["has_funding_data"] = merged_df["funding_total_usd"].notna().astype(int)

        merged_df["industry"] = merged_df["industry_yc"].combine_first(merged_df["industry_cb"])
        merged_df["status"] = merged_df["status_cb"].combine_first(merged_df["status_yc"])
        merged_df["company_age"] = merged_df["company_age_yc"].combine_first(merged_df["company_age_cb"])
        merged_df["region"] = merged_df["region_yc"].combine_first(merged_df["region_cb"])
        # Convert funding_total_usd to numeric
        merged_df["funding_total_usd"] = pd.to_numeric(merged_df["funding_total_usd"], errors="coerce")

        # Final selection
        final_df = merged_df[[
            "name", "industry", "funding_total_usd", "status", "company_age", "region"
        ]]
        return final_df
