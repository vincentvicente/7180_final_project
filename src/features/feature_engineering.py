"""
Feature engineering module for startup success prediction.
Implements feature creation based on instructor feedback, particularly:
- Converting "year founded" to "company age"
- Creating temporal and funding-related features
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List


class FeatureEngineer:
    """
    Feature engineering class for creating meaningful features
    from raw startup data.
    """
    
    def __init__(self, reference_year: Optional[int] = None):
        """
        Initialize the FeatureEngineer.
        
        Args:
            reference_year: Reference year for calculating company age
                          (defaults to current year)
        """
        self.reference_year = reference_year or datetime.now().year
        self.created_features = []
        
    def create_company_age(self, 
                          df: pd.DataFrame,
                          year_founded_col: str = 'year_founded') -> pd.DataFrame:
        """
        Create company age feature from year founded.
        ADDRESSES INSTRUCTOR FEEDBACK: Convert "year founded" to "company age"
        
        Args:
            df: Input DataFrame
            year_founded_col: Name of the year founded column
            
        Returns:
            DataFrame with company_age feature
        """
        df_new = df.copy()
        
        if year_founded_col in df_new.columns:
            # Calculate company age
            df_new['company_age'] = self.reference_year - df_new[year_founded_col]
            
            # Handle negative ages (future years) - set to 0
            df_new['company_age'] = df_new['company_age'].clip(lower=0)
            
            self.created_features.append('company_age')
            print(f"Created company_age feature (reference year: {self.reference_year})")
            print(f"Company age stats:\n{df_new['company_age'].describe()}")
        else:
            print(f"Warning: {year_founded_col} column not found")
        
        return df_new
    
    def create_age_categories(self, 
                             df: pd.DataFrame,
                             age_col: str = 'company_age') -> pd.DataFrame:
        """
        Create categorical age groups for analysis.
        
        Args:
            df: Input DataFrame
            age_col: Name of the age column
            
        Returns:
            DataFrame with age_category feature
        """
        df_new = df.copy()
        
        if age_col in df_new.columns:
            # Define age bins
            bins = [0, 2, 5, 10, 20, 100]
            labels = ['0-2 years', '3-5 years', '6-10 years', '11-20 years', '20+ years']
            
            df_new['age_category'] = pd.cut(df_new[age_col], 
                                           bins=bins, 
                                           labels=labels,
                                           include_lowest=True)
            
            self.created_features.append('age_category')
            print(f"Created age_category feature")
            print(f"Age category distribution:\n{df_new['age_category'].value_counts()}")
        
        return df_new
    
    def create_funding_features(self, 
                               df: pd.DataFrame,
                               total_funding_col: str = 'total_funding',
                               funding_rounds_col: str = 'funding_rounds',
                               investor_count_col: str = 'investor_count') -> pd.DataFrame:
        """
        Create funding-related features.
        
        Args:
            df: Input DataFrame
            total_funding_col: Name of total funding column
            funding_rounds_col: Name of funding rounds column
            investor_count_col: Name of investor count column
            
        Returns:
            DataFrame with funding features
        """
        df_new = df.copy()
        
        # Average funding per round
        if total_funding_col in df_new.columns and funding_rounds_col in df_new.columns:
            df_new['avg_funding_per_round'] = (
                df_new[total_funding_col] / df_new[funding_rounds_col].replace(0, np.nan)
            )
            self.created_features.append('avg_funding_per_round')
            print("Created avg_funding_per_round feature")
        
        # Funding per investor
        if total_funding_col in df_new.columns and investor_count_col in df_new.columns:
            df_new['funding_per_investor'] = (
                df_new[total_funding_col] / df_new[investor_count_col].replace(0, np.nan)
            )
            self.created_features.append('funding_per_investor')
            print("Created funding_per_investor feature")
        
        # Log transform of total funding (to handle skewness)
        if total_funding_col in df_new.columns:
            df_new['log_total_funding'] = np.log1p(df_new[total_funding_col])
            self.created_features.append('log_total_funding')
            print("Created log_total_funding feature")
        
        return df_new
    
    def create_temporal_features(self, 
                                df: pd.DataFrame,
                                year_founded_col: str = 'year_founded',
                                first_funding_year_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create temporal features related to founding and funding timing.
        
        Args:
            df: Input DataFrame
            year_founded_col: Name of year founded column
            first_funding_year_col: Name of first funding year column
            
        Returns:
            DataFrame with temporal features
        """
        df_new = df.copy()
        
        # Years since founding
        if year_founded_col in df_new.columns:
            df_new['years_since_founding'] = self.reference_year - df_new[year_founded_col]
            self.created_features.append('years_since_founding')
        
        # Time to first funding
        if first_funding_year_col and first_funding_year_col in df_new.columns:
            df_new['time_to_first_funding'] = (
                df_new[first_funding_year_col] - df_new[year_founded_col]
            )
            # Handle negative values (funded before official founding)
            df_new['time_to_first_funding'] = df_new['time_to_first_funding'].clip(lower=0)
            self.created_features.append('time_to_first_funding')
            print("Created time_to_first_funding feature")
        
        # Decade founded (for cohort analysis)
        if year_founded_col in df_new.columns:
            df_new['founding_decade'] = (df_new[year_founded_col] // 10) * 10
            self.created_features.append('founding_decade')
            print("Created founding_decade feature")
        
        return df_new
    
    def create_team_features(self, 
                           df: pd.DataFrame,
                           team_size_col: str = 'team_size') -> pd.DataFrame:
        """
        Create team-related features.
        
        Args:
            df: Input DataFrame
            team_size_col: Name of team size column
            
        Returns:
            DataFrame with team features
        """
        df_new = df.copy()
        
        if team_size_col in df_new.columns:
            # Team size categories
            bins = [0, 1, 2, 5, 10, 100]
            labels = ['Solo', 'Pair', 'Small (3-5)', 'Medium (6-10)', 'Large (10+)']
            
            df_new['team_size_category'] = pd.cut(df_new[team_size_col],
                                                  bins=bins,
                                                  labels=labels,
                                                  include_lowest=True)
            
            # Is solo founder
            df_new['is_solo_founder'] = (df_new[team_size_col] == 1).astype(int)
            
            self.created_features.extend(['team_size_category', 'is_solo_founder'])
            print("Created team features")
        
        return df_new
    
    def create_location_features(self, 
                                df: pd.DataFrame,
                                location_col: str = 'location',
                                major_hubs: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create location-related features.
        
        Args:
            df: Input DataFrame
            location_col: Name of location column
            major_hubs: List of major startup hubs
            
        Returns:
            DataFrame with location features
        """
        df_new = df.copy()
        
        if major_hubs is None:
            major_hubs = ['San Francisco', 'New York', 'Boston', 'Los Angeles', 
                         'Seattle', 'Austin', 'London', 'Berlin']
        
        if location_col in df_new.columns:
            # Is in major hub
            df_new['is_major_hub'] = df_new[location_col].apply(
                lambda x: 1 if any(hub.lower() in str(x).lower() for hub in major_hubs) else 0
            )
            
            self.created_features.append('is_major_hub')
            print("Created location features")
            print(f"Companies in major hubs: {df_new['is_major_hub'].sum()}")
        
        return df_new
    
    def create_interaction_features(self, 
                                   df: pd.DataFrame,
                                   feature_pairs: Optional[List[tuple]] = None) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of tuples containing feature pairs to interact
            
        Returns:
            DataFrame with interaction features
        """
        df_new = df.copy()
        
        if feature_pairs is None:
            # Default interactions
            feature_pairs = [
                ('company_age', 'total_funding'),
                ('funding_rounds', 'investor_count'),
                ('team_size', 'total_funding')
            ]
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df_new.columns and feat2 in df_new.columns:
                interaction_name = f'{feat1}_x_{feat2}'
                df_new[interaction_name] = df_new[feat1] * df_new[feat2]
                self.created_features.append(interaction_name)
                print(f"Created interaction feature: {interaction_name}")
        
        return df_new
    
    def get_created_features(self) -> List[str]:
        """
        Get list of all created features.
        
        Returns:
            List of feature names
        """
        return self.created_features

