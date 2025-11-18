"""
Model training module for startup success prediction.
ADDRESSES INSTRUCTOR FEEDBACK: Handle class imbalance (71% in "active" class).

Implements multiple strategies for handling imbalanced data:
1. Class weighting
2. SMOTE (Synthetic Minority Over-sampling Technique)
3. Undersampling
4. Combination approaches
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek


class ModelTrainer:
    """
    Model training class with built-in support for
    handling imbalanced datasets.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelTrainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def split_data(self,
                   X: pd.DataFrame,
                   y: pd.Series,
                   test_size: float = 0.2,
                   stratify: bool = True) -> Tuple:
        """
        Split data into train and test sets.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            stratify: Whether to use stratified split (recommended for imbalanced data)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if stratify else None
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"\nClass distribution in training set:")
        print(y_train.value_counts())
        print(f"\nClass distribution proportions:")
        print(y_train.value_counts(normalize=True))
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self,
                      X_train: pd.DataFrame,
                      X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled,
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled
    
    def handle_imbalance_smote(self,
                               X_train: pd.DataFrame,
                               y_train: pd.Series,
                               sampling_strategy: str = 'auto') -> Tuple:
        """
        Handle class imbalance using SMOTE.
        METHOD 1: Synthetic Minority Over-sampling Technique
        
        Args:
            X_train: Training features
            y_train: Training target
            sampling_strategy: Sampling strategy for SMOTE
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        smote = SMOTE(random_state=self.random_state, sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"Original dataset shape: {X_train.shape}")
        print(f"Resampled dataset shape: {X_resampled.shape}")
        print(f"Class distribution after SMOTE:")
        print(pd.Series(y_resampled).value_counts())
        
        return X_resampled, y_resampled
    
    def handle_imbalance_undersample(self,
                                    X_train: pd.DataFrame,
                                    y_train: pd.Series,
                                    sampling_strategy: str = 'auto') -> Tuple:
        """
        Handle class imbalance using undersampling.
        METHOD 2: Random Under-sampling
        
        Args:
            X_train: Training features
            y_train: Training target
            sampling_strategy: Sampling strategy
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        rus = RandomUnderSampler(random_state=self.random_state, sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        
        print(f"Original dataset shape: {X_train.shape}")
        print(f"Resampled dataset shape: {X_resampled.shape}")
        print(f"Class distribution after undersampling:")
        print(pd.Series(y_resampled).value_counts())
        
        return X_resampled, y_resampled
    
    def handle_imbalance_combined(self,
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 method: str = 'smote_tomek') -> Tuple:
        """
        Handle class imbalance using combined methods.
        METHOD 3: Combined Over/Under-sampling
        
        Args:
            X_train: Training features
            y_train: Training target
            method: Method ('smote_tomek' or 'smote_enn')
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if method == 'smote_tomek':
            resampler = SMOTETomek(random_state=self.random_state)
        else:
            resampler = SMOTEENN(random_state=self.random_state)
        
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        
        print(f"Original dataset shape: {X_train.shape}")
        print(f"Resampled dataset shape: {X_resampled.shape}")
        print(f"Class distribution after {method}:")
        print(pd.Series(y_resampled).value_counts())
        
        return X_resampled, y_resampled
    
    def train_logistic_regression(self,
                                  X_train: pd.DataFrame,
                                  y_train: pd.Series,
                                  class_weight: Optional[str] = 'balanced',
                                  **kwargs) -> LogisticRegression:
        """
        Train Logistic Regression model.
        Uses class_weight='balanced' to handle imbalance.
        
        Args:
            X_train: Training features
            y_train: Training target
            class_weight: Class weight strategy ('balanced' or None)
            **kwargs: Additional model parameters
            
        Returns:
            Trained model
        """
        model = LogisticRegression(
            class_weight=class_weight,
            random_state=self.random_state,
            max_iter=1000,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        self.models['LogisticRegression'] = model
        
        print("Trained Logistic Regression")
        return model
    
    def train_random_forest(self,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           class_weight: Optional[str] = 'balanced',
                           n_estimators: int = 100,
                           **kwargs) -> RandomForestClassifier:
        """
        Train Random Forest model.
        Uses class_weight='balanced' to handle imbalance.
        
        Args:
            X_train: Training features
            y_train: Training target
            class_weight: Class weight strategy ('balanced' or None)
            n_estimators: Number of trees
            **kwargs: Additional model parameters
            
        Returns:
            Trained model
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        self.models['RandomForest'] = model
        
        print("Trained Random Forest")
        return model
    
    def train_xgboost(self,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     scale_pos_weight: Optional[float] = None,
                     n_estimators: int = 100,
                     **kwargs) -> XGBClassifier:
        """
        Train XGBoost model.
        Uses scale_pos_weight to handle imbalance.
        
        Args:
            X_train: Training features
            y_train: Training target
            scale_pos_weight: Weight for positive class
                             (recommended: sum(negative)/sum(positive))
            n_estimators: Number of boosting rounds
            **kwargs: Additional model parameters
            
        Returns:
            Trained model
        """
        # Calculate scale_pos_weight if not provided
        if scale_pos_weight is None:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
            print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
        
        model = XGBClassifier(
            n_estimators=n_estimators,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        self.models['XGBoost'] = model
        
        print("Trained XGBoost")
        return model
    
    def train_lightgbm(self,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      class_weight: Optional[str] = 'balanced',
                      n_estimators: int = 100,
                      **kwargs) -> LGBMClassifier:
        """
        Train LightGBM model.
        Uses class_weight='balanced' to handle imbalance.
        
        Args:
            X_train: Training features
            y_train: Training target
            class_weight: Class weight strategy ('balanced' or None)
            n_estimators: Number of boosting rounds
            **kwargs: Additional model parameters
            
        Returns:
            Trained model
        """
        model = LGBMClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=self.random_state,
            verbose=-1,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        self.models['LightGBM'] = model
        
        print("Trained LightGBM")
        return model
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        use_class_weights: bool = True) -> Dict[str, Any]:
        """
        Train all available models.
        
        Args:
            X_train: Training features
            y_train: Training target
            use_class_weights: Whether to use class weighting
            
        Returns:
            Dictionary of trained models
        """
        print("Training all models...\n")
        
        # Logistic Regression
        self.train_logistic_regression(
            X_train, y_train,
            class_weight='balanced' if use_class_weights else None
        )
        
        # Random Forest
        self.train_random_forest(
            X_train, y_train,
            class_weight='balanced' if use_class_weights else None
        )
        
        # XGBoost
        self.train_xgboost(X_train, y_train)
        
        # LightGBM
        self.train_lightgbm(
            X_train, y_train,
            class_weight='balanced' if use_class_weights else None
        )
        
        print(f"\nTrained {len(self.models)} models")
        return self.models
    
    def cross_validate_model(self,
                            model: Any,
                            X: pd.DataFrame,
                            y: pd.Series,
                            cv: int = 5,
                            scoring: str = 'f1_weighted') -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to validate
            X: Features
            y: Target
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with mean and std of scores
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        
        results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
        
        print(f"Cross-validation ({cv} folds):")
        print(f"  Mean {scoring}: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        
        return results
    
    def save_model(self, model: Any, filepath: str) -> None:
        """
        Save trained model to file.
        
        Args:
            model: Trained model
            filepath: Path to save model
        """
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load trained model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model

