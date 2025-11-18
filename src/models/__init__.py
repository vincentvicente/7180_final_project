"""
Model module
Contains model training and evaluation functionality
"""

from .train_model import ModelTrainer
from .evaluate_model import ModelEvaluator

__all__ = ['ModelTrainer', 'ModelEvaluator']
