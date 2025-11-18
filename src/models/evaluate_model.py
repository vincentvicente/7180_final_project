"""
Model evaluation module for startup success prediction.
ADDRESSES INSTRUCTOR FEEDBACK: Use confusion matrix and appropriate metrics
for imbalanced dataset (71% in "active" class).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


class ModelEvaluator:
    """
    Model evaluation class with focus on metrics appropriate
    for imbalanced datasets.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.evaluation_results = {}
        
    def evaluate_model(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      y_pred_proba: Optional[np.ndarray] = None,
                      model_name: str = 'Model') -> Dict[str, Any]:
        """
        Comprehensive model evaluation with all relevant metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for AUC calculation)
            model_name: Name of the model
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {'model_name': model_name}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        results['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
        results['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
        results['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                results['average_precision'] = average_precision_score(y_true, y_pred_proba)
            except ValueError as e:
                print(f"Warning: Could not calculate AUC: {e}")
                results['roc_auc'] = None
                results['average_precision'] = None
        
        # Classification report
        results['classification_report'] = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )
        
        # Store results
        self.evaluation_results[model_name] = results
        
        return results
    
    def print_evaluation_report(self, results: Dict[str, Any]) -> None:
        """
        Print formatted evaluation report.
        EMPHASIZES CONFUSION MATRIX as requested by instructor.
        
        Args:
            results: Dictionary of evaluation results
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT: {results['model_name']}")
        print(f"{'='*60}\n")
        
        # Overall metrics
        print("Overall Metrics:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        
        if results.get('roc_auc'):
            print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
        
        # Per-class metrics (important for imbalanced data)
        print("\nPer-Class Metrics:")
        for i, (prec, rec, f1) in enumerate(zip(
            results['precision_per_class'],
            results['recall_per_class'],
            results['f1_per_class']
        )):
            print(f"  Class {i}:")
            print(f"    Precision: {prec:.4f}")
            print(f"    Recall:    {rec:.4f}")
            print(f"    F1-Score:  {f1:.4f}")
        
        # CONFUSION MATRIX - Primary focus for instructor
        print("\n" + "="*60)
        print("CONFUSION MATRIX (As requested by instructor)")
        print("="*60)
        cm = results['confusion_matrix']
        print(cm)
        print("\nNote: Rows represent true labels, columns represent predictions")
        
        # Calculate per-class accuracy from confusion matrix
        print("\nPer-Class Analysis from Confusion Matrix:")
        for i in range(cm.shape[0]):
            total = cm[i].sum()
            correct = cm[i, i]
            class_accuracy = correct / total if total > 0 else 0
            print(f"  Class {i}: {correct}/{total} correct ({class_accuracy:.2%})")
        
        print(f"\n{'='*60}\n")
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = 'Confusion Matrix',
                             figsize: Tuple[int, int] = (8, 6),
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix as heatmap.
        ADDRESSES INSTRUCTOR FEEDBACK: Visualize confusion matrix clearly.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names or ['Failure', 'Success'],
                   yticklabels=class_names or ['Failure', 'Success'])
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add percentage annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1%})',
                       ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_pred_proba: np.ndarray,
                      title: str = 'ROC Curve',
                      figsize: Tuple[int, int] = (8, 6),
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self,
                                    y_true: np.ndarray,
                                    y_pred_proba: np.ndarray,
                                    title: str = 'Precision-Recall Curve',
                                    figsize: Tuple[int, int] = (8, 6),
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve (important for imbalanced datasets).
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot PR curve
        ax.plot(recall, precision, color='darkgreen', lw=2,
               label=f'PR curve (AP = {avg_precision:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")
        
        return fig
    
    def compare_models(self,
                      results_list: List[Dict[str, Any]],
                      metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
                      figsize: Tuple[int, int] = (12, 6),
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare multiple models visually.
        
        Args:
            results_list: List of evaluation result dictionaries
            metrics: List of metrics to compare
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        model_names = [r['model_name'] for r in results_list]
        
        # Prepare data
        data = {metric: [r[metric] for r in results_list] for metric in metrics}
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bars
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            offset = width * (i - len(metrics)/2 + 0.5)
            ax.bar(x + offset, data[metric], width, label=metric.capitalize())
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison saved to {save_path}")
        
        return fig
    
    def create_evaluation_summary_table(self,
                                       results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a summary table of model evaluations.
        
        Args:
            results_list: List of evaluation result dictionaries
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for results in results_list:
            row = {
                'Model': results['model_name'],
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            }
            
            if results.get('roc_auc'):
                row['ROC-AUC'] = results['roc_auc']
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df

