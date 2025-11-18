"""
Visualization module for startup success prediction.
ADDRESSES INSTRUCTOR FEEDBACK: Create pre-curated metrics and EDA visualizations
for user consumption.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Tuple, Dict, Any

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class PlotGenerator:
    """
    Visualization class for generating pre-curated metrics
    and exploratory data analysis plots.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the PlotGenerator.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 8)
        
    def plot_class_distribution(self,
                               y: pd.Series,
                               title: str = 'Class Distribution',
                               labels: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot target class distribution.
        IMPORTANT: Shows the imbalance in the dataset.
        
        Args:
            y: Target variable
            title: Plot title
            labels: Class labels
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        value_counts = y.value_counts()
        ax1.bar(range(len(value_counts)), value_counts.values, color=self.color_palette[:len(value_counts)])
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Absolute Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(value_counts)))
        if labels:
            ax1.set_xticklabels(labels)
        
        # Add count labels
        for i, v in enumerate(value_counts.values):
            ax1.text(i, v + max(value_counts.values) * 0.01, str(v),
                    ha='center', va='bottom', fontweight='bold')
        
        # Percentage plot
        value_props = y.value_counts(normalize=True) * 100
        ax2.pie(value_props.values, labels=labels or value_props.index,
               autopct='%1.1f%%', startangle=90, colors=self.color_palette[:len(value_props)])
        ax2.set_title('Relative Class Distribution', fontsize=14, fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_company_age_distribution(self,
                                     df: pd.DataFrame,
                                     age_col: str = 'company_age',
                                     target_col: str = 'target',
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot company age distribution by success/failure.
        ADDRESSES FEEDBACK: Visualize the engineered "company age" feature.
        
        Args:
            df: Input DataFrame
            age_col: Company age column name
            target_col: Target column name
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Overall distribution
        axes[0, 0].hist(df[age_col], bins=30, color=self.color_palette[0], alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Company Age (years)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Overall Company Age Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].axvline(df[age_col].median(), color='red', linestyle='--',
                          label=f'Median: {df[age_col].median():.1f}')
        axes[0, 0].legend()
        
        # Distribution by target
        for target in df[target_col].unique():
            subset = df[df[target_col] == target][age_col]
            axes[0, 1].hist(subset, bins=20, alpha=0.6,
                           label=f'Class {target}', edgecolor='black')
        axes[0, 1].set_xlabel('Company Age (years)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Company Age by Outcome', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        
        # Box plot
        df.boxplot(column=age_col, by=target_col, ax=axes[1, 0])
        axes[1, 0].set_xlabel('Outcome', fontsize=11)
        axes[1, 0].set_ylabel('Company Age (years)', fontsize=11)
        axes[1, 0].set_title('Company Age Distribution by Outcome', fontsize=12, fontweight='bold')
        plt.sca(axes[1, 0])
        plt.xticks([1, 2], ['Failure', 'Success'])
        
        # Violin plot
        sns.violinplot(data=df, x=target_col, y=age_col, ax=axes[1, 1], palette=self.color_palette[:2])
        axes[1, 1].set_xlabel('Outcome', fontsize=11)
        axes[1, 1].set_ylabel('Company Age (years)', fontsize=11)
        axes[1, 1].set_title('Company Age Density by Outcome', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticklabels(['Failure', 'Success'])
        
        plt.suptitle('Company Age Analysis', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_success_rate_by_category(self,
                                     df: pd.DataFrame,
                                     category_col: str,
                                     target_col: str = 'target',
                                     title: Optional[str] = None,
                                     top_n: int = 20,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot success rate by categorical variable.
        PRE-CURATED METRIC for user consumption.
        
        Args:
            df: Input DataFrame
            category_col: Categorical column name
            target_col: Target column name
            title: Plot title
            top_n: Show top N categories by count
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Calculate success rate by category
        success_rate = df.groupby(category_col)[target_col].agg(['mean', 'count'])
        
        # Filter to top N by count to avoid overcrowding
        success_rate = success_rate.nlargest(top_n, 'count')
        success_rate = success_rate.sort_values('mean', ascending=False)
        success_rate['percentage'] = success_rate['mean'] * 100
        
        # Adjust figure size based on number of categories
        figsize = (12, max(6, len(success_rate) * 0.4))
        fig, ax = plt.subplots(figsize=figsize)
        
        # Bar plot
        bars = ax.barh(range(len(success_rate)), success_rate['percentage'],
                      color=self.color_palette[0], alpha=0.8)
        
        ax.set_yticks(range(len(success_rate)))
        ax.set_yticklabels(success_rate.index, fontsize=10)
        ax.set_xlabel('Success Rate (%)', fontsize=12)
        ax.set_ylabel(category_col.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title or f'Success Rate by {category_col.replace("_", " ").title()} (Top {top_n})',
                    fontsize=14, fontweight='bold')
        
        # Add percentage labels and count
        for i, (idx, row) in enumerate(success_rate.iterrows()):
            ax.text(row['percentage'] + 1, i, f"{row['percentage']:.1f}% (n={int(row['count'])})",
                   va='center', fontsize=9)
        
        ax.set_xlim(0, min(100, success_rate['percentage'].max() + 15))
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_funding_analysis(self,
                             df: pd.DataFrame,
                             funding_col: str = 'total_funding',
                             target_col: str = 'target',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot funding amount analysis.
        PRE-CURATED METRIC for user consumption.
        
        Args:
            df: Input DataFrame
            funding_col: Funding column name
            target_col: Target column name
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distribution by outcome
        for target in df[target_col].unique():
            subset = df[df[target_col] == target][funding_col]
            axes[0, 0].hist(np.log1p(subset), bins=30, alpha=0.6,
                           label=f'Class {target}', edgecolor='black')
        axes[0, 0].set_xlabel('Log(Total Funding + 1)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Funding Distribution by Outcome', fontsize=12, fontweight='bold')
        axes[0, 0].legend(['Failure', 'Success'])
        
        # Box plot
        df.boxplot(column=funding_col, by=target_col, ax=axes[0, 1])
        axes[0, 1].set_xlabel('Outcome', fontsize=11)
        axes[0, 1].set_ylabel('Total Funding', fontsize=11)
        axes[0, 1].set_title('Funding by Outcome', fontsize=12, fontweight='bold')
        axes[0, 1].set_yscale('log')
        plt.sca(axes[0, 1])
        plt.xticks([1, 2], ['Failure', 'Success'])
        
        # Funding bins
        funding_bins = pd.qcut(df[funding_col], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                              duplicates='drop')
        success_by_funding = df.groupby(funding_bins)[target_col].mean() * 100
        axes[1, 0].bar(range(len(success_by_funding)), success_by_funding.values,
                      color=self.color_palette[2], alpha=0.8)
        axes[1, 0].set_xticks(range(len(success_by_funding)))
        axes[1, 0].set_xticklabels(success_by_funding.index, rotation=45, ha='right')
        axes[1, 0].set_xlabel('Funding Level', fontsize=11)
        axes[1, 0].set_ylabel('Success Rate (%)', fontsize=11)
        axes[1, 0].set_title('Success Rate by Funding Level', fontsize=12, fontweight='bold')
        
        # Average funding by outcome
        avg_funding = df.groupby(target_col)[funding_col].mean()
        axes[1, 1].bar([0, 1], avg_funding.values, color=self.color_palette[:2], alpha=0.8)
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_xticklabels(['Failure', 'Success'])
        axes[1, 1].set_xlabel('Outcome', fontsize=11)
        axes[1, 1].set_ylabel('Average Total Funding', fontsize=11)
        axes[1, 1].set_title('Average Funding by Outcome', fontsize=12, fontweight='bold')
        for i, v in enumerate(avg_funding.values):
            axes[1, 1].text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Funding Analysis', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_heatmap(self,
                                df: pd.DataFrame,
                                features: Optional[List[str]] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation heatmap of features.
        
        Args:
            df: Input DataFrame
            features: List of features to include (None for all numeric)
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = df[features].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self,
                               feature_importance: pd.DataFrame,
                               top_n: int = 20,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance from model.
        
        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Sort and get top N
        top_features = feature_importance.nlargest(top_n, 'importance')
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'],
                      color=self.color_palette[1], alpha=0.8)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(top_features['importance']):
            ax.text(v + max(top_features['importance']) * 0.01, i, f'{v:.4f}',
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self,
                                    df: pd.DataFrame,
                                    target_col: str = 'target') -> Dict[str, go.Figure]:
        """
        Create interactive Plotly visualizations for Streamlit dashboard.
        PRE-CURATED INTERACTIVE METRICS for user consumption.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Dictionary of Plotly figures
        """
        figures = {}
        
        # 1. Class distribution
        class_counts = df[target_col].value_counts()
        fig1 = go.Figure(data=[go.Pie(labels=['Failure', 'Success'], values=class_counts.values,
                                      hole=0.3)])
        fig1.update_layout(title='Class Distribution', height=400)
        figures['class_distribution'] = fig1
        
        # 2. Success rate by category (if categorical columns exist)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            col = categorical_cols[0]
            success_rate = df.groupby(col)[target_col].mean() * 100
            fig2 = px.bar(x=success_rate.index, y=success_rate.values,
                         labels={'x': col, 'y': 'Success Rate (%)'},
                         title=f'Success Rate by {col}')
            figures['success_rate_by_category'] = fig2
        
        return figures

