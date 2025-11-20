"""
Execution script for model training and comprehensive evaluation on the test set.
This script orchestrates the full model pipeline using the modular functions 
in ModelTrainer, including data splitting, scaling, SMOTE resampling, 
multi-model training, and final best model selection/saving for app.py.
"""
import pandas as pd
import sys
import os
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Ensure src.models module is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator

# --- Configuration ---
PROCESSED_DATA_PATH = 'data/processed/final_dataset.csv' # Path to the final file saved by run_pipeline.py

def execute_model_evaluation():
    # 1. Load the processed dataset
    print(f"--- 1. Loading Processed Data from {PROCESSED_DATA_PATH} ---")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Processed data file not found. Please run run_pipeline.py first.")
        return
    
    # 2. Drop rows with missing values BEFORE train/test split
    # Prepare Data for Modeling
    # Drop non-feature columns
    df = df.dropna()

    X_raw = df.drop(columns=['target', 'name', 'status'], errors='ignore')
    y = df['target']
    
    # 3. Initialize ModelTrainer
    trainer = ModelTrainer(random_state=42)
    
    # 4. Split Data (using the trainer's split logic)
    X_train, X_test, y_train, y_test = trainer.split_data(X_raw, y)

    # 5. Scale Features
    print("\n--- 5. Scaling Features ---")
    X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
    
    # 6. Handle Class Imbalance using SMOTE (as an example strategy)
    print("\n--- 6. Resampling Training Data (SMOTE) ---")
    X_resampled, y_resampled = trainer.handle_imbalance_smote(X_train_scaled, y_train)

    # 7. Train All Models
    print("\n--- 7. Training All Models with Resampled Data (using class_weights) ---")
    # Note: train_all_models uses class weights where applicable.
    trainer.train_all_models(X_resampled, y_resampled, use_class_weights=True)
    
    # 8. Initialize ModelEvaluator
    evaluator = ModelEvaluator()

    # 9. Evaluate each model on the SCALED Test Set (X_test_scaled)
    print("\n--- 9. Evaluating Models on SCALED Test Set ---")
    for model_name, model in trainer.models.items():
        print(f"\n--- Evaluating model: {model_name} ---")
        
        y_pred = model.predict(X_test_scaled)
        y_proba = None
        
        # Get probability (for ROC AUC calculation)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test_scaled)
        
        # Print classification report
        report = classification_report(y_test, y_pred)
        print(report)
        
        # Show Confusion Matrix (Uncomment plt.show() to display plots)
        # evaluator.plot_confusion_matrix(y_test, y_pred, title=f"{model_name} Confusion Matrix")
        # plt.show()
        
        # Show ROC Curve (Uncomment plt.show() to display plots)
        # if y_proba is not None and len(y_test.unique()) > 1:
        #     evaluator.plot_roc_curve(y_test, y_proba, title=f"{model_name} ROC Curve")
        #     plt.show()
    
    # 10. Select the best model and save all necessary artifacts for app.py
    print("\n--- 10. Selecting and Saving Best Model Artifacts ---")
    trainer.select_and_save_best_artifacts(X_test_scaled, y_test)
        
    print("\n✅ Model Evaluation Pipeline Completed.")


if __name__ == "__main__":
    execute_model_evaluation()