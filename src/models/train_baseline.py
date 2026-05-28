import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt

# Configure matplotlib for headless environment
import matplotlib
matplotlib.use('Agg')

def load_data():
    data_path = '../../data/processed/model_input.csv'
    df = pd.read_csv(data_path)
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    return X, y

def train_and_evaluate():
    X, y = load_data()
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Stratified K-Fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Model
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    
    # Set MLflow tracking URI (local directory)
    os.makedirs('../../mlruns', exist_ok=True)
    mlflow.set_tracking_uri('sqlite:///../../mlruns/mlflow.db')
    mlflow.set_experiment('Churn_Prediction_Baseline')
    
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        mlflow.log_params({
            "model_type": "RandomForest",
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced"
        })
        
        # Cross-validation AUC
        cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        
        # Train on full train set
        rf.fit(X_train, y_train)
        
        # Predict on test set
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        y_pred = rf.predict(X_test)
        
        # Metrics
        test_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        print(f"Test ROC-AUC: {test_auc:.4f}")
        print(f"Test PR-AUC: {pr_auc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        mlflow.log_metrics({
            "test_roc_auc": test_auc,
            "test_pr_auc": pr_auc
        })
        
        # Log Model
        mlflow.sklearn.log_model(rf, "baseline_rf_model")
        
        # Explainability (SHAP)
        print("Generating SHAP baseline explainability...")
        explainer = shap.TreeExplainer(rf)
        # Use a small sample for SHAP to save time
        X_sample = X_test.sample(200, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        os.makedirs('../../docs/model_plots', exist_ok=True)
        
        # For Random Forest in newer shap versions, shap_values might be a list
        if isinstance(shap_values, list):
            sv = shap_values[1] # SHAP values for class 1 (Churn)
        elif len(shap_values.shape) == 3:
            sv = shap_values[:,:,1]
        else:
            sv = shap_values
            
        plt.figure()
        shap.summary_plot(sv, X_sample, show=False)
        plt.savefig('../../docs/model_plots/shap_summary_rf.png', bbox_inches='tight')
        plt.close()
        print("SHAP summary plot saved to docs/model_plots/shap_summary_rf.png")

if __name__ == "__main__":
    train_and_evaluate()
