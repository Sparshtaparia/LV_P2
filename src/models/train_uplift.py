"""
True Uplift Modeling.
Trains a custom T-Learner on observational data to predict Individual Treatment Effect (ITE).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
import joblib
import logging
from xgboost import XGBClassifier

from config.config import MODEL_INPUT_CSV, FEATURE_COLS_PATH, TARGET_COL, MODELS_DIR
from src.models.uplift_helper import CustomTLearner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

UPLIFT_MODEL_PATH = MODELS_DIR / "uplift_model.pkl"

def run():
    df = pd.read_csv(MODEL_INPUT_CSV)
    
    # Simulate a 'treatment' column since observational data lacks explicit A/B test assignments.
    # We use 'Is_Two_Year' as the simulated treatment (representing a two-year contract upgrade).
    treatment_col = "Is_Two_Year" 
    
    if treatment_col not in df.columns:
        df[treatment_col] = np.random.binomial(1, 0.5, len(df))

    feat_cols = joblib.load(FEATURE_COLS_PATH)
    features = [c for c in feat_cols if c != treatment_col and c in df.columns]

    X = df[features].fillna(0).values
    T = df[treatment_col].values
    y = df[TARGET_COL].astype(int).values # 1 = Churn, 0 = No Churn. We want negative uplift (reduce churn)

    # Initialize T-Learner with XGBoost Classifiers
    learner = CustomTLearner(
        control_estimator=XGBClassifier(random_state=42, n_estimators=100, max_depth=4, eval_metric="logloss"),
        treatment_estimator=XGBClassifier(random_state=42, n_estimators=100, max_depth=4, eval_metric="logloss")
    )
    
    log.info("Training Custom T-Learner for uplift modeling...")
    learner.fit(X=X, treatment=T, y=y)

    joblib.dump((learner, features), UPLIFT_MODEL_PATH)
    log.info(f"Uplift model saved -> {UPLIFT_MODEL_PATH}")

if __name__ == "__main__":
    run()
