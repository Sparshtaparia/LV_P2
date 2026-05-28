"""
Uplift helper classes.
"""
import logging

log = logging.getLogger(__name__)

class CustomTLearner:
    """
    A custom two-model uplift learner (T-Learner).
    Trains separate classifiers for the treatment and control groups.
    """
    def __init__(self, control_estimator, treatment_estimator):
        self.control_estimator = control_estimator
        self.treatment_estimator = treatment_estimator

    def fit(self, X, treatment, y):
        # Control group (treatment == 0)
        X_0 = X[treatment == 0]
        y_0 = y[treatment == 0]
        log.info(f"Fitting control estimator with {len(X_0)} samples...")
        self.control_estimator.fit(X_0, y_0)

        # Treatment group (treatment == 1)
        X_1 = X[treatment == 1]
        y_1 = y[treatment == 1]
        log.info(f"Fitting treatment estimator with {len(X_1)} samples...")
        self.treatment_estimator.fit(X_1, y_1)

    def predict(self, X):
        # ITE = P(Y=1 | X, T=1) - P(Y=1 | X, T=0)
        pred_treatment = self.treatment_estimator.predict_proba(X)[:, 1]
        pred_control = self.control_estimator.predict_proba(X)[:, 1]
        return pred_treatment - pred_control
