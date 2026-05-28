"""
Data Drift Detection using Evidently AI.
Compares a reference dataset (training data) against current data (production).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from config.config import MODEL_INPUT_CSV, PROC_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DRIFT_REPORT_PATH = PROC_DIR / "drift_report.html"

def detect_drift():
    # In a real scenario, current_data would be pulled from the database or Kafka consumer output
    # For demonstration, we split the model_input into reference and current
    df = pd.read_csv(MODEL_INPUT_CSV)
    
    # Simulate a split: 80% reference, 20% current
    split_idx = int(len(df) * 0.8)
    reference_data = df.iloc[:split_idx]
    current_data = df.iloc[split_idx:]
    
    # Introduce artificial drift to the current data for demonstration
    if "MonthlyCharges" in current_data.columns:
        current_data["MonthlyCharges"] = current_data["MonthlyCharges"] * 1.5

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    
    report.save_html(str(DRIFT_REPORT_PATH))
    log.info(f"Drift report generated at {DRIFT_REPORT_PATH}")
    
    # Parse JSON for alerting
    drift_result = report.as_dict()
    dataset_drift = drift_result["metrics"][0]["result"]["dataset_drift"]
    
    if dataset_drift:
        log.warning("🚨 DATA DRIFT DETECTED! Model retraining may be required.")
    else:
        log.info("✅ No significant data drift detected.")

if __name__ == "__main__":
    detect_drift()
