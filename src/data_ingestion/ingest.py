"""
Data ingestion pipeline:
  - Load raw Telco CSV
  - Synthetic augmentation with Faker
  - Write to SQLite (or PostgreSQL)
  - Produce model_input.csv with feature engineering
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from faker import Faker
from sqlalchemy import create_engine
import logging

from config.config import (
    RAW_CSV, AUGMENTED_CSV, DATABASE_URL, PROC_DIR,
    NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL, ID_COL, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SYNTHETIC_N = 2957   # pad to 10 000 total

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"], inplace=True)
    log.info(f"Loaded raw dataset: {df.shape}")
    return df

def generate_synthetic(n: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    fake = Faker()
    Faker.seed(seed)
    np.random.seed(seed)

    INTERNET = ("DSL", "Fiber optic", "No")
    PHONE    = ("Yes", "No")
    LINES    = ("Yes", "No", "No phone service")
    YSVC     = ("Yes", "No", "No internet service")
    CONTR    = ("Month-to-month", "One year", "Two year")
    PAY      = ("Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)")

    rows = []
    for _ in range(n):
        tenure = int(np.random.beta(2, 2) * 72)
        monthly = round(np.random.uniform(18.25, 118.75), 2)
        churn = "Yes" if np.random.rand() < 0.27 else "No"
        rows.append({
            "customerID":        fake.unique.uuid4()[:10],
            "gender":            np.random.choice(["Male", "Female"]),
            "SeniorCitizen":     int(np.random.rand() < 0.16),
            "Partner":           np.random.choice(PHONE),
            "Dependents":        np.random.choice(PHONE),
            "tenure":            tenure,
            "PhoneService":      np.random.choice(PHONE),
            "MultipleLines":     np.random.choice(LINES),
            "InternetService":   np.random.choice(INTERNET),
            "OnlineSecurity":    np.random.choice(YSVC),
            "OnlineBackup":      np.random.choice(YSVC),
            "DeviceProtection":  np.random.choice(YSVC),
            "TechSupport":       np.random.choice(YSVC),
            "StreamingTV":       np.random.choice(YSVC),
            "StreamingMovies":   np.random.choice(YSVC),
            "Contract":          np.random.choice(CONTR),
            "PaperlessBilling":  np.random.choice(PHONE),
            "PaymentMethod":     np.random.choice(PAY),
            "MonthlyCharges":    monthly,
            "TotalCharges":      round(monthly * tenure, 2),
            "Churn":             churn,
        })
    return pd.DataFrame(rows)

def save_to_db(df: pd.DataFrame):
    engine = create_engine(DATABASE_URL)
    customers = df[["customerID","gender","SeniorCitizen","Partner","Dependents","Churn"]]
    services  = df[["customerID","tenure","PhoneService","MultipleLines","InternetService",
                    "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
                    "StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]]
    billing   = df[["customerID","MonthlyCharges","TotalCharges"]]
    customers.to_sql("customers", engine, if_exists="replace", index=False)
    services .to_sql("services",  engine, if_exists="replace", index=False)
    billing  .to_sql("billing",   engine, if_exists="replace", index=False)
    log.info("Saved to database (3 tables)")

def run():
    raw = load_raw()
    synth = generate_synthetic(SYNTHETIC_N)
    combined = pd.concat([raw, synth], ignore_index=True)
    combined.to_csv(AUGMENTED_CSV, index=False)
    log.info(f"Augmented dataset saved -> {AUGMENTED_CSV}  shape={combined.shape}")
    save_to_db(combined)

if __name__ == "__main__":
    run()
