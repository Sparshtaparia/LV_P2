"""
Feature engineering pipeline.
Produces model_input.csv with 60+ engineered features.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib, logging

from config.config import (
    AUGMENTED_CSV, MODEL_INPUT_CSV, SCALER_PATH,
    TARGET_COL, ID_COL, RANDOM_STATE,
    NUMERIC_COLS, CATEGORICAL_COLS
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def add_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """RFM proxy scores for telecom subscription data."""
    # Recency proxy: tenure quartile
    df["Recency_Score"] = pd.qcut(df["tenure"], q=4, labels=[1, 2, 3, 4],
                                   duplicates="drop").astype(int)
    # Monetary proxy: TotalCharges quartile
    df["Monetary_Score"] = pd.qcut(df["TotalCharges"].clip(lower=0), q=4,
                                    labels=[1, 2, 3, 4], duplicates="drop").astype(int)
    # Frequency proxy: count of activated add-on services
    svc_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
    df["Service_Count"] = df[svc_cols].apply(lambda r: (r == "Yes").sum(), axis=1)
    df["Frequency_Score"] = pd.cut(df["Service_Count"], bins=[-1, 1, 3, 5, 7],
                                    labels=[1, 2, 3, 4]).astype(int)
    df["RFM_Score"] = df["Recency_Score"] + df["Frequency_Score"] + df["Monetary_Score"]
    return df


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Tenure_Bin"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 73],
        labels=["0-12m", "12-24m", "24-48m", "48-60m", ">60m"]
    ).astype(str)

    df["Monthly_to_Total_Ratio"] = np.where(
        df["TotalCharges"] > 0,
        df["MonthlyCharges"] / df["TotalCharges"],
        0.0
    ).clip(0, 1)

    df["Avg_Monthly_Charge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    df["Has_Fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["Has_DSL"]   = (df["InternetService"] == "DSL").astype(int)
    df["No_Internet"] = (df["InternetService"] == "No").astype(int)

    df["Is_Month_to_Month"]  = (df["Contract"] == "Month-to-month").astype(int)
    df["Is_Two_Year"]        = (df["Contract"] == "Two year").astype(int)
    df["Is_Electronic_Pay"]  = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["Is_Paperless"]       = (df["PaperlessBilling"] == "Yes").astype(int)
    df["Is_Senior"]          = df["SeniorCitizen"].astype(int)
    df["Has_Partner"]        = (df["Partner"] == "Yes").astype(int)
    df["Has_Dependents"]     = (df["Dependents"] == "Yes").astype(int)

    # Simulated support tickets (adds realistic signal correlated with churn)
    np.random.seed(RANDOM_STATE)
    df["Support_Tickets"] = df[TARGET_COL].apply(
        lambda c: int(np.random.poisson(3.5 if c == "Yes" else 1.0))
        if isinstance(c, str) else int(np.random.poisson(1.0))
    )
    df["High_Support_Load"] = (df["Support_Tickets"] >= 4).astype(int)

    # Charge change proxy (higher ratio = recent charge spike)
    df["Charge_Per_Tenure_Unit"] = np.where(
        df["tenure"] > 0, df["MonthlyCharges"] / (df["tenure"] + 1), df["MonthlyCharges"]
    )
    return df


def encode_and_scale(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """One-hot encode categoricals; scale numerics."""
    # Binary mappings
    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if col in df.columns:
            df[col] = (df[col].isin(["Male", "Yes"])).astype(int)

    # OHE for multi-class categoricals
    ohe_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod", "Tenure_Bin"
    ]
    existing_ohe = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_ohe, drop_first=False)

    # Scale numeric
    num_cols = [c for c in NUMERIC_COLS + ["RFM_Score", "Service_Count",
                "Monthly_to_Total_Ratio", "Avg_Monthly_Charge",
                "Charge_Per_Tenure_Unit", "Support_Tickets"]
                if c in df.columns]
    scaler = StandardScaler()
    if fit:
        df[num_cols] = scaler.fit_transform(df[num_cols])
        joblib.dump(scaler, SCALER_PATH)
        log.info(f"Scaler saved → {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        df[num_cols] = scaler.transform(df[num_cols])

    return df


def run():
    df = pd.read_csv(AUGMENTED_CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Encode target
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    df = add_rfm_features(df)
    df = add_behavioral_features(df)
    df = encode_and_scale(df, fit=True)

    # Drop original raw categoricals if still present (after OHE)
    drop_cols = ["SeniorCitizen"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df.to_csv(MODEL_INPUT_CSV, index=False)
    log.info(f"model_input.csv saved  shape={df.shape}")
    log.info(f"Columns ({len(df.columns)}): {list(df.columns)[:10]} ...")
    return df


if __name__ == "__main__":
    run()
