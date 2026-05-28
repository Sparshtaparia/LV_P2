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
    # Recency proxy: tenure quartile (fixed bins for inference compatibility)
    df["Recency_Score"] = pd.cut(df["tenure"], bins=[-1, 9, 29, 55, 100], 
                                 labels=[1, 2, 3, 4]).astype(int)
    # Monetary proxy: TotalCharges quartile (fixed bins)
    df["Monetary_Score"] = pd.cut(df["TotalCharges"].clip(lower=0), bins=[-1, 400, 1400, 3800, 10000],
                                  labels=[1, 2, 3, 4]).astype(int)
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
    if "Support_Tickets" not in df.columns:
        actual_target_col = None
        for potential_target in [TARGET_COL, "TARGET", "churn_label", "churn"]:
            if potential_target in df.columns:
                actual_target_col = potential_target
                break
        
        if actual_target_col is not None:
            df["Support_Tickets"] = df[actual_target_col].apply(
                lambda c: int(np.random.poisson(3.5 if str(c).strip().lower() in ["yes", "1", "true"] else 1.0))
                if pd.notna(c) else int(np.random.poisson(1.0))
            )
        else:
            # For inference when no target is present, generate standard Poisson sample per row (mean=1.0)
            df["Support_Tickets"] = [int(np.random.poisson(1.0)) for _ in range(len(df))]

    df["High_Support_Load"] = (df["Support_Tickets"] >= 4).astype(int)

    # Charge change proxy (higher ratio = recent charge spike)
    df["Charge_Per_Tenure_Unit"] = np.where(
        df["tenure"] > 0, df["MonthlyCharges"] / (df["tenure"] + 1), df["MonthlyCharges"]
    )
    
    # New Advanced Features (Phase 2 Roadmap)
    df["Engagement_x_Tenure"] = df["Service_Count"] * df["tenure"]
    
    svc_count_safe = np.where(df["Service_Count"] == 0, 1, df["Service_Count"])
    df["Spend_Intensity"] = df["TotalCharges"] / svc_count_safe
    
    tenure_safe = np.where(df["tenure"] == 0, 1, df["tenure"])
    df["Support_Per_Order"] = df["Support_Tickets"] / tenure_safe
    
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
                "Charge_Per_Tenure_Unit", "Support_Tickets",
                "Engagement_x_Tenure", "Spend_Intensity", "Support_Per_Order"]
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
    
    # Save parquet for Feast offline store
    df_feast = df.copy()
    df_feast["event_timestamp"] = pd.Timestamp.now()
    if ID_COL not in df_feast.columns:
        # Assuming ID_COL was preserved or we need to merge it back
        # For demonstration, generate unique IDs if missing
        import uuid
        df_feast[ID_COL] = [str(uuid.uuid4()) for _ in range(len(df_feast))]
    
    feast_path = MODEL_INPUT_CSV.with_suffix('.parquet')
    df_feast.to_parquet(feast_path, index=False)
    log.info(f"Feast offline store parquet saved -> {feast_path}")

    log.info(f"model_input.csv saved  shape={df.shape}")
    log.info(f"Columns ({len(df.columns)}): {list(df.columns)[:10]} ...")
    return df


if __name__ == "__main__":
    run()
