"""
Central configuration for the Churn Platform.
All paths, hyperparameters, and constants live here.
"""
import os
from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR   = ROOT_DIR / "docs"
PLOTS_DIR  = DOCS_DIR / "plots"
LOGS_DIR   = ROOT_DIR / "logs"

for d in [PROC_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
RAW_CSV          = RAW_DIR  / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
AUGMENTED_CSV    = PROC_DIR / "telco_churn_augmented.csv"
MODEL_INPUT_CSV  = PROC_DIR / "model_input.csv"
DB_PATH          = PROC_DIR / "churn_db.sqlite"
DATABASE_URL     = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# ── Model artefacts ──────────────────────────────────────────────────────────
BEST_MODEL_PATH    = MODELS_DIR / "best_model.pkl"
CALIBRATED_MODEL   = MODELS_DIR / "calibrated_model.pkl"
SCALER_PATH        = MODELS_DIR / "scaler.pkl"
ENCODER_PATH       = MODELS_DIR / "encoder.pkl"
CLUSTER_MODEL_PATH = MODELS_DIR / "kmeans_cluster.pkl"
FEATURE_COLS_PATH  = MODELS_DIR / "feature_columns.pkl"

# ── MLflow ───────────────────────────────────────────────────────────────────
_MLRUNS_DIR = ROOT_DIR / "mlruns"
_MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", _MLRUNS_DIR.as_uri())
MLFLOW_EXPERIMENT   = "Churn_Prediction_v4"

# ── Training ─────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
VAL_SIZE      = 0.10
TARGET_COL    = "Churn"
ID_COL        = "customerID"
N_CLUSTERS    = 6          # K-Means segments
SMOTE_RATIO   = 0.5

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("PORT", 8000))
SECRET_KEY = os.getenv("SECRET_KEY", "churn-platform-secret-2026")

# ── Numeric & categorical columns ────────────────────────────────────────────
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
