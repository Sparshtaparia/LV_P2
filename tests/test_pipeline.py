"""
Unit tests for core pipeline components.
Run: venv\Scripts\pytest.exe tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np


# ── Test data ingestion ────────────────────────────────────────────────────────
def test_raw_csv_exists():
    from config.config import RAW_CSV
    assert RAW_CSV.exists(), f"Raw CSV not found: {RAW_CSV}"


def test_augmented_csv_exists():
    from config.config import AUGMENTED_CSV
    assert AUGMENTED_CSV.exists(), "Run ingest.py first"


def test_augmented_shape():
    from config.config import AUGMENTED_CSV
    df = pd.read_csv(AUGMENTED_CSV)
    assert len(df) >= 7043, "Augmented dataset should have ≥7043 rows"
    assert "Churn" in df.columns


# ── Test feature engineering ──────────────────────────────────────────────────
def test_model_input_exists():
    from config.config import MODEL_INPUT_CSV
    assert MODEL_INPUT_CSV.exists(), "Run build_features.py first"


def test_model_input_no_nulls():
    from config.config import MODEL_INPUT_CSV, TARGET_COL
    df = pd.read_csv(MODEL_INPUT_CSV)
    assert df[TARGET_COL].isnull().sum() == 0, "Target column has nulls"


def test_churn_binary():
    from config.config import MODEL_INPUT_CSV, TARGET_COL
    df = pd.read_csv(MODEL_INPUT_CSV)
    assert set(df[TARGET_COL].unique()).issubset({0, 1}), "Churn must be binary 0/1"


# ── Test model artifacts ───────────────────────────────────────────────────────
def test_best_model_exists():
    from config.config import BEST_MODEL_PATH
    assert BEST_MODEL_PATH.exists(), "Run train_churn.py first"


def test_calibrated_model_exists():
    from config.config import CALIBRATED_MODEL
    assert CALIBRATED_MODEL.exists(), "Calibrated model not found"


def test_model_predict():
    from config.config import CALIBRATED_MODEL, FEATURE_COLS_PATH
    import joblib
    model = joblib.load(CALIBRATED_MODEL)
    f_cols = joblib.load(FEATURE_COLS_PATH)
    X = pd.DataFrame([{c: 0 for c in f_cols}])
    proba = model.predict_proba(X)[0, 1]
    assert 0.0 <= proba <= 1.0, "Probability out of [0,1] range"


# ── Test scored output ────────────────────────────────────────────────────────
def test_scored_csv_exists():
    from config.config import PROC_DIR
    assert (PROC_DIR / "customers_scored.csv").exists(), "Run segment.py first"


def test_scored_columns():
    from config.config import PROC_DIR
    df = pd.read_csv(PROC_DIR / "customers_scored.csv")
    for col in ["churn_probability", "churn_label", "segment_name"]:
        assert col in df.columns, f"Missing column: {col}"


def test_churn_prob_range():
    from config.config import PROC_DIR
    df = pd.read_csv(PROC_DIR / "customers_scored.csv")
    assert df["churn_probability"].between(0, 1).all(), "Probabilities out of range"


# ── Test retention simulator ──────────────────────────────────────────────────
def test_retention_csv_exists():
    from config.config import PLOTS_DIR
    assert (PLOTS_DIR / "retention_simulation.csv").exists(), "Run retention_simulator.py first"


def test_retention_results():
    from config.config import PLOTS_DIR
    df = pd.read_csv(PLOTS_DIR / "retention_simulation.csv")
    assert len(df) == 4, "Should have 4 intervention results"
    assert "mc_mean_reduction" in df.columns
    assert (df["mc_mean_reduction"] > 0).all(), "All reductions should be > 0"
