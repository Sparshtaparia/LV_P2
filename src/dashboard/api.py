"""
FastAPI inference service.
Endpoints: GET /health | POST /predict | POST /explain | POST /segment
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib, shap
import time
from prometheus_client import make_asgi_app, Counter, Histogram

from src.features.build_features import add_rfm_features, add_behavioral_features, encode_and_scale
from src.dashboard.auth import get_current_user

from config.config import CALIBRATED_MODEL, FEATURE_COLS_PATH, CLUSTER_MODEL_PATH

app = FastAPI(
    title="Churn Intelligence API",
    description="Production-grade churn prediction & explainability endpoints",
    version="4.0",
)

# Prometheus metrics
app.mount("/metrics", make_asgi_app())
PREDICTION_COUNTER = Counter("churn_predictions_total", "Total number of predictions made")
PREDICTION_LATENCY = Histogram("churn_prediction_latency_seconds", "Latency of prediction requests")

# Lazy-load models once at startup
_model    = None
_f_cols   = None
_km       = None
_explainer= None

def _load():
    global _model, _f_cols, _km, _explainer
    if _model is None:
        _model  = joblib.load(CALIBRATED_MODEL)
        _f_cols = joblib.load(FEATURE_COLS_PATH)
        try:
            _km = joblib.load(CLUSTER_MODEL_PATH)
        except Exception:
            _km = None
        try:
            _explainer = shap.TreeExplainer(_model.calibrated_classifiers_[0].estimator)
        except Exception:
            _explainer = None


class CustomerIn(BaseModel):
    tenure:           float = 12
    MonthlyCharges:   float = 65.0
    TotalCharges:     float = 780.0
    Contract:         str   = "Month-to-month"
    InternetService:  str   = "Fiber optic"
    OnlineSecurity:   str   = "No"
    TechSupport:      str   = "No"
    PaperlessBilling: str   = "Yes"
    PaymentMethod:    str   = "Electronic check"
    SeniorCitizen:    int   = 0
    Partner:          str   = "No"
    Dependents:       str   = "No"
    extra_features:   Optional[dict] = None


def _build_df(c: CustomerIn) -> pd.DataFrame:
    _load()
    raw_data = {
        "tenure": c.tenure,
        "MonthlyCharges": c.MonthlyCharges,
        "TotalCharges": c.TotalCharges,
        "Contract": c.Contract,
        "InternetService": c.InternetService,
        "OnlineSecurity": c.OnlineSecurity,
        "TechSupport": c.TechSupport,
        "PaperlessBilling": c.PaperlessBilling,
        "PaymentMethod": c.PaymentMethod,
        "SeniorCitizen": c.SeniorCitizen,
        "Partner": c.Partner,
        "Dependents": c.Dependents,
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "TARGET": "No"
    }
    
    if c.extra_features:
        raw_data.update(c.extra_features)
        
    df_raw = pd.DataFrame([raw_data])
    df_raw = add_rfm_features(df_raw)
    df_raw = add_behavioral_features(df_raw)
    df_processed = encode_and_scale(df_raw, fit=False)
    
    final_row = {}
    for col in _f_cols:
        if col in df_processed.columns:
            final_row[col] = df_processed[col].iloc[0]
        else:
            final_row[col] = 0
            
    return pd.DataFrame([final_row])


@app.get("/health")
def health():
    return {"status": "ok", "version": "4.0"}


@app.post("/predict")
def predict(customer: CustomerIn, user: dict = Depends(get_current_user)):
    start_time = time.time()
    _load()
    X   = _build_df(customer)
    p   = float(_model.predict_proba(X)[0, 1])
    lbl = int(p >= 0.5)
    
    PREDICTION_COUNTER.inc()
    PREDICTION_LATENCY.observe(time.time() - start_time)
    
    return {
        "churn_probability": round(p, 4),
        "churn_label":       lbl,
        "risk_band":         "high" if p >= 0.7 else ("medium" if p >= 0.4 else "low"),
    }


@app.post("/explain")
def explain(customer: CustomerIn, user: dict = Depends(get_current_user)):
    _load()
    X = _build_df(customer)
    p = float(_model.predict_proba(X)[0, 1])

    top_features = {}
    if _explainer is not None:
        try:
            sv = _explainer.shap_values(X)
            if isinstance(sv, list): sv = sv[1]
            elif sv.ndim == 3:       sv = sv[:, :, 1]
            sv_flat = sv[0]
            idx     = np.argsort(np.abs(sv_flat))[::-1][:5]
            top_features = {_f_cols[i]: round(float(sv_flat[i]), 4) for i in idx}
        except Exception as e:
            top_features = {"error": str(e)}
    else:
        # Fallback: feature importances if available
        try:
            base = _model.calibrated_classifiers_[0].estimator
            fi   = base.feature_importances_
            idx  = np.argsort(fi)[::-1][:5]
            top_features = {_f_cols[i]: round(float(fi[i]), 4) for i in idx}
        except Exception:
            pass

    return {
        "churn_probability": round(p, 4),
        "top_risk_drivers":  top_features,
    }


@app.post("/segment")
def segment(customer: CustomerIn, user: dict = Depends(get_current_user)):
    _load()
    X = _build_df(customer)
    p = float(_model.predict_proba(X)[0, 1])

    seg_id   = -1
    seg_name = "Unknown"
    if _km is not None:
        try:
            seg_id = int(_km.predict(X)[0])
            SEGMENT_NAMES = {
                0: "High-Value Loyal",
                1: "At-Risk Decliners",
                2: "New Explorers",
                3: "Price-Sensitive",
                4: "Long-Tenured Stable",
                5: "Churned Likely",
            }
            seg_name = SEGMENT_NAMES.get(seg_id, f"Segment {seg_id}")
        except Exception:
            pass

    return {
        "churn_probability": round(p, 4),
        "segment_id":        seg_id,
        "segment_name":      seg_name,
    }
