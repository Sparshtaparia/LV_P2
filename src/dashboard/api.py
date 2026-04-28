"""
FastAPI inference service.
Endpoints: GET /health | POST /predict | POST /explain | POST /segment
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib, shap

from config.config import CALIBRATED_MODEL, FEATURE_COLS_PATH, CLUSTER_MODEL_PATH

app = FastAPI(
    title="Churn Intelligence API",
    description="Production-grade churn prediction & explainability endpoints",
    version="4.0",
)

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
    row = {col: 0 for col in _f_cols}
    # Numeric (z-score approximation using training stats)
    row["tenure"]          = (c.tenure - 32) / 24
    row["MonthlyCharges"]  = (c.MonthlyCharges - 64) / 30
    row["TotalCharges"]    = (c.TotalCharges - 2280) / 2266

    # Contract
    if c.Contract == "Month-to-month":
        for k in ["Contract_Month-to-month", "Is_Month_to_Month"]:
            if k in row: row[k] = 1
    elif c.Contract == "Two year":
        for k in ["Contract_Two year", "Is_Two_Year"]:
            if k in row: row[k] = 1

    # Internet
    if c.InternetService == "Fiber optic":
        for k in ["InternetService_Fiber optic", "Has_Fiber"]:
            if k in row: row[k] = 1
    elif c.InternetService == "DSL":
        for k in ["InternetService_DSL", "Has_DSL"]:
            if k in row: row[k] = 1

    # Security / support
    for col, val, prefix in [
        ("OnlineSecurity", c.OnlineSecurity, "OnlineSecurity_"),
        ("TechSupport",    c.TechSupport,    "TechSupport_"),
        ("PaymentMethod",  c.PaymentMethod,  "PaymentMethod_"),
    ]:
        k = prefix + val
        if k in row: row[k] = 1

    row["Is_Paperless"]     = 1 if c.PaperlessBilling == "Yes" else 0
    row["Is_Electronic_Pay"]= 1 if c.PaymentMethod == "Electronic check" else 0
    row["Is_Senior"]        = c.SeniorCitizen
    row["Has_Partner"]      = 1 if c.Partner == "Yes" else 0
    row["Has_Dependents"]   = 1 if c.Dependents == "Yes" else 0

    # Extra features override
    if c.extra_features:
        for k, v in c.extra_features.items():
            if k in row: row[k] = v

    return pd.DataFrame([row])


@app.get("/health")
def health():
    return {"status": "ok", "version": "4.0"}


@app.post("/predict")
def predict(customer: CustomerIn):
    _load()
    X   = _build_df(customer)
    p   = float(_model.predict_proba(X)[0, 1])
    lbl = int(p >= 0.5)
    return {
        "churn_probability": round(p, 4),
        "churn_label":       lbl,
        "risk_band":         "high" if p >= 0.7 else ("medium" if p >= 0.4 else "low"),
    }


@app.post("/explain")
def explain(customer: CustomerIn):
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
def segment(customer: CustomerIn):
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
