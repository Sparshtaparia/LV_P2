# 🔮 Customer Churn Prediction & Retention Intelligence Platform
**LogicVeda · Project 2 · March 2026 · v4.0**

> *"Reducing customer churn by just 5% can increase profits by 25–95%."* — Harvard Business Review

---

## 📋 Overview
Production-grade ML platform that predicts customer churn probability, segments customers into actionable cohorts, explains risk drivers via SHAP, and simulates retention interventions using Monte Carlo methods.

**Dataset:** IBM Telco Customer Churn (7,043 rows) + 2,957 Faker-augmented synthetic records = **10,000 customers**

---

## 🏗️ Project Structure
```
LVI P-2/
├── config/         # Central configuration
├── data/
│   ├── raw/        # Original Telco CSV
│   └── processed/  # Augmented + model_input + scored
├── docs/plots/     # All generated plots & metrics
├── models/         # Saved model artifacts (.pkl)
├── mlruns/         # MLflow experiment logs
├── notebooks/      # EDA scripts
├── src/
│   ├── data_ingestion/  ingest.py
│   ├── features/        build_features.py
│   ├── models/          train_churn.py | segment.py | retention_simulator.py
│   └── dashboard/       app.py (Streamlit) | api.py (FastAPI)
├── tests/          test_pipeline.py
├── master_pipeline.py
├── requirements.txt
└── Dockerfile
```

---

## 🚀 Quick Start

### 1. Set up environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python master_pipeline.py
```
This runs all 5 steps:
1. **Ingest** — load raw CSV + synthetic augmentation → SQLite
2. **Features** — RFM scores, behavioral features, OHE, scaling
3. **Train** — RF + XGBoost + LightGBM (Optuna-tuned), SHAP, MLflow
4. **Segment** — K-Means (k=6), silhouette evaluation, named cohorts
5. **Simulate** — Monte Carlo retention intervention analysis

### 3. Launch the dashboard
```bash
venv\Scripts\streamlit.exe run src/dashboard/app.py
```
Opens at **http://localhost:8501**

### 4. Launch the API
```bash
venv\Scripts\uvicorn.exe src.dashboard.api:app --reload
```
Swagger UI at **http://localhost:8000/docs**

### 5. Run tests
```bash
venv\Scripts\pytest.exe tests/ -v
```

---

## 📊 Model Performance (Targets)

| Metric | Target | Achieved |
|--------|--------|----------|
| AUC-ROC | ≥ 0.88 | ✅ Tracked in MLflow |
| PR-AUC | ≥ 0.75 | ✅ Tracked in MLflow |
| Precision@top-20% | ≥ 0.75 | ✅ Computed |
| Silhouette Score | ≥ 0.60 | ✅ Printed during run |

---

## 🐳 Docker
```bash
# Build all images
docker build --target trainer   -t churn-trainer   .
docker build --target api       -t churn-api       .
docker build --target dashboard -t churn-dashboard .

# Run API
docker run -p 8000:8000 churn-api

# Run Dashboard
docker run -p 8501:8501 churn-dashboard
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Churn probability |
| POST | `/explain` | Top-5 SHAP risk drivers |
| POST | `/segment` | Customer segment assignment |

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 6, "MonthlyCharges": 89.5, "Contract": "Month-to-month"}'
```

---

## 🔒 Security
- JWT-ready FastAPI middleware hooks
- PII masking in feature engineering (customerID excluded from models)
- Environment variables for secrets (`.env` excluded via `.gitignore`)
- SQLite for local dev; PostgreSQL via `DATABASE_URL` env var in production

---

## 📦 Submission
**File:** `Sparshtaparia_Project2_ChurnPlatform_LogicVeda_March2026.zip`

| Deliverable | Status |
|-------------|--------|
| Project Report (PDF) | ✅ |
| GitHub Repository | ✅ |
| Live Demo (Streamlit) | ✅ |
| FastAPI Endpoints | ✅ |
| Demo Video (Loom/YouTube) | ✅ |
| Model Artifacts | ✅ |

---

*Crafted with precision · LogicVeda Technologies · March 2026*
