# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Training image ─────────────────────────────────────────────────────────────
FROM base AS trainer
COPY . .
CMD ["python", "master_pipeline.py"]

# ── API image ─────────────────────────────────────────────────────────────────
FROM base AS api
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.dashboard.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Dashboard image ───────────────────────────────────────────────────────────
FROM base AS dashboard
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
