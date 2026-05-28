# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Create a non-privileged user (UID 1000) for Hugging Face security compliance
RUN useradd -m -u 1000 user
WORKDIR /app

# Pre-install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all repository files and change ownership to user
COPY --chown=user . /app

# Switch to user context
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Hugging Face Spaces binds to port 7860 by default
EXPOSE 7860

# Start the FastAPI serving app on port 7860
CMD ["uvicorn", "src.dashboard.api:app", "--host", "0.0.0.0", "--port", "7860"]
