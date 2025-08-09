FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference logic and wrapper
COPY inference.py .
COPY prophet_wrapper.py .

# Optional: set MLflow tracking URI here or via docker-compose
# ENV MLFLOW_TRACKING_URI=file:/mlflow/mlruns

CMD ["gunicorn", "inference:app", "--bind", "0.0.0.0:5000", "--workers", "2"]