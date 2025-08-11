import os
import logging
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime

# Setup logging
logging.basicConfig(filename="logs/server.log", level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(title="Energy Forecasting API")

# Detect environment
is_azure = "AZUREML_EXPERIMENT_ID" in os.environ or "AZUREML_RUN_ID" in os.environ

if is_azure:
    logging.info("Running in Azure ML environment")
else:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:mlruns"))
    logging.info("Running in local environment")

# Set default model path
default_model_path = (
    "models:/transformer_load_forecast/Production"
    if is_azure else
    "models:/transformer_load_forecast@production"
)

model_path = os.getenv("MODEL_PATH", default_model_path)
logging.info(f"Loading model from: {model_path}")

# Load model
try:
    model = mlflow.pyfunc.load_model(model_path)
except Exception as e:
    logging.exception("Model loading failed")
    model = None

# Define input schema
class InputRecord(BaseModel):
    ds: datetime  # Prophet expects datetime column named 'ds'

@app.post("/predict")
def predict(data: List[InputRecord]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Build DataFrame
        input_df = pd.DataFrame([{"ds": r.ds} for r in data])

        # Normalize ds: make UTC-aware then drop tz to satisfy MLflow schema
        # - Handles both naive and tz-aware inputs consistently
        input_df["ds"] = pd.to_datetime(input_df["ds"], utc=True).dt.tz_localize(None)

        forecast = model.predict(input_df)

        # Return yhat column if available
        if isinstance(forecast, pd.DataFrame) and "yhat" in forecast.columns:
            # Optional: include ds alongside yhat for traceability
            return [
                {"ds": ds.isoformat(), "yhat": float(y)}
                for ds, y in zip(input_df["ds"], forecast["yhat"])
            ]

        # Fallback: return all numeric columns
        return forecast.select_dtypes(include="number").to_dict(orient="list")

    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/version")
def version():
    return {"model_path": model_path}