import json
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.pyfunc
import pandas as pd
import pickle
from sklearn.metrics import root_mean_squared_error
import os
from prophet import Prophet
from prophet_wrapper import ProphetWrapper  # ✅ Custom wrapper
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient
import tempfile

print("📂 Current working directory:", os.getcwd())

# Detect environment
is_azure = "AZUREML_EXPERIMENT_ID" in os.environ or "AZUREML_RUN_ID" in os.environ

print("Running in Azure ML:", is_azure)

experiment_name = "transformer-load-exp"

# ✅ Set tracking URI for Docker container
if not is_azure:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/mlflow/mlruns"))

print("MLflow Tracking URI:", mlflow.get_tracking_uri())

# Ensure experiment exists
if not is_azure:
    client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(experiment_name)
    if existing is None:
        client.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

print("Tracking URI:", mlflow.get_tracking_uri())

# Disable autologging (Prophet isn't natively supported)
mlflow.autolog(disable=True)

# Download blob from storage if running in Azure or from the inputs folder if training locally.
if is_azure:
    blob_uri = "https://transformerloadstorage.blob.core.windows.net/training-data/peak_load.csv"
    print(f"📦 Loading data from blob: {blob_uri}")
    credential = DefaultAzureCredential()    
    blob_client = BlobClient.from_blob_url(blob_uri, credential=credential)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        blob_client.download_blob().readinto(tmp)
        data_path = tmp.name
else:
    print("📁 Loading data from local path")
    data_path = "data/peak_load.csv"

df = pd.read_csv(data_path, parse_dates=["timestamp"])

# Feature engineering
df["hour"] = df["hour"].astype(int)
df["is_weekend"] = df["is_weekend"].astype(int)
df = pd.get_dummies(df, columns=["season", "day_of_week"], drop_first=True)

# Prepare data for Prophet
df_prophet = df[["timestamp", "kwh"]].rename(columns={"timestamp": "ds", "kwh": "y"})

# Train/val/test split
split_1 = int(len(df_prophet) * 0.8)
split_2 = int(len(df_prophet) * 0.9)
train_df = df_prophet.iloc[:split_1]
val_df = df_prophet.iloc[split_1:split_2]
test_df = df_prophet.iloc[split_2:]

# Start MLflow run
with mlflow.start_run(run_name="prophet_load_forecast"):
    # Train model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive"
    )
    model.fit(train_df)

    # Forecast on test set
    future = test_df[["ds"]].copy()
    forecast = model.predict(future)

    # Evaluate
    y_true = test_df["y"].values
    y_pred = forecast["yhat"].values
    rmse = root_mean_squared_error(y_true, y_pred)
    mlflow.log_metric("rmse", rmse)

    # Save model locally inside container
    model_path = "transformer_load_model_prophet.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save feature list
    feature_path = "model_features.json"
    with open(feature_path, "w") as f:
        json.dump(["ds"], f)

    # Log artifacts
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(feature_path)

    # Set MLflow tags
    mlflow.set_tag("model_type", "Prophet")
    mlflow.set_tag("use_case", "Energy Load Forecasting")
    mlflow.set_tag("owner", "Michael Maio")

    # Infer model signature using only 'ds' as input
    future_df = train_df[["ds"]].copy()
    signature = infer_signature(future_df, model.predict(future_df)[["yhat"]])

    # ✅ Log and register the model using the wrapper
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ProphetWrapper(),
        artifacts={"model": model_path},
        registered_model_name="transformer_load_forecast",
        signature=signature
    )

    print(f"✅ Prophet model trained and logged with RMSE: {rmse:.2f} kWh")