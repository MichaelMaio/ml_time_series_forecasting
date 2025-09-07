# Standard library.
import json
import os
import pickle
import shutil
import tempfile

# Third-party.
import mlflow
import mlflow.pyfunc
from mlflow.artifacts import download_artifacts
from mlflow.models.signature import infer_signature
import pandas as pd
from prophet import Prophet
from sklearn.metrics import root_mean_squared_error

# Azure SDK.
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobClient
from azureml.core import Run

# Local modules.
from prophet_wrapper import ProphetWrapper

print("\n*** STARTING TRAINING SCRIPT ***")
print("Current working directory:", os.getcwd())

# Detect if we're running in Azure ML.
is_azure = "AZUREML_EXPERIMENT_ID" in os.environ or "AZUREML_RUN_ID" in os.environ
print("Running in Azure ML:", is_azure)

# Set tracking URI if doing a local run.
if not is_azure:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/mlflow/mlruns"))

print("MLflow Tracking URI:", mlflow.get_tracking_uri())

# Ensure experiment exists.
if not is_azure:

    experiment_name = "transformer-load-exp"
    client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(experiment_name)

    if existing is None:
        client.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)

# Disable autologging (Prophet isn't natively supported)
mlflow.autolog(disable=True)

if is_azure:
    
    # Training data will be downloaded from blob storage.
    blob_uri = "https://transformerloadstorage.blob.core.windows.net/training-data/peak_load.csv"
    print(f"Downloading training data from blob: {blob_uri}")

    # Get the managed identity client ID from environment variable.
    client_id = os.environ.get("MANAGED_IDENTITY_CLIENT_ID")

    if not client_id:
        raise RuntimeError("MANAGED_IDENTITY_CLIENT_ID environment variable not set.")

    # Create the managed identity credential..
    credential = ManagedIdentityCredential(client_id=client_id)

    # Create the blob client and download the blob to a temporary file.
    blob_client = BlobClient.from_blob_url(blob_uri, credential=credential)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        blob_client.download_blob().readinto(tmp)
        data_path = tmp.name

else:
    # Training data will be read from local path.
    data_path = "data/peak_load.csv"

# Load the training data.
print(f"Loading training data from local path {data_path}")
df = pd.read_csv(data_path, parse_dates=["timestamp"])

# Prepare the training data for Prophet.
df_prophet = df[["timestamp", "kw"]].rename(columns={"timestamp": "ds", "kw": "y"})

# Split the data into the training, validation, and test sets (80/10/10).
split_1 = int(len(df_prophet) * 0.8)
split_2 = int(len(df_prophet) * 0.9)
train_df = df_prophet.iloc[:split_1]
val_df = df_prophet.iloc[split_1:split_2]
test_df = df_prophet.iloc[split_2:]

# Start the MLflow run.
print("Starting the MLflow run.")

if is_azure:
    # In Azure, use the existing Run context.
    run = Run.get_context()
else:
    # Suppress MLflow-related git warnings.
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    mlflow.start_run(run_name="prophet_load_forecast")

# Create and train the model.
print("Training the model.")

model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode="additive"
)

model.fit(train_df)

# Validate the model.
print("Validating the model.")
val_future = val_df[["ds"]].copy()
val_forecast = model.predict(val_future)

# Calculate the validation rmse and log it.
val_rmse = root_mean_squared_error(val_df["y"], val_forecast["yhat"])
mlflow.log_metric("val_rmse", val_rmse)
print(f"Validation RMSE is {val_rmse:.2f} kw.")

RMSE_THRESHOLD = 3.0

if val_rmse >= RMSE_THRESHOLD:
    raise RuntimeError(f"Validation RMSE {val_rmse:.2f} exceeds threshold of {RMSE_THRESHOLD}.")

# Test the model.
print("Testing the model.")
test_future = test_df[["ds"]].copy()
test_forecast = model.predict(test_future)

# Calculate the test rmse and log it.
rmse = root_mean_squared_error(test_df["y"], test_forecast["yhat"])
mlflow.log_metric("rmse", rmse)
print(f"Test RMSE is {rmse:.2f} kw.")

# Dump the model locally.
print("Dumping the model.")
model_path = "transformer_load_model_prophet.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Dump the feature list locally.
print("Dumping the feature list.")
feature_path = "model_features.json"

with open(feature_path, "w") as f:
    json.dump(["ds"], f)

# Log the model and feature list as artifacts.
print("Logging model artifacts.")
mlflow.log_artifact(model_path)
mlflow.log_artifact(feature_path)

# Set tags for the model.
print("Setting model tags.")
mlflow.set_tag("model_type", "Prophet")
mlflow.set_tag("use_case", "Electrical Load Forecasting")
mlflow.set_tag("owner", "Michael Maio")

# Infer the model signature from the inputs and outputs.
future_df = train_df[["ds"]].copy()
signature = infer_signature(future_df, model.predict(future_df)[["yhat"]])

# Log the model.
print("Logging the model.")

if is_azure:
    
    run.tag("stage", "training")

    mlflow.pyfunc.log_model(
        artifact_path="trained_model",
        python_model=ProphetWrapper(),
        artifacts={"model": model_path},
        signature=signature
    )

    # Get the model uri so we can download it to the output dataset.
    model_uri = f"runs:/{run.id}/trained_model"
    print(f"Model URI is: {model_uri}")

    # Download the model artifacts to a local path.
    local_path = download_artifacts(model_uri)
    print(f"Model local path is: {local_path}")
    print("Local model contents:", os.listdir(local_path))

    # Get the model output path from the output dataset.
    model_output_path = run.output_datasets["trained_model"]
    print(f"Model output path is: {model_output_path}")

    # Copy the downloaded model to the output path.
    # This allows it to be picked up by the next step in the pipeline (promotion).
    shutil.copytree(local_path, model_output_path, dirs_exist_ok=True)
    print("Copied model contents:", os.listdir(model_output_path))

else:
    
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=ProphetWrapper(),
        artifacts={"model": model_path},
        registered_model_name="transformer_load_forecast",
        signature=signature
    )

    mlflow.end_run()