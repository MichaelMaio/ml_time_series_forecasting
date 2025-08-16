import json
import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient
import os
from azureml.core import Workspace, Model, Run

print("Current working directory:", os.getcwd())

# Detect environment
is_azure = "AZUREML_EXPERIMENT_ID" in os.environ or "AZUREML_RUN_ID" in os.environ

print("Running in Azure ML:", is_azure)

# Load model from MLflow
if is_azure:

    run = Run.get_context()
    ws = run.experiment.workspace

    model_input_path = run.input_datasets["model_input"]
    print(f"model_input is {model_input_path}")

    if not os.path.exists(model_input_path):
        raise RuntimeError(f"Model input path not found: {model_input_path}")

    required_files = ["model_features.json", "transformer_load_model_prophet.pkl"]

    for fname in required_files:
        fpath = os.path.join(model_input_path, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Expected artifact missing: {fpath}")

    model = mlflow.pyfunc.load_model(model_input_path)

    print("Model metadata:", model.metadata.to_dict())
    print("Model input schema:", model.metadata.signature.inputs)

else:
    model = mlflow.pyfunc.load_model("models:/transformer_load_forecast@production")

# Load feature list from artifact
if is_azure:
    feature_path = os.path.join(model_path, "model_features.json")
else:
    client = MlflowClient()
    model_version = client.get_model_version_by_alias("transformer_load_forecast", "production")

    print(f"Using model version: {model_version.version}, run ID: {model_version.run_id}")

    feature_path = mlflow.artifacts.download_artifacts(run_id=model_version.run_id, artifact_path="model_features.json")

    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Expected artifact missing: {feature_path}")

with open(feature_path, "r") as f:
    feature_cols = json.load(f)

# Generate hourly timestamps from 2025 to 2030
timestamps = pd.date_range(start="2025-01-01", end="2030-01-01", freq="h", inclusive="left")

# Prepare input DataFrame for Prophet
df_input = pd.DataFrame({"ds": timestamps})

# ProphetWrapper expects a DataFrame with 'ds' column only
print("Running predictions.")
forecast = model.predict(df_input)

# Extract predictions
df_input["predicted_kwh"] = forecast["yhat"]

# Save results
print("Saving predictions.")
os.makedirs("outputs", exist_ok=True)
df_input.to_csv("outputs/predicted_kwh.csv", index=False)

# Check for transformer overload
print("Checking for transformer overload.")
transformer_limit = 85.0
overload = df_input[df_input["predicted_kwh"] > transformer_limit]

if not overload.empty:
    print(f"Transformer overload predicted on: {overload.iloc[0]['ds']}")
else:
    print("No overload predicted between 2025 and 2029.")

# Plot forecast
print("Plotting predictions.")
plt.figure(figsize=(14, 6))
plt.plot(df_input["ds"], df_input["predicted_kwh"], label="Predicted kWh", color="steelblue")
plt.xlabel("Timestamp")
plt.ylabel("Predicted kWh")
plt.title("Hourly Energy Forecast (2025–2030)")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/predicted_kwh_trend.png")

if is_azure:
    print("Uploading blobs.")

    # Define blob paths
    storage_account_url = "https://transformerloadstorage.blob.core.windows.net"
    container_name = "predictions"
    csv_blob_name = "predicted_kwh.csv"
    plot_blob_name = "predicted_kwh_trend.png"
    overload_blob_name = "overload_events.csv"

    print(f"Uploading to container '{container_name}' in account '{storage_account_url}'")

    # Authenticate
    credential = DefaultAzureCredential()

    def upload_to_blob(blob_name):
        blob = BlobClient(
            account_url=storage_account_url,
            container_name=container_name,
            blob_name=blob_name,
            credential=credential
        )

        local_path = f"outputs/{blob_name}"
        
        try:
            with open(local_path, "rb") as f:
                blob.upload_blob(f, overwrite=True)
                print(f"Uploaded {blob_name} to {blob.url}.")
        except Exception as e:
            print(f"Failed to upload {blob_name} to {blob.url}: {e}")

    upload_to_blob(csv_blob_name)
    upload_to_blob(plot_blob_name)

    if not overload.empty:
        overload.to_csv(f"outputs/{overload_blob_name}", index=False)
        upload_to_blob(overload_blob_name)

    print("Forecast results uploaded to blob storage.")

    predictions_path = run.output_datasets["predictions"]
  
    df_input.to_csv(os.path.join(predictions_path, "predicted_kwh.csv"), index=False)
    plt.savefig(os.path.join(predictions_path, "predicted_kwh_trend.png"))

    if not overload.empty:
        overload.to_csv(os.path.join(predictions_path, "overload_events.csv"), index=False)

    print("Forecast results saved to AzureML output folder.")
