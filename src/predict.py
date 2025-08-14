import json
import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient
import os

print("📂 Current working directory:", os.getcwd())

# Detect environment
is_azure = "AZUREML_EXPERIMENT_ID" in os.environ or "AZUREML_RUN_ID" in os.environ

print("Running in Azure ML:", is_azure)

# Load model from MLflow
model = mlflow.pyfunc.load_model("models:/transformer_load_forecast@production")

# Load feature list from artifact
client = MlflowClient()
model_version = client.get_model_version_by_alias("transformer_load_forecast", "production")
feature_path = mlflow.artifacts.download_artifacts(run_id=model_version.run_id, artifact_path="model_features.json")

with open(feature_path, "r") as f:
    feature_cols = json.load(f)

# Generate hourly timestamps from 2025 to 2030
timestamps = pd.date_range(start="2025-01-01", end="2030-01-01", freq="h", inclusive="left")

# Prepare input DataFrame for Prophet
df_input = pd.DataFrame({"ds": timestamps})

# ProphetWrapper expects a DataFrame with 'ds' column only
forecast = model.predict(df_input)

# Extract predictions
df_input["predicted_kwh"] = forecast["yhat"]

# Save results
df_input.to_csv("predicted_kwh.csv", index=False)

os.makedirs("outputs", exist_ok=True)
df_input.to_csv("outputs/predicted_kwh.csv", index=False)

# Check for transformer overload
transformer_limit = 85.0
overload = df_input[df_input["predicted_kwh"] > transformer_limit]

if not overload.empty:
    print(f"⚠️ Transformer overload predicted on: {overload.iloc[0]['ds']}")
else:
    print("✅ No overload predicted between 2025 and 2029.")

# Plot forecast
plt.figure(figsize=(14, 6))
plt.plot(df_input["ds"], df_input["predicted_kwh"], label="Predicted kWh", color="steelblue")
plt.xlabel("Timestamp")
plt.ylabel("Predicted kWh")
plt.title("Hourly Energy Forecast (2025–2030)")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/predicted_kwh_trend.png")

if is_azure:
    # Define blob paths
    storage_account_url = "https://transformerloadstorage.blob.core.windows.net"
    container_name = "predictions"
    csv_blob_name = "predicted_kwh.csv"
    plot_blob_name = "predicted_kwh_trend.png"
    overload_blob_name = "overload_events.csv"

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
                print(f"📤 Uploaded {blob_name} to {blob.url}.")
        except Exception as e:
            print(f"❌ Failed to upload {blob_name} to {blob.url}: {e}")

    upload_to_blob(csv_blob_name)
    upload_to_blob(plot_blob_name)

    if not overload.empty:
        overload.to_csv(f"outputs/{overload_blob_name}", index=False)
        upload_to_blob(overload_blob_name)

    print("✅ Forecast results uploaded to blob storage.")
