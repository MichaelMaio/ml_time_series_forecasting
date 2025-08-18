import json
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobClient
import os
from azureml.core import Run

print("Current working directory:", os.getcwd())

# Detect environment
is_azure = "AZUREML_EXPERIMENT_ID" in os.environ or "AZUREML_RUN_ID" in os.environ

print("Running in Azure ML:", is_azure)

# Load model from MLflow
if is_azure:

    run = Run.get_context()

    model_input_path = run.input_datasets["promoted_model"]
    print(f"model_input is {model_input_path}")

    if not os.path.exists(model_input_path):
        raise RuntimeError(f"Model input path not found: {model_input_path}")

    model = mlflow.pyfunc.load_model(model_input_path)

    print("Model metadata:", model.metadata.to_dict())
    print("Model input schema:", model.metadata.signature.inputs)

else:
    model = mlflow.pyfunc.load_model("models:/transformer_load_forecast@production")

# Generate hourly timestamps from 2025 to 2030
timestamps = pd.date_range(start="2025-01-01", end="2030-01-01", freq="h", inclusive="left")

# Prepare input DataFrame for Prophet
df_input = pd.DataFrame({"ds": timestamps})

# ProphetWrapper expects a DataFrame with 'ds' column only
print("Running predictions.")
forecast = model.predict(df_input)

# Extract predictions
df_input["predicted_kw"] = forecast["yhat"]

# Save results
print("Saving predictions.")
os.makedirs("outputs", exist_ok=True)
df_input.to_csv("outputs/predicted_kw.csv", index=False)

# Check for transformer overload
print("Checking for transformer overload.")
transformer_limit = 85.0
overload = df_input[df_input["predicted_kw"] > transformer_limit]

if not overload.empty:
    print(f"Transformer overload predicted on: {overload.iloc[0]['ds']}")
else:
    print("No overload predicted between 2025 and 2029.")

# Plot forecast
print("Plotting predictions.")
plt.figure(figsize=(14, 6))
plt.plot(df_input["ds"], df_input["predicted_kw"], label="Predicted kw", color="steelblue")
plt.xlabel("Timestamp")
plt.ylabel("Predicted kw")
plt.title("Hourly Peak Transformer Load Forecast (2025–2030)")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/predicted_kw_trend.png")

if is_azure:

    print("Logging prediction metrics")
    mlflow.log_metric("max_predicted_kw", df_input["predicted_kw"].max())
    mlflow.log_metric("overload_event_count", len(overload))
    mlflow.log_metric("first_overload_timestamp", overload.iloc[0]["ds"].timestamp())

    predictions_path = run.output_datasets["predictions"]
  
    df_input.to_csv(os.path.join(predictions_path, "predicted_kw.csv"), index=False)
    plt.savefig(os.path.join(predictions_path, "predicted_kw_trend.png"))

    if not overload.empty:
        overload.to_csv(os.path.join(predictions_path, "overload_events.csv"), index=False)

    print("Forecast results saved to AzureML output folder.")
