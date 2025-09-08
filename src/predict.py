# Standard library
import logging
import os

# Third-party
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

# Azure SDK
from azureml.core import Run

print("\n*** STARTING PREDICTION SCRIPT ***")
print("Current working directory:", os.getcwd())

# Disable Prophet logging to suppress errors about plotting in headless mode.
logging.getLogger("prophet.plot").disabled = True

# Detect if we're running in Azure ML.
is_azure = "AZUREML_EXPERIMENT_ID" in os.environ or "AZUREML_RUN_ID" in os.environ
print("Running in Azure ML:", is_azure)

if is_azure:

    # Get the current run.
    run = Run.get_context()

    # Get the path to the promoted model from the input datasets.
    model_input_path = run.input_datasets["promoted_model"]
    print(f"model input is path is {model_input_path}")

    # Check that the model path exists.
    if not os.path.exists(model_input_path):
        raise RuntimeError(f"Model input path not found: {model_input_path}")

    # Load the model.
    print("Loading the model.")
    model = mlflow.pyfunc.load_model(model_input_path)

    print("Model metadata:", model.metadata.to_dict())
    print("Model input schema:", model.metadata.signature.inputs)

else:
    MODEL_PATH = "models:/transformer_load_forecast@production"
    print(f"Loading model from MLflow Model Registry: {MODEL_PATH}")
    model = mlflow.pyfunc.load_model(MODEL_PATH)

# Generate hourly timestamps from 2025 to 2030 (for future predicitons).
PREDICTION_START_DAY = "2025-01-01"
PREDICTION_END_DAY = "2030-01-01"
print(f"Generating hourly timestamps from {PREDICTION_START_DAY} to {PREDICTION_END_DAY}.")
timestamps = pd.date_range(start=PREDICTION_START_DAY, end=PREDICTION_END_DAY, freq="h", inclusive="left")

# Initialize the input DataFrame.
df_input = pd.DataFrame({"ds": timestamps})

# Run the predictions.
print(f"Running predictions against {len(timestamps)} hourly timestamps.")
forecast = model.predict(df_input)

# Write the predictions to a file.
PREDICTIONS_FILE_PATH = "outputs/predicted_kw.csv"
print(f"Writing predctions to {PREDICTIONS_FILE_PATH}.")
df_input["predicted_kw"] = forecast["yhat"]
os.makedirs("outputs", exist_ok=True)
df_input.to_csv("outputs/predicted_kw.csv", index=False)

# Check if the transformer is predicted to overload.
print("Checking for transformer overload.")
TRANSFORMER_LIMIT = 85.0
overload = df_input[df_input["predicted_kw"] > TRANSFORMER_LIMIT]

if not overload.empty:
    print(f"Transformer overload predicted on: {overload.iloc[0]['ds']}")
else:
    print(f"No overload predicted between the start of {PREDICTION_START_DAY} and the end of {PREDICTION_END_DAY}.")

# Plot the forecast.
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

    # Log the max predicted kw usage and the count of overload events.
    print("Logging prediction metrics.")
    mlflow.log_metric("max_predicted_kw", df_input["predicted_kw"].max())
    mlflow.log_metric("overload_event_count", len(overload))

    # Save the predictions to so they're viewable in Azure ML.
    predictions_path = run.output_datasets["predictions"]
    df_input.to_csv(os.path.join(predictions_path, "predicted_kw.csv"), index=False)
    plt.savefig(os.path.join(predictions_path, "predicted_kw_trend.png"))

    # Check if there are any overload events predicted.
    if not overload.empty:

        # Save the overload events so they're viewable in Azure ML.
        overload.to_csv(os.path.join(predictions_path, "overload_events.csv"), index=False)

        # Log the timestamp of the first predicted overload as a tag.
        ts = overload.iloc[0]["ds"]
        formatted_ts = ts.strftime("%Y-%m-%d %H:%M:%S")
        mlflow.set_tag("first_overload_timestamp", formatted_ts)

    print("Forecast results saved to Azure ML output folder.")
