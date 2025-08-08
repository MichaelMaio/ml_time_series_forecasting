import json
import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
import pandas as pd

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
plt.savefig("predicted_kwh_trend.png")
plt.show()