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

def create_feature_df(timestamps: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"timestamp": timestamps})
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    
    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall"
    })
    
    df = pd.get_dummies(df, columns=["season", "day_of_week"], drop_first=True)
    
    # Ensure all expected features are present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]

# Prepare input features
X = create_feature_df(timestamps)

# Predict and check for overload
X["predicted_kwh"] = model.predict(X)
X["timestamp"] = timestamps  # restore timestamp for reporting

X[["timestamp", "predicted_kwh"]].to_csv("predicted_kwh.csv", index=False)

transformer_limit = 85.0
overload = X[X["predicted_kwh"] > transformer_limit]

if not overload.empty:
    print(f"⚠️ Transformer overload predicted on: {overload.iloc[0]['timestamp']}")
else:
    print("✅ No overload predicted between 2025 and 2029.")

plt.figure(figsize=(14, 6))
plt.plot(X["timestamp"], X["predicted_kwh"], label="Predicted kWh", color="steelblue")
plt.xlabel("Timestamp")
plt.ylabel("Predicted kWh")
plt.title("Hourly Energy Forecast (2025–2030)")
plt.grid(True)
plt.tight_layout()
plt.savefig("predicted_kwh_trend.png")  # Save to file
plt.show()  # Display interactively
