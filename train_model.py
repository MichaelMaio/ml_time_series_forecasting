import json
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

# Enable auto-logging for XGBoost
mlflow.xgboost.autolog()

# Load data
df = pd.read_csv("data/peak_load.csv", parse_dates=["timestamp"])

# Feature engineering
df["hour"] = df["hour"].astype(int)
df["is_weekend"] = df["is_weekend"].astype(int)
df = pd.get_dummies(df, columns=["season", "day_of_week"], drop_first=True)

# Cast integer columns to float64 to avoid schema enforcement issues
int_cols = df.select_dtypes(include="int").columns
df[int_cols] = df[int_cols].astype("float64")

# Split features and target
X = df.drop(columns=["timestamp", "kwh"])
y = df["kwh"]

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# Start MLflow run
with mlflow.start_run(run_name="transformer_load_forecast"):
    # Train model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, early_stopping_rounds=10)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mlflow.log_metric("rmse", rmse)

    # Save model and feature list locally
    model.save_model("transformer_load_model.json")
    with open("model_features.json", "w") as f:
        json.dump(list(X.columns), f)

    # Log artifacts
    mlflow.log_artifact("transformer_load_model.json")
    mlflow.log_artifact("model_features.json")

    # Set MLflow tags
    mlflow.set_tag("model_type", "XGBoost")
    mlflow.set_tag("use_case", "Energy Load Forecasting")
    mlflow.set_tag("owner", "Michael Maio")

    # Infer model signature for input/output schema
    signature = infer_signature(X_train, model.predict(X_train))

    # Log and register the model
    mlflow.xgboost.log_model(
        model,
        name="transformer_load_model",
        registered_model_name="transformer_load_forecast",
        signature=signature
    )

    print(f"✅ Model trained and logged with RMSE: {rmse:.2f} kWh")