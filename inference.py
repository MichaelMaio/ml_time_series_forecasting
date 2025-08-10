import os
import logging
import mlflow
import mlflow.pyfunc
import pandas as pd
from flask import Flask, request, jsonify

logging.basicConfig(filename="logs/server.log", level=logging.INFO)

app = Flask(__name__)

# Detect environment
is_azure = "AZUREML_EXPERIMENT_ID" in os.environ or "AZUREML_RUN_ID" in os.environ

if is_azure:
    logging.info("Running in Azure ML environment")
else:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:mlruns"))
    logging.info("Running in local environment")

# Set default model path based on environment
default_model_path = (
    "models:/transformer_load_forecast/Production"
    if is_azure else
    "models:/transformer_load_forecast@production"
)

model_path = os.getenv("MODEL_PATH", default_model_path)
logging.info(f"Loading model from: {model_path}")

# Load model
model = mlflow.pyfunc.load_model(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_json = request.get_json()
        if not input_json:
            return jsonify({"error": "Empty or invalid JSON"}), 400

        input_df = pd.DataFrame(input_json)
        predictions = model.predict(input_df)
        return jsonify(predictions.tolist())

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/version", methods=["GET"])
def version():
    try:
        return jsonify({
            "model_path": model_path
        })
    except Exception as e:
        logging.exception("Version check failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)