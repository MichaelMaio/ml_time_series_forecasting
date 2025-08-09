import os
import logging
import mlflow
import mlflow.pyfunc
import pandas as pd
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Set MLflow tracking URI from environment or default to local registry
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:mlruns"))

# Load model using alias (preferred over deprecated stage)
model_path = os.getenv("MODEL_PATH", "models:/transformer_load_forecast@production")
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)