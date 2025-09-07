# Standard library
import os
import shutil

# Third-party
import mlflow
from mlflow.tracking import MlflowClient
import yaml

# Azure SDK
from azureml.core import Run

print("\n*** STARTING PROMOTION SCRIPT ***")
print("Current working directory:", os.getcwd())

# Detect environment
is_azure = "AZUREML_EXPERIMENT_ID" in os.environ or "AZUREML_RUN_ID" in os.environ

print("Running in Azure ML:", is_azure)

model_name = "transformer_load_forecast"
alias_name = "production"

# Set tracking URI only for local
if not is_azure:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/mlflow/mlruns"))

print("Tracking URI:", mlflow.get_tracking_uri())
print("Promoting model:", model_name)

if is_azure:

    print("Using AzureML model registry for promotion.")
    
    run = Run.get_context()
    parent_run = run.parent
    training_run = None

    # Find the training run among child runs
    for child in parent_run.get_children():
        if child.get_tags().get("stage") == "training":
            training_run = child
            break

    # If no training run found, log available child runs and abort.
    if training_run is None:
        print("Available child runs and tags:")
        for child in parent_run.get_children():
            print(f" - {child.name}: {child.get_tags()}")
        raise RuntimeError("training run not found.")

    # Log RMSE from training run.
    print("Found training run:", training_run.name)
    rmse = training_run.get_metrics().get("rmse")

    if rmse is None:
        raise RuntimeError("RMSE metric not found in training run.")

    print(f"Retrieved RMSE from training run: {rmse}")

    # Check RMSE threshold
    print("Checking RMSE threshold for model promotion.")
    RMSE_THRESHOLD = 5.0

    if rmse > RMSE_THRESHOLD:
        print(f"RMSE {rmse:.2f} exceeds threshold of {RMSE_THRESHOLD:2f}. Skipping registration.")
        run.fail("Model rejected due to high RMSE.")
        exit(0)

    # Get model input path and output dataset
    model_input_path = run.input_datasets["trained_model"]
    promoted_model_path = run.output_datasets["promoted_model"]
    print(f"Received model input path: {model_input_path}")
    print(f"Writing promoted model to: {promoted_model_path}")

    if not os.path.exists(model_input_path):
        raise RuntimeError(f"Model input path not found: {model_input_path}")

    # Copy model to output path
    print("Copying model to output path.")
    shutil.copytree(model_input_path, promoted_model_path, dirs_exist_ok=True)

else:

    client = MlflowClient()

    # Validate model exists
    try:
        client.get_registered_model(model_name)
    except Exception as e:
        raise Exception(f"Model '{model_name}' not found in registry.") from e

    # Local: use alias-based promotion
    versions = client.search_model_versions(f"name='{model_name}'")

    # Filter unaliased versions
    aliased_versions = set()
    for v in versions:
        if alias_name in v.aliases:
            aliased_versions.add(v.version)

    unaliased = [v for v in versions if v.version not in aliased_versions]

    if not unaliased:
        print(f"No unaliased versions found for model '{model_name}'. Nothing to promote.")
        exit(0)

    latest_version_obj = sorted(unaliased, key=lambda v: v.creation_timestamp, reverse=True)[0]

    # Validate registry snapshot path using meta.yaml
    version = latest_version_obj.version
    tracking_root = mlflow.get_tracking_uri().replace("file:", "")
    meta_path = os.path.join(tracking_root, "models", model_name, f"version-{version}", "meta.yaml")

    print("Checking meta.yaml path:", meta_path)

    if not os.path.exists(meta_path):
        raise Exception(f"meta.yaml not found for model version {version}")

    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    storage_uri = meta.get("storage_location", "")
    if not storage_uri.startswith("file:"):
        raise Exception(f"Unexpected storage_location format: {storage_uri}")

    registry_path = os.path.normpath(storage_uri.replace("file:", ""))

    print(f"Registry snapshot path: {registry_path}")

    if not os.path.exists(registry_path):
        raise Exception(f"Registry snapshot path does not exist: {registry_path}")

    client.set_registered_model_alias(
        name=model_name,
        alias=alias_name,
        version=latest_version_obj.version
    )

    print(f"Promoted version {latest_version_obj.version} of '{model_name}' to alias '{alias_name}' (local).")