# Standard library
import os
import shutil

# Third-party
import mlflow
from mlflow.tracking import MlflowClient

# Azure SDK
from azureml.core import Run

print("\n*** STARTING PROMOTION SCRIPT ***")
print("Current working directory:", os.getcwd())

# Detect if we're running in Azure ML.
is_azure = "AZUREML_EXPERIMENT_ID" in os.environ or "AZUREML_RUN_ID" in os.environ
print("Running in Azure ML:", is_azure)

# Set tracking URI if doing a local run.
if not is_azure:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/mlflow/mlruns"))

print("Tracking URI:", mlflow.get_tracking_uri())

if is_azure:

    print("Using AzureML model registry for promotion.")
    
    # Get the current run and its parent.
    run = Run.get_context()
    parent_run = run.parent
    training_run = None

    # Find the training run among child runs.
    for child in parent_run.get_children():
        if child.get_tags().get("stage") == "training":
            training_run = child
            break

    # If no training run was found, log the available child runs and abort.
    if training_run is None:

        print("Training run not found. Here are the available child runs and tags:")

        for child in parent_run.get_children():
            print(f" - {child.name}: {child.get_tags()}")

        raise RuntimeError("Training run not found.")

    print("Found training run:", training_run.name)

    # Log the RMSE from the training run.
    rmse = training_run.get_metrics().get("rmse")

    if rmse is None:
        raise RuntimeError("RMSE metric not found in the training run.")

    print(f"Retrieved RMSE from training run: {rmse}")

    # Check that the RMSE is under the threshold for promotion.
    RMSE_THRESHOLD = 3.0

    if rmse > RMSE_THRESHOLD:
        ERROR_MESSAGE = f"RMSE {rmse:.2f} exceeds threshold of {RMSE_THRESHOLD:.2f}. Aborting promotion."
        print(ERROR_MESSAGE)
        run.fail(ERROR_MESSAGE)
        exit(0)
    
    # Get trained model input path and the promoted model output path.
    model_input_path = run.input_datasets["trained_model"]
    promoted_model_path = run.output_datasets["promoted_model"]
    print(f"Received model input path: {model_input_path}")
    print(f"Copying promoted model to: {promoted_model_path}")

    # Check that the model input path exists.
    if not os.path.exists(model_input_path):
        raise RuntimeError(f"Model input path not found: {model_input_path}")

    # Copy the model to the output path.
    # This allows it to be picked up by the next step in the pipeline (prediction).
    print("Copying model to output path.")
    shutil.copytree(model_input_path, promoted_model_path, dirs_exist_ok=True)

else:

    model_name = "transformer_load_forecast"
    alias_name = "production"

    client = MlflowClient()

    # Get all versions of the model.
    print(f"Searching for all versions of model {model_name}.")
    versions = client.search_model_versions(f"name='{model_name}'")

    # Find all unaliased versions of the model.
    unaliased = [v for v in versions if not v.aliases]
    print(f"Found {len(unaliased)} unaliased versions of model '{model_name}'.")

    if not unaliased:
        raise RuntimeError(f"No unaliased versions found for model '{model_name}'. Nothing to promote.")
    else:
        # Promote the latest unaliased version of the model to production.
        latest = max(unaliased, key=lambda v: v.creation_timestamp)
        print(f"Latest unaliased version is: {latest.version}")

        client.set_registered_model_alias(
            name=model_name,
            alias=alias_name,
            version=latest.version
        )
        print(f"Promoted version {latest.version} of model '{model_name}' to alias '{alias_name}'.")