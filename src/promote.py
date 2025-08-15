import os
import mlflow
from mlflow.tracking import MlflowClient
import yaml
import glob

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
    from azureml.core import Workspace, Model

    ws = Workspace.from_config()
    models = Model.list(ws, name=model_name)

    if not models:
        raise Exception(f"No registered models found for '{model_name}' in AzureML.")

    print(f"Found model '{model_name}' with {len(models)} registered versions.")

    # Sort by creation time descending
    latest_model = sorted(models, key=lambda m: m.created_time, reverse=True)[0]

    existing_tags = latest_model.tags or {}
    existing_tags["stage"] = "production"
    latest_model.update(tags=existing_tags)

    print(f"Latest registered model version: {latest_model.version}")

    # AzureML doesn't use MLflow stages directly, but you can tag or version-control here
    print(f"Promoted version {latest_model.version} of '{model_name}' to stage 'production' (AzureML).")

    if latest_model.path and os.path.exists(latest_model.path):
        print("Model artifacts:")
        for path in glob.glob(os.path.join(latest_model.path, "*")):
            print(" -", path)
    else:
        print("Model path not available or inaccessible.")

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