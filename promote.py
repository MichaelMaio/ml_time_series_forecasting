import os
import mlflow
from mlflow.tracking import MlflowClient
import yaml

# Detect environment
is_azure = "AZUREML_EXPERIMENT_ID" in os.environ or "AZUREML_RUN_ID" in os.environ
model_name = "transformer_load_forecast"
alias_name = "production"

# Set tracking URI only for local
if not is_azure:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/mlflow/mlruns"))

print("Tracking URI:", mlflow.get_tracking_uri())
print("Promoting model:", model_name)

client = MlflowClient()

# Validate model exists
try:
    client.get_registered_model(model_name)
except Exception as e:
    raise Exception(f"Model '{model_name}' not found in registry.") from e

if is_azure:
    # ✅ Azure: use stage-based promotion
    unpromoted_versions = client.get_latest_versions(model_name, stages=["None"])
    if not unpromoted_versions:
        raise Exception(f"No unpromoted versions found for model '{model_name}'.")

    latest_version = unpromoted_versions[0].version

    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"✅ Promoted version {latest_version} of '{model_name}' to stage 'Production' (Azure).")

else:
    # ✅ Local: use alias-based promotion
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

    # ✅ Validate registry snapshot path using meta.yaml
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

    print(f"🔍 Registry snapshot path: {registry_path}")

    if not os.path.exists(registry_path):
        raise Exception(f"Registry snapshot path does not exist: {registry_path}")

    client.set_registered_model_alias(
        name=model_name,
        alias=alias_name,
        version=latest_version_obj.version
    )

    print(f"✅ Promoted version {latest_version_obj.version} of '{model_name}' to alias '{alias_name}' (local).")