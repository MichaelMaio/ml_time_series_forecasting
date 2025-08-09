resource "azurerm_app_service_plan" "ml_plan" {
  name                = var.app_service_plan_name
  location            = azurerm_resource_group.ml_rg.location
  resource_group_name = azurerm_resource_group.ml_rg.name
  kind                = "Linux"
  reserved            = true

  sku {
    tier = "Basic"
    size = "B1"
  }
}

resource "azurerm_app_service" "ml_app" {
  name                = var.app_service_name
  location            = azurerm_resource_group.ml_rg.location
  resource_group_name = azurerm_resource_group.ml_rg.name
  app_service_plan_id = azurerm_app_service_plan.ml_plan.id

  site_config {
    linux_fx_version = "DOCKER|mlacrregistry.azurecr.io/transformer-api:latest"
  }

  app_settings = {
    # Docker registry credentials
    "DOCKER_REGISTRY_SERVER_URL"      = "https://${azurerm_container_registry.acr.login_server}"
    "DOCKER_REGISTRY_SERVER_USERNAME" = azurerm_container_registry.acr.admin_username
    "DOCKER_REGISTRY_SERVER_PASSWORD" = azurerm_container_registry.acr.admin_password

    # MLflow tracking and model path
    "MLFLOW_TRACKING_URI" = "https://transformer-load-ws.azureml.ms"
    "MODEL_PATH"          = "models:/transformer_load_forecast/Production"
  }

  depends_on = [azurerm_container_registry.acr]
}