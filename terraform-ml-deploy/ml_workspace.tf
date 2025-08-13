# Reference the current Azure client configuration
data "azurerm_client_config" "current" {}

# Reference existing resource group instead of creating it
data "azurerm_resource_group" "ml_rg" {
  name = var.resource_group_name
}

# Reference existing identity instead of creating it.
data "azurerm_user_assigned_identity" "ml_identity" {
  name                = var.identity_name
  resource_group_name = data.azurerm_resource_group.ml_rg.name
}

# Reference existing container registry instead of creating it
data "azurerm_container_registry" "acr" {
  name                = var.acr_name
  resource_group_name = data.azurerm_resource_group.ml_rg.name
}

# Referencing the existing storage account instead of creating it
data "azurerm_storage_account" "ml_storage" {
  name                = var.storage_account_name
  resource_group_name = data.azurerm_resource_group.ml_rg.name
}

resource "azurerm_key_vault" "ml_kv" {
  name                       = var.keyvault_name
  location                   = data.azurerm_resource_group.ml_rg.location
  resource_group_name        = data.azurerm_resource_group.ml_rg.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 0
  purge_protection_enabled   = true
}

resource "azurerm_key_vault_access_policy" "ml_policy" {
  key_vault_id = azurerm_key_vault.ml_kv.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_user_assigned_identity.ml_identity.principal_id

  secret_permissions = ["Get", "List"]
}

resource "azurerm_log_analytics_workspace" "ml_la" {
  name                = var.log_workspace_name
  location            = azurerm_resource_group.ml_rg.location
  resource_group_name = azurerm_resource_group.ml_rg.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_application_insights" "ml_ai" {
  name                = var.appinsights_name
  location            = data.azurerm_resource_group.ml_rg.location
  resource_group_name = data.azurerm_resource_group.ml_rg.name
  application_type    = "web"
  workspace_id        = azurerm_log_analytics_workspace.ml_la.id
}

# Create ML workspace using existing ACR and storage
resource "azurerm_machine_learning_workspace" "ml_ws" {
  name                   = var.ml_workspace_name
  location               = var.location
  resource_group_name    = data.azurerm_resource_group.ml_rg.name
  container_registry_id  = data.azurerm_container_registry.acr.id
  storage_account_id     = data.azurerm_storage_account.ml_storage.id
  key_vault_id           = azurerm_key_vault.ml_kv.id
  application_insights_id = azurerm_application_insights.ml_ai.id

  identity {
    type         = "UserAssigned"
    identity_ids = [data.azurerm_user_assigned_identity.ml_identity.id]
  }

  depends_on = [
  azurerm_key_vault.ml_kv,
  azurerm_application_insights.ml_ai,
  azurerm_log_analytics_workspace.ml_la
  ]
}

# Assign AzureML Compute Operator access to the ML workspace
resource "azurerm_role_assignment" "ml_compute_operator" {
  name                 = "77B92631-C0D9-4B13-B92A-F6FF4A8055F2"
  principal_id         = data.azurerm_user_assigned_identity.ml_identity.principal_id
  role_definition_name = "AzureML Compute Operator"
  scope                = azurerm_machine_learning_workspace.ml_ws.id
  depends_on           = [azurerm_machine_learning_workspace.ml_ws]
}

# Assign AzureML Data Scientist access to the ML workspace
resource "azurerm_role_assignment" "ml_data_scientist" {
  name                 = "FF58279F-C7C8-4861-BE5C-BBC1B89C7137"
  principal_id         = data.azurerm_user_assigned_identity.ml_identity.principal_id
  role_definition_name = "AzureML Data Scientist"
  scope                = azurerm_machine_learning_workspace.ml_ws.id
  depends_on           = [azurerm_machine_learning_workspace.ml_ws]
}

# Create compute cluster
resource "azurerm_machine_learning_compute_cluster" "cpu_cluster" {
  name                          = var.compute_name
  location                      = var.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.ml_ws.id
  vm_size                       = "Standard_B1ms"
  vm_priority                   = "LowPriority"

  identity {
    type         = "UserAssigned"
    identity_ids = [data.azurerm_user_assigned_identity.ml_identity.id]
  }

  scale_settings {
    max_node_count                   = 1
    min_node_count                   = 0
    scale_down_nodes_after_idle_duration = "PT60S"
  }

  depends_on = [azurerm_role_assignment.ml_workspace_access]
}