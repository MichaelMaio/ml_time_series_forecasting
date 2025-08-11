# Reference existing resource group instead of creating it
data "azurerm_resource_group" "ml_rg" {
  name     = var.resource_group_name
}

# Reference existing ACR instead of creating it
data "azurerm_container_registry" "acr" {
  name                = var.acr_name
  resource_group_name = var.resource_group_name
}

# Assign AcrPull access to existing ACR
resource "azurerm_role_assignment" "acr_pull" {
  principal_id         = var.identity_principal_id
  role_definition_name = "AcrPull"
  scope                = data.azurerm_container_registry.acr.id
}

# Create storage account
resource "azurerm_storage_account" "ml_storage" {
  name                     = var.storage_account_name
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

# Assign blob access to storage account
resource "azurerm_role_assignment" "blob_access" {
  principal_id         = var.identity_principal_id
  role_definition_name = "Storage Blob Data Contributor"
  scope                = azurerm_storage_account.ml_storage.id
}

# Create ML workspace using existing ACR and new storage
resource "azurerm_machine_learning_workspace" "ml_ws" {
  name                   = var.workspace_name
  location               = var.location
  resource_group_name    = var.resource_group_name
  container_registry_id  = data.azurerm_container_registry.acr.id
  storage_account_id     = azurerm_storage_account.ml_storage.id

  identity {
    type         = "UserAssigned"
    identity_ids = [var.identity_id]
  }
}

# Assign Machine Learning Compute Operator access to the ML workspace
resource "azurerm_role_assignment" "ml_workspace_access" {
  principal_id         = var.identity_principal_id
  role_definition_name = "Machine Learning Compute Operator"
  scope                = azurerm_machine_learning_workspace.ml_ws.id
  depends_on           = [azurerm_machine_learning_workspace.ml_ws]
}

# Create compute cluster
resource "azurerm_machine_learning_compute_cluster" "cpu_cluster" {
  name                = var.compute_name
  location            = var.location
  resource_group_name = var.resource_group_name
  workspace_name      = azurerm_machine_learning_workspace.ml_ws.name
  vm_size             = "Standard_B2ms"

  identity {
    type         = "UserAssigned"
    identity_ids = [var.identity_id]
  }

  scale_settings {
    max_node_count                   = 2
    min_node_count                   = 0
    node_idle_time_before_scale_down = "PT120S"
  }

  depends_on = [azurerm_role_assignment.ml_workspace_access]
}