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

# Create ML workspace using existing ACR and storage
resource "azurerm_machine_learning_workspace" "ml_ws" {
  name                   = var.workspace_name
  location               = var.location
  resource_group_name    = data.azurerm_resource_group.ml_rg.name
  container_registry_id  = data.azurerm_container_registry.acr.id
  storage_account_id     = data.azurerm_storage_account.ml_storage.id

  identity {
    type         = "UserAssigned"
    identity_ids = [data.azurerm_user_assigned_identity.ml_identity.id]
  }
}

# Assign Machine Learning Compute Operator access to the ML workspace
resource "azurerm_role_assignment" "ml_workspace_access" {
  name                 = "77B92631-C0D9-4B13-B92A-F6FF4A8055F2"

  principal_id         = data.azurerm_user_assigned_identity.ml_identity.principal_id
  role_definition_name = "Machine Learning Compute Operator"
  scope                = azurerm_machine_learning_workspace.ml_ws.id
  depends_on           = [azurerm_machine_learning_workspace.ml_ws]
}

# Create compute cluster
resource "azurerm_machine_learning_compute_cluster" "cpu_cluster" {
  name                = var.compute_name
  location            = var.location
  resource_group_name = data.azurerm_resource_group.ml_rg.name
  workspace_name      = azurerm_machine_learning_workspace.ml_ws.name
  vm_size             = "Standard_B2ms"

  identity {
    type         = "UserAssigned"
    identity_ids = [data.azurerm_user_assigned_identity.ml_identity.id]
  }

  scale_settings {
    max_node_count                   = 1
    min_node_count                   = 0
    node_idle_time_before_scale_down = "PT60S"
  }

  depends_on = [azurerm_role_assignment.ml_workspace_access]
}