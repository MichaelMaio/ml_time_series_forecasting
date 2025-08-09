resource "azurerm_container_registry" "acr" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.ml_rg.name
  location            = azurerm_resource_group.ml_rg.location
  sku                 = "Basic"
  admin_enabled       = true
}

resource "azurerm_machine_learning_workspace" "ml_ws" {
  name                = var.workspace_name
  location            = azurerm_resource_group.ml_rg.location
  resource_group_name = azurerm_resource_group.ml_rg.name
  container_registry_id = azurerm_container_registry.acr.id
}