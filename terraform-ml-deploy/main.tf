provider "azurerm" {
  features {}
}

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# Reference existing identity instead of creating it.
data "azurerm_user_assigned_identity" "ml_identity" {
  name                = var.identity_name
  resource_group_name = data.azurerm_resource_group.ml_rg.name
}

identity_id           = data.azurerm_user_assigned_identity.ml_identity.id
identity_principal_id = data.azurerm_user_assigned_identity.ml_identity.principal_id

module "ml_workspace" {
  source = "./ml_workspace"

  resource_group_name   = var.resource_group_name
  location              = var.location
  workspace_name        = var.workspace_name
  acr_name              = var.acr_name
  storage_account_name  = var.storage_account_name
  compute_name          = var.compute_name
  identity_name         = var.identity_name
  identity_id           = identity_id
  identity_principal_id = identity_principal_id
}