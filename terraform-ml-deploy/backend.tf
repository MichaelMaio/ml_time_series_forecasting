terraform {
  backend "azurerm" {
    resource_group_name  = "transformer-load-rg"
    storage_account_name = "transformerloadstorage"
    container_name       = "tfstate"
    key                  = "pipeline.tfstate"
  }
}