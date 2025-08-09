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

resource "azurerm_resource_group" "ml_rg" {
  name     = var.resource_group_name
  location = var.location
}

module "ml_workspace" {
  source = "./"
}

module "app_service" {
  source = "./"
}