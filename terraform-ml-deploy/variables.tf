variable "location" {
  description = "Azure region"
  default     = "West US"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  default     = "transformer-load-rg"
}

variable "identity_name" {
  description = "User-assigned managed identity name"
  default     = "transformer-load-identity"
}

variable "keyvault_name" {
  description = "Name of the key vault"
  default     = "transformer-load-kv"
}

variable "log_workspace_name" {
  description = "Name of the Log Analytics workspace"
  default     = "transformer-load-loganalytics"
}

variable "appinsights_name" {
  description = "Name of the application insights resource"
  default     = "transformer-load-appinsights"
}

variable "ml_workspace_name" {
  description = "Name of the ML workspace"
  default     = "transformer-load-ws"
}

variable "acr_name" {
  description = "Name of the Azure Container Registry"
  default     = "transformerloadacr"
}

variable "storage_account_name" {
  description = "Name of the Azure Storage Account"
  default     = "transformerloadstorage"
}

variable "compute_name" {
  description = "Name of the Azure ML compute cluster"
  default     = "transformer-load-compute"
}
