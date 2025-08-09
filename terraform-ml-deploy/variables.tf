variable "location" {
  description = "Azure region"
  default     = "West US"
}

variable "resource_group_name" {
  description = "Name of the resource group"
  default     = "ml-rg"
}

variable "workspace_name" {
  description = "Name of the ML workspace"
  default     = "transformer-load-ws"
}

variable "app_service_plan_name" {
  description = "Name of the App Service Plan"
  default     = "ml-app-plan"
}

variable "app_service_name" {
  description = "Name of the App Service"
  default     = "ml-app-service"
}

variable "acr_name" {
  description = "Name of the Azure Container Registry"
  default     = "mlacrregistry"
}

variable "docker_image_tag" {
  description = "Tag of the Docker image in ACR"
  default     = "latest"
}