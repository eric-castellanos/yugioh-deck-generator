variable "cluster_name" {
  default = "mlflow-cluster"
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.29"
}

variable "create_resources" {
  type        = bool
  description = "Set to true to create resources, false to skip creation"
  default     = true
}

variable "node_instance_type" {
  description = "EC2 instance type for EKS worker nodes"
  type        = string
  default     = "t3.medium"
}

variable "vpc_id" {}

variable "subnet_ids" {
  type = list(string)
}

variable "environment" {
  default = "dev"
}

variable "create_cloudwatch_log_group" {
  description = "Determines whether to create CloudWatch Log Group for EKS cluster"
  type        = bool
  default     = true
}

locals {
  full_cluster_name = "${var.cluster_name}-${var.environment}"
}