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
  description = "List of subnet IDs for EKS cluster and node groups"
  type        = list(string)
  default     = []
}

variable "control_plane_subnet_ids" {
  description = "Subnet IDs for the EKS control plane"
  type        = list(string)
  default     = []
}

variable "environment" {
  default = "dev"
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "create_cloudwatch_log_group" {
  description = "Determines whether to create CloudWatch Log Group for EKS cluster"
  type        = bool
  default     = true
}

variable "account_id" {
  description = "AWS Account ID"
  type        = string
}

variable "github_actions_role_arn" {
  description = "ARN of the GitHub Actions role"
  type        = string
}

output "effective_control_plane_subnet_ids" {
  description = "Effective subnet IDs for the EKS control plane"
  value       = coalescelist(var.control_plane_subnet_ids, var.subnet_ids)
}