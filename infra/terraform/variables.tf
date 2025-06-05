variable "db_password" {
  description = "Password for the RDS database"
  type        = string
  sensitive   = true
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.29"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "mlflow-eks-cluster"
}

variable "node_instance_type" {
  description = "EC2 instance type for EKS worker nodes"
  type        = string
  default     = "t3.medium"
}