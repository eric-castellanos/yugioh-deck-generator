variable "cluster_name" {
  default = "mlflow-eks-cluster"
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.29"
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