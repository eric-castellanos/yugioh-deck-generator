variable "cluster_name" {
  default = "mlflow-eks-cluster"
}

variable "vpc_id" {}

variable "subnet_ids" {
  type = list(string)
}

variable "environment" {
  default = "dev"
}