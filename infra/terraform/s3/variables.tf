variable "region" {
    type = string
    description = "AWS region for S3"
    default = "us-east-1"
}

variable "bucket_name" {
  type = string
  description = "S3 bucket where backend info/data is stored."
}

variable "mlflow_user" {
  type = string
  description = "username for signing into MLFlow"
  default = "user"
}

variable "environment" {
  type        = string
  description = "Environment tag (e.g., dev, prod)"
  default = "dev"
}