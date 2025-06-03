variable "aws_region" {
    type = string
    description = "Region where AWS account is located"
    default = "us-east-1"
}

variable "bucket_name" {
  type = string
  description = "S3 bucket where backend info/data is stored."
  default = "mlflow-backend"
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