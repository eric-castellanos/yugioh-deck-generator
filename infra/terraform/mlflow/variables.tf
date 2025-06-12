variable "vpc_id" {
  description = "The ID of the VPC"
  type        = string
}

variable "rds_instance_name" {
  description = "Name of the existing RDS instance"
  type        = string
}

variable "s3_bucket_name" {
  description = "Name of the existing S3 bucket for MLflow artifacts"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "mlflow_version" {
  description = "Version of MLflow to deploy"
  type        = string
  default     = "latest"
}

variable "enable_efs" {
  description = "Enable EFS for model registry"
  type        = bool
  default     = true
}

variable "private_subnet_tags" {
  description = "Tags to identify private subnets"
  type        = map(string)
  default = {
    Name = "private-subnet-*"
  }
}
