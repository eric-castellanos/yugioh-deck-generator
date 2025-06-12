variable "db_name" {
  description = "Database name"
  default     = "mlflowdb"
}

variable "db_username" {
  description = "Master DB username"
  default     = "mlflowadmin"
}

variable "db_password" {
  description = "Master DB password"
  sensitive   = true
}

variable "vpc_id" {}

variable "subnet_ids" {
  type        = list(string)
  description = "List of subnet IDs for RDS instance"
}

variable "region" {
  description = "AWS region for RDS"
  type        = string
}

variable "create_resources" {
  type        = bool
  description = "Set to true to create resources, false to skip creation"
  default     = true
}

variable "db_instance_class" {
  default = "db.t3.micro"
}

variable "allocated_storage" {
  default = 20
}

variable "environment" {
  default = "dev"
}

