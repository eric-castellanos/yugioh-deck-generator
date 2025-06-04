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
  type = list(string)
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

