output "db_endpoint" {
  value = var.create_resources ? aws_db_instance.mlflow[0].endpoint : ""
}

output "db_name" {
  value = var.create_resources ? aws_db_instance.mlflow[0].db_name : ""
}

output "db_username" {
  value = var.create_resources ? aws_db_instance.mlflow[0].username : ""
}