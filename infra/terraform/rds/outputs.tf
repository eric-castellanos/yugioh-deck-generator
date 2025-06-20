output "db_endpoint" {
  value = aws_db_instance.mlflow.endpoint
}

output "db_name" {
  value = aws_db_instance.mlflow.db_name
}

output "db_username" {
  value = aws_db_instance.mlflow.username
}