output "mlflow_db_endpoint" {
  value = aws_db_instance.mlflow.endpoint
}

output "mlflow_db_name" {
  value = aws_db_instance.mlflow.db_name
}