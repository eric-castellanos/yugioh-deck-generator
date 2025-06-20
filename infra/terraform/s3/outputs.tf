output "mlflow_bucket_name" {
  value = var.bucket_name
}

output "mlflow_user_arn" {
  value = aws_iam_user.mlflow_user.arn
}