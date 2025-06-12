output "mlflow_bucket_name" {
  value = var.bucket_name
}

output "mlflow_user_arn" {
  value = var.create_resources ? aws_iam_user.mlflow_user[0].arn : ""
}