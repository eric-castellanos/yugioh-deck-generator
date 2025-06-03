output "mlflow_bucket_name" {
  value = aws_s3_bucket.mlflow_bucket.bucket
}

output "mlflow_user_arn" {
  value = aws_iam_user.mlflow_user.arn
}