output "mlflow_dev_sg_id" {
  description = "Security Group ID for MLflow development"
  value       = aws_security_group.mlflow_dev_sg.id
}

output "mlflow_prod_sg_id" {
  description = "Security Group ID for MLflow production"
  value       = aws_security_group.mlflow_prod_sg.id
}

output "mlflow_db_host" {
  description = "Database host for MLflow"
  value       = data.aws_db_instance.mlflow_db.endpoint
}

output "mlflow_db_password" {
  description = "Database password for MLflow"
  value       = data.aws_secretsmanager_secret_version.mlflow_db_creds.secret_string_json["password"]
  sensitive   = true
}

output "mlflow_s3_bucket" {
  description = "S3 bucket name for MLflow artifacts"
  value       = data.aws_s3_bucket.mlflow_artifacts.bucket
}

output "mlflow_dev_efs_id" {
  description = "EFS volume ID for MLflow development"
  value       = aws_efs_file_system.mlflow_dev_efs.id
}

output "mlflow_prod_efs_id" {
  description = "EFS volume ID for MLflow production"
  value       = aws_efs_file_system.mlflow_prod_efs.id
}

output "mlflow_dev_efs_mount_target_ids" {
  description = "EFS mount target IDs for MLflow development"
  value       = aws_efs_mount_target.mlflow_dev_efs_mount_target[*].id
}

output "mlflow_prod_efs_mount_target_ids" {
  description = "EFS mount target IDs for MLflow production"
  value       = aws_efs_mount_target.mlflow_prod_efs_mount_target[*].id
}
