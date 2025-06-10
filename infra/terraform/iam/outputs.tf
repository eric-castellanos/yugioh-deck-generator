output "mlflow_role_arn" {
  description = "ARN of the MLflow IAM role"
  value       = aws_iam_role.mlflow.arn
}

output "mlflow_instance_profile_arn" {
  description = "ARN of the MLflow instance profile"
  value       = aws_iam_instance_profile.mlflow.arn
}

output "eks_mlflow_role_arn" {
  description = "ARN of the EKS MLflow IAM role"
  value       = aws_iam_role.eks_mlflow.arn
}

output "mlflow_role_name" {
  description = "Name of the MLflow IAM role"
  value       = aws_iam_role.mlflow.name
}

output "eks_mlflow_role_name" {
  description = "Name of the EKS MLflow IAM role"
  value       = aws_iam_role.eks_mlflow.name
}

output "mlflow_instance_profile_name" {
  description = "Name of the MLflow instance profile"
  value       = aws_iam_instance_profile.mlflow.name
}
