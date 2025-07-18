output "cluster_name" {
  value = "mlflow-cluster-${var.environment}"
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  value = module.eks.cluster_primary_security_group_id
}

output "node_group_role_arn" {
  value = aws_iam_role.eks_node_group.arn
}

output "cluster_oidc_issuer_url" {
  value = module.eks.cluster_oidc_issuer_url
}

output "oidc_provider_arn" {
  value = module.eks.oidc_provider_arn
}

output "kms_key_id" {
  description = "The KMS key ID used for EKS encryption"
  value       = aws_kms_key.this.key_id
}

output "kms_key_arn" {
  description = "The KMS key ARN used for EKS encryption"  
  value       = aws_kms_key.this.arn
}