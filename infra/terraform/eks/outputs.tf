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