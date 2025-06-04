output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  value = module.eks.cluster_primary_security_group_id
}

output "node_group_role_arn" {
  value = module.eks.eks_managed_node_groups["default_node_group"].iam_role_arn
}