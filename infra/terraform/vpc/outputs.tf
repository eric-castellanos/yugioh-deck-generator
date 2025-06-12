output "vpc_id" {
  description = "ID of the VPC in use"
  value       = local.final_vpc_id
}

output "public_subnet_ids" {
  value = module.vpc[0].public_subnets
}

output "private_subnet_ids" {
  value = module.vpc[0].private_subnets
}