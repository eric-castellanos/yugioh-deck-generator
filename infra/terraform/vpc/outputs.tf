output "vpc_id" {
  description = "ID of the VPC in use"
  value       = local.final_vpc_id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = local.use_existing_vpc ? (length(data.aws_subnets.private.ids) > 0 ? data.aws_subnets.private.ids : []) : module.vpc[0].private_subnets
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = local.use_existing_vpc ? (length(data.aws_subnets.public.ids) > 0 ? data.aws_subnets.public.ids : []) : module.vpc[0].public_subnets
}