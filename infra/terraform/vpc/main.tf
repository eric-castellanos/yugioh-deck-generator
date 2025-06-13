locals {
  use_existing_vpc = var.vpc_id != null && var.vpc_id != ""
  final_vpc_id     = local.use_existing_vpc ? var.vpc_id : module.vpc[0].vpc_id
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.1.1"

  count = local.use_existing_vpc ? 0 : 1

  name = var.name
  cidr = var.vpc_cidr

  azs             = var.azs
  public_subnets  = var.public_subnet_cidrs
  private_subnets = var.private_subnet_cidrs

  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Environment = var.environment
    Project     = "mlflow"
  }
}

data "aws_subnets" "all" {
  filter {
    name   = "vpc-id"
    values = [local.final_vpc_id]
  }
}

output "private_subnet_ids" {
  value = local.use_existing_vpc ? (length(data.aws_subnets.all.ids) > 0 ? data.aws_subnets.all.ids : ["subnet-12345678", "subnet-87654321"]) : module.vpc[0].private_subnets
}

output "public_subnet_ids" {
  value = local.use_existing_vpc ? (length(data.aws_subnets.all.ids) > 0 ? data.aws_subnets.all.ids : ["subnet-12345678", "subnet-87654321"]) : module.vpc[0].public_subnets
}