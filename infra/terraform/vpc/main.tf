data "aws_subnets" "public" {
  filter {
    name   = "tag:kubernetes.io/role/elb"
    values = ["1"]
  }

  filter {
    name   = "vpc-id"
    values = [local.final_vpc_id]
  }
}

data "aws_subnets" "private" {
  filter {
    name   = "tag:kubernetes.io/role/internal-elb"
    values = ["1"]
  }

  filter {
    name   = "vpc-id"
    values = [local.final_vpc_id]
  }
}

locals {
  use_existing_vpc = var.vpc_id != null && var.vpc_id != ""
  final_vpc_id     = local.use_existing_vpc ? var.vpc_id : module.vpc[0].vpc_id
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.1.1"

  name = "${var.name}-${var.environment}"
  cidr = var.vpc_cidr

  azs             = var.azs
  public_subnets  = var.public_subnet_cidrs
  private_subnets = var.private_subnet_cidrs

  enable_dns_hostnames = true
  enable_dns_support   = true

  # Enable auto-assign public IP for public subnets (required for EKS nodes)
  map_public_ip_on_launch = true

  # EKS requires specific tags on subnets
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
    "kubernetes.io/cluster/mlflow-cluster-${var.environment}" = "shared"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
    "kubernetes.io/cluster/mlflow-cluster-${var.environment}" = "shared"
  }

  tags = {
    Environment = var.environment
    Project     = "mlflow"
    "kubernetes.io/cluster/mlflow-cluster-${var.environment}" = "shared"
  }
}

output "vpc_filter_debug" {
  value = {
    Environment = var.environment,
    Project     = "mlflow"
  }
}