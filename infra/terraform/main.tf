# terraform/main.tf - Root module to orchestrate all child modules

locals {
  use_existing_vpc = var.vpc_id != null && var.vpc_id != ""
  final_vpc_id     = local.use_existing_vpc ? var.vpc_id : (var.existing_vpc_id != null ? var.existing_vpc_id : module.vpc.vpc_id)
  full_cluster_name = "${var.cluster_name}-${var.environment}"  # e.g. "mlflow-cluster-dev"
}

module "vpc" {
  source               = "./vpc"
  name                 = "mlflow-vpc"
  vpc_cidr             = "10.0.0.0/16"
  azs                  = ["us-east-1a", "us-east-1b", "us-east-1c"]
  public_subnet_cidrs  = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  private_subnet_cidrs = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  environment          = var.environment
  vpc_id               = var.vpc_id
  create_vpc           = !var.existing_resources && !local.use_existing_vpc && var.existing_vpc_id == null
}

# Only create S3 bucket and IAM resources if they don't exist
module "s3" {
  source           = "./s3"
  region           = var.region
  bucket_name      = "mlflow-backend"
  mlflow_user      = var.mlflow_user
  create_resources = !var.existing_resources
}

module "rds" {
  source           = "./rds"
  environment      = var.environment
  subnet_ids       = module.vpc.private_subnet_ids
  db_password      = var.db_password
  region           = var.region
  vpc_id           = local.final_vpc_id
  create_resources = !var.existing_resources
}

module "eks" {
  source                      = "./eks"
  cluster_name                = var.cluster_name
  cluster_version             = var.cluster_version
  vpc_id                      = local.final_vpc_id
  subnet_ids                  = module.vpc.private_subnet_ids
  create_resources            = !var.existing_resources
  create_cloudwatch_log_group = false # Disable creation of CloudWatch Log Group
  account_id                  = var.account_id
  github_actions_role_arn     = var.github_actions_role_arn
  environment                 = var.environment
}

output "mlflow_db_endpoint" {
  value = module.rds.db_endpoint
}

output "mlflow_db_name" {
  value = module.rds.db_name
}

output "mlflow_db_username" {
  value = module.rds.db_username
}

output "mlflow_bucket_name" {
  value = module.s3.mlflow_bucket_name
}

output "eks_cluster_name" {
  value = module.eks.cluster_name
}
