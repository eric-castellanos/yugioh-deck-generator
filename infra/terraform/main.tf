# terraform/main.tf - Root module to orchestrate all child modules

module "vpc" {
  source = "./vpc"
}

module "s3" {
  source = "./s3"
}

module "rds" {
  source      = "./rds"
  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.private_subnet_ids
  db_password = var.db_password
}

module "eks" {
  source            = "./eks"
  cluster_name      = var.cluster_name
  cluster_version   = var.cluster_version
  vpc_id            = module.vpc.vpc_id
  subnet_ids        = module.vpc.private_subnet_ids
}

output "rds_endpoint" {
  value = module.rds.db_endpoint
}

output "s3_bucket_name" {
  value = module.s3.bucket_name
}

output "eks_cluster_name" {
  value = module.eks.cluster_name
}
