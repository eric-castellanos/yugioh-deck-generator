module "vpc" {
  source = "./vpc"
}

module "eks" {
  source     = "./eks"
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
}

module "rds" {
  source     = "./rds"
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
  db_password = var.db_password
}

module "s3" {
  source = "./s3"
}