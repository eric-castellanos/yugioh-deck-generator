module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name = local.full_cluster_name
  vpc_id       = var.vpc_id
  subnet_ids   = var.subnet_ids

  create_cloudwatch_log_group = var.create_cloudwatch_log_group

  tags = {
    Environment = var.environment
    Project     = "mlflow"
  }
}

variable "log_retention_in_days" {
  type    = number
  default = 30
}

resource "aws_kms_key" "this" {
  count = var.create_resources ? 1 : 0
}

resource "aws_kms_alias" "this" {
  count = var.create_resources ? 1 : 0

  name          = "alias/eks/${local.full_cluster_name}"
  target_key_id = aws_kms_key.this[0].id

  lifecycle {
    ignore_changes = [name]
  }
}

resource "aws_cloudwatch_log_group" "this" {
  count = var.create_resources ? 1 : 0

  name              = "/aws/eks/${local.full_cluster_name}/cluster"
  retention_in_days = var.log_retention_in_days

  tags = {
    Name = var.cluster_name
  }

  lifecycle {
    ignore_changes = [name]
  }
}

resource "aws_eks_node_group" "default_node_group" {
  cluster_name    = local.full_cluster_name
  node_group_name = "default_node_group"
  node_role_arn   = aws_iam_role.eks_node_group.arn
  subnet_ids      = var.subnet_ids

  scaling_config {
    desired_size = 2
    max_size     = 3
    min_size     = 1
  }

  instance_types = ["t3.medium"]
  capacity_type  = "ON_DEMAND"
}

resource "aws_iam_role" "eks_node_group" {
  name        = "eks_node_group"
  description = "EKS Node Group Role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Effect = "Allow"
        Sid    = ""
      },
    ]
  })
}