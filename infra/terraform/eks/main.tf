locals {
  full_cluster_name = "${var.cluster_name}-${var.environment}"
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name = local.full_cluster_name
  vpc_id       = var.vpc_id
  
  # Control plane should have access to both public and private subnets
  subnet_ids = concat(var.private_subnet_ids, var.public_subnet_ids)

  create_cloudwatch_log_group = var.create_cloudwatch_log_group

  # Use our custom KMS key (existing or newly created)
  create_kms_key = false
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.this.arn
    resources        = ["secrets"]
  }

  # Enable public access for personal project
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access_cidrs = ["0.0.0.0/0"]

  tags = {
    Environment = var.environment
    Project     = "mlflow"
  }
}

variable "log_retention_in_days" {
  type    = number
  default = 30
}

resource "aws_cloudwatch_log_group" "this" {
  name              = "/aws/eks/${local.full_cluster_name}/cluster"
  retention_in_days = var.log_retention_in_days

  tags = {
    Name = local.full_cluster_name
  }

  lifecycle {
    ignore_changes = [name]
  }
}

resource "aws_eks_node_group" "default_node_group" {
  cluster_name    = local.full_cluster_name
  node_group_name = "default_node_group"
  node_role_arn   = aws_iam_role.eks_node_group.arn
  
  # Temporarily use public subnets to troubleshoot networking issues
  # TODO: Switch back to private subnets once working
  subnet_ids      = var.public_subnet_ids

  scaling_config {
    desired_size = 1  # Start with just 1 node for testing
    max_size     = 3
    min_size     = 1
  }

  instance_types = [var.node_instance_type]
  capacity_type  = "ON_DEMAND"
  
  # Use Amazon Linux 2 instead of AL2023 for better compatibility
  ami_type = "AL2_x86_64"
  
  # Add explicit update config
  update_config {
    max_unavailable = 1
  }

  tags = {
    Environment = var.environment
    Debug = "troubleshooting-node-join"
  }
  
  # Ensure proper dependencies
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.ecr_read_only,
  ]
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

# Required policy attachments for EKS worker nodes
resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node_group.name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node_group.name
}

resource "aws_iam_role_policy_attachment" "ecr_read_only" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node_group.name
}