terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0.0, < 6.0.0"
    }
  }
}

# Data sources for existing infrastructure
data "aws_db_instance" "mlflow" {
  name = var.rds_instance_name
}

data "aws_s3_bucket" "mlflow" {
  bucket = var.s3_bucket_name
}

data "aws_subnet_ids" "private" {
  filter {
    name   = "tag:Name"
    values = [for k, v in var.private_subnet_tags : v]
  }
}

data "aws_security_group" "mlflow_rds" {
  name = "mlflow-rds-sg-${var.environment}"
}

data "aws_security_group" "mlflow" {
  name = "mlflow-sg-${var.environment}"
}

# Use the data sources in the rest of the configuration
resource "aws_lb" "mlflow" {
  name               = "mlflow-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [data.aws_security_group.mlflow.id]
  subnets            = data.aws_subnet_ids.private.ids
}

resource "aws_lb_target_group" "mlflow" {
  name     = "mlflow-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = data.aws_subnet_ids.private.vpc_id
}

# MLflow Load Balancer
resource "aws_lb" "mlflow" {
  name               = "mlflow-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.mlflow.id]
  subnets            = data.aws_subnet_ids.default.ids
}

resource "aws_lb_target_group" "mlflow" {
  name     = "mlflow-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = data.aws_vpc.default.id
}

resource "aws_lb_listener" "mlflow" {
  load_balancer_arn = aws_lb.mlflow.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlflow.arn
  }
}

# MLflow Security Group
resource "aws_security_group" "mlflow" {
  name        = "mlflow-sg"
  description = "Security group for MLflow"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # Consider restricting this in production
  }

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # MLflow default tracking port
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# MLflow IAM Role
resource "aws_iam_role" "mlflow" {
  name = "mlflow-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      },
    ]
  })
}

# MLflow IAM Role Policies
resource "aws_iam_role_policy_attachment" "mlflow_s3" {
  role       = aws_iam_role.mlflow.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "mlflow_rds" {
  role       = aws_iam_role.mlflow.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonRDSFullAccess"
}

resource "aws_iam_role_policy_attachment" "mlflow_ecs" {
  role       = aws_iam_role.mlflow.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonECS_FullAccess"
}

# EFS for Model Registry (Optional)
resource "aws_efs_file_system" "mlflow" {
  creation_token = "mlflow-efs"
  encrypted      = true
  kms_key_id     = aws_kms_key.mlflow.id
}

resource "aws_efs_mount_target" "mlflow" {
  count = length(data.aws_subnet_ids.default.ids)

  file_system_id  = aws_efs_file_system.mlflow.id
  subnet_id       = data.aws_subnet_ids.default.ids[count.index]
  security_groups = [aws_security_group.mlflow.id]
}

resource "aws_kms_key" "mlflow" {
  description             = "KMS key for MLflow EFS"
  deletion_window_in_days = 10
}

# Outputs
output "mlflow_load_balancer_dns" {
  value = aws_lb.mlflow.dns_name
}

output "mlflow_security_group_id" {
  value = aws_security_group.mlflow.id
}

output "mlflow_iam_role_arn" {
  value = aws_iam_role.mlflow.arn
}

output "mlflow_efs_id" {
  value = aws_efs_file_system.mlflow.id
}
