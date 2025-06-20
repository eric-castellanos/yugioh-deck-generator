locals {
  resource_names = {
    subnet_group   = "mlflow-db-subnet-group-${var.environment}"
    security_group = "mlflow-rds-sg-${var.environment}"
    db_instance    = "mlflow-db-${var.environment}"
  }
}

resource "aws_db_subnet_group" "mlflow" {
  name        = local.resource_names.subnet_group
  description = "Subnet group for MLFlow RDS instance"
  subnet_ids  = var.subnet_ids

  tags = {
    Name        = "mlflow-db-subnet-group"
    Environment = var.environment
  }
}

resource "aws_security_group" "mlflow_rds_sg" {
  name        = local.resource_names.security_group
  description = "Allow inbound access to RDS"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # TODO: Restrict in prod
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "mlflow-rds-sg"
    Environment = var.environment
  }
}

resource "aws_db_instance" "mlflow" {
  identifier             = local.resource_names.db_instance
  allocated_storage      = var.allocated_storage
  engine                 = "postgres"
  engine_version         = "14.13"
  instance_class         = var.db_instance_class
  db_name                = "${replace(var.db_name, "-", "_")}_${var.environment}"
  username               = var.db_username
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids = [aws_security_group.mlflow_rds_sg.id]
  skip_final_snapshot    = true
  publicly_accessible    = true # False in production

  tags = {
    Name        = "mlflow-db"
    Environment = var.environment
  }
}