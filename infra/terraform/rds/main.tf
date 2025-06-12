locals {
  resource_names = {
    subnet_group   = "mlflow-db-subnet-group-${var.environment}"
    security_group = "mlflow-rds-sg-${var.environment}"
    db_instance    = "mlflow-db-${var.environment}"
  }
}

resource "aws_db_subnet_group" "mlflow" {
  count = var.create_resources ? 1 : 0

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
  count = var.create_resources ? 1 : 0

  identifier             = local.resource_names.db_instance
  allocated_storage      = var.allocated_storage
  engine                 = "postgres"
  engine_version         = "14.7"
  instance_class         = var.db_instance_class
  db_name                = var.db_name
  username               = var.db_username
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.mlflow[0].name
  vpc_security_group_ids = [aws_security_group.mlflow_rds_sg.id]
  skip_final_snapshot    = true
  publicly_accessible    = true # False in production

  tags = {
    Name        = "mlflow-db"
    Environment = var.environment
  }
}