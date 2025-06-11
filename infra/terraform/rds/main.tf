resource "aws_db_subnet_group" "mlflow" {
  name       = "mlflow-db-subnet-group-${var.environment}"
  subnet_ids = var.subnet_ids
}

resource "aws_security_group" "mlflow_rds_sg" {
  name        = "mlflow-rds-sg-${var.environment}"
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
}

resource "aws_db_instance" "mlflow" {
  identifier              = "mlflow-db-${var.environment}"
  allocated_storage       = var.allocated_storage
  engine                  = "postgres"
  engine_version          = "14.7"
  instance_class          = var.db_instance_class
  db_name                    = var.db_name
  username                = var.db_username
  password                = var.db_password
  db_subnet_group_name    = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids  = [aws_security_group.mlflow_rds_sg.id]
  skip_final_snapshot     = true
  publicly_accessible     = true # False in production
}