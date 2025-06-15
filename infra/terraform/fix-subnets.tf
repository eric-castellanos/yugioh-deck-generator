# Fix existing public subnets to auto-assign public IPs for EKS
resource "aws_subnet" "public_subnet_fix_1" {
  subnet_id               = "subnet-05ae996ac53899096"
  map_public_ip_on_launch = true

  tags = {
    Name = "mlflow-vpc-dev-public-us-east-1a"
    Environment = "dev"
    Project = "mlflow"
    "kubernetes.io/cluster/mlflow-cluster-dev" = "shared"
    "kubernetes.io/role/elb" = "1"
  }

  lifecycle {
    ignore_changes = [
      vpc_id,
      cidr_block,
      availability_zone
    ]
  }
}

resource "aws_subnet" "public_subnet_fix_2" {
  subnet_id               = "subnet-0eb6c63c02fadf2a8"
  map_public_ip_on_launch = true

  tags = {
    Name = "mlflow-vpc-dev-public-us-east-1b"
    Environment = "dev"
    Project = "mlflow"
    "kubernetes.io/cluster/mlflow-cluster-dev" = "shared"
    "kubernetes.io/role/elb" = "1"
  }

  lifecycle {
    ignore_changes = [
      vpc_id,
      cidr_block,
      availability_zone
    ]
  }
}

resource "aws_subnet" "public_subnet_fix_3" {
  subnet_id               = "subnet-0177633d8090f0d00"
  map_public_ip_on_launch = true

  tags = {
    Name = "mlflow-vpc-dev-public-us-east-1c"
    Environment = "dev"
    Project = "mlflow"
    "kubernetes.io/cluster/mlflow-cluster-dev" = "shared"
    "kubernetes.io/role/elb" = "1"
  }

  lifecycle {
    ignore_changes = [
      vpc_id,
      cidr_block,
      availability_zone
    ]
  }
}
