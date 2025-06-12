resource "aws_s3_bucket" "mlflow_bucket" {
  count = var.create_resources ? 1 : 0

  bucket = var.bucket_name

  tags = {
    Name        = "mlflow_artifacts"
    environment = var.environment
  }
}

resource "aws_iam_user" "mlflow_user" {
  count = var.create_resources ? 1 : 0

  name = var.mlflow_user
  path = "/"

  tags = {
    Name        = "mlflow_user"
    environment = var.environment
  }
}

data "aws_iam_policy_document" "mlflow_s3_policy" {
  version = "2012-10-17"
  statement {
    actions = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
    effect  = "Allow"
    resources = [
      "arn:aws:s3:::${var.bucket_name}",
      "arn:aws:s3:::${var.bucket_name}/*"
    ]
  }
}

resource "aws_iam_policy" "mlflow_s3_policy" {
  count = var.create_resources ? 1 : 0

  name        = "mlflow-s3-access"
  description = "Allow MLflow to access its artifact S3 bucket"
  policy      = data.aws_iam_policy_document.mlflow_s3_policy.json
}

resource "aws_iam_user_policy_attachment" "mlflow_s3_policy_attachment" {
  count = var.create_resources ? 1 : 0

  user       = aws_iam_user.mlflow_user[0].name
  policy_arn = aws_iam_policy.mlflow_s3_policy[0].arn
}