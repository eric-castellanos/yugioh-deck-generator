resource "aws_s3_bucket" "mlflow_bucket" {
    bucket = var.bucket_name

    tags = {
        Name = "mlflow_artifacts"
        environment = var.environment
    }
}

resource "aws_iam_user" "mlflow_user" {
    name = var.mlflow_user
}

resource "aws_iam_policy" "mlflow_s3_policy" {
    name        = "mlflow-s3-access"
    description = "Allow MLflow to access its artifact S3 bucket"
    policy      = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = [ "s3:PutObject", "s3:GetObject", "s3:ListBucket" ],
                Effect = "Allow",
                Resource = [
                    "arn:aws:s3:::${var.bucket_name}",
                    "arn:aws:s3:::${var.bucket_name}/*"
                ]
            }
        ]
    })
}

resource "aws_iam_user_policy_attachment" "attach_mlflow_policy" {
    user = aws_iam_user.mlflow_user
    policy_arn = aws_iam_policy.mlflow_s3_policy.arn
}