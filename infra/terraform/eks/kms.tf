resource "aws_kms_key" "this" {
  description         = "KMS key for EKS cluster encryption"
  enable_key_rotation = true

  policy = jsonencode({
    Version = "2012-10-17",
    Id = "key-default-1",
    Statement = [
      {
        Sid    = "EnableRootPermissions",
        Effect = "Allow",
        Principal = {
          AWS = "arn:aws:iam::${var.account_id}:root"
        },
        Action   = [
          "kms:*",
          "kms:PutKeyPolicy"
        ],
        Resource = "*"
      },
      {
        Sid    = "AllowGitHubActionsAccess",
        Effect = "Allow",
        Principal = {
          AWS = var.github_actions_role_arn
        },
        Action = [
          "kms:DescribeKey",
          "kms:GetKeyPolicy",
          "kms:ListAliases",
          "kms:CreateAlias",
          "kms:UpdateAlias",
          "kms:TagResource",
          "kms:PutKeyPolicy"
        ],
        Resource = "*"
      }
    ]
  })
}

data "aws_kms_alias" "existing" {
  name = "alias/eks/${var.cluster_name}-${var.environment}"
}

resource "aws_kms_alias" "this" {
  name          = "alias/eks/${var.cluster_name}"
  target_key_id = aws_kms_key.this.key_id
}