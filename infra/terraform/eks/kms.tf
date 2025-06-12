resource "aws_kms_key" "this" {
  description         = "KMS key for EKS cluster encryption"
  enable_key_rotation = true

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      # Allow full admin control to the root user
      {
        Sid       = "AllowRootAccount",
        Effect    = "Allow",
        Principal = {
          AWS = "arn:aws:iam::${var.account_id}:root"
        },
        Action    = "kms:*",
        Resource  = "*"
      },
      # Allow GitHub Actions role to perform needed actions
      {
        Sid       = "AllowGitHubActionsAccess",
        Effect    = "Allow",
        Principal = {
          AWS = var.github_actions_role_arn
        },
        Action = [
          "kms:DescribeKey",
          "kms:GetKeyPolicy",
          "kms:ListAliases",
          "kms:CreateAlias",
          "kms:UpdateAlias",
          "kms:TagResource"
        ],
        Resource = "*"
      }
    ]
  })
}

resource "aws_kms_alias" "this" {
  name          = "alias/eks/${local.full_cluster_name}"
  target_key_id = aws_kms_key.this.key_id
}