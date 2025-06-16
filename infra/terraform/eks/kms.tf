# Try to find existing KMS key by alias first
data "aws_kms_alias" "existing" {
  name = "alias/eks/${var.cluster_name}-${var.environment}"
  
  # Handle case where alias doesn't exist
  count = 1
}

# Check if the alias exists without failing
locals {
  alias_exists = try(data.aws_kms_alias.existing[0].target_key_id != null, false)
  existing_key_id = try(data.aws_kms_alias.existing[0].target_key_id, null)
}

# Create new KMS key only if existing one is not found
resource "aws_kms_key" "this" {
  count = local.alias_exists ? 0 : 1
  
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

  tags = {
    Name = "eks-${var.cluster_name}-${var.environment}"
    Environment = var.environment
  }
}

# Create alias only if it doesn't exist
resource "aws_kms_alias" "this" {
  count = local.alias_exists ? 0 : 1
  
  name          = "alias/eks/${var.cluster_name}-${var.environment}"
  target_key_id = aws_kms_key.this[0].key_id
}

# Output the KMS key ID (either existing or newly created)
locals {
  final_kms_key_id = local.alias_exists ? local.existing_key_id : aws_kms_key.this[0].key_id
}