#!/bin/bash

echo "🔄 Switching to long-term IAM user profile: mlflow-user"

# Step 1: Clear any lingering temporary credentials or assumed-role sessions
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN
unset AWS_PROFILE
unset AWS_DEFAULT_PROFILE

# Step 2: Load the IAM user credentials into environment variables
export AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile mlflow-user)
export AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile mlflow-user)
export AWS_DEFAULT_REGION=$(aws configure get region --profile mlflow-user || echo "us-east-1")

# Step 3: Confirm the identity
echo "🔍 Verifying AWS identity..."
IDENTITY_JSON=$(aws sts get-caller-identity 2>/dev/null)

if [ $? -eq 0 ]; then
    ARN=$(echo "$IDENTITY_JSON" | jq -r '.Arn')
    echo "✅ Now using IAM user: $ARN"
else
    echo "❌ Failed to authenticate using profile 'mlflow-user'"
    echo "💡 Check your ~/.aws/credentials and ~/.aws/config files"
    exit 1
fi




