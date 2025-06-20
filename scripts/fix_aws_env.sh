#!echo "🔄 Setting up AWS credentials for MLflow..."

# First, unset any existing environment variables
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN

# Export direct IAM user credentials with the necessary permissions
echo "🔑 Using mlflow-user IAM credentials with direct S3 and RDS access..."

# Set AWS_PROFILE to use mlflow-user for all AWS CLI commands
export AWS_PROFILE=mlflow-user
export AWS_DEFAULT_PROFILE=mlflow-user

# Load credentials from AWS profile
export AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile mlflow-user)
export AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile mlflow-user)
export AWS_DEFAULT_REGION=$(aws configure get region --profile mlflow-user || echo "us-east-1")cho "🔄 Setting up AWS credentials for MLflow..."

# First, unset any existing environment variables
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN

# Export direct IAM user credentials with the necessary permissions
echo "� Using mlflow-user IAM credentials with direct S3 and RDS access..."

# Load credentials from AWS profile
export AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile mlflow-user)
export AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile mlflow-user)
export AWS_DEFAULT_REGION=$(aws configure get region --profile mlflow-user)

# No session token needed for direct IAM user
unset AWS_SESSION_TOKEN

# Update the .env.mlflow-tunnel file with the mlflow-user credentials
if [ -f .env.mlflow-tunnel ]; then
    sed -i "/^AWS_ACCESS_KEY_ID=/ c\AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" .env.mlflow-tunnel
    sed -i "/^AWS_SECRET_ACCESS_KEY=/ c\AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" .env.mlflow-tunnel
    # Remove any session token line and replace with a comment
    sed -i '/AWS_SESSION_TOKEN=/ c\# No session token needed for IAM user with direct access' .env.mlflow-tunnel
    echo "✅ Updated .env.mlflow-tunnel with mlflow-user credentials"
fi
# Verify the credentials work
if aws sts get-caller-identity --output json > /dev/null 2>&1; then
    AWS_IDENTITY=$(aws sts get-caller-identity --output json)
    echo "ℹ️  Current identity: $(echo $AWS_IDENTITY | jq -r '.Arn')"
    echo ""
    echo "✅ AWS credentials set up successfully!"
    
    # Test S3 access
    if aws s3 ls s3://${S3_BUCKET} > /dev/null 2>&1; then
        echo "✅ S3 bucket access verified"
    else
        echo "⚠️  Could not access S3 bucket. Check S3 permissions"
    fi
    
    echo ""
    echo "ℹ️  To use these credentials with MLflow, run:"
    echo "    ./scripts/mlflow-local.sh restart"
else
    echo "❌ No valid AWS credentials found"
    echo ""
    echo "💡 Please check your mlflow-user AWS profile"
    echo "💡 Make sure you have valid AWS access key, secret key, and region configured"
    exit 1
fi
