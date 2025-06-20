#!/bin/bash

# MLflow Local Development Setup Script

set -e

echo "🚀 Setting up MLflow Local Development Environment"
echo "=================================================="

# Load environment variables
if [ -f .env.mlflow-tunnel ]; then
    echo "📝 Loading environment variables..."
    source .env.mlflow-tunnel
elif [ -f .env.mlflow-local ]; then
    echo "📝 Loading environment variables..."
    source .env.mlflow-local
else
    echo "❌ Environment file not found. Please create .env.mlflow-tunnel or .env.mlflow-local first."
    exit 1
fi

# Function to get AWS credentials
setup_aws_credentials() {
    echo "🔑 Setting up AWS credentials..."
    
    # Check if we have the AWS credentials in the environment variables already
    if [ ! -z "$AWS_ACCESS_KEY_ID" ] && [ ! -z "$AWS_SECRET_ACCESS_KEY" ] && [ ! -z "$AWS_SESSION_TOKEN" ]; then
        echo "✅ Using AWS credentials from environment variables"
        
        # Verify the credentials work
        if aws sts get-caller-identity &> /dev/null; then
            echo "✅ AWS credentials are verified and working"
            return 0
        else
            echo "⚠️  AWS credentials in environment are invalid or expired"
        fi
    fi
    
    # If not, check if credentials are in the .env.mlflow-tunnel file
    if [ -f .env.mlflow-tunnel ]; then
        echo "� Checking for AWS credentials in .env.mlflow-tunnel..."
        
        # Extract credentials from the .env file
        export AWS_ACCESS_KEY_ID=$(grep AWS_ACCESS_KEY_ID .env.mlflow-tunnel | cut -d '=' -f2)
        export AWS_SECRET_ACCESS_KEY=$(grep AWS_SECRET_ACCESS_KEY .env.mlflow-tunnel | cut -d '=' -f2)
        
        if [ ! -z "$AWS_ACCESS_KEY_ID" ] && [ ! -z "$AWS_SECRET_ACCESS_KEY" ] && [ ! -z "$AWS_SESSION_TOKEN" ]; then
            echo "✅ Found AWS credentials in .env.mlflow-tunnel"
            
            # Verify the credentials work
            if aws sts get-caller-identity &> /dev/null; then
                echo "✅ AWS credentials are verified and working"
                return 0
            else
                echo "⚠️  AWS credentials in .env.mlflow-tunnel are invalid or expired"
            fi
        fi
    fi
    
    # If we still don't have working credentials, try the AWS CLI
    if command -v aws &> /dev/null; then
        echo "AWS CLI found. Attempting to use existing credentials..."
        
        # Try to get current credentials and test them
        if aws sts get-caller-identity &> /dev/null; then
            echo "✅ AWS credentials from AWS CLI are working"
            return 0
        else
            echo "⚠️  Current AWS CLI credentials are expired or invalid"
            echo "💡 Please run './fix_aws_env.sh' to refresh your AWS credentials"
            return 1
        fi
    else
        echo "❌ AWS CLI not found. Please install it first"
        return 1
    fi
}

# Function to test RDS connectivity
test_rds_connection() {
    echo "🔗 Testing RDS connection..."
    
    if command -v psql &> /dev/null; then
        # Use DB_PORT if defined, otherwise default to 5432
        PORT="${DB_PORT:-5432}"
        echo "Attempting to connect to $RDS_ENDPOINT:$PORT..."
        
        if PGPASSWORD="$MLFLOW_DB_PASSWORD" psql -h "$RDS_ENDPOINT" -p "$PORT" -U "mlflowadmin" -d "mlflow_db_dev" -c "SELECT 1;" &> /dev/null; then
            echo "✅ RDS connection successful"
        else
            echo "❌ RDS connection failed. Check your credentials and network connectivity"
            exit 1
        fi
    else
        echo "⚠️  psql not found. Skipping RDS connection test"
    fi
}

# Function to test S3 connectivity
test_s3_connection() {
    echo "📦 Testing S3 connection..."
    
    if aws s3 ls "s3://$S3_BUCKET" &> /dev/null; then
        echo "✅ S3 connection successful"
    else
        echo "❌ S3 connection failed. Check your credentials and bucket permissions"
        exit 1
    fi
}

# Function to start MLflow local server
start_mlflow_local() {
    echo "🐳 Starting MLflow local server (connected to AWS)..."
    
    # Create a temporary .env file for docker-compose
    cat > .env.mlflow-docker <<EOF
RDS_ENDPOINT=${RDS_ENDPOINT}
MLFLOW_DB_PASSWORD=${MLFLOW_DB_PASSWORD}
S3_BUCKET=${S3_BUCKET}
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
EOF
    
    # Start the service
    docker-compose --env-file .env.mlflow-docker -f docker-compose.local-mlflow.yml up -d mlflow-local
    
    # Clean up temporary env file
    rm -f .env.mlflow-docker
    
    echo "⏳ Waiting for MLflow server to start..."
    sleep 10
    
    # Test the server
    for i in {1..30}; do
        if curl -f http://localhost:5000/health &> /dev/null; then
            echo "✅ MLflow server is running!"
            echo "🌐 Access MLflow UI at: http://localhost:5000"
            return 0
        fi
        echo "  Attempt $i/30: Waiting for server..."
        sleep 2
    done
    
    echo "❌ MLflow server failed to start properly"
    docker-compose -f docker-compose.local-mlflow.yml logs mlflow-local
    exit 1
}

# Function to start offline MLflow (no AWS dependencies)
start_mlflow_offline() {
    echo "🐳 Starting MLflow offline server (local database)..."
    
    docker-compose -f docker-compose.local-mlflow.yml --profile offline up -d
    
    echo "⏳ Waiting for MLflow server to start..."
    sleep 15
    
    # Test the server
    for i in {1..30}; do
        if curl -f http://localhost:5001/health &> /dev/null; then
            echo "✅ MLflow offline server is running!"
            echo "🌐 Access MLflow UI at: http://localhost:5001"
            return 0
        fi
        echo "  Attempt $i/30: Waiting for server..."
        sleep 2
    done
    
    echo "❌ MLflow offline server failed to start properly"
    docker-compose -f docker-compose.local-mlflow.yml --profile offline logs
    exit 1
}

# Function to stop all MLflow services
stop_mlflow() {
    echo "🛑 Stopping MLflow services..."
    docker-compose -f docker-compose.local-mlflow.yml down
    docker-compose -f docker-compose.local-mlflow.yml --profile offline down
    echo "✅ MLflow services stopped"
}

# Function to show status
show_status() {
    echo "📊 MLflow Services Status"
    echo "========================"
    docker-compose -f docker-compose.local-mlflow.yml ps
    docker-compose -f docker-compose.local-mlflow.yml --profile offline ps
}

# Function to show logs
show_logs() {
    echo "📝 MLflow Service Logs"
    echo "====================="
    docker-compose -f docker-compose.local-mlflow.yml logs --tail=50 -f
}

# Main menu
case "${1:-menu}" in
    "start")
        if setup_aws_credentials; then
            echo "⚠️  Skipping RDS connection test (will test in container)"
            echo "⚠️  Skipping S3 connection test (will test in container)"
            start_mlflow_local
        else
            echo "❌ AWS credentials setup failed. Consider using offline mode instead:"
            echo "    ./scripts/mlflow-local.sh start-offline"
            exit 1
        fi
        ;;
    "start-offline")
        start_mlflow_offline
        ;;
    "stop")
        stop_mlflow
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    "restart")
        stop_mlflow
        sleep 2
        if setup_aws_credentials; then
            test_rds_connection
            test_s3_connection
            start_mlflow_local
        else
            echo "❌ AWS credentials setup failed. Cannot restart in AWS mode."
            exit 1
        fi
        ;;
    "menu"|*)
        echo "MLflow Local Development Commands:"
        echo "=================================="
        echo "./scripts/mlflow-local.sh start          - Start MLflow connected to AWS (direct)"
        echo "./scripts/mlflow-local.sh start-offline  - Start MLflow with local database"
        echo "./scripts/mlflow-local.sh stop           - Stop all MLflow services"
        echo "./scripts/mlflow-local.sh restart        - Restart MLflow (AWS mode)"
        echo "./scripts/mlflow-local.sh status         - Show service status"
        echo "./scripts/mlflow-local.sh logs           - Show service logs"
        echo ""
        echo "🚀 QUICK START (Daily Use):"
        echo "============================="
        echo "1. ./scripts/mlflow-local.sh start-offline  # Start offline MLflow (recommended)"
        echo "2. Open: http://localhost:5001"
        echo ""
        echo "OR for AWS integration:"
        echo "1. ./scripts/mlflow-local.sh start          # Start AWS-connected MLflow"
        echo "2. Open: http://localhost:5000"
        echo ""
        echo "Offline Mode (recommended for development):"
        echo "- Uses local PostgreSQL database"
        echo "- Uses local file storage"
        echo "- Available at: http://localhost:5001"
        echo "- No AWS credentials required"
        echo ""
        echo "AWS Mode:"
        echo "- Direct connection to public RDS database"
        echo "- Uses your existing S3 bucket"
        echo "- Available at: http://localhost:5000"
        echo "- Requires valid AWS credentials"
        ;;
esac
