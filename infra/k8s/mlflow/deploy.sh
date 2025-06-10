#!/bin/bash

# Set colors for output
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
NC="\033[0m" # No Color

# Function to print colored messages
print_info() {
    echo -e "${YELLOW}INFO: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
    exit 1
}

print_success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install kubectl first."
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install AWS CLI first."
fi

# Check if we're in the correct directory
if [ ! -f "deployment.yaml" ]; then
    print_error "Please run this script from the mlflow directory"
fi

# Get EFS ID from AWS
EFS_ID=$(aws efs describe-file-systems --query 'FileSystems[?Name==`mlflow-efs`].FileSystemId' --output text)
if [ -z "$EFS_ID" ]; then
    print_error "Could not find EFS file system. Please create the EFS file system first."
fi

# Create the storage class configuration file with the actual EFS ID
cat > storageclass.yaml << EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
  namespace: mlflow
  labels:
    app: mlflow
    component: storage
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: ${EFS_ID}
  directoryPerms: "700"
  gidRangeStart: "1000"
  gidRangeEnd: "2000"
  basePath: "/dynamic-provisioning"
reclaimPolicy: Retain
volumeBindingMode: Immediate
allowVolumeExpansion: true
EOF

print_info "Deploying MLflow..."

# Deploy in order
print_info "Creating namespace and RBAC..."
kubectl apply -f namespace.yaml
sleep 5

print_info "Creating storage class..."
kubectl apply -f storageclass.yaml
sleep 5

print_info "Creating PVC..."
kubectl apply -f pvc.yaml
sleep 5

print_info "Creating service account and secrets..."
kubectl apply -f serviceaccount.yaml
sleep 5

print_info "Creating configmap..."
kubectl apply -f configmap.yaml
sleep 5

print_info "Creating deployment..."
kubectl apply -f deployment.yaml
sleep 5

print_info "Creating services..."
kubectl apply -f service.yaml
sleep 5

print_info "Waiting for pods to be ready..."
while true; do
    READY=$(kubectl get pods -n mlflow | grep Running | wc -l)
    if [ "$READY" -ge "2" ]; then
        break
    fi
    sleep 5
    print_info "Waiting for pods to be ready..."
done

# Get the external IP
EXTERNAL_IP=$(kubectl get svc mlflow-ingress -n mlflow -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

print_success "MLflow deployment completed successfully!"
print_success "MLflow UI is available at: http://${EXTERNAL_IP}"
print_success "API endpoint: http://${EXTERNAL_IP}:5000"

# Print important environment variables
print_info "\nImportant environment variables:"
print_info "AWS_ACCESS_KEY_ID: $(kubectl get secret mlflow-secrets -n mlflow -o jsonpath='{.data.aws_access_key_id}' | base64 --decode)"
print_info "AWS_SECRET_ACCESS_KEY: $(kubectl get secret mlflow-secrets -n mlflow -o jsonpath='{.data.aws_secret_access_key}' | base64 --decode)"
print_info "BACKEND_STORE_URI: $(kubectl get secret mlflow-secrets -n mlflow -o jsonpath='{.data.backend_store_uri}' | base64 --decode)"
print_info "ARTIFACT_ROOT: $(kubectl get secret mlflow-secrets -n mlflow -o jsonpath='{.data.artifact_root}' | base64 --decode)"
