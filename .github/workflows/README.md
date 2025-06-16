# MLflow on AWS - GitHub Actions Workflows

This repository contains GitHub Actions workflows for deploying and managing MLflow on AWS infrastructure using Terraform and Kubernetes.

## 🏗️ **Workflows Overview**

> **🆕 NEW**: Use the modern workflows below for all new deployments. Legacy workflows are kept for backward compatibility.

### 1. **Infrastructure Deploy** (`infrastructure-deploy.yml`) 🆕
Deploys the complete AWS infrastructure for MLflow from scratch or manages existing resources.

**Prerequisites:**
- ⚠️ **Terraform Backend**: Assumes S3 backend bucket and DynamoDB lock table already exist

**Capabilities:**
- ✅ **Clean Setup**: Can deploy to AWS accounts with existing Terraform backend
- ✅ **Resume Deployment**: Can resume infrastructure from suspended state
- ✅ **Environment Support**: Supports dev/staging/prod environments
- ✅ **Approval Options**: Manual approval or auto-approve modes

### 2. **Infrastructure Destroy Complete** (`infrastructure-destroy-complete.yml`)
Safely tears down all MLflow infrastructure including optional state backend cleanup.

**Capabilities:**
- ✅ **Complete Teardown**: Destroys all infrastructure resources
- ✅ **Kubernetes Cleanup**: Removes K8s resources before destroying EKS
- ✅ **Enhanced IAM Cleanup**: Works around GitHub Actions permission limitations
- ✅ **State Backend Cleanup**: Optional destruction of Terraform state storage
- ✅ **Safety Checks**: Requires explicit confirmation (`destroy-all`)

### 3. **MLflow Kubernetes Deploy** (`mlflow-k8s-deploy.yml`)
Deploys, redeploys, or destroys MLflow application on existing EKS infrastructure.

**Capabilities:**
- ✅ **MLflow Deployment**: Deploys MLflow with PostgreSQL and S3 integration
- ✅ **IAM Integration**: Sets up IRSA (IAM Roles for Service Accounts)
- ✅ **LoadBalancer Setup**: Creates AWS LoadBalancer for external access
- ✅ **Health Checks**: Verifies deployment health

## 🚀 **Deployment Workflows**

### **Scenario 1: Fresh AWS Account (Clean Setup)**

**Prerequisites:**
- Create Terraform state backend first (S3 bucket + DynamoDB table)
- Or run any destroy workflow once with `clean_state_backend: false` to create backend

1. **Deploy Infrastructure**:
   ```
   Workflow: Infrastructure Deploy
   Inputs:
   - environment: dev
   - deployment_type: clean_setup
   - auto_approve: false (recommended for first run)
   ```

2. **Deploy MLflow**:
   ```
   Workflow: MLflow Kubernetes Deploy
   Inputs:
   - environment: dev
   - action: deploy
   ```

### **Scenario 2: Existing Infrastructure**

1. **Deploy MLflow Only**:
   ```
   Workflow: MLflow Kubernetes Deploy
   Inputs:
   - environment: dev
   - action: deploy
   ```

### **Scenario 3: Complete Teardown**

1. **Destroy MLflow**:
   ```
   Workflow: MLflow Kubernetes Deploy
   Inputs:
   - environment: dev
   - action: destroy
   ```

2. **Destroy Infrastructure**:
   ```
   Workflow: Infrastructure Destroy Complete
   Inputs:
   - environment: dev
   - confirm_destroy: destroy-all
   - clean_state_backend: true
   ```

## ⚙️ **Required Secrets**

Configure these secrets in your GitHub repository:

| Secret Name | Description | Example |
|------------|-------------|---------|
| `AWS_ROLE_TO_ASSUME` | GitHub Actions OIDC role ARN | `arn:aws:iam::123456789012:role/github-actions-role` |
| `AWS_ACCOUNT_ID` | Your AWS Account ID | `123456789012` |
| `MLFLOW_DB_PASSWORD` | Password for RDS PostgreSQL | `YourSecurePassword123!` |

## 🔧 **Workflow Features**

### **Infrastructure Deploy Workflow**

**Prerequisites:**
- Terraform S3 backend bucket and DynamoDB lock table must already exist
- Use `infrastructure-destroy-complete.yml` with `clean_state_backend: false` to create backend if needed

**Key Features:**
- **Assumes Existing Backend**: Uses pre-configured S3 bucket and DynamoDB table for Terraform state
- **Environment Isolation**: Each environment uses its own state backend configuration
- **Flexible Configuration**: Supports both clean setup and existing resources  
- **Error Recovery**: Handles state lock issues automatically
- **Plan Artifacts**: Uploads Terraform plans for review

**Environment Variables Created:**
```bash
# For clean setup
existing_resources = false
vpc_id = null  # Will create new VPC

# For resume deployment
existing_resources = true  # Uses existing tfvars configuration
```

### **Infrastructure Destroy Complete Workflow**

**Key Features:**
- **Safe Destruction**: Requires explicit "destroy-all" confirmation
- **K8s Cleanup First**: Removes Kubernetes resources before destroying EKS
- **Enhanced IAM Cleanup**: Works around GitHub Actions permission limitations  
- **LoadBalancer Cleanup**: Ensures AWS LoadBalancers are deleted
- **Optional State Cleanup**: Can destroy Terraform state backend
- **Detailed Logging**: Uploads destruction logs as artifacts

### **MLflow K8s Deploy Workflow**

**Key Features:**
- **IRSA Setup**: Automatically creates IAM roles for service accounts
- **S3 Integration**: Configures MLflow for S3 artifact storage
- **Database Integration**: Connects to RDS PostgreSQL
- **LoadBalancer**: Creates AWS LoadBalancer for external access
- **Health Monitoring**: Built-in health checks and monitoring

## 📁 **Generated Resources**

### **AWS Resources Created:**
- **VPC & Networking**: VPC, subnets, route tables, internet gateway
- **EKS Cluster**: Kubernetes cluster with managed node groups
- **RDS PostgreSQL**: Database for MLflow metadata
- **S3 Bucket**: Artifact storage for MLflow
- **IAM Roles**: Service accounts and OIDC integration
- **LoadBalancer**: External access to MLflow UI

### **Kubernetes Resources:**
- **Namespace**: `mlflow`
- **Deployment**: MLflow server pods
- **Services**: ClusterIP and LoadBalancer services  
- **Secrets**: Database and S3 credentials
- **ServiceAccount**: With IAM role annotations

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **State Lock Errors**: Workflows automatically handle and retry
2. **VPC Limits**: Set `existing_resources = true` to use existing VPC
3. **LoadBalancer Pending**: Wait 5-10 minutes for AWS to provision
4. **Pod CrashLoop**: Check database connectivity and credentials

### **Debug Commands:**
```bash
# Check MLflow pods
kubectl get pods -n mlflow

# Check service status  
kubectl get svc -n mlflow

# View pod logs
kubectl logs -f deployment/mlflow -n mlflow

# Test database connection
kubectl run postgres-client --rm -it --restart=Never --image=postgres:13 --env="PGPASSWORD=<password>" -- psql -h <endpoint> -U mlflowadmin -d mlflow_db_dev
```

## 🎯 **Next Steps After Deployment**

1. **Access MLflow UI**: Get LoadBalancer URL with `kubectl get svc mlflow-ingress -n mlflow`
2. **Test Functionality**: Create experiments, log metrics, upload artifacts
3. **Configure DNS**: Optionally set up custom domain with Route53
4. **SSL/TLS**: Add certificate for HTTPS access
5. **Monitoring**: Set up CloudWatch monitoring and alerts

## 🔐 **Security Considerations**

- **IAM Roles**: Uses IRSA instead of access keys
- **Network Security**: EKS in private subnets with security groups
- **Database Security**: RDS with VPC security groups
- **State Security**: Encrypted S3 backend with DynamoDB locking
- **Secrets Management**: Kubernetes secrets for sensitive data

## 📚 **Additional Resources**

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [EKS User Guide](https://docs.aws.amazon.com/eks/latest/userguide/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## 🔄 **Legacy Workflows (Deprecated)**

> **⚠️ DEPRECATED**: These workflows are kept for backward compatibility but are not recommended for new deployments.

### **Legacy Plan & Apply** (`plan-and-apply.yml`)
- Use only if you need the exact old behavior
- Limited to existing resources and manual configuration
- Will be removed in future updates

### **Legacy Destroy** (`destroy.yml`) 
- Use only for infrastructure created with legacy workflow
- Limited cleanup capabilities
- Will be removed in future updates

**👉 Recommendation**: Migrate to the new workflows above for better features and reliability.

---
