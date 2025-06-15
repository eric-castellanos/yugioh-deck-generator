# 🔄 Workflow Migration Checklist

## 📅 **When to Delete Legacy Workflows**

**❌ Don't delete yet if:**
- [ ] You haven't tested the new workflows in your environment
- [ ] Team members are still using the legacy workflows  
- [ ] You have critical infrastructure managed by legacy workflows
- [ ] You want to keep them as reference/backup

**✅ Safe to delete when:**
- [ ] New workflows tested successfully in dev environment
- [ ] All team members trained on new workflows
- [ ] No critical infrastructure depends on legacy workflows
- [ ] At least 1-2 months of successful usage of new workflows

## 🧪 **Testing New Workflows**

### **Phase 1: Test Infrastructure Deploy**
```bash
# Test in dev environment
Workflow: Infrastructure Deploy
- environment: dev  
- clean_setup: true (if testing clean setup)
- auto_approve: false
```

### **Phase 2: Test MLflow Deploy**
```bash
# Test MLflow deployment
Workflow: MLflow Kubernetes Deploy
- environment: dev
- action: deploy
```

### **Phase 3: Test Complete Teardown**
```bash
# Test cleanup (use test environment!)
Workflow: MLflow Kubernetes Deploy
- action: destroy

Workflow: Infrastructure Destroy  
- environment: dev
- confirm_destroy: destroy
- clean_state_backend: true
```

## 📋 **Migration Steps**

### **Step 1: Update Team Documentation (Done ✅)**
- [x] Mark legacy workflows as deprecated
- [x] Add migration notices to workflow files
- [x] Update README with recommendations

### **Step 2: Test New Workflows**
- [ ] Test clean setup in dev environment
- [ ] Test MLflow deployment and functionality  
- [ ] Test complete teardown and cleanup
- [ ] Verify all features work as expected

### **Step 3: Gradual Migration**
- [ ] Use new workflows for all new environments
- [ ] Migrate existing environments one by one
- [ ] Train team members on new workflows

### **Step 4: Full Migration**
- [ ] All infrastructure managed by new workflows
- [ ] Team comfortable with new workflows
- [ ] No dependencies on legacy workflows

### **Step 5: Cleanup (Future)**
- [ ] Delete `plan-and-apply.yml`
- [ ] Delete `destroy.yml`  
- [ ] Update documentation
- [ ] Archive migration documents

## 🔍 **What to Test**

### **Infrastructure Deploy Workflow:**
- [ ] Clean setup from fresh AWS account
- [ ] State backend creation (S3 + DynamoDB)
- [ ] VPC and networking creation
- [ ] EKS cluster deployment
- [ ] RDS database creation
- [ ] S3 bucket creation
- [ ] Plan review and approval process

### **MLflow Deploy Workflow:**
- [ ] MLflow pod deployment
- [ ] Database connectivity
- [ ] S3 artifact storage  
- [ ] LoadBalancer creation
- [ ] Health checks and monitoring
- [ ] UI accessibility
- [ ] Experiment creation and logging

### **Destroy Workflows:**
- [ ] Kubernetes cleanup before EKS destruction
- [ ] Complete infrastructure teardown
- [ ] State backend cleanup
- [ ] No leftover resources

## 🚨 **Red Flags - Don't Delete Legacy Yet**

- ❌ New workflows failing in testing
- ❌ Missing functionality compared to legacy
- ❌ Team not comfortable with new workflows  
- ❌ Critical production infrastructure on legacy
- ❌ Unexplained differences in behavior

## ✅ **Green Flags - Safe to Delete Legacy**

- ✅ All tests passing with new workflows
- ✅ Feature parity or better than legacy
- ✅ Team trained and comfortable
- ✅ All infrastructure migrated successfully  
- ✅ Consistent behavior and results

## 📞 **Support During Migration**

If you encounter issues during migration:

1. **Check the logs** in GitHub Actions artifacts
2. **Compare with legacy workflow** behavior  
3. **Review the README.md** for troubleshooting
4. **Use the debug commands** provided in documentation
5. **Keep legacy workflows** as fallback during transition

---

**Recommendation: Keep legacy workflows for at least 1-2 months while you validate the new ones work perfectly for your use case.**
