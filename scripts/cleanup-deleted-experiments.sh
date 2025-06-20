#!/bin/bash

# Cleanup script for deleted MLflow experiments
# This script removes S3 artifacts for experiments that have been deleted from MLflow UI

set -e

# Configuration
S3_BUCKET="${S3_BUCKET:-mlflow-backend-dev}"
ARTIFACT_PATH="mlflow-artifacts"

echo "🧹 MLflow S3 Artifact Cleanup Tool"
echo "=================================="
echo "Bucket: s3://$S3_BUCKET/$ARTIFACT_PATH"
echo ""

# Function to list all S3 artifact paths
list_s3_artifacts() {
    echo "📦 Listing S3 artifacts..."
    aws s3 ls "s3://$S3_BUCKET/$ARTIFACT_PATH/" --recursive | awk '{print $4}' | grep -E '^mlflow-artifacts/[0-9]+/' | cut -d'/' -f2 | sort -u
}

# Function to get active experiment IDs from MLflow (requires MLflow server to be running)
get_active_experiments() {
    echo "🔍 Getting active experiments from MLflow..."
    python3 -c "
import mlflow
import sys

try:
    mlflow.set_tracking_uri('http://localhost:5000')
    experiments = mlflow.search_experiments(view_type=mlflow.entities.ViewType.ALL)
    
    active_exp_ids = set()
    for exp in experiments:
        if exp.lifecycle_stage != 'deleted':
            # Get all runs for this experiment
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format='list')
            for run in runs:
                active_exp_ids.add(run.info.run_id)
    
    for run_id in sorted(active_exp_ids):
        print(run_id)
        
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Function to find orphaned artifacts
find_orphaned_artifacts() {
    echo "🔍 Finding orphaned artifacts..."
    
    # Get lists
    s3_artifacts=$(list_s3_artifacts)
    active_experiments=$(get_active_experiments)
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to get active experiments. Make sure MLflow server is running at http://localhost:5000"
        exit 1
    fi
    
    # Find orphaned artifacts
    orphaned_artifacts=""
    for s3_artifact in $s3_artifacts; do
        if ! echo "$active_experiments" | grep -q "^$s3_artifact$"; then
            orphaned_artifacts="$orphaned_artifacts $s3_artifact"
        fi
    done
    
    echo "$orphaned_artifacts"
}

# Function to delete orphaned artifacts
delete_orphaned_artifacts() {
    local orphaned_artifacts="$1"
    
    if [ -z "$orphaned_artifacts" ]; then
        echo "✅ No orphaned artifacts found!"
        return 0
    fi
    
    echo "🗑️  Found orphaned artifacts:"
    for artifact in $orphaned_artifacts; do
        echo "  - $artifact"
    done
    echo ""
    
    read -p "Do you want to delete these artifacts? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Deleting orphaned artifacts..."
        for artifact in $orphaned_artifacts; do
            echo "  Deleting s3://$S3_BUCKET/$ARTIFACT_PATH/$artifact/"
            aws s3 rm "s3://$S3_BUCKET/$ARTIFACT_PATH/$artifact/" --recursive
        done
        echo "✅ Cleanup completed!"
    else
        echo "❌ Cleanup cancelled"
    fi
}

# Main execution
case "${1:-find}" in
    "find")
        orphaned_artifacts=$(find_orphaned_artifacts)
        if [ -z "$orphaned_artifacts" ]; then
            echo "✅ No orphaned artifacts found!"
        else
            echo "🗑️  Found orphaned artifacts:"
            for artifact in $orphaned_artifacts; do
                echo "  - $artifact"
            done
            echo ""
            echo "Run with 'clean' argument to delete them:"
            echo "$0 clean"
        fi
        ;;
    "clean")
        orphaned_artifacts=$(find_orphaned_artifacts)
        delete_orphaned_artifacts "$orphaned_artifacts"
        ;;
    "help"|"--help"|"-h")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  find     Find orphaned artifacts (default)"
        echo "  clean    Find and delete orphaned artifacts"
        echo "  help     Show this help"
        echo ""
        echo "This script finds S3 artifacts that belong to deleted MLflow experiments."
        echo "Make sure MLflow server is running at http://localhost:5000 before running."
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac
