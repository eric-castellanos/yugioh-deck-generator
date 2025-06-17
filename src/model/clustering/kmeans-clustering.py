import io
import logging
import json
import tempfile
import os
import argparse
from typing import Literal

import boto3
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import mlflow
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
#from kneed import KneeLocator

from src.utils.s3_utils import read_parquet_from_s3
from src.utils.clustering_metrics_summary import calculate_enhanced_clustering_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configure MLflow tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("/yugioh_card_clustering")

# Parameters
S3_BUCKET = "yugioh-data"
N_CLUSTERS = 5 # (based on elbow plot) 
PCA_COMPONENTS = 2
RANDOM_SEED = 42

# Archetype cardinality reduction parameters  
ARCHETYPE_MIN_COUNT = 2  # Only combine archetypes with 1 card (singletons)

# Feature type to S3 key mapping
FEATURE_CONFIGS = {
    "tfidf": {
        "s3_key": "processed/feature_engineered/2025-06/feature_engineered_tfidf.parquet",
        "description": "TF-IDF features only from card descriptions",
        "feature_prefix": "desc_feat_"
    },
    "embeddings": {
        "s3_key": "processed/feature_engineered/2025-06/feature_engineered_embeddings.parquet", 
        "description": "Word embedding features only from card descriptions",
        "feature_prefix": "embed_feat_"
    },
    "combined": {
        "s3_key": "processed/feature_engineered/2025-06/feature_engineered_combined.parquet",
        "description": "Combined TF-IDF and word embedding features from card descriptions", 
        "feature_prefix": ["tfidf_feat_", "embed_feat_"]
    }
}


def run_clustering_experiment(feature_type: Literal["tfidf", "embeddings", "combined"]):
    """Run K-means clustering experiment with specified feature type."""
    
    config = FEATURE_CONFIGS[feature_type]
    s3_key = config["s3_key"]
    
    # Start MLflow run
    with mlflow.start_run():
        # Set tags for easy querying and organization
        mlflow.set_tag("model_type", "clustering")
        mlflow.set_tag("algorithm", "kmeans")
        mlflow.set_tag("dataset", "yugioh_cards")
        mlflow.set_tag("preprocessing", "pca")
        mlflow.set_tag("version", "v1.0")
        mlflow.set_tag("purpose", "card_clustering")
        mlflow.set_tag("data_source", "s3")
        mlflow.set_tag("stage", "exploration")  # exploration, development, production
        
        # Feature-specific tags
        mlflow.set_tag("feature_type", feature_type)
        mlflow.set_tag("feature_description", config["description"])
        
        # Log parameters
        mlflow.log_param("s3_bucket", S3_BUCKET)
        mlflow.log_param("s3_key", s3_key)
        mlflow.log_param("feature_type", feature_type)
        mlflow.log_param("n_clusters", N_CLUSTERS)
        mlflow.log_param("pca_components", PCA_COMPONENTS)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("archetype_min_count", ARCHETYPE_MIN_COUNT)
        
        logging.info(f"Loading {feature_type} feature dataset from S3")
        df = read_parquet_from_s3(S3_BUCKET, s3_key)
        
        # Keep processing in Polars for efficiency
        meta_cols = ["id", "name", "archetype"]
        meta_df = df.select(meta_cols)
        feature_df = df.drop(meta_cols)

        logging.info(f"Data loaded. Original feature shape: {feature_df.shape}")
        
        # Log dataset metrics (using Polars)
        mlflow.log_metric("num_samples", len(df))

        # Drop non-numeric columns (keep all numeric and boolean types for features)
        # This includes: floats, ints, booleans (for categorical dummies), and unsigned ints
        feature_df = feature_df.select([
            col for col in feature_df.columns
            if feature_df[col].dtype in (
                pl.Float32, pl.Float64, pl.Int32, pl.Int64, 
                pl.UInt32, pl.UInt64, pl.UInt8, pl.UInt16,
                pl.Int8, pl.Int16, pl.Boolean
            )
        ])
        
        logging.info(f"After filtering for numeric/boolean columns: {feature_df.shape}")
        logging.info(f"Feature columns included: {feature_df.columns[:10]}...")  # Show first 10 columns

        # Convert to pandas ONLY for MLflow dataset logging (at the very end)
        pandas_df = df.to_pandas()
        
        # Log dataset using MLflow Datasets with enhanced schema
        dataset_source = f"s3://{S3_BUCKET}/{s3_key}"
        
        # Categorize features by type for comprehensive schema
        all_columns = df.columns
        text_features = [col for col in feature_df.columns if col.startswith(('desc_feat_', 'tfidf_feat_', 'embed_feat_'))]
        numeric_features = [col for col in feature_df.columns if col in ['atk', 'def', 'level', 'atk_norm', 'def_norm', 'level_norm']]
        categorical_features = [col for col in feature_df.columns if col not in text_features + numeric_features + meta_cols]
        
        # Create detailed schema information
        schema_info = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "feature_breakdown": {
                "text_features": {
                    "count": len(text_features),
                    "columns": text_features,  # First 10 for brevity
                    "description": f"{feature_type} text features from card descriptions"
                },
                "numeric_features": {
                    "count": len(numeric_features),
                    "columns": numeric_features,
                    "description": "Normalized card stats (attack, defense, level)"
                },
                "categorical_features": {
                    "count": len(categorical_features),
                    "columns": categorical_features,  # First 20 for brevity  
                    "description": "One-hot encoded categorical features (type, attribute, archetype)"
                }
            },
            "metadata_columns": meta_cols,
            "total_features_used": len(feature_df.columns),
            "feature_type": feature_type,
            "feature_description": config["description"],
            "processing_engine": "polars",
            "dtypes_sample": {str(col): str(dtype) for col, dtype in list(zip(df.columns, df.dtypes))[:20]}
        }
        
        # Log enhanced schema information
        mlflow.log_dict(schema_info, "dataset_schema.json")
        
        # Create MLflow dataset (pandas required here)
        dataset = mlflow.data.from_pandas(
            pandas_df, 
            source=dataset_source,
            name=f"yugioh_cards_{feature_type}",
            targets="archetype"  # String, not list
        )
        mlflow.log_input(dataset, context="training")

        # Use Polars DataFrame directly with scikit-learn (if supported)
        # Convert to numpy only when necessary for sklearn operations
        try:
            # Try to use Polars DataFrame directly with sklearn
            X = feature_df
            logging.info("Using Polars DataFrame directly with scikit-learn")
        except Exception as e:
            # Fallback to numpy if Polars support isn't available
            logging.info("Converting to numpy array for scikit-learn compatibility")
            X = feature_df.to_numpy()
        
        # Log feature count after filtering
        mlflow.log_metric("num_features", X.shape[1])

        logging.info(f"Applying PCA to reduce to {PCA_COMPONENTS} components")
        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
        X_reduced = pca.fit_transform(X)
        logging.info(f"PCA complete. Reduced feature shape: {X_reduced.shape}")

        logging.info(f"Running KMeans with {N_CLUSTERS} clusters")
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
        labels = kmeans.fit_predict(X_reduced)

        meta_df = meta_df.with_columns(pl.Series(name="cluster", values=labels))

        # Calculate enhanced clustering metrics
        enhanced_metrics = calculate_enhanced_clustering_metrics(meta_df)
        
        # Log enhanced metrics to MLflow
        for metric_name, metric_value in enhanced_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"enhanced_{metric_name}", metric_value)

        # Calculate and log standard metrics
        if "archetype" in meta_df.columns:
            score = adjusted_rand_score(meta_df["archetype"].to_list(), labels)
            logging.info(f"Adjusted Rand Index (vs. archetype): {score:.4f}")
            mlflow.log_metric("adjusted_rand_index", score)

        silhouette = silhouette_score(X_reduced, labels)
        logging.info(f"Silhouette Score: {silhouette:.4f}")
        mlflow.log_metric("silhouette_score", silhouette)

        # Plot explained variance (scree plot)
        explained_var = pca.explained_variance_ratio_
        mlflow.log_metric("explained_variance_ratio", explained_var.sum())
        
        # Create scree plot as temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(explained_var) + 1), explained_var.cumsum(), marker='o')
            plt.xlabel("Number of Components")
            plt.ylabel("Cumulative Explained Variance")
            plt.title(f"Scree Plot (PCA) - {feature_type.upper()} Features")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow and clean up
            mlflow.log_artifact(tmp_file.name, "plots")
            os.unlink(tmp_file.name)

        logging.info(f"Total explained variance by {PCA_COMPONENTS} components: {explained_var.sum():.4f}")

        logging.info("Projecting to 2D using t-SNE...")
        X_tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED).fit_transform(X_reduced)

        # t-SNE plot as temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=10)

            # Manually create legend based on unique cluster labels
            unique_labels = np.unique(labels)
            legend_patches = [mpatches.Patch(color=scatter.cmap(scatter.norm(i)), label=f"Cluster {i}") for i in unique_labels]
            plt.legend(handles=legend_patches, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.title(f"t-SNE Projection of KMeans Clusters - {feature_type.upper()} Features")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow and clean up
            mlflow.log_artifact(tmp_file.name, "plots")
            os.unlink(tmp_file.name)
        
        # Calculate elbow plot
        inertias = []
        k_values = range(2, 20)

        for k in k_values:
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            kmeans_temp.fit(X_reduced)  # use PCA-reduced features
            inertias.append(kmeans_temp.inertia_)
            print(f"K: {k} -> WCSS: {kmeans_temp.inertia_}")

        # Elbow plot as temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.figure(figsize=(8, 4))
            plt.plot(k_values, inertias, marker='o')
            plt.title(f"Elbow Plot: KMeans Inertia vs. Number of Clusters - {feature_type.upper()} Features")
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Inertia (WCSS)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow and clean up
            mlflow.log_artifact(tmp_file.name, "plots")
            os.unlink(tmp_file.name)

        logging.info("Example cards per cluster:")
        for cluster_id in range(N_CLUSTERS):
            logging.info(f"Cluster {cluster_id}:")
            cluster_cards = meta_df.filter(pl.col("cluster") == cluster_id).head(5)
            for row in cluster_cards.iter_rows():
                logging.info(f" - {row[1]} (Archetype: {row[2]})")

        # Create and log metrics as temporary file
        metrics = {
            "feature_type": feature_type,
            "n_clusters": N_CLUSTERS,
            "pca_components": PCA_COMPONENTS,
            "explained_variance_ratio": float(explained_var.sum()),
            "silhouette_score": float(silhouette),
            "adjusted_rand_index": float(score) if "archetype" in meta_df.columns else None,
            "enhanced_metrics": enhanced_metrics
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(metrics, tmp_file, indent=2)
            tmp_file.flush()
            
            # Log to MLflow and clean up
            mlflow.log_artifact(tmp_file.name, "metrics")
            os.unlink(tmp_file.name)
        
        logging.info(f"MLflow run completed successfully for {feature_type} features")

def main():
    parser = argparse.ArgumentParser(description="Run K-means clustering with different feature types")
    parser.add_argument(
        "--feature-type", 
        type=str, 
        choices=["tfidf", "embeddings", "combined", "all"],
        default="all",
        help="Feature type to use for clustering (default: all)"
    )
    
    args = parser.parse_args()
    
    if args.feature_type == "all":
        # Run all three experiments
        for feature_type in ["tfidf", "embeddings", "combined"]:
            logging.info(f"🚀 Starting {feature_type} clustering experiment...")
            run_clustering_experiment(feature_type)
            logging.info(f"✅ Completed {feature_type} clustering experiment")
    else:
        # Run single experiment
        logging.info(f"🚀 Starting {args.feature_type} clustering experiment...")
        run_clustering_experiment(args.feature_type)
        logging.info(f"✅ Completed {args.feature_type} clustering experiment")


if __name__ == "__main__":
    main()