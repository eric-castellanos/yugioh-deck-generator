"""
DBSCAN clustering experiment for Yu-Gi-Oh card data with comprehensive MLflow tracking.
"""

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
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.data

from src.utils.s3_utils import read_parquet_from_s3

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configure MLflow tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("/yugioh_card_clustering")

# Parameters
S3_BUCKET = "yugioh-data"
EPS = 0.5  # Maximum distance between two samples for one to be considered as in the neighborhood of the other
MIN_SAMPLES = 5  # Number of samples in a neighborhood for a point to be considered as a core point
PCA_COMPONENTS = 2
RANDOM_SEED = 42

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
    """Run DBSCAN clustering experiment with specified feature type."""
    
    config = FEATURE_CONFIGS[feature_type]
    s3_key = config["s3_key"]
    
    # Start MLflow run
    with mlflow.start_run():
        # Set tags for easy querying and organization
        mlflow.set_tag("model_type", "clustering")
        mlflow.set_tag("algorithm", "dbscan")
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
        mlflow.log_param("eps", EPS)
        mlflow.log_param("min_samples", MIN_SAMPLES)
        mlflow.log_param("pca_components", PCA_COMPONENTS)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("archetype_min_count", 2)
        
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

        # Convert to pandas ONLY for MLflow dataset logging
        pandas_df = df.to_pandas()
        
        # Log dataset using MLflow Datasets with enhanced schema
        dataset_source = f"s3://{S3_BUCKET}/{s3_key}"
        
        # Categorize features by type for comprehensive schema
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
                    "columns": text_features[:20],  # Limit for readability
                    "description": f"{feature_type} text features from card descriptions"
                },
                "numeric_features": {
                    "count": len(numeric_features),
                    "columns": numeric_features,
                    "description": "Normalized card stats (attack, defense, level)"
                },
                "categorical_features": {
                    "count": len(categorical_features),
                    "columns": categorical_features[:20],  # Limit for readability
                    "description": "One-hot encoded categorical features (type, attribute, archetype)"
                }
            },
            "metadata_columns": meta_cols,
            "total_features_used": len(feature_df.columns),
            "feature_type": feature_type,
            "feature_description": config["description"],
            "processing_engine": "polars"
        }
        
        # Log enhanced schema information
        mlflow.log_dict(schema_info, "dataset_schema.json")
        
        # Create MLflow dataset (pandas required here)
        dataset = mlflow.data.from_pandas(
            pandas_df, 
            source=dataset_source,
            name=f"yugioh_cards_{feature_type}",
            targets="archetype"
        )
        mlflow.log_input(dataset, context="training")

        # Use Polars DataFrame directly with scikit-learn (if supported)
        try:
            X = feature_df
            logging.info("Using Polars DataFrame directly with scikit-learn")
        except Exception as e:
            logging.info("Converting to numpy array for scikit-learn compatibility")
            X = feature_df.to_numpy()
        
        # Log feature count after filtering
        mlflow.log_metric("num_features", X.shape[1])

        # Apply PCA for dimensionality reduction before DBSCAN
        logging.info(f"Applying PCA to reduce to {PCA_COMPONENTS} components")
        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
        X_reduced = pca.fit_transform(X)
        logging.info(f"PCA complete. Reduced feature shape: {X_reduced.shape}")

        # DBSCAN works better with standardized features
        logging.info("Standardizing features for DBSCAN")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)

        logging.info(f"Running DBSCAN with eps={EPS}, min_samples={MIN_SAMPLES}")
        dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
        labels = dbscan.fit_predict(X_scaled)

        # Count clusters and noise points
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logging.info(f"Number of clusters: {n_clusters}")
        logging.info(f"Number of noise points: {n_noise}")
        
        # Log cluster metrics
        mlflow.log_metric("n_clusters_found", n_clusters)
        mlflow.log_metric("n_noise_points", n_noise)
        mlflow.log_metric("noise_percentage", n_noise / len(labels) * 100)

        meta_df = meta_df.with_columns(pl.Series(name="cluster", values=labels))

        # Calculate and log metrics (only if we have more than 1 cluster and not all noise)
        if n_clusters > 1 and n_noise < len(labels):
            if "archetype" in meta_df.columns:
                # Filter out noise points for ARI calculation
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 0:
                    archetype_labels = meta_df["archetype"].to_list()
                    filtered_archetype = [archetype_labels[i] for i in range(len(archetype_labels)) if non_noise_mask[i]]
                    filtered_cluster_labels = labels[non_noise_mask]
                    
                    score = adjusted_rand_score(filtered_archetype, filtered_cluster_labels)
                    logging.info(f"Adjusted Rand Index (vs. archetype, excluding noise): {score:.4f}")
                    mlflow.log_metric("adjusted_rand_index", score)

            # Calculate silhouette score (excluding noise points)
            if np.sum(labels != -1) > 1:
                non_noise_indices = labels != -1
                if len(set(labels[non_noise_indices])) > 1:  # Need at least 2 clusters
                    silhouette = silhouette_score(X_scaled[non_noise_indices], labels[non_noise_indices])
                    logging.info(f"Silhouette Score (excluding noise): {silhouette:.4f}")
                    mlflow.log_metric("silhouette_score", silhouette)
                else:
                    logging.info("Only one cluster found (excluding noise), silhouette score not calculated")
                    mlflow.log_metric("silhouette_score", 0.0)
            else:
                logging.info("All points classified as noise, silhouette score not calculated")
                mlflow.log_metric("silhouette_score", 0.0)
        else:
            logging.info("No meaningful clusters found, skipping clustering metrics")
            mlflow.log_metric("adjusted_rand_index", 0.0)
            mlflow.log_metric("silhouette_score", 0.0)

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
        X_tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED).fit_transform(X_scaled)

        # t-SNE plot as temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.figure(figsize=(10, 6))
            
            # Use a different colormap that handles -1 (noise) well
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                if label == -1:
                    # Plot noise points in gray
                    mask = labels == label
                    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c='gray', s=10, alpha=0.6, label='Noise')
                else:
                    mask = labels == label
                    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors[i]], s=10, label=f'Cluster {label}')

            plt.title(f"t-SNE Projection of DBSCAN Clusters - {feature_type.upper()} Features")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow and clean up
            mlflow.log_artifact(tmp_file.name, "plots")
            os.unlink(tmp_file.name)

        logging.info("Example cards per cluster:")
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                logging.info(f"Noise points (cluster -1):")
                cluster_cards = meta_df.filter(pl.col("cluster") == cluster_id).head(5)
            else:
                logging.info(f"Cluster {cluster_id}:")
                cluster_cards = meta_df.filter(pl.col("cluster") == cluster_id).head(5)
            
            for row in cluster_cards.iter_rows():
                logging.info(f" - {row[1]} (Archetype: {row[2]})")

        # Create and log metrics as temporary file
        metrics = {
            "feature_type": feature_type,
            "eps": EPS,
            "min_samples": MIN_SAMPLES,
            "pca_components": PCA_COMPONENTS,
            "n_clusters_found": int(n_clusters),
            "n_noise_points": int(n_noise),
            "noise_percentage": float(n_noise / len(labels) * 100),
            "explained_variance_ratio": float(explained_var.sum()),
            "silhouette_score": float(silhouette) if 'silhouette' in locals() else 0.0,
            "adjusted_rand_index": float(score) if 'score' in locals() else 0.0
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(metrics, tmp_file, indent=2)
            tmp_file.flush()
            
            # Log to MLflow and clean up
            mlflow.log_artifact(tmp_file.name, "metrics")
            os.unlink(tmp_file.name)
        
        logging.info(f"MLflow run completed successfully for {feature_type} features")

def main():
    parser = argparse.ArgumentParser(description="Run DBSCAN clustering with different feature types")
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
            logging.info(f"🚀 Starting {feature_type} DBSCAN clustering experiment...")
            run_clustering_experiment(feature_type)
            logging.info(f"✅ Completed {feature_type} DBSCAN clustering experiment")
    else:
        # Run single experiment
        logging.info(f"🚀 Starting {args.feature_type} DBSCAN clustering experiment...")
        run_clustering_experiment(args.feature_type)
        logging.info(f"✅ Completed {args.feature_type} DBSCAN clustering experiment")


if __name__ == "__main__":
    main()
