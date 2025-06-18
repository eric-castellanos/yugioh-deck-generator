"""
Enhanced HDBSCAN clustering experiments for Yu-Gi-Oh card data with additional configurations        pca_components_override = 10  # Reduce to just 10 components for memory efficiency
1. HDBSCAN with no PCA + L2 Normalization + cosine distance metric
2. HDBSCAN with UMAP + Euclidean distance metric
Plus comprehensive MLflow tracking and enhanced metrics.
"""

import io
import logging
import json
import tempfile
import os
import argparse
import itertools
from typing import Literal

import boto3
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    import umap
except ImportError:
    raise ImportError("umap-learn is required. Install with: pip install umap-learn")

from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize
import mlflow
import mlflow.data

import sys
import os

from src.utils.s3_utils import read_parquet_from_s3
from src.utils.clustering.clustering_metrics_summary import calculate_enhanced_clustering_metrics
from src.utils.clustering.cluster_evaluation import (
    analyze_cluster_top_features,
    find_cluster_representative_cards,
    analyze_archetype_distribution,
    analyze_cluster_stats,
    evaluate_clustering_results
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configure MLflow tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("/yugioh_card_clustering")

# Parameters
S3_BUCKET = "yugioh-data"
MIN_CLUSTER_SIZE = 4  # Minimum size of clusters
MIN_SAMPLES = 3  # Number of samples in a neighborhood for a point to be considered a core point
CLUSTER_SELECTION_EPSILON = 0.1  # Distance threshold for cluster selection
PCA_COMPONENTS = 2
UMAP_COMPONENTS = 30
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
RANDOM_SEED = 42

# UMAP Hyperparameter Grid - User Specified Grid
UMAP_GRID = [
    #{"n_neighbors": 10, "min_dist": 0.0, "n_components": 10},
    #{"n_neighbors": 15, "min_dist": 0.1, "n_components": 20},
    #{"n_neighbors": 30, "min_dist": 0.1, "n_components": 20},
    #{"n_neighbors": 30, "min_dist": 0.25, "n_components": 50},
    #{"n_neighbors": 50, "min_dist": 0.25, "n_components": 50},
    #{"n_neighbors": 30, "min_dist": 0.1, "n_components": 70},
    #{"n_neighbors": 50, "min_dist": 0.5, "n_components": 100}, # this seems to be the best UMAP parameter combination
    {"n_neighbors": 100, "min_dist": 0.0, "n_components": 50}, # or this one
]


# HDBSCAN GRID
HDBSCAN_GRID = [
    {"min_cluster_size": 3, "min_samples": 3, "cluster_selection_epsilon": 0.085}, # smaller values for all params seem to work better
    #{"min_cluster_size": 4, "min_samples": 3, "cluster_selection_epsilon": 0.05},
    #{"min_cluster_size": 5, "min_samples": 5, "cluster_selection_epsilon": 0.1},
    #{"min_cluster_size": 8, "min_samples": 5, "cluster_selection_epsilon": 0.15},
    #{"min_cluster_size": 10, "min_samples": 8, "cluster_selection_epsilon": 0.2},
]


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


def run_enhanced_hdbscan_experiment(
    feature_type: Literal["tfidf", "embeddings", "combined"],
    experiment_config: str,
    umap_params: dict = None,
    hdbscan_params: dict = None
):
    """
    Run enhanced HDBSCAN clustering experiment with specified feature type and configuration.
    
    Args:
        feature_type: Type of features to use
        experiment_config: Configuration name ("pca_euclidean", "umap_euclidean", "umap_grid_search", or "combined_grid_search")
        umap_params: Optional dictionary with UMAP parameters for grid search
        hdbscan_params: Optional dictionary with HDBSCAN parameters for grid search
    """
    
    config = FEATURE_CONFIGS[feature_type]
    s3_key = config["s3_key"]
    
    if experiment_config == "umap_euclidean":
        use_pca = False
        use_umap = True
        use_l2_norm = False
        metric = "euclidean"
        config_description = "UMAP Reduction, Euclidean Distance"
        # Use default UMAP parameters if not specified
        if umap_params is None:
            umap_params = {
                "n_neighbors": UMAP_N_NEIGHBORS,
                "min_dist": UMAP_MIN_DIST,
                "n_components": UMAP_COMPONENTS
            }
    elif experiment_config == "umap_grid_search":
        use_pca = False
        use_umap = True
        use_l2_norm = False
        metric = "euclidean"
        # umap_params must be provided for grid search
        if umap_params is None:
            raise ValueError("umap_params must be provided for grid search")
        config_description = f"UMAP Grid Search (n_neighbors={umap_params['n_neighbors']}, min_dist={umap_params['min_dist']}, n_components={umap_params['n_components']})"
    elif experiment_config == "combined_grid_search":
        use_pca = False
        use_umap = True
        use_l2_norm = False
        metric = "euclidean"
        # Both umap_params and hdbscan_params must be provided for combined grid search
        if umap_params is None or hdbscan_params is None:
            raise ValueError("Both umap_params and hdbscan_params must be provided for combined grid search")
        config_description = f"Combined Grid Search (UMAP: n_neighbors={umap_params['n_neighbors']}, min_dist={umap_params['min_dist']}, n_components={umap_params['n_components']}) (HDBSCAN: min_cluster_size={hdbscan_params['min_cluster_size']}, min_samples={hdbscan_params['min_samples']}, epsilon={hdbscan_params['cluster_selection_epsilon']})"
    elif experiment_config == "pca_euclidean":
        use_pca = True
        use_umap = False
        use_l2_norm = False
        metric = "euclidean"
        config_description = "PCA Reduction, Euclidean Distance"
        umap_params = None  # Not used for PCA
    else:
        raise ValueError(f"Unknown experiment configuration: {experiment_config}")
    
    # Start MLflow run
    with mlflow.start_run():
        # Set tags for easy querying and organization
        mlflow.set_tag("model_type", "clustering")
        mlflow.set_tag("algorithm", "hdbscan_enhanced")
        mlflow.set_tag("dataset", "yugioh_cards")
        mlflow.set_tag("preprocessing", experiment_config)
        mlflow.set_tag("version", "v2.0")
        mlflow.set_tag("purpose", "card_clustering")
        mlflow.set_tag("data_source", "s3")
        mlflow.set_tag("stage", "exploration")
        mlflow.set_tag("distance_metric", metric)
        mlflow.set_tag("experiment_config", experiment_config)

        # Feature-specific tags
        mlflow.set_tag("feature_type", feature_type)
        mlflow.set_tag("feature_description", config["description"])
        mlflow.set_tag("config_description", config_description)
        
        # Log parameters - use grid search parameters if provided
        if hdbscan_params is not None:
            actual_min_cluster_size = hdbscan_params["min_cluster_size"]
            actual_min_samples = hdbscan_params["min_samples"]
            actual_cluster_selection_epsilon = hdbscan_params["cluster_selection_epsilon"]
        else:
            actual_min_cluster_size = MIN_CLUSTER_SIZE
            actual_min_samples = MIN_SAMPLES
            actual_cluster_selection_epsilon = CLUSTER_SELECTION_EPSILON
            
        mlflow.log_param("s3_bucket", S3_BUCKET)
        mlflow.log_param("s3_key", s3_key)
        mlflow.log_param("feature_type", feature_type)
        mlflow.log_param("experiment_config", experiment_config)
        mlflow.log_param("min_cluster_size", actual_min_cluster_size)
        mlflow.log_param("min_samples", actual_min_samples)
        mlflow.log_param("cluster_selection_epsilon", actual_cluster_selection_epsilon)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("metric", metric)
        mlflow.log_param("use_pca", use_pca)
        mlflow.log_param("use_umap", use_umap)
        mlflow.log_param("use_l2_norm", use_l2_norm)
        
        if use_pca:
            components_to_use = locals().get('pca_components_override', PCA_COMPONENTS)
            mlflow.log_param("pca_components", components_to_use)
        if use_umap:
            if umap_params is not None:
                mlflow.log_param("umap_components", umap_params["n_components"])
                mlflow.log_param("umap_n_neighbors", umap_params["n_neighbors"])
                mlflow.log_param("umap_min_dist", umap_params["min_dist"])
            else:
                mlflow.log_param("umap_components", UMAP_COMPONENTS)
                mlflow.log_param("umap_n_neighbors", UMAP_N_NEIGHBORS)
                mlflow.log_param("umap_min_dist", UMAP_MIN_DIST)
        
        logging.info(f"Loading {feature_type} feature dataset from S3")
        df = read_parquet_from_s3(S3_BUCKET, s3_key)
        
        # Keep processing in Polars for efficiency
        meta_cols = ["id", "name", "archetype"]
        meta_df = df.select(meta_cols)
        feature_df = df.drop(meta_cols)

        logging.info(f"Data loaded. Original feature shape: {feature_df.shape}")
        
        # Log dataset metrics
        mlflow.log_metric("num_samples", len(df))

        # Drop non-numeric columns
        feature_df = feature_df.select([
            col for col in feature_df.columns
            if feature_df[col].dtype in (
                pl.Float32, pl.Float64, pl.Int32, pl.Int64, 
                pl.UInt32, pl.UInt64, pl.UInt8, pl.UInt16,
                pl.Int8, pl.Int16, pl.Boolean
            )
        ])
        
        logging.info(f"After filtering for numeric/boolean columns: {feature_df.shape}")

        # Convert to pandas for MLflow dataset logging
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
                    "columns": text_features[:20],  # First 20 for brevity
                    "description": f"{feature_type} text features from card descriptions"
                },
                "numeric_features": {
                    "count": len(numeric_features),
                    "columns": numeric_features,
                    "description": "Normalized card stats (attack, defense, level)"
                },
                "categorical_features": {
                    "count": len(categorical_features),
                    "columns": categorical_features[:20],  # First 20 for brevity  
                    "description": "One-hot encoded categorical features (type, attribute, archetype)"
                }
            },
            "metadata_columns": meta_cols,
            "total_features_used": len(feature_df.columns),
            "feature_type": feature_type,
            "feature_description": config["description"],
            "processing_engine": "polars",
            "experiment_config": experiment_config,
            "dtypes_sample": {str(col): str(dtype) for col, dtype in list(zip(df.columns, df.dtypes))[:20]}
        }
        
        # Log enhanced schema information
        mlflow.log_dict(schema_info, "dataset_schema.json")
        
        # Create MLflow dataset (pandas required here)
        dataset = mlflow.data.from_pandas(
            pandas_df, 
            source=dataset_source,
            name=f"yugioh_cards_{feature_type}_{experiment_config}",
            targets="archetype"  # String, not list
        )
        mlflow.log_input(dataset, context="training")
        
        # Get feature matrix as numpy array for scikit-learn
        X = feature_df.to_numpy()
        mlflow.log_metric("num_features", X.shape[1])
        
        logging.info(f"Feature matrix shape: {X.shape}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply dimensionality reduction and/or normalization based on config
        X_final = X_scaled
        reducer = None
        explained_variance = None
        
        if use_pca:
            # Use override for PCA components if specified (for memory optimization)
            components = locals().get('pca_components_override', PCA_COMPONENTS)
            logging.info(f"Applying PCA with {components} components...")
            pca = PCA(n_components=components, random_state=RANDOM_SEED)
            X_final = pca.fit_transform(X_scaled)
            explained_variance = pca.explained_variance_ratio_
            reducer = pca
            # Store explained variance for artifacts but don't log as metric
            
        elif use_umap:
            if umap_params is not None:
                n_components = umap_params["n_components"]
                n_neighbors = umap_params["n_neighbors"]
                min_dist = umap_params["min_dist"]
                logging.info(f"Applying UMAP with {n_components} components, {n_neighbors} neighbors, {min_dist} min_dist...")
                umap_reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=RANDOM_SEED,
                    metric='cosine'  # Use euclidean for UMAP
                )
            else:
                logging.info(f"Applying UMAP with {UMAP_COMPONENTS} components...")
                umap_reducer = umap.UMAP(
                    n_components=UMAP_COMPONENTS,
                    n_neighbors=UMAP_N_NEIGHBORS,
                    min_dist=UMAP_MIN_DIST,
                    random_state=RANDOM_SEED,
                    metric='cosine'
                )
            X_final = umap_reducer.fit_transform(X_scaled)
            reducer = umap_reducer
            
        if use_l2_norm:
            logging.info("Applying L2 normalization...")
            X_final = normalize(X_final, norm='l2')
        
        logging.info(f"Final feature matrix shape: {X_final.shape}")
        # Store final feature count for artifacts but don't log as metric

        # Fit HDBSCAN
        logging.info(f"Running HDBSCAN with config: {config_description}")
        clusterer = HDBSCAN(
            min_cluster_size=actual_min_cluster_size,
            min_samples=actual_min_samples,
            metric=metric,
            cluster_selection_epsilon=actual_cluster_selection_epsilon
        )
        labels = clusterer.fit_predict(X_final)

        # Count clusters and noise points
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logging.info(f"Number of clusters: {n_clusters}")
        logging.info(f"Number of noise points: {n_noise}")
        
        # Log essential cluster metrics only
        mlflow.log_metric("total_clusters", n_clusters)  # Essential metric
        mlflow.log_metric("noise_percentage", n_noise / len(labels) * 100)
        
        # Store HDBSCAN-specific metrics for artifacts but don't log to MLflow
        persistence_metrics = {}
        if hasattr(clusterer, 'cluster_persistence_'):
            persistence_scores = clusterer.cluster_persistence_
            if len(persistence_scores) > 0:
                persistence_metrics = {
                    "avg_cluster_persistence": float(np.mean(persistence_scores)),
                    "max_cluster_persistence": float(np.max(persistence_scores)),
                    "min_cluster_persistence": float(np.min(persistence_scores))
                }

        # Add cluster labels to metadata
        meta_df = meta_df.with_columns(pl.Series(name="cluster", values=labels))

        # Calculate enhanced clustering metrics
        enhanced_metrics = calculate_enhanced_clustering_metrics(meta_df)
        
        # Store enhanced metrics for artifacts but don't log them to MLflow
        # (Only keeping essential metrics: total_clusters, silhouette_score, num_samples, num_features, noise_percentage)

        # Calculate detailed cluster evaluation metrics
        if n_clusters > 0:
            logging.info("Calculating detailed cluster evaluation metrics...")
            
            # Get feature names if available
            feature_names = None
            if hasattr(feature_df, 'columns'):
                feature_names = feature_df.columns
            elif isinstance(feature_df, pl.DataFrame):
                feature_names = feature_df.columns
            
            # Analyze top features per cluster
            top_features_analysis = analyze_cluster_top_features(
                X_final, labels, top_n=10, feature_names=feature_names
            )
            
            # Calculate metrics from top features analysis
            if top_features_analysis:
                feature_diversity_scores = []
                for cluster_id, features in top_features_analysis.items():
                    # Calculate feature diversity within cluster (how spread out the top features are)
                    if len(features) > 1:
                        # For numerical features, calculate std of feature indices as diversity measure
                        if isinstance(features[0], int):
                            feature_diversity = np.std(features)
                        else:
                            feature_diversity = len(set(features)) / len(features)  # Unique ratio for string features
                        feature_diversity_scores.append(feature_diversity)
                
                # Feature diversity scores calculated but only stored in artifacts, not logged as metrics
            
            # Find representative cards and calculate representativeness metrics
            try:
                representative_cards = find_cluster_representative_cards(
                    X_final, labels, meta_df, top_n=5
                )
                
                if representative_cards:
                    # Calculate average similarity to centroid across all clusters
                    similarity_scores = []
                    cluster_cohesion_scores = []
                    
                    for cluster_id, cards in representative_cards.items():
                        cluster_similarities = [card.get('similarity_to_centroid', 0) for card in cards]
                        if cluster_similarities:
                            avg_similarity = np.mean(cluster_similarities)
                            similarity_scores.append(avg_similarity)
                            # Cohesion: how similar the top cards are to each other (lower std = more cohesive)
                            cohesion = 1.0 - np.std(cluster_similarities) if len(cluster_similarities) > 1 else 1.0
                            cluster_cohesion_scores.append(cohesion)
                    
                    # Similarity and cohesion scores calculated but only stored in artifacts, not logged as metrics
                        
            except Exception as e:
                logging.warning(f"Could not calculate representative card metrics: {e}")
            
            # Analyze archetype distribution and calculate purity metrics
            if "archetype" in meta_df.columns:
                try:
                    archetype_dist = analyze_archetype_distribution(labels, meta_df, normalize=True)
                    
                    if archetype_dist:
                        # Calculate archetype purity (how dominant the top archetype is in each cluster)
                        purity_scores = []
                        archetype_coverage_scores = []
                        
                        for cluster_id, arch_dist in archetype_dist.items():
                            if arch_dist:
                                # Purity: proportion of the most common archetype
                                max_proportion = max(arch_dist.values())
                                purity_scores.append(max_proportion)
                                
                                # Coverage: how many different archetypes are represented
                                num_archetypes = len(arch_dist)
                                archetype_coverage_scores.append(num_archetypes)
                        
                        # Purity and coverage scores calculated but only stored in artifacts, not logged as metrics
                            
                except Exception as e:
                    logging.warning(f"Could not calculate archetype distribution metrics: {e}")
            
            # Analyze cluster stats for numeric columns
            try:
                numeric_columns = ['atk', 'def', 'level']
                available_numeric_cols = [col for col in numeric_columns if col in meta_df.columns]
                
                if available_numeric_cols:
                    cluster_stats = analyze_cluster_stats(labels, meta_df, available_numeric_cols)
                    
                    if cluster_stats:
                        # Calculate variance in stats across clusters (higher = more diverse clusters)
                        for col in available_numeric_cols:
                            col_values = []
                            for cluster_id, stats in cluster_stats.items():
                                if col in stats and not np.isnan(stats[col]):
                                    col_values.append(stats[col])
                            
                            # Cluster diversity and stats calculated but only stored in artifacts, not logged as metrics
                                
            except Exception as e:
                logging.warning(f"Could not calculate cluster stats metrics: {e}")
            
            logging.info("Detailed cluster evaluation metrics calculated and logged")

        # Calculate and log standard metrics (only if we have meaningful clusters)
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
                    # Store ARI for artifacts but don't log as metric

            # Calculate silhouette score (excluding noise points)
            if np.sum(labels != -1) > 1:
                non_noise_indices = labels != -1
                if len(set(labels[non_noise_indices])) > 1:  # Need at least 2 clusters
                    silhouette = silhouette_score(X_final[non_noise_indices], labels[non_noise_indices])
                    logging.info(f"Silhouette Score (excluding noise): {silhouette:.4f}")
                    mlflow.log_metric("silhouette_score", silhouette)
                else:
                    mlflow.log_metric("silhouette_score", 0.0)
            else:
                mlflow.log_metric("silhouette_score", 0.0)
        else:
            logging.info("No meaningful clusters found, skipping standard clustering metrics")
            mlflow.log_metric("silhouette_score", 0.0)

        # Create and save visualizations
        create_visualizations(
            X_final, labels, feature_type, experiment_config, 
            reducer, explained_variance, clusterer
        )

        # Log example cards per cluster
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

        # Create comprehensive metrics summary
        metrics_summary = {
            "feature_type": feature_type,
            "experiment_config": experiment_config,
            "config_description": config_description,
            "min_cluster_size": actual_min_cluster_size,
            "min_samples": actual_min_samples,
            "cluster_selection_epsilon": actual_cluster_selection_epsilon,
            "metric": metric,
            "n_clusters_found": int(n_clusters),
            "n_noise_points": int(n_noise),
            "noise_percentage": float(n_noise / len(labels) * 100),
            "silhouette_score": float(silhouette) if 'silhouette' in locals() else 0.0,
            "adjusted_rand_index": float(score) if 'score' in locals() else 0.0,
            "use_pca": use_pca,
            "use_umap": use_umap,
            "use_l2_norm": use_l2_norm
        }

        # Add explained variance if applicable
        if explained_variance is not None:
            metrics_summary["explained_variance_ratio"] = float(explained_variance.sum())

        # Add HDBSCAN-specific metrics if available
        if hasattr(clusterer, 'cluster_persistence_') and len(clusterer.cluster_persistence_) > 0:
            metrics_summary.update({
                "avg_cluster_persistence": float(np.mean(clusterer.cluster_persistence_)),
                "max_cluster_persistence": float(np.max(clusterer.cluster_persistence_)),
                "min_cluster_persistence": float(np.min(clusterer.cluster_persistence_))
            })

        # Add condensed enhanced metrics to summary (only key stats)
        condensed_enhanced_metrics = {
            'total_cards': enhanced_metrics.get('total_cards', 0),
            'unique_archetypes': enhanced_metrics.get('unique_archetypes', 0),
            'archetype_diversity_score': enhanced_metrics.get('archetype_diversity_score', 0),
            'avg_stats': enhanced_metrics.get('overall_stats', {})
        }
        metrics_summary["enhanced_metrics"] = condensed_enhanced_metrics            # Add detailed cluster analysis
        if n_clusters > 0:
            try:
                # Get comprehensive cluster evaluation
                feature_names = None
                if hasattr(feature_df, 'columns'):
                    feature_names = list(feature_df.columns)
                elif isinstance(feature_df, pl.DataFrame):
                    feature_names = list(feature_df.columns)
                
                cluster_evaluation = evaluate_clustering_results(
                    X_final, labels, meta_df, feature_names
                )
                
                # Create a condensed version of cluster evaluation
                condensed_evaluation = create_condensed_cluster_analysis(cluster_evaluation)
                
                # Save condensed cluster evaluation as artifact
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                    json.dump(condensed_evaluation, tmp_file, indent=2)
                    tmp_file.flush()
                    mlflow.log_artifact(tmp_file.name, "cluster_analysis")
                    os.unlink(tmp_file.name)
                
                # Add summary of cluster evaluation to metrics summary
                metrics_summary["cluster_evaluation_summary"] = {
                    "total_clusters_analyzed": len(cluster_evaluation.get('top_features', {})),
                    "has_representative_cards": len(cluster_evaluation.get('representative_cards', {})) > 0,
                    "has_archetype_analysis": len(cluster_evaluation.get('archetype_distribution', {})) > 0,
                    "has_cluster_stats": len(cluster_evaluation.get('cluster_stats', {})) > 0
                }
                
                logging.info("Condensed cluster evaluation saved as artifact")
                
            except Exception as e:
                logging.warning(f"Could not save detailed cluster evaluation: {e}")
                metrics_summary["cluster_evaluation_summary"] = {"error": str(e)}

        # Save metrics summary
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(metrics_summary, tmp_file, indent=2)
            tmp_file.flush()
            mlflow.log_artifact(tmp_file.name, "metrics")
            os.unlink(tmp_file.name)
        
        logging.info(f"Enhanced HDBSCAN run completed successfully for {feature_type} features with {experiment_config}")


def create_visualizations(X_final, labels, feature_type, experiment_config, reducer, explained_variance, clusterer):
    """Create and log visualization plots."""
    
    # Create dimensionality reduction plot if applicable
    if explained_variance is not None:  # PCA case
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o')
            plt.xlabel("Number of Components")
            plt.ylabel("Cumulative Explained Variance")
            plt.title(f"Scree Plot (PCA) - {feature_type.upper()} Features - {experiment_config}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(tmp_file.name, "plots")
            os.unlink(tmp_file.name)

    # Create cluster hierarchy plot (HDBSCAN-specific)
    # Note: sklearn's HDBSCAN may have different attributes than hdbscan package
    try:
        if hasattr(clusterer, 'condensed_tree_') and clusterer.condensed_tree_ is not None:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                plt.figure(figsize=(12, 8))
                clusterer.condensed_tree_.plot(select_clusters=True, selection_palette='viridis')
                plt.title(f"HDBSCAN Cluster Hierarchy - {feature_type.upper()} Features - {experiment_config}")
                plt.tight_layout()
                plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(tmp_file.name, "plots")
                os.unlink(tmp_file.name)
        else:
            logging.info("Cluster hierarchy plot not available with sklearn's HDBSCAN")
    except Exception as e:
        logging.warning(f"Could not create cluster hierarchy plot: {e}")

    # Create t-SNE visualization for high-dimensional data
    if X_final.shape[1] > 2:
        logging.info("Projecting to 2D using t-SNE for visualization...")
        X_tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED).fit_transform(X_final)
    else:
        X_tsne = X_final

    # t-SNE/2D cluster plot
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        plt.figure(figsize=(12, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                mask = labels == label
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c='gray', s=10, alpha=0.6, label='Noise')
            else:
                mask = labels == label
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors[i]], s=10, label=f'Cluster {label}')

        plt.title(f"HDBSCAN Clusters - {feature_type.upper()} Features - {experiment_config}")
        plt.xlabel("Component 1" if X_final.shape[1] <= 2 else "t-SNE 1")
        plt.ylabel("Component 2" if X_final.shape[1] <= 2 else "t-SNE 2")
        # Legend removed for HDBSCAN due to large number of clusters
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(tmp_file.name, "plots")
        os.unlink(tmp_file.name)


def run_combined_grid_search(feature_type: Literal["tfidf", "embeddings", "combined"]):
    """
    Run combined UMAP + HDBSCAN hyperparameter grid search for the specified feature type.
    
    Args:
        feature_type: Type of features to use for grid search
    """
    # Create all combinations of UMAP and HDBSCAN parameters
    param_combinations = list(itertools.product(UMAP_GRID, HDBSCAN_GRID))
    
    logging.info(f"🔍 Starting combined UMAP + HDBSCAN grid search for {feature_type} features")
    logging.info(f"Grid search will test {len(param_combinations)} parameter combinations:")
    logging.info(f"  UMAP parameters: {len(UMAP_GRID)} combinations")
    logging.info(f"  HDBSCAN parameters: {len(HDBSCAN_GRID)} combinations")
    logging.info(f"  Total combinations: {len(UMAP_GRID)} × {len(HDBSCAN_GRID)} = {len(param_combinations)}")
    
    for i, (umap_params, hdbscan_params) in enumerate(param_combinations):
        logging.info(f"🔬 Grid Search {i+1}/{len(param_combinations)}: {feature_type}")
        logging.info(f"   UMAP: n_neighbors={umap_params['n_neighbors']}, min_dist={umap_params['min_dist']}, n_components={umap_params['n_components']}")
        logging.info(f"   HDBSCAN: min_cluster_size={hdbscan_params['min_cluster_size']}, min_samples={hdbscan_params['min_samples']}, epsilon={hdbscan_params['cluster_selection_epsilon']}")
        
        try:
            run_enhanced_hdbscan_experiment(feature_type, "combined_grid_search", umap_params, hdbscan_params)
            logging.info(f"✅ Completed grid search {i+1}/{len(param_combinations)}")
        except Exception as e:
            logging.error(f"❌ Failed grid search {i+1}/{len(param_combinations)}: {e}")
            continue
    
    logging.info(f"🎯 Completed combined UMAP + HDBSCAN grid search for {feature_type} features")


def run_umap_grid_search(feature_type: Literal["tfidf", "embeddings", "combined"]):
    """
    Run UMAP hyperparameter grid search for the specified feature type.
    
    Args:
        feature_type: Type of features to use for grid search
    """
    logging.info(f"🔍 Starting UMAP grid search for {feature_type} features")
    logging.info(f"Grid search will test {len(UMAP_GRID)} parameter combinations:")
    
    for i, params in enumerate(UMAP_GRID):
        logging.info(f"  {i+1}. n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}, n_components={params['n_components']}")
    
    for i, umap_params in enumerate(UMAP_GRID):
        logging.info(f"🔬 Grid Search {i+1}/{len(UMAP_GRID)}: {feature_type} with {umap_params}")
        try:
            run_enhanced_hdbscan_experiment(feature_type, "umap_grid_search", umap_params)
            logging.info(f"✅ Completed grid search {i+1}/{len(UMAP_GRID)}")
        except Exception as e:
            logging.error(f"❌ Failed grid search {i+1}/{len(UMAP_GRID)}: {e}")
            continue
    
    logging.info(f"🎯 Completed UMAP grid search for {feature_type} features")


def create_condensed_cluster_analysis(cluster_evaluation):
    """
    Create a condensed version of cluster evaluation to reduce file size.
    
    Args:
        cluster_evaluation: Full cluster evaluation results
        
    Returns:
        Condensed version with only key information (JSON serializable)
    """
    def convert_to_json_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    # Convert summary stats to ensure JSON serialization
    summary_stats = convert_to_json_serializable(cluster_evaluation.get('summary_stats', {}))
    
    condensed = {
        'summary_stats': summary_stats,
        'clusters': {}
    }
    
    # Get all cluster data
    top_features = cluster_evaluation.get('top_features', {})
    representative_cards = cluster_evaluation.get('representative_cards', {})
    archetype_dist = cluster_evaluation.get('archetype_distribution', {})
    cluster_stats = cluster_evaluation.get('cluster_stats', {})
    
    # Process each cluster
    for cluster_id in top_features.keys():
        cluster_info = {
            'size': 0,  # Will be calculated from representative cards
            'top_features': top_features.get(cluster_id, [])[:5],  # Limit to top 5 features
            'representative_cards': [],
            'top_archetypes': [],
            'avg_stats': {}
        }
        
        # Add representative cards (limit to top 3, with essential info only)
        rep_cards = representative_cards.get(cluster_id, [])
        cluster_info['size'] = len(rep_cards) if rep_cards else 0
        for card in rep_cards[:3]:
            condensed_card = {
                'name': str(card.get('name', '')),
                'similarity': round(float(card.get('similarity_to_centroid', 0)), 3)
            }
            # Add key attributes if available
            for attr in ['archetype', 'type']:
                if attr in card:
                    condensed_card[attr] = str(card[attr])
            cluster_info['representative_cards'].append(condensed_card)
        
        # Add top 3 archetypes with percentages
        arch_dist_cluster = archetype_dist.get(cluster_id, {})
        if arch_dist_cluster:
            sorted_archetypes = sorted(arch_dist_cluster.items(), key=lambda x: x[1], reverse=True)[:3]
            cluster_info['top_archetypes'] = [
                {'archetype': str(arch), 'percentage': round(float(pct) * 100, 1)} 
                for arch, pct in sorted_archetypes
            ]
        
        # Add simplified cluster stats (averages only)
        stats = cluster_stats.get(cluster_id, {})
        if stats:
            cluster_info['avg_stats'] = {
                str(col): round(float(val), 1) for col, val in stats.items() 
                if col in ['atk', 'def', 'level'] and not (isinstance(val, float) and np.isnan(val))
            }
        
        condensed['clusters'][str(cluster_id)] = cluster_info
    
    return condensed


def main():
    parser = argparse.ArgumentParser(description="Run enhanced HDBSCAN clustering experiments")
    parser.add_argument(
        "--feature-type", 
        type=str, 
        choices=["tfidf", "embeddings", "combined", "all"],
        default="all",
        help="Feature type to use for clustering (default: all)"
    )
    parser.add_argument(
        "--experiment-config",
        type=str,
        choices=["pca_euclidean", "umap_euclidean", "umap_grid_search", "combined_grid_search", "all"],
        default="all",
        help="Experiment configuration to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Define experiment configurations to run
    if args.experiment_config == "all":
        configs = ["pca_euclidean", "umap_euclidean"]
    elif args.experiment_config == "umap_grid_search":
        # Special handling for UMAP-only grid search
        if args.feature_type == "all":
            feature_types = ["tfidf", "embeddings", "combined"]
        else:
            feature_types = [args.feature_type]
        
        # Run UMAP grid search for each feature type
        for feature_type in feature_types:
            run_umap_grid_search(feature_type)
        return
    elif args.experiment_config == "combined_grid_search":
        # Special handling for combined UMAP + HDBSCAN grid search
        if args.feature_type == "all":
            feature_types = ["tfidf", "embeddings", "combined"]
        else:
            feature_types = [args.feature_type]
        
        # Run combined grid search for each feature type
        for feature_type in feature_types:
            run_combined_grid_search(feature_type)
        return
    else:
        configs = [args.experiment_config]
    
    # Define feature types to run
    if args.feature_type == "all":
        feature_types = ["tfidf", "embeddings", "combined"]
    else:
        feature_types = [args.feature_type]
    
    # Run all combinations
    for config in configs:
        for feature_type in feature_types:
            logging.info(f"🚀 Starting enhanced HDBSCAN experiment: {feature_type} features with {config}")
            run_enhanced_hdbscan_experiment(feature_type, config)
            logging.info(f"✅ Completed enhanced HDBSCAN experiment: {feature_type} features with {config}")


if __name__ == "__main__":
    main()
