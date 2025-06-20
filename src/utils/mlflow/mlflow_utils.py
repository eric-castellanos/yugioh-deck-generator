"""
MLflow utilities for Yu-Gi-Oh card clustering experiments.
"""

import mlflow
import mlflow.sklearn
import tempfile
import json
import os
import pickle
import numpy as np
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the tracking URI to connect to the MLflow server
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize MLflow client
client = MlflowClient(MLFLOW_TRACKING_URI)


def setup_clustering_experiment(experiment_name: str = "/yugioh_card_clustering") -> str:
    """
    Set up or get the clustering experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        Experiment ID
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        logger.error(f"Error setting up experiment: {e}")
        raise


def log_clustering_tags(
    algorithm: str,
    feature_type: str,
    experiment_config: str,
    stage: str = "exploration",
    version: str = "v2.0",
    **kwargs
) -> None:
    """
    Log standard tags for clustering experiments.
    
    Args:
        algorithm: Clustering algorithm used
        feature_type: Type of features (tfidf, embeddings, combined)
        experiment_config: Configuration description
        stage: Stage of the experiment (exploration, production, etc.)
        version: Version tag
        **kwargs: Additional tags to log
    """
    standard_tags = {
        "model_type": "clustering",
        "algorithm": algorithm,
        "dataset": "yugioh_cards",
        "preprocessing": experiment_config,
        "version": version,
        "purpose": "card_clustering",
        "data_source": "s3",
        "stage": stage,
        "feature_type": feature_type,
        "experiment_config": experiment_config
    }
    
    # Add any additional tags
    standard_tags.update(kwargs)
    
    for key, value in standard_tags.items():
        mlflow.set_tag(key, value)


def log_clustering_params(
    s3_bucket: str,
    s3_key: str,
    algorithm_params: Dict[str, Any],
    preprocessing_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Log parameters for clustering experiments.
    
    Args:
        s3_bucket: S3 bucket name
        s3_key: S3 key for the dataset
        algorithm_params: Algorithm-specific parameters
        preprocessing_params: Preprocessing parameters (PCA, UMAP, etc.)
        **kwargs: Additional parameters to log
    """
    # Helper function to convert numpy types to Python types
    def convert_value(value):
        if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif hasattr(value, 'item'):  # other numpy scalars
            return value.item()
        elif isinstance(value, (list, tuple)):
            return [convert_value(v) for v in value]
        elif isinstance(value, dict):
            return {str(k): convert_value(v) for k, v in value.items()}
        else:
            return value
    
    # Log data source params
    mlflow.log_param("s3_bucket", s3_bucket)
    mlflow.log_param("s3_key", s3_key)
    
    # Log algorithm parameters
    for key, value in algorithm_params.items():
        mlflow.log_param(key, convert_value(value))
    
    # Log preprocessing parameters
    if preprocessing_params:
        for key, value in preprocessing_params.items():
            mlflow.log_param(key, convert_value(value))
    
    # Log any additional parameters
    for key, value in kwargs.items():
        mlflow.log_param(key, convert_value(value))


def log_essential_clustering_metrics(
    total_clusters: int,
    silhouette_score: float,
    num_samples: int,
    num_features: int,
    noise_percentage: float
) -> None:
    """
    Log the essential clustering metrics.
    
    Args:
        total_clusters: Number of clusters found
        silhouette_score: Silhouette score
        num_samples: Number of data samples
        num_features: Number of features
        noise_percentage: Percentage of noise points
    """
    # Convert numpy types to Python native types
    mlflow.log_metric("total_clusters", int(total_clusters))
    mlflow.log_metric("silhouette_score", float(silhouette_score))
    mlflow.log_metric("num_samples", int(num_samples))
    mlflow.log_metric("num_features", int(num_features))
    mlflow.log_metric("noise_percentage", float(noise_percentage))


def log_dataset_info(dataset_info: Dict[str, Any]) -> None:
    """
    Log dataset schema and information as MLflow artifact.
    
    Args:
        dataset_info: Dictionary containing dataset information
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump(dataset_info, tmp_file, indent=2)
        tmp_file.flush()
        mlflow.log_artifact(tmp_file.name, "dataset")
        os.unlink(tmp_file.name)


def save_clustering_artifacts(
    metrics_summary: Dict[str, Any],
    cluster_analysis: Optional[Dict[str, Any]] = None,
    plots: Optional[List[str]] = None
) -> None:
    """
    Save clustering artifacts to MLflow.
    
    Args:
        metrics_summary: Summary of all metrics
        cluster_analysis: Detailed cluster analysis results
        plots: List of plot file paths to log
    """
    # Save metrics summary with safe JSON serialization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        safe_metrics = safe_json_serialize(metrics_summary)
        json.dump(safe_metrics, tmp_file, indent=2)
        tmp_file.flush()
        mlflow.log_artifact(tmp_file.name, "metrics")
        os.unlink(tmp_file.name)
    
    # Save cluster analysis if provided with safe JSON serialization
    if cluster_analysis:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            safe_analysis = safe_json_serialize(cluster_analysis)
            json.dump(safe_analysis, tmp_file, indent=2)
            tmp_file.flush()
            mlflow.log_artifact(tmp_file.name, "cluster_analysis")
            os.unlink(tmp_file.name)
    
    # Save plots if provided
    if plots:
        for plot_path in plots:
            if os.path.exists(plot_path):
                mlflow.log_artifact(plot_path, "plots")


def register_best_clustering_model(
    model,
    model_name: str,
    version: str = "1.0",
    description: str = "Yu-Gi-Oh Card Clustering Model",
    stage: Optional[str] = None
) -> str:
    """
    Register the best clustering model to MLflow Model Registry.
    
    Args:
        model: The trained clustering model
        model_name: Name for the registered model
        version: Version tag
        description: Model description
        stage: Model stage (Staging, Production, etc.)
        
    Returns:
        Model version URI
    """
    try:
        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="clustering_model",
            registered_model_name=model_name,
            await_registration_for=300  # Wait up to 5 minutes for registration
        )
        
        # Add version and description tags
        client.set_registered_model_tag(
            model_name, 
            "version", 
            version
        )
        
        client.update_registered_model(
            model_name,
            description=description
        )
        
        # Transition to specified stage
        if stage:
            client.transition_model_version_stage(
                name=model_name,
                version=model_info.registered_model_version,
                stage=stage,
                archive_existing_versions=True
            )
        
        logger.info(f"Model registered: {model_name} v{version} -> {stage}")
        return model_info.model_uri
        
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise


def get_best_clustering_run(experiment_name: str = "/yugioh_card_clustering") -> Optional[Any]:
    """
    Get the best clustering run based on composite score.
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        Best run object or None if no runs found
    """
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.warning(f"Experiment {experiment_name} not found")
            return None
        
        runs = client.search_runs(
            experiment.experiment_id,
            filter_string="",
            max_results=2000
        )
        
        if not runs:
            logger.warning("No runs found in experiment")
            return None
        
        # Calculate composite score for each run
        def run_score(run):
            silhouette = run.data.metrics.get("silhouette_score", -1)
            noise = run.data.metrics.get("noise_percentage", 0)
            n_clusters = run.data.metrics.get("total_clusters", 0)
            
            # Primary score based on silhouette
            score = 0.7 * silhouette
            
            # Noise adjustment (prefer moderate noise levels for outlier detection)
            if 5 <= noise <= 15:
                score += 0.2
            elif 15 < noise <= 25:
                score += 0.05
            elif noise > 25:
                score -= 0.1 * (noise - 25) / 100
            
            # Cluster count bonus (prefer reasonable number of clusters)
            if 10 <= n_clusters <= 50:
                score += 0.1
            elif n_clusters > 50:
                score -= 0.05
            
            return score
        
        best_run = max(runs, key=run_score)
        logger.info(f"Best run found: {best_run.info.run_id} with score {run_score(best_run):.4f}")
        return best_run
        
    except Exception as e:
        logger.error(f"Error finding best run: {e}")
        return None


# Legacy functions for backwards compatibility
def run_score(run):
    """Legacy function - use get_best_clustering_run instead."""
    silhouette = run.data.metrics.get("silhouette_score", -1)
    adjusted_rand = run.data.metrics.get("adjusted_rand_index", 0)
    
    noise = run.data.metrics.get("enhanced_noise_percentage", run.data.metrics.get("noise_percentage", 0))
    cluster_std = run.data.metrics.get("enhanced_cluster_size_std", 0)
    entropy = run.data.metrics.get("enhanced_archetype_entropy_mean", 0)
    n_clusters = run.data.metrics.get("enhanced_num_clusters", run.data.metrics.get("n_clusters_found", 0))

    is_kmeans = noise == 0 and cluster_std < 10
    
    score = 0.7 * silhouette

    if 5 <= noise <= 15:
        score += 0.2
    elif 15 < noise <= 25:
        score += 0.05
    elif noise > 25:
        score -= 0.1 * (noise - 25) / 100

    if is_kmeans:
        score -= 0.1

    score += min(0.1, cluster_std / 100)

    if entropy > 0:
        score += max(0, 0.1 * (3.0 - entropy))

    return score


# Initialize experiment on import
try:
    setup_clustering_experiment()
except Exception as e:
    logger.warning(f"Could not initialize experiment on import: {e}")


def log_dataset_with_schema(
    pandas_df,
    feature_df,
    s3_bucket: str,
    s3_key: str,
    feature_type: str,
    experiment_config: str,
    available_meta_cols: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Log dataset with comprehensive schema information to MLflow.
    
    Args:
        pandas_df: Pandas dataframe for MLflow dataset creation
        feature_df: Feature dataframe (polars) for schema analysis
        s3_bucket: S3 bucket name
        s3_key: S3 key for the dataset
        feature_type: Type of features (tfidf, embeddings, combined)
        experiment_config: Experiment configuration
        available_meta_cols: Available metadata columns
        config: Feature configuration dictionary
    """
    # Log dataset using MLflow Datasets with enhanced schema
    dataset_source = f"s3://{s3_bucket}/{s3_key}"
    
    # Categorize features by type for comprehensive schema
    all_columns = pandas_df.columns
    text_features = [col for col in feature_df.columns if col.startswith(('desc_feat_', 'tfidf_feat_', 'embed_feat_'))]
    numeric_features = [col for col in feature_df.columns if col in ['atk', 'def', 'level', 'atk_norm', 'def_norm', 'level_norm']]
    categorical_features = [col for col in feature_df.columns if col not in text_features + numeric_features + available_meta_cols]
    
    # Create detailed schema information
    schema_info = {
        "num_rows": len(pandas_df),
        "num_columns": len(pandas_df.columns),
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
        "metadata_columns": available_meta_cols,
        "total_features_used": len(feature_df.columns),
        "feature_type": feature_type,
        "feature_description": config["description"],
        "processing_engine": "polars",
        "experiment_config": experiment_config,
        "dtypes_sample": {str(col): str(dtype) for col, dtype in list(zip(pandas_df.columns, pandas_df.dtypes))[:20]}
    }
    
    # Log enhanced schema information with safe serialization
    safe_schema_info = safe_json_serialize(schema_info)
    mlflow.log_dict(safe_schema_info, "dataset_schema.json")
    
    # Create MLflow dataset (pandas required here)
    dataset = mlflow.data.from_pandas(
        pandas_df, 
        source=dataset_source,
        name=f"yugioh_cards_{feature_type}_{experiment_config}",
        targets="archetype"  # String, not list
    )
    mlflow.log_input(dataset, context="training")


def log_clustering_model_and_artifacts(
    clusterer,
    metrics_summary: Dict[str, Any],
    cluster_evaluation: Optional[Dict[str, Any]] = None,
    plot_paths: Optional[List[str]] = None,
    register_model: bool = False,
    model_name: str = "yugioh_clustering_model"
) -> Optional[str]:
    """
    Log clustering model and artifacts to MLflow, optionally registering the model.
    
    Args:
        clusterer: Trained clustering model
        metrics_summary: Summary of all metrics
        cluster_evaluation: Detailed cluster analysis results
        plot_paths: List of plot file paths to log
        register_model: Whether to register the model in Model Registry
        model_name: Name for the registered model
        
    Returns:
        Model URI if registered, None otherwise
    """
    model_uri = None
    
    # Log the clustering model
    try:
        if register_model:
            # Register the model in the Model Registry
            model_info = mlflow.sklearn.log_model(
                sk_model=clusterer,
                artifact_path="clustering_model",
                registered_model_name=model_name,
                await_registration_for=300  # Wait up to 5 minutes for registration
            )
            model_uri = model_info.model_uri
            
            # Add model tags and description
            try:
                client.set_registered_model_tag(model_name, "version", "1.0")
                client.set_registered_model_tag(model_name, "algorithm", "hdbscan")
                client.set_registered_model_tag(model_name, "dataset", "yugioh_cards")
                client.set_registered_model_tag(model_name, "purpose", "card_clustering")
                
                client.update_registered_model(
                    model_name,
                    description="Yu-Gi-Oh Card Clustering Model using HDBSCAN - Final production model v1.0"
                )
                
                # Transition to Production stage
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_info.registered_model_version,
                    stage="Production",
                    archive_existing_versions=True
                )
                
                logger.info(f"Model registered and promoted to Production: {model_name} v1.0")
                
            except Exception as e:
                logger.warning(f"Could not set model tags/description: {e}")
        else:
            # Just log the model without registration
            mlflow.sklearn.log_model(
                sk_model=clusterer,
                artifact_path="clustering_model"
            )
        
    except Exception as e:
        logger.error(f"Error logging clustering model: {e}")
    
    # Save metrics summary
    save_clustering_artifacts(metrics_summary, cluster_evaluation, plot_paths)
    
    return model_uri


def create_and_log_visualization_plots(
    X_final, 
    labels, 
    feature_type: str, 
    experiment_config: str, 
    reducer=None, 
    explained_variance=None, 
    clusterer=None
) -> List[str]:
    """
    Create and log visualization plots for clustering results.
    
    Args:
        X_final: Final feature matrix
        labels: Cluster labels
        feature_type: Type of features used
        experiment_config: Experiment configuration
        reducer: Dimensionality reduction model (PCA/UMAP)
        explained_variance: Explained variance from PCA
        clusterer: Clustering model
        
    Returns:
        List of plot file paths that were created
    """
    import tempfile
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    
    plot_paths = []
    
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
            plot_paths.append(tmp_file.name)

    # Create cluster hierarchy plot (HDBSCAN-specific)
    if clusterer is not None:
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
                    plot_paths.append(tmp_file.name)
            else:
                logger.info("Cluster hierarchy plot not available with sklearn's HDBSCAN")
        except Exception as e:
            logger.warning(f"Could not create cluster hierarchy plot: {e}")

    # Create t-SNE visualization for high-dimensional data
    if X_final.shape[1] > 2:
        logger.info("Projecting to 2D using t-SNE for visualization...")
        try:
            X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_final)
        except Exception as e:
            logger.warning(f"t-SNE failed, using first 2 components: {e}")
            X_tsne = X_final[:, :2]
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
        plot_paths.append(tmp_file.name)
    
    return plot_paths


def create_condensed_cluster_analysis(cluster_evaluation, labels=None):
    """
    Create a condensed version of cluster evaluation to reduce file size and add type purity metrics.
    
    Args:
        cluster_evaluation: Full cluster evaluation results
        labels: Optional cluster labels array to calculate actual cluster sizes
    """
    def convert_to_json_serializable(obj):
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

    summary_stats = convert_to_json_serializable(cluster_evaluation.get('summary_stats', {}))
    top_features = cluster_evaluation.get('top_features', {})
    representative_cards = cluster_evaluation.get('representative_cards', {})
    archetype_dist = cluster_evaluation.get('archetype_distribution', {})
    cluster_stats = cluster_evaluation.get('cluster_stats', {})

    clusters_unsorted = {}
    type_purity_list = []
    mostly_pure_type_list = []
    for cluster_id in top_features.keys():
        cluster_info = {
            'size': 0,
            'top_features': top_features.get(cluster_id, [])[:5],
            'representative_cards': [],
            'top_archetypes': [],
            'avg_stats': {},
            'type_distribution': {},
            'type_purity': 0.0,
            'dominant_type': None
        }
        rep_cards = representative_cards.get(cluster_id, [])
        
        # Calculate actual cluster size from labels if available
        if labels is not None:
            cluster_info['size'] = int(np.sum(labels == cluster_id))
        else:
            # Fallback to representative cards count (not ideal but better than nothing)
            cluster_info['size'] = len(rep_cards) if rep_cards else 0
        
        # Type distribution and purity
        type_counts = {}
        for card in rep_cards:
            t = card.get('type', None)
            if t:
                type_counts[t] = type_counts.get(t, 0) + 1
        if type_counts:
            total = sum(type_counts.values())
            dominant_type, dominant_count = max(type_counts.items(), key=lambda x: x[1])
            purity = dominant_count / total if total > 0 else 0.0
            cluster_info['type_distribution'] = {k: v/total for k, v in type_counts.items()}
            cluster_info['type_purity'] = round(purity, 3)
            cluster_info['dominant_type'] = dominant_type
            if purity == 1.0:
                type_purity_list.append(cluster_id)
            if purity >= 0.9:
                mostly_pure_type_list.append(cluster_id)
        
        # Add representative cards (limit to top 3, with essential info only)
        for card in rep_cards[:3]:
            condensed_card = {
                'name': str(card.get('name', '')),
                'similarity': round(float(card.get('similarity_to_centroid', 0)), 3)
            }
            for attr in ['archetype', 'type']:
                if attr in card:
                    condensed_card[attr] = str(card[attr])
            cluster_info['representative_cards'].append(condensed_card)
        
        # Add top 3 archetypes with nonzero percentages
        arch_dist_cluster = archetype_dist.get(cluster_id, {})
        if arch_dist_cluster:
            sorted_archetypes = sorted(arch_dist_cluster.items(), key=lambda x: x[1], reverse=True)
            filtered_archetypes = [
                {'archetype': str(arch), 'percentage': round(float(pct) * 100, 1)}
                for arch, pct in sorted_archetypes[:3] if pct > 0
            ]
            cluster_info['top_archetypes'] = filtered_archetypes
        
        # Add simplified cluster stats (averages only)
        stats = cluster_stats.get(cluster_id, {})
        if stats:
            cluster_info['avg_stats'] = {
                str(col): round(float(val), 1) for col, val in stats.items()
                if col in ['atk', 'def', 'level'] and not (isinstance(val, float) and np.isnan(val))
            }
        
        clusters_unsorted[str(cluster_id)] = cluster_info
    
    # Sort clusters by size (descending)
    sorted_clusters = dict(sorted(clusters_unsorted.items(), key=lambda x: x[1]['size'], reverse=True))
    
    condensed = {
        'summary_stats': summary_stats,
        'clusters': sorted_clusters,
        'type_purity_metrics': {
            'num_pure_type_clusters': len(type_purity_list),
            'num_mostly_pure_type_clusters': len(mostly_pure_type_list),
            'pure_type_cluster_ids': [str(cid) for cid in type_purity_list],
            'mostly_pure_type_cluster_ids': [str(cid) for cid in mostly_pure_type_list]
        }
    }
    return condensed


def log_condensed_cluster_analysis(cluster_evaluation: Dict[str, Any], labels) -> None:
    """
    Log condensed cluster analysis as MLflow artifact.
    
    Args:
        cluster_evaluation: Full cluster evaluation results
        labels: Cluster labels for size calculation
    """
    # Create a condensed version of cluster evaluation
    condensed_evaluation = create_condensed_cluster_analysis(cluster_evaluation, labels)
    
    # Save condensed cluster evaluation as artifact with safe JSON serialization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        safe_evaluation = safe_json_serialize(condensed_evaluation)
        json.dump(safe_evaluation, tmp_file, indent=2)
        tmp_file.flush()
        mlflow.log_artifact(tmp_file.name, "cluster_analysis")
        os.unlink(tmp_file.name)


def determine_if_final_model(run_data: Dict[str, Any]) -> bool:
    """
    Determine if this is the final model that should be registered.
    
    Args:
        run_data: Dictionary containing run metrics and parameters
        
    Returns:
        Boolean indicating if this should be the final registered model
    """
    # Criteria for final model (updated for HDBSCAN):
    # 1. Good silhouette score (> 0.3)
    # 2. Reasonable number of clusters (10-1000, HDBSCAN can find many fine-grained clusters)
    # 3. Not too much noise (< 30%)
    # 4. Combined features (most comprehensive)
    
    silhouette = run_data.get('silhouette_score', 0)
    n_clusters = run_data.get('total_clusters', 0)
    noise_pct = run_data.get('noise_percentage', 100)
    feature_type = run_data.get('feature_type', '')
    
    is_final = (
        silhouette > 0.3 and
        10 <= n_clusters <= 1000 and  # Increased upper limit for HDBSCAN
        noise_pct < 30 and
        feature_type == 'combined'
    )
    
    logger.info(f"Final model check: silhouette={silhouette:.3f}, clusters={n_clusters}, "
                f"noise={noise_pct:.1f}%, features={feature_type} -> {'YES' if is_final else 'NO'}")
    
    return is_final


def convert_to_native_types(obj):
    """Convert numpy types to native Python types for MLflow compatibility."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert both keys and values, ensuring keys are strings
        return {str(convert_to_native_types(k)): convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj


def safe_json_serialize(obj):
    """
    Safely serialize objects to JSON, converting numpy types to native Python types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif hasattr(obj, 'item'):  # other numpy scalars
        return obj.item()
    else:
        return obj


def force_register_model_from_run(
    run_id: str, 
    model_name: str = "yugioh_clustering_final_model",
    version: str = "1.0",
    stage: str = "Production"
) -> str:
    """
    Force register a model from a specific MLflow run ID.
    
    Args:
        run_id: MLflow run ID to register the model from
        model_name: Name for the registered model
        version: Version tag
        stage: Model stage
        
    Returns:
        Model URI
    """
    try:
        # Get the run
        run = client.get_run(run_id)
        
        # Register the model from the run's artifacts
        model_uri = f"runs:/{run_id}/clustering_model"
        
        # Register the model
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        client.set_registered_model_tag(model_name, "version", version)
        client.set_registered_model_tag(model_name, "algorithm", "hdbscan")
        client.set_registered_model_tag(model_name, "dataset", "yugioh_cards")
        client.set_registered_model_tag(model_name, "purpose", "card_clustering")
        
        # Update description
        client.update_registered_model(
            model_name,
            description=f"Yu-Gi-Oh Card Clustering Model using HDBSCAN - Force registered v{version}"
        )
        
        # Transition to stage
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage=stage,
            archive_existing_versions=True
        )
        
    except Exception as e:
        logger.error(f"Error forcing model registration: {e}")
        raise
        logger.info(f"Model force registered: {model_name} v{version} -> {stage}")
        return model_uri
        
    except Exception as e:
        logger.error(f"Error force registering model: {e}")
        raise


# =====================================
# DECK GENERATION MLFLOW UTILITIES
# =====================================

def setup_deck_generation_experiment(experiment_name: str = "/yugioh_deck_generation") -> str:
    """
    Set up or get the deck generation experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        Experiment ID
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        logger.error(f"Error setting up experiment: {e}")
        raise


def log_deck_generation_tags(
    generation_mode: str,
    target_archetype: Optional[str] = None,
    stage: str = "exploration",
    version: str = "v1.0",
    **kwargs
) -> None:
    """
    Log standard tags for deck generation experiments.
    
    Args:
        generation_mode: Mode used (meta_aware or novel)
        target_archetype: Target archetype (for meta_aware mode)
        stage: Stage of the experiment (exploration, production, etc.)
        version: Version tag
        **kwargs: Additional tags to log
    """
    standard_tags = {
        "model_type": "deck_generation",
        "dataset": "yugioh_cards",
        "version": version,
        "purpose": "deck_generation",
        "stage": stage,
        "generation_mode": generation_mode
    }
    
    if target_archetype:
        standard_tags["target_archetype"] = target_archetype
    
    # Add any additional tags
    standard_tags.update(kwargs)
    
    for key, value in standard_tags.items():
        mlflow.set_tag(key, value)


def log_deck_generation_params(
    generation_mode: str,
    target_archetype: Optional[str] = None,
    clustering_run_id: Optional[str] = None,
    **kwargs
) -> None:
    """
    Log parameters for deck generation experiments.
    
    Args:
        generation_mode: Generation mode (meta_aware or novel)
        target_archetype: Target archetype (for meta_aware mode)
        clustering_run_id: Run ID of the clustering model used
        **kwargs: Additional parameters to log
    """
    mlflow.log_param("generation_mode", generation_mode)
    
    if target_archetype:
        mlflow.log_param("target_archetype", target_archetype)
    
    if clustering_run_id:
        mlflow.log_param("clustering_run_id", clustering_run_id)
    
    # Log any additional parameters
    for key, value in kwargs.items():
        mlflow.log_param(key, str(value))


def log_deck_metrics(
    main_deck: List[Dict],
    extra_deck: List[Dict],
    metadata: Any
) -> None:
    """
    Log deck generation metrics.
    
    Args:
        main_deck: List of main deck cards
        extra_deck: List of extra deck cards
        metadata: Deck metadata object
    """
    # Main deck metrics
    mlflow.log_metric("main_deck_size", len(main_deck))
    mlflow.log_metric("extra_deck_size", len(extra_deck))
    
    # Card type ratios for main deck
    total_main = len(main_deck)
    if total_main > 0:
        monster_count = sum(1 for card in main_deck if card.get('type') in ['Monster', 'Ritual Monster'])
        spell_count = sum(1 for card in main_deck if card.get('type') == 'Spell')
        trap_count = sum(1 for card in main_deck if card.get('type') == 'Trap')
        
        mlflow.log_metric("monster_ratio", monster_count / total_main)
        mlflow.log_metric("spell_ratio", spell_count / total_main)
        mlflow.log_metric("trap_ratio", trap_count / total_main)
        mlflow.log_metric("monster_count", monster_count)
        mlflow.log_metric("spell_count", spell_count)
        mlflow.log_metric("trap_count", trap_count)
    
    # Extra deck type ratios
    if len(extra_deck) > 0:
        fusion_count = sum(1 for card in extra_deck if 'Fusion' in card.get('type', ''))
        synchro_count = sum(1 for card in extra_deck if 'Synchro' in card.get('type', ''))
        xyz_count = sum(1 for card in extra_deck if 'Xyz' in card.get('type', ''))
        link_count = sum(1 for card in extra_deck if 'Link' in card.get('type', ''))
        
        total_extra = len(extra_deck)
        mlflow.log_metric("fusion_ratio", fusion_count / total_extra)
        mlflow.log_metric("synchro_ratio", synchro_count / total_extra)
        mlflow.log_metric("xyz_ratio", xyz_count / total_extra)
        mlflow.log_metric("link_ratio", link_count / total_extra)
        mlflow.log_metric("fusion_count", fusion_count)
        mlflow.log_metric("synchro_count", synchro_count)
        mlflow.log_metric("xyz_count", xyz_count)
        mlflow.log_metric("link_count", link_count)
    
    # Archetype and diversity metrics
    metadata_dict = metadata.to_dict()
    mlflow.log_metric("cluster_entropy", metadata_dict.get('cluster_entropy', 0.0))
    
    # Log archetype distribution as metrics
    archetype_dist = metadata_dict.get('archetype_distribution', {})
    if archetype_dist:
        # Log the diversity of archetypes
        total_cards = sum(archetype_dist.values())
        archetype_count = len(archetype_dist)
        mlflow.log_metric("archetype_diversity", archetype_count)
        mlflow.log_metric("dominant_archetype_ratio", max(archetype_dist.values()) / total_cards if total_cards > 0 else 0)
    
    # Log cluster distribution metrics
    cluster_dist = metadata_dict.get('cluster_distribution', {})
    if cluster_dist:
        mlflow.log_metric("cluster_diversity", len(cluster_dist))


def log_deck_artifacts(
    main_deck: List[Dict],
    extra_deck: List[Dict],
    metadata: Any
) -> None:
    """
    Log deck generation artifacts.
    
    Args:
        main_deck: List of main deck cards
        extra_deck: List of extra deck cards
        metadata: Deck metadata object
    """
    # Create deck list artifact
    deck_list = {
        "main_deck": main_deck,
        "extra_deck": extra_deck,
        "metadata": metadata.to_dict()
    }
    
    # Save deck list as JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump(deck_list, tmp_file, indent=2, default=str)
        tmp_file.flush()
        mlflow.log_artifact(tmp_file.name, "deck_list.json")
        os.unlink(tmp_file.name)
    
    # Create human-readable deck list
    deck_text = "=== YU-GI-OH! DECK LIST ===\n\n"
    deck_text += f"Main Deck ({len(main_deck)} cards):\n"
    deck_text += "=" * 30 + "\n"
    
    # Group cards by type
    monsters = [c for c in main_deck if c.get('type') in ['Monster', 'Ritual Monster']]
    spells = [c for c in main_deck if c.get('type') == 'Spell']
    traps = [c for c in main_deck if c.get('type') == 'Trap']
    
    if monsters:
        deck_text += f"\nMonsters ({len(monsters)}):\n"
        for card in monsters:
            deck_text += f"  - {card['name']}\n"
    
    if spells:
        deck_text += f"\nSpells ({len(spells)}):\n"
        for card in spells:
            deck_text += f"  - {card['name']}\n"
    
    if traps:
        deck_text += f"\nTraps ({len(traps)}):\n"
        for card in traps:
            deck_text += f"  - {card['name']}\n"
    
    if extra_deck:
        deck_text += f"\n\nExtra Deck ({len(extra_deck)} cards):\n"
        deck_text += "=" * 30 + "\n"
        for card in extra_deck:
            deck_text += f"  - {card['name']} ({card.get('type', 'Unknown')})\n"
    
    # Add metadata
    deck_text += f"\n\nDeck Statistics:\n"
    deck_text += "=" * 30 + "\n"
    deck_text += f"Dominant Archetype: {metadata.dominant_archetype}\n"
    deck_text += f"Cluster Entropy: {metadata.cluster_entropy:.3f}\n"
    deck_text += f"Archetype Distribution: {metadata.archetype_distribution}\n"
    
    # Save human-readable deck list
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(deck_text)
        tmp_file.flush()
        mlflow.log_artifact(tmp_file.name, "deck_list.txt")
        os.unlink(tmp_file.name)


# Deck Generation MLflow Integration Functions


# Use get_clustering_model_from_registry from the imported module instead


# Import the clustering model loading and card clustering functions
from src.utils.mlflow.get_clustering_model_from_registry import (
    get_clustering_model_from_registry,
    get_card_data_with_clusters
)

# For backwards compatibility
def get_clustered_cards_from_model(model_name: str = "yugioh_clustering_final_model",
                                 stage: Optional[str] = None):
    """
    Load the clustering model object from MLflow model registry and applies it to card data.
    This function is deprecated, use get_card_data_with_clusters from get_clustering_model_from_registry instead.
    
    Args:
        model_name: Name of the registered model
        stage: Model stage (None, "Staging", "Production", etc.)
        
    Returns:
        Dictionary mapping cluster IDs to lists of card dictionaries
    """
    logger.warning("get_clustered_cards_from_model is deprecated, use get_card_data_with_clusters instead")
    try:
        # Get the model
        model = get_clustering_model_from_registry(model_name, stage)
        
        # Apply the model to card data
        return get_card_data_with_clusters(model)
    except Exception as e:
        logger.error(f"Error getting clustered cards from model {model_name}: {e}")
        raise



def _log_deck_artifacts(main_deck: List[Dict], extra_deck: List[Dict], metadata: Any) -> None:
    """
    Log deck lists and metadata as MLflow artifacts.
    
    Args:
        main_deck: Main deck card list
        extra_deck: Extra deck card list
        metadata: Deck metadata
    """
    try:
        # Create deck list in readable format
        deck_list = {
            "main_deck": {
                "size": len(main_deck),
                "cards": [
                    {
                        "name": card.get('name', 'Unknown'),
                        "type": card.get('type', 'Unknown'),
                        "archetype": card.get('archetype'),
                        "cluster_id": card.get('cluster_id')
                    }
                    for card in main_deck
                ]
            },
            "extra_deck": {
                "size": len(extra_deck),
                "cards": [
                    {
                        "name": card.get('name', 'Unknown'),
                        "type": card.get('type', 'Unknown'),
                        "archetype": card.get('archetype'),
                        "cluster_id": card.get('cluster_id')
                    }
                    for card in extra_deck
                ]
            },
            "metadata": metadata.to_dict()
        }
        
        # Save deck list as JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(deck_list, tmp_file, indent=2)
            tmp_file.flush()
            mlflow.log_artifact(tmp_file.name, "deck_list.json")
            os.unlink(tmp_file.name)
        
        # Create human-readable deck list
        readable_deck = []
        readable_deck.append("=" * 50)
        readable_deck.append("YU-GI-OH! GENERATED DECK")
        readable_deck.append("=" * 50)
        readable_deck.append("")
        
        # Main deck by type
        readable_deck.append(f"MAIN DECK ({len(main_deck)} cards):")
        readable_deck.append("-" * 30)
        
        # Group by card type
        cards_by_type = {}
        for card in main_deck:
            card_type = card.get('type', 'Unknown')
            if card_type not in cards_by_type:
                cards_by_type[card_type] = []
            cards_by_type[card_type].append(card)
        
        for card_type in ['Monster', 'Ritual Monster', 'Spell', 'Trap']:
            if card_type in cards_by_type:
                readable_deck.append(f"\n{card_type}s ({len(cards_by_type[card_type])}):")
                for card in sorted(cards_by_type[card_type], key=lambda x: x.get('name', '')):
                    archetype = card.get('archetype', 'Generic')
                    readable_deck.append(f"  - {card.get('name', 'Unknown')} [{archetype}]")
        
        # Extra deck
        if extra_deck:
            readable_deck.append(f"\nEXTRA DECK ({len(extra_deck)} cards):")
            readable_deck.append("-" * 30)
            
            extra_by_type = {}
            for card in extra_deck:
                card_type = card.get('type', 'Unknown')
                if card_type not in extra_by_type:
                    extra_by_type[card_type] = []
                extra_by_type[card_type].append(card)
            
            for card_type, cards in extra_by_type.items():
                readable_deck.append(f"\n{card_type}s ({len(cards)}):")
                for card in sorted(cards, key=lambda x: x.get('name', '')):
                    archetype = card.get('archetype', 'Generic')
                    readable_deck.append(f"  - {card.get('name', 'Unknown')} [{archetype}]")
        
        # Metadata summary
        readable_deck.append(f"\nDECK STATISTICS:")
        readable_deck.append("-" * 30)
        readable_deck.append(f"Dominant Archetype: {metadata.dominant_archetype}")
        readable_deck.append(f"Cluster Entropy: {metadata.cluster_entropy:.3f}")
        readable_deck.append(f"Monster/Spell/Trap: {metadata.monster_count}/{metadata.spell_count}/{metadata.trap_count}")
        
        # Save readable deck list
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write('\n'.join(readable_deck))
            tmp_file.flush()
            mlflow.log_artifact(tmp_file.name, "deck_list.txt")
            os.unlink(tmp_file.name)
            
    except Exception as e:
        logger.error(f"Error logging deck artifacts: {e}")
        raise


def get_recent_deck_runs(experiment_name: str = "yugioh_deck_generation", 
                        limit: int = 10) -> List[Dict]:
    """
    Get recent deck generation runs from MLflow.
    
    Args:
        experiment_name: Name of the experiment
        limit: Maximum number of runs to return
        
    Returns:
        List of run information dictionaries
    """
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if not experiment:
            logger.warning(f"Experiment {experiment_name} not found")
            return []
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=limit
        )
        
        run_info = []
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                "start_time": run.info.start_time,
                "status": run.info.status,
                "metrics": run.data.metrics,
                "params": run.data.params
            }
            run_info.append(run_data)
        
        logger.info(f"Retrieved {len(run_info)} recent deck generation runs")
        return run_info
        
    except Exception as e:
        logger.error(f"Error getting recent deck runs: {e}")
        return []
