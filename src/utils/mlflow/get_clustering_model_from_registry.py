"""
Get the clustering model from the MLflow model registry.
"""

import logging
import mlflow
from typing import Any, Dict, List, Optional, Tuple
from src.utils.s3_utils import read_parquet_from_s3
import polars as pl
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# MLflow tracking URI
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Feature type to S3 key mapping
FEATURE_CONFIGS = {
    "tfidf": {
        "s3_key": "processed/feature_engineered/2025-06/feature_engineered_tfidf.parquet",
        "description": "TF-IDF features only from card descriptions"
    },
    "embeddings": {
        "s3_key": "processed/feature_engineered/2025-06/feature_engineered_embeddings.parquet", 
        "description": "Word embedding features only from card descriptions"
    },
    "combined": {
        "s3_key": "processed/feature_engineered/2025-06/feature_engineered_combined.parquet",
        "description": "Combined TF-IDF and word embedding features from card descriptions"
    }
}

S3_BUCKET = "yugioh-data"

def get_clustering_model_from_registry(
    model_name: str = "yugioh_clustering_final_model", 
    stage: str = None
) -> Any:
    """
    Get the clustering model from MLflow model registry.
    
    Args:
        model_name: Name of the registered model
        stage: Model stage (Production, Staging, etc.). If None, loads latest version.
        
    Returns:
        The loaded clustering model
    """
    try:
        # First try with the specified stage if provided
        if stage:
            try:
                model_uri = f"models:/{model_name}/{stage}"
                logger.info(f"Attempting to load {model_name} model (stage: {stage}) from MLflow model registry")
                model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Model {model_name} (stage: {stage}) loaded successfully")
                return model
            except Exception as stage_error:
                logger.warning(f"Could not load model with stage {stage}: {stage_error}. Trying latest version.")
        
        # If no stage provided or stage load failed, try with latest version
        model_uri = f"models:/{model_name}/latest"
        logger.info(f"Loading {model_name} model (latest version) from MLflow model registry")
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Model {model_name} (latest version) loaded successfully")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name} from registry: {e}")
        raise

def get_card_data_with_clusters(clustering_model = None) -> Dict[int, List[Dict]]:
    """
    Get card data with cluster assignments from the trained clustering model.
    
    Instead of re-clustering, this function retrieves the original cluster labels 
    assigned during model training.
    
    Args:
        clustering_model: Optional clustering model. If not provided, will load from registry.
    
    Returns:
        Dictionary mapping cluster IDs to lists of card dictionaries
    """
    try:
        # Load model if not provided
        model = clustering_model
        if model is None:
            # Load the model without specifying a stage (will use latest version)
            model = get_clustering_model_from_registry(stage=None)
        
        # Get the stored cluster labels from the trained model
        logger.info("Retrieving cluster labels from the trained model")
        
        # Check various attributes where labels might be stored
        labels_attrs = ['labels_', '_labels', 'labels', 'cluster_labels_']
        labels = None
        
        for attr in labels_attrs:
            if hasattr(model, attr):
                potential_labels = getattr(model, attr)
                if potential_labels is not None and len(potential_labels) > 0:
                    labels = potential_labels
                    logger.info(f"Found cluster labels in model.{attr}")
                    break
        
        # If no labels found, try to see if the model has a labels method
        if labels is None and hasattr(model, 'predict'):
            logger.info("No stored labels found, will use model.predict() method")
            try:
                # We'll need to predict using our feature data from S3
                feature_type = "combined"
                s3_key = FEATURE_CONFIGS[feature_type]["s3_key"]
                
                # Load the data
                logger.info(f"Loading card data from S3 with key: {s3_key}")
                df = read_parquet_from_s3(S3_BUCKET, s3_key)
                
                # Extract metadata columns
                metadata_cols = ["id", "name", "type", "desc", "atk", "def", "level", 
                                "attribute", "archetype", "tcgplayer_price"]
                available_meta_cols = [col for col in metadata_cols if col in df.columns]
                
                # Extract feature columns (non-metadata)
                feature_df = df.drop(available_meta_cols)
                
                # Preprocess features for prediction
                X = feature_df.to_numpy()
                
                # Call predict method
                labels = model.predict(X)
                logger.info(f"Generated labels using model.predict() on {len(labels)} samples")
            except Exception as predict_err:
                logger.error(f"Error predicting labels: {predict_err}")
                raise AttributeError("Could not get labels from model attributes or predict method")
        
        # Final check for labels
        if labels is None:
            # Last resort - try to extract all attributes and see what might be usable
            model_attrs = dir(model)
            logger.error(f"Could not find labels. Available model attributes: {model_attrs}")
            raise AttributeError("The clustering model does not have stored labels and predict failed. "
                                "This likely means the model was not saved correctly.")
        logger.info(f"Retrieved {len(labels)} cluster labels from the model")
        
        # Get model metadata from MLflow to determine what feature type was used
        # By default, use the combined features which are most likely to be used in production
        feature_type = "combined"
        s3_key = FEATURE_CONFIGS[feature_type]["s3_key"]
        
        # Load the original data that was used for training
        logger.info(f"Loading original card data from S3 with key: {s3_key}")
        df = read_parquet_from_s3(S3_BUCKET, s3_key)
        
        # Verify the number of samples matches the number of labels
        if len(df) != len(labels):
            raise ValueError(f"Number of data points ({len(df)}) doesn't match "
                            f"number of labels ({len(labels)}). Dataset may have changed.")
        
        # Extract metadata columns for the card information
        metadata_cols = ["id", "name", "type", "desc", "atk", "def", "level", 
                        "attribute", "archetype", "tcgplayer_price"]
        available_meta_cols = [col for col in metadata_cols if col in df.columns]
        meta_df = df.select(available_meta_cols)
        
        # Add the cluster labels to the metadata dataframe
        logger.info("Adding cluster labels to card data")
        meta_df = meta_df.with_columns(pl.Series(name="cluster_id", values=labels))
        
        # Create a dictionary of cards by cluster
        clustered_cards = {}
        
        # Get unique cluster IDs (excluding noise if desired)
        unique_clusters = np.unique(labels)
        logger.info(f"Found {len(unique_clusters)} unique clusters including noise points")
        
        # Process each cluster
        for cluster_id in unique_clusters:
            # Skip noise points (cluster -1) if they exist
            if cluster_id == -1:
                continue
            
            # Get all cards in this cluster
            cluster_cards = meta_df.filter(pl.col("cluster_id") == cluster_id)
            
            # Convert to list of dictionaries
            cards_list = []
            for card in cluster_cards.to_dicts():
                # Create a clean dictionary with only the needed fields
                card_dict = {
                    "id": card.get("id"),
                    "name": card.get("name"),
                    "type": card.get("type"),
                    "desc": card.get("desc", ""),
                    "cluster_id": int(card.get("cluster_id")),
                }
                
                # Add optional fields if present
                for field in ["atk", "def", "level", "attribute", "archetype", "tcgplayer_price"]:
                    if field in card and card[field] is not None:
                        card_dict[field] = card[field]
                
                # Handle archetypes as a list for consistency with expected format
                if "archetype" in card and card["archetype"]:
                    card_dict["archetypes"] = [card["archetype"]]
                else:
                    card_dict["archetypes"] = []
                    
                cards_list.append(card_dict)
            
            clustered_cards[int(cluster_id)] = cards_list
            
        # Calculate some statistics for logging
        non_noise_clusters = len([c for c in unique_clusters if c != -1])
        total_cards = sum(len(cards) for cards in clustered_cards.values())
        avg_cluster_size = total_cards / non_noise_clusters if non_noise_clusters > 0 else 0
        
        logger.info(f"Created clustered cards dictionary with {len(clustered_cards)} clusters")
        logger.info(f"Total of {total_cards} cards across all clusters, average cluster size: {avg_cluster_size:.2f}")
        
        return clustered_cards
    
    except Exception as e:
        logger.error(f"Error getting card data with clusters: {e}")
        raise

def diagnose_mlflow_model(model_name: str = "yugioh_clustering_final_model") -> Dict[str, Any]:
    """
    Diagnose a model from the MLflow model registry.
    
    This function provides detailed information about a model in the registry
    to help troubleshoot issues with loading models or accessing labels.
    
    Args:
        model_name: Name of the registered model to diagnose
        
    Returns:
        Dictionary with diagnostic information about the model
    """
    from mlflow.tracking import MlflowClient
    
    try:
        client = MlflowClient(MLFLOW_TRACKING_URI)
        diagnostics = {"model_name": model_name}
        
        # Check if model exists
        try:
            model_details = client.get_registered_model(model_name)
            diagnostics["model_exists"] = True
            diagnostics["creation_timestamp"] = model_details.creation_timestamp
            diagnostics["last_updated_timestamp"] = model_details.last_updated_timestamp
            diagnostics["description"] = model_details.description
            
            # Extract tags safely
            try:
                if hasattr(model_details, 'tags') and model_details.tags:
                    if isinstance(model_details.tags, dict):
                        diagnostics["tags"] = model_details.tags
                    else:
                        diagnostics["tags"] = {tag.key: tag.value for tag in model_details.tags}
                else:
                    diagnostics["tags"] = {}
            except Exception as tag_error:
                diagnostics["tags_error"] = str(tag_error)
                diagnostics["tags"] = {}
                
        except Exception as e:
            diagnostics["model_exists"] = False
            diagnostics["error"] = str(e)
            return diagnostics
        
        # Get model versions
        try:
            model_versions = client.search_model_versions(f"name='{model_name}'")
            diagnostics["versions"] = []
            
            for version in model_versions:
                version_info = {
                    "version": version.version,
                    "current_stage": version.current_stage,
                    "run_id": version.run_id,
                    "status": version.status,
                    "source": version.source
                }
                diagnostics["versions"].append(version_info)
        except Exception as e:
            diagnostics["versions_error"] = str(e)
        
        # Try to load the latest model version to check for labels
        try:
            model_uri = f"models:/{model_name}/latest"
            model = mlflow.sklearn.load_model(model_uri)
            diagnostics["model_load_success"] = True
            diagnostics["model_type"] = type(model).__name__
            
            # Check all possible attributes where labels might be stored
            labels_attrs = ['labels_', '_labels', 'labels', 'cluster_labels_']
            for attr in labels_attrs:
                diagnostics[f"has_{attr}"] = hasattr(model, attr)
            
            # Try to get the labels
            for attr in labels_attrs:
                if hasattr(model, attr):
                    labels = getattr(model, attr)
                    if labels is not None:
                        diagnostics[f"{attr}_length"] = len(labels)
                        diagnostics[f"{attr}_unique_clusters"] = len(np.unique(labels))
                        # Count of items per cluster
                        cluster_counts = {int(cluster_id): int(np.sum(labels == cluster_id)) 
                                         for cluster_id in np.unique(labels)}
                        diagnostics[f"{attr}_cluster_counts"] = cluster_counts
                        diagnostics["found_labels"] = True
                        diagnostics["labels_attribute"] = attr
                        break
            else:
                diagnostics["found_labels"] = False
                
            # Check all available attributes of the model
            diagnostics["model_attributes"] = dir(model)
            
        except Exception as e:
            diagnostics["model_load_success"] = False
            diagnostics["load_error"] = str(e)
        
        return diagnostics
    
    except Exception as e:
        logger.error(f"Error diagnosing model {model_name}: {e}")
        return {"error": str(e)}
