from typing import List, Tuple, Dict, Any
import logging
import os

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import mlflow

from src.utils.s3_utils import read_parquet_from_s3
from src.utils.mlflow.mlflow_utils import setup_experiment, log_params, log_metrics, log_ml_model, log_artifact, log_tags

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S3_BUCKET = "yugioh-data"
S3_KEY = "processed/feature_engineered/deck_scoring/2025-06/feature_engineered.parquet"

def load_data(bucket: str, key: str) -> pd.DataFrame:
    logger.info("Loading data from S3")
    return read_parquet_from_s3(bucket, key).to_pandas()

def preprocess_data(deck_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    logger.info("Preprocessing data")

    embedding_df = pd.DataFrame(
        deck_df['main_deck_mean_embedding'].to_list(),
        index=deck_df.index
    ).add_prefix("embedding_")

    deck_df = pd.concat([deck_df.drop(columns=['main_deck_mean_embedding']), embedding_df], axis=1)

    embedding_cols = [col for col in deck_df.columns if col.startswith("embedding_")]
    bow_cols = [col for col in deck_df.columns if col.startswith("bow_")]

    feature_cols = [
        'has_tuner', 'num_tuners', 'has_same_level_monsters', 'max_same_level_count',
        'avg_monster_level', 'has_pendulum_monsters', 'has_synchro_monsters',
        'has_xyz_monsters', 'has_link_monsters', 'max_copies_per_card',
        'avg_copies_per_monster', 'num_unique_monsters', 'main_deck_mean_tfidf',
        'mentions_banish', 'mentions_graveyard', 'mentions_draw', 'mentions_search',
        'mentions_special_summon', 'mentions_negate', 'mentions_destroy', 'mentions_shuffle'
    ] + embedding_cols + bow_cols

    X = deck_df[feature_cols]
    y = deck_df['composite_score']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dense_cols = [col for col in X.columns if col not in bow_cols and col not in embedding_cols]
    X_train_bow = X_train_raw[bow_cols]
    X_test_bow = X_test_raw[bow_cols]

    X_train_dense = X_train_raw[dense_cols]
    X_test_dense = X_test_raw[dense_cols]

    svd = TruncatedSVD(n_components=50, random_state=42)
    X_train_bow_reduced = svd.fit_transform(X_train_bow)
    X_test_bow_reduced = svd.transform(X_test_bow)

    scaler = StandardScaler()
    X_train_dense_scaled = scaler.fit_transform(X_train_dense)
    X_test_dense_scaled = scaler.transform(X_test_dense)

    embedding_train = np.vstack(X_train_raw[embedding_cols].values)
    embedding_test = np.vstack(X_test_raw[embedding_cols].values)

    embedding_svd = TruncatedSVD(n_components=20, random_state=42)
    embedding_train_reduced = embedding_svd.fit_transform(embedding_train)
    embedding_test_reduced = embedding_svd.transform(embedding_test)

    X_train_final = np.hstack([X_train_dense_scaled, X_train_bow_reduced, embedding_train_reduced])
    X_test_final = np.hstack([X_test_dense_scaled, X_test_bow_reduced, embedding_test_reduced])

    dense_feature_names = X_train_dense.columns.tolist()
    bow_feature_names = [f"svd_bow_{i}" for i in range(X_train_bow_reduced.shape[1])]
    embedding_feature_names = [f"svd_embed_{i}" for i in range(embedding_train_reduced.shape[1])]
    final_feature_names = dense_feature_names + bow_feature_names + embedding_feature_names

    # Log preprocessing parameters
    preprocessing_params = {
        "bow_svd_components": 50,
        "embedding_svd_components": 20,
        "train_test_split_ratio": 0.8,
        "dense_features_count": len(dense_feature_names),
        "bow_features_count": len(bow_feature_names),
        "embedding_features_count": len(embedding_feature_names),
        "total_features_count": len(final_feature_names)
    }
    log_params(preprocessing_params)
    
    return X_train_final, X_test_final, y_train, y_test, final_feature_names

def train_and_evaluate(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    logger.info("Training XGBoost model")
    
    # Define model parameters
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Log model parameters to MLflow using our utility function
    log_params(params)
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    logger.info("Evaluating model")
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics to MLflow using our utility function
    metrics = {
        "rmse": rmse,
        "r2_score": r2
    }
    log_metrics(metrics)

    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Create and save feature importance visualization
    model.get_booster().feature_names = feature_names
    logger.info("Plotting feature importances")
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=20, ax=ax)
    plt.tight_layout()
    plt.title("Top Feature Importances")
    
    # Save plot locally 
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    plt_path = os.path.join(artifacts_dir, "xgb_feature_importance.png")
    plt.savefig(plt_path, dpi=300, bbox_inches="tight")
    
    # Log the model artifact and feature importance plot using our utility function
    model_uri = log_ml_model(
        model, 
        artifact_path="xgboost_model", 
        framework="xgboost",
        registered_model_name="deck_scoring_xgboost"
    )
    
    # Log the plot as artifact
    log_artifact(plt_path, "visualizations")
    
    return {
        "model": model,
        "metrics": metrics,
        "model_uri": model_uri,
        "artifacts": {
            "feature_importance": plt_path
        }
    }

def log_deck_scoring_prediction_tags(version: str = "v1.0") -> None:
    """
    Log standard tags for deck scoring experiments
    """
    tags = {
        "model_type": "regression",
        "algorithm": "xgboost",
        "dataset": "yugioh_decks",
        "purpose": "deck_scoring_prediction",
        "data_source": f"s3://{S3_BUCKET}/{S3_KEY}",
        "stage": "development",
        "version": version
    }
    
    # Use the general-purpose log_tags function
    log_tags(tags)

if __name__ == "__main__":
    experiment_name = "deck_scoring_xgboost"
    experiment_id = setup_experiment(experiment_name)

    # Use MLflow run context to group all logged items
    with mlflow.start_run(experiment_id=experiment_id) as run:
        logger.info(f"MLflow run started: {run.info.run_id}")
        
        # Log standard tags
        log_deck_scoring_prediction_tags()
        
        # Log data source and dataset information
        dataset_params = {
            "data_source": f"s3://{S3_BUCKET}/{S3_KEY}",
        }
        
        df = load_data(S3_BUCKET, S3_KEY)
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
        
        # Add dataset information to parameters
        dataset_params.update({
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(feature_names)
        })
        
        # Log all dataset parameters using our utility function
        log_params(dataset_params)
        
        # Train and evaluate the model
        results = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)
        
        logger.info(f"MLflow run completed: {run.info.run_id}")
        logger.info(f"MLflow UI: http://localhost:5000/experiments/{experiment_id}")
