from typing import List, Tuple
import logging

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
import mlflow

from src.utils.s3_utils import read_parquet_from_s3
from src.utils.mlflow.mlflow_utils import setup_experiment, log_params, log_metrics, log_tags
from src.model.tuning.optuna_tuner import tune_xgboost_model
from src.utils.plotting.plotting_utils import plot_scree_curve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S3_BUCKET = "yugioh-data"
S3_KEY = "processed/feature_engineered/deck_scoring/2025-08/feature_engineered.parquet"

def load_data(bucket: str, key: str) -> pd.DataFrame:
    logger.info("Loading data from S3")
    return read_parquet_from_s3(bucket, key).to_pandas()

def preprocess_data(deck_df: pd.DataFrame, target_col: str = 'composite_score', use_pca: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    logger.info(f"Preprocessing data using {'PCA' if use_pca else 'TruncatedSVD'} with target variable: {target_col}")

    embedding_df = pd.DataFrame(
        deck_df['main_deck_mean_embedding'].to_list(),
        index=deck_df.index
    ).add_prefix("embedding_")
    deck_df = pd.concat([deck_df.drop(columns=['main_deck_mean_embedding']), embedding_df], axis=1)

    embedding_cols = [col for col in deck_df.columns if col.startswith("embedding_")]
    feature_cols = [
        # Basic deck features
        'has_tuner', 'num_tuners', 'has_same_level_monsters', 'max_same_level_count',
        'avg_monster_level', 'has_pendulum_monsters', 'has_synchro_monsters',
        'has_xyz_monsters', 'has_link_monsters', 'max_copies_per_card',
        'avg_copies_per_monster', 'num_unique_monsters',
        # Advanced Mechanic Features
        'tuner_count', 'non_tuner_count', 'can_synchro', 'matched_synchro_levels',
        'xyz_level_mode', 'xyz_level_mode_count', 'can_xyz', 'p_two_of_mode_lvl_in_7',
        'fusion_enabler_count', 'can_fusion', 'low_level_count', 'monster_count',
        'max_link_rating_in_extra', 'can_link', 'pendulum_count', 'pendulum_span_max',
        'pendulum_inrange_monsters', 'can_pendulum',
        # NLP and Strategy Features
        'main_deck_mean_tfidf', 'num_banish', 'num_graveyard', 'num_draw', 'num_search',
        'num_special_summon', 'num_negate', 'num_destroy', 'num_shuffle'
        # Cluster related features
        'cluster_entropy', 'intra_deck_cluster_distance', 'cluster_co_occurrence_rarity',
        'noise_card_percentage'
    ] + embedding_cols

    # Filter out features that don't exist in the dataframe
    available_features = [col for col in feature_cols if col in deck_df.columns]
    if len(available_features) != len(feature_cols):
        missing_features = set(feature_cols) - set(available_features)
        logger.warning(f"Missing features: {missing_features}")
        feature_cols = available_features

    X = deck_df[feature_cols]
    y = deck_df[target_col]
    test_size = 0.2
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    dense_cols = [col for col in X.columns if col not in embedding_cols]
    X_train_dense = X_train_raw[dense_cols]
    X_test_dense = X_test_raw[dense_cols]

    scaler = StandardScaler()
    X_train_dense_scaled = scaler.fit_transform(X_train_dense)
    X_test_dense_scaled = scaler.transform(X_test_dense)

    embedding_train = np.vstack(X_train_raw[embedding_cols].values)
    embedding_test = np.vstack(X_test_raw[embedding_cols].values)

    embedding_n_components = 40
    if use_pca:
        reducer = PCA(n_components=embedding_n_components, random_state=42)
    else:
        reducer = TruncatedSVD(n_components=embedding_n_components, random_state=42)

    embedding_train_reduced = reducer.fit_transform(embedding_train)
    embedding_test_reduced = reducer.transform(embedding_test)

    plot_scree_curve(reducer, output_path=f'src/model/deck_scoring/scree.png')

    X_train_final = np.hstack([X_train_dense_scaled, embedding_train_reduced])
    X_test_final = np.hstack([X_test_dense_scaled, embedding_test_reduced])

    dense_feature_names = X_train_dense.columns.tolist()
    embedding_feature_names = [f"{'pca' if use_pca else 'svd'}_embed_{i}" for i in range(embedding_n_components)]
    final_feature_names = dense_feature_names + embedding_feature_names

    log_params({
        "target_variable": target_col,
        "embedding_reduction_method": "PCA" if use_pca else "TruncatedSVD",
        "embedding_components": embedding_n_components,
        "train_test_split_ratio": 1 - test_size,
        "dense_features_count": len(dense_feature_names),
        "embedding_features_count": len(embedding_feature_names),
        "total_features_count": len(final_feature_names)
    })

    return X_train_final, X_test_final, y_train, y_test, final_feature_names

def log_deck_scoring_prediction_tags(target_col: str, version: str = "v1.0", method: str = "TruncatedSVD") -> None:
    log_tags({
        "model_type": "regression",
        "algorithm": "xgboost",
        "dataset": "yugioh_decks",
        "purpose": "deck_scoring_prediction",
        "target_variable": target_col,
        "data_source": f"s3://{S3_BUCKET}/{S3_KEY}",
        "stage": "development",
        "version": version,
        "dimensionality_reduction": method
    })

def run_pipeline(target_col: str = 'composite_score', use_pca: bool = False) -> None:
    df = load_data(S3_BUCKET, S3_KEY)
    
    # Validate target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    # import pdb; pdb.set_trace()
    method = "PCA" if use_pca else "TruncatedSVD"
    experiment_name = f"deck_scoring_model_{target_col}"
    experiment_id = setup_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        logger.info(f"MLflow run started: {run.info.run_id} [{method}] Target: {target_col}")
        log_deck_scoring_prediction_tags(target_col=target_col, method=method)

        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df, target_col=target_col, use_pca=use_pca)

        # Split off validation from training
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        log_params({
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(feature_names)
        })

        best_model, best_params = tune_xgboost_model(X_train_final, y_train_final, X_val, y_val, feature_names=feature_names)

        logger.info("Evaluating tuned model on test set")
        y_pred = best_model.predict(X_test)
        log_metrics({
            "test_rmse": mean_squared_error(y_test, y_pred),
            "test_r2": r2_score(y_test, y_pred)
        })

        logger.info(f"Run completed: {run.info.run_id}")

if __name__ == "__main__":
    # Examples of running with different target variables
    # run_pipeline(target_col='composite_score', use_pca=False)    # Original composite score
    # run_pipeline(target_col='win_rate', use_pca=False)          # Raw win rate  
    run_pipeline(target_col='adjusted_win_rate', use_pca=True)   # Bayesian adjusted win rate
