import mlflow
import tempfile
import json
import os
import numpy as np
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List, Union
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow tracking setup
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(MLFLOW_TRACKING_URI)

# === General Utilities ===
def safe_json_serialize(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj

def log_params(params: Dict[str, Any]) -> None:
    for key, value in params.items():
        value = safe_json_serialize(value)
        mlflow.log_param(key, value)
    logger.info(f"Logged {len(params)} parameters to MLflow")

def log_metrics(metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
    for key, value in metrics.items():
        value = safe_json_serialize(value)
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value, step=step)
    logger.info(f"Logged {len(metrics)} metrics to MLflow")

def log_tags(tags: Dict[str, str]) -> None:
    for key, value in tags.items():
        mlflow.set_tag(key, str(value))
    logger.info(f"Logged {len(tags)} tags to MLflow")

def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    mlflow.log_artifact(local_path, artifact_path)
    logger.info(f"Logged artifact from {local_path} to MLflow" + (f" under {artifact_path}" if artifact_path else ""))

def log_dict_as_artifact(data: Dict[str, Any], filename: str, artifact_subdir: str) -> None:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump(safe_json_serialize(data), tmp_file, indent=2)
        tmp_file.flush()
        mlflow.log_artifact(tmp_file.name, artifact_path=artifact_subdir)
        os.unlink(tmp_file.name)

# === Experiment Setup ===
def setup_experiment(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name)
    return experiment_id

# === Model Logging and Registration ===
def log_ml_model(model: Any, artifact_path: str = "model", framework: str = "sklearn",
                 save_input_example: bool = False, input_example: Optional[Any] = None,
                 signature: Optional[Any] = None) -> str:
    logger.info(f"Logging {framework} model to MLflow")
    try:
        log_fn_map = {
            "xgboost": mlflow.xgboost.log_model,
            "sklearn": mlflow.sklearn.log_model,
            "pytorch": mlflow.pytorch.log_model
        }
        if framework.lower() in log_fn_map:
            return log_fn_map[framework.lower()] (
                model,
                artifact_path=artifact_path,
                input_example=input_example if save_input_example else None,
                signature=signature
            ).model_uri
        else:
            return mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                input_example=input_example if save_input_example else None,
                signature=signature
            ).model_uri
    except Exception as e:
        logger.error(f"Error logging {framework} model: {e}")
        raise

def register_model(model_uri: str, registered_model_name: str) -> str:
    try:
        result = client.create_model_version(
            name=registered_model_name,
            source=model_uri,
            run_id=mlflow.active_run().info.run_id
        )
        logger.info(f"Model registered: {result.name} v{result.version}")
        return f"models:/{result.name}/{result.version}"
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise

# === Clustering Utilities ===
def log_clustering_tags(algorithm: str, feature_type: str, experiment_config: str, 
                        stage: str = "exploration", version: str = "v2.0", **kwargs) -> None:
    tags = {
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
    tags.update(kwargs)
    log_tags(tags)

def log_clustering_params(s3_bucket: str, s3_key: str, algorithm_params: Dict[str, Any], 
                          preprocessing_params: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    params = {
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
        **algorithm_params
    }
    if preprocessing_params:
        params.update(preprocessing_params)
    params.update(kwargs)
    log_params(params)

def log_essential_clustering_metrics(total_clusters: int, silhouette_score: float, num_samples: int, 
                                     num_features: int, noise_percentage: float) -> None:
    metrics = {
        "total_clusters": total_clusters,
        "silhouette_score": silhouette_score,
        "num_samples": num_samples,
        "num_features": num_features,
        "noise_percentage": noise_percentage
    }
    log_metrics(metrics)

def save_clustering_artifacts(metrics_summary: Dict[str, Any], 
                               cluster_analysis: Optional[Dict[str, Any]] = None,
                               plots: Optional[List[str]] = None) -> None:
    log_dict_as_artifact(metrics_summary, "metrics_summary.json", "metrics")
    if cluster_analysis:
        log_dict_as_artifact(cluster_analysis, "cluster_analysis.json", "cluster_analysis")
    if plots:
        for path in plots:
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="plots")

# === Deck Generation Utilities ===
def log_deck_generation_tags(generation_mode: str, target_archetype: Optional[str] = None,
                             stage: str = "exploration", version: str = "v1.0", **kwargs) -> None:
    tags = {
        "model_type": "deck_generation",
        "dataset": "yugioh_cards",
        "version": version,
        "purpose": "deck_generation",
        "stage": stage,
        "generation_mode": generation_mode
    }
    if target_archetype:
        tags["target_archetype"] = target_archetype
    tags.update(kwargs)
    log_tags(tags)

def log_deck_generation_params(generation_mode: str, target_archetype: Optional[str] = None,
                                clustering_run_id: Optional[str] = None, **kwargs) -> None:
    params = {"generation_mode": generation_mode}
    if target_archetype:
        params["target_archetype"] = target_archetype
    if clustering_run_id:
        params["clustering_run_id"] = clustering_run_id
    params.update(kwargs)
    log_params(params)

def log_deck_metrics(main_deck: List[Dict], extra_deck: List[Dict], metadata: Any) -> None:
    mlflow.log_metric("main_deck_size", len(main_deck))
    mlflow.log_metric("extra_deck_size", len(extra_deck))

    if main_deck:
        monster = sum('Monster' in card.get('type', '') for card in main_deck)
        spell = sum('Spell' in card.get('type', '') for card in main_deck)
        trap = sum('Trap' in card.get('type', '') for card in main_deck)
        total = len(main_deck)
        mlflow.log_metric("monster_ratio", monster / total)
        mlflow.log_metric("spell_ratio", spell / total)
        mlflow.log_metric("trap_ratio", trap / total)
        mlflow.log_metric("monster_count", monster)
        mlflow.log_metric("spell_count", spell)
        mlflow.log_metric("trap_count", trap)

    if extra_deck:
        fusion = sum('Fusion' in card.get('type', '') for card in extra_deck)
        synchro = sum('Synchro' in card.get('type', '') for card in extra_deck)
        xyz = sum('Xyz' in card.get('type', '') for card in extra_deck)
        link = sum('Link' in card.get('type', '') for card in extra_deck)
        total = len(extra_deck)
        mlflow.log_metric("fusion_ratio", fusion / total)
        mlflow.log_metric("synchro_ratio", synchro / total)
        mlflow.log_metric("xyz_ratio", xyz / total)
        mlflow.log_metric("link_ratio", link / total)
        mlflow.log_metric("fusion_count", fusion)
        mlflow.log_metric("synchro_count", synchro)
        mlflow.log_metric("xyz_count", xyz)
        mlflow.log_metric("link_count", link)

    meta = metadata.to_dict()
    mlflow.log_metric("cluster_entropy", meta.get("cluster_entropy", 0.0))
    mlflow.log_metric("archetype_diversity", len(meta.get("archetype_distribution", {})))
    total_cards = sum(meta.get("archetype_distribution", {}).values())
    max_pct = max(meta.get("archetype_distribution", {}).values(), default=0)
    mlflow.log_metric("dominant_archetype_ratio", max_pct / total_cards if total_cards > 0 else 0)
    mlflow.log_metric("cluster_diversity", len(meta.get("cluster_distribution", {})))
    mlflow.log_metric("intra_deck_cluster_distance", meta.get("intra_deck_cluster_distance", 0.0))
    mlflow.log_metric("cluster_co_occurrence_rarity", meta.get("cluster_co_occurrence_rarity", 0.0))
    mlflow.log_metric("noise_card_percentage", meta.get("noise_card_percentage", 0.0))

def log_deck_artifacts(main_deck: List[Dict], extra_deck: List[Dict], metadata: Any) -> None:
    deck_dict = {
        "main_deck": main_deck,
        "extra_deck": extra_deck,
        "metadata": metadata.to_dict()
    }
    log_dict_as_artifact(deck_dict, "deck_list.json", "")

    text = f"=== YU-GI-OH! DECK LIST ===\n\nMain Deck ({len(main_deck)} cards):\n" + "="*30 + "\n"
    for group, name in [("Monster", "Monsters"), ("Spell", "Spells"), ("Trap", "Traps")]:
        cards = [c for c in main_deck if group in c.get("type", "")]
        if cards:
            text += f"\n{name} ({len(cards)}):\n" + ''.join([f"  - {c['name']}\n" for c in cards])

    if extra_deck:
        text += f"\n\nExtra Deck ({len(extra_deck)} cards):\n" + "="*30 + "\n"
        for card in extra_deck:
            text += f"  - {card['name']} ({card.get('type', 'Unknown')})\n"

    meta = metadata.to_dict()
    text += "\n\nDeck Statistics:\n" + "="*30 + "\n"
    text += f"Dominant Archetype: {meta.get('dominant_archetype', '')}\n"
    text += f"Cluster Entropy: {meta.get('cluster_entropy', 0.0):.3f}\n"
    text += f"Archetype Distribution: {meta.get('archetype_distribution', {})}\n"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(text)
        tmp_file.flush()
        mlflow.log_artifact(tmp_file.name, "deck_list.txt")
        os.unlink(tmp_file.name)

# === Force Model Registration Utility ===
def force_register_model_from_run(run_id: str, model_name: str = "yugioh_clustering_final_model",
                                  version: str = "1.0", stage: str = "Production") -> str:
    try:
        model_uri = f"runs:/{run_id}/clustering_model"
        mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)

        tags = {
            "version": version,
            "algorithm": "hdbscan",
            "dataset": "yugioh_cards",
            "purpose": "card_clustering"
        }
        for k, v in tags.items():
            client.set_registered_model_tag(model_name, k, v)

        client.update_registered_model(
            name=model_name,
            description=f"Yu-Gi-Oh Card Clustering Model using HDBSCAN - Force registered v{version}"
        )

        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage=stage,
            archive_existing_versions=True
        )

        logger.info(f"Model force registered: {model_name} v{version} -> {stage}")
        return model_uri

    except Exception as e:
        logger.error(f"Error force registering model: {e}")
        raise

# === Best Run Evaluation ===
def get_best_clustering_run(experiment_name: str = "/yugioh_card_clustering") -> Optional[Any]:
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

        def run_score(run):
            silhouette = run.data.metrics.get("silhouette_score", -1)
            noise = run.data.metrics.get("noise_percentage", 0)
            n_clusters = run.data.metrics.get("total_clusters", 0)
            score = 0.7 * silhouette
            if 5 <= noise <= 15:
                score += 0.2
            elif 15 < noise <= 25:
                score += 0.05
            elif noise > 25:
                score -= 0.1 * (noise - 25) / 100
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

# === Final Model Check ===
def determine_if_final_model(run_data: Dict[str, Any]) -> bool:
    silhouette = run_data.get('silhouette_score', 0)
    n_clusters = run_data.get('total_clusters', 0)
    noise_pct = run_data.get('noise_percentage', 100)
    feature_type = run_data.get('feature_type', '')

    is_final = (
        silhouette > 0.3 and
        10 <= n_clusters <= 1000 and
        noise_pct < 30 and
        feature_type == 'combined'
    )

    logger.info(
        f"Final model check: silhouette={silhouette:.3f}, clusters={n_clusters}, "
        f"noise={noise_pct:.1f}%, features={feature_type} -> {'YES' if is_final else 'NO'}"
    )
    return is_final