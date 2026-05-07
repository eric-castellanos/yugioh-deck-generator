from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

UMAP_N_COMPONENTS_GRID = [10, 15, 25, 35, 50, 75, 100]
UMAP_N_NEIGHBORS_GRID = [10, 15, 30, 50, 75, 100]
UMAP_MIN_DIST_GRID = [0.0, 0.01, 0.05, 0.1, 0.25]
UMAP_METRIC_GRID = ["cosine", "euclidean"]
MIN_CLUSTER_SIZE_GRID = [3, 5, 8, 10, 15, 20, 25, 35, 50]
MIN_SAMPLES_GRID = [1, 2, 3, 5, 8, 10]
CLUSTER_SELECTION_METHOD_GRID = ["eom", "leaf"]
CLUSTER_SELECTION_EPSILON_GRID = [0.0, 0.01, 0.05, 0.1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization for UMAP + HDBSCAN clustering."
    )
    parser.add_argument("--embeddings-file", default="data/features/card_embeddings.parquet")
    parser.add_argument("--output-dir", default="data/features/clusters/hdbscan_sweep")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument(
        "--mlflow-experiment",
        default="yugioh-deck-generator/clustering-embeddings/hdbscan-sweep",
    )
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument(
        "--use-umap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable UMAP before HDBSCAN (default: true). Use --no-use-umap to disable.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--tiny-cluster-max-size", type=int, default=3)
    parser.add_argument("--tiny-cluster-ratio-threshold", type=float, default=0.20)

    parser.add_argument("--noise-ratio-threshold", type=float, default=0.30)
    parser.add_argument("--min-clusters-threshold", type=int, default=50)
    parser.add_argument("--largest-cluster-ratio-threshold", type=float, default=0.25)

    parser.add_argument("--num-clusters-normalizer", type=int, default=100)
    parser.add_argument("--weight-num-clusters", type=float, default=0.40)
    parser.add_argument("--weight-noise-ratio", type=float, default=0.25)
    parser.add_argument("--weight-largest-cluster-ratio", type=float, default=0.15)
    parser.add_argument("--weight-tiny-cluster-ratio", type=float, default=0.10)
    parser.add_argument("--weight-membership-probability", type=float, default=0.10)
    parser.add_argument("--penalty-low-clusters", type=float, default=1.0)
    parser.add_argument("--penalty-high-noise", type=float, default=1.0)

    parser.add_argument("--weight-semantic-archetype", type=float, default=0.10)
    parser.add_argument("--weight-semantic-functional", type=float, default=0.20)
    parser.add_argument("--weight-semantic-label-confidence", type=float, default=0.05)
    parser.add_argument("--weight-semantic-generic-balance", type=float, default=0.10)
    return parser.parse_args()


def _normalize_l2(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / safe_norms


def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    existing = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in existing:
            return df[existing[candidate.lower()]]
    return None


def _to_archetype_series(df: pd.DataFrame) -> pd.Series:
    archetype_col = _find_first_column(
        df, ["archetype", "archetypes", "primary_archetype", "card_archetype"]
    )
    if archetype_col is None:
        return pd.Series(["unknown"] * len(df), index=df.index, dtype="string")
    return archetype_col.fillna("unknown").astype("string").str.strip().replace("", "unknown")


def _to_label_confidence_series(df: pd.DataFrame) -> pd.Series:
    confidence_col = _find_first_column(
        df, ["label_confidence", "llm_label_confidence", "cluster_label_confidence"]
    )
    if confidence_col is None:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index, dtype=float)
    return pd.to_numeric(confidence_col, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)


def load_embeddings(embeddings_file: str) -> tuple[pd.Series, np.ndarray, pd.DataFrame]:
    df = pd.read_parquet(embeddings_file)
    required = {"card_id", "embedding_vector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    card_ids = df["card_id"].astype("int64")
    vectors = np.asarray(df["embedding_vector"].tolist(), dtype=float)
    if vectors.ndim != 2:
        raise ValueError("Expected 2D vectors")
    semantic_df = pd.DataFrame(
        {
            "card_id": card_ids.to_numpy(),
            "archetype": _to_archetype_series(df).to_numpy(),
            "label_confidence": _to_label_confidence_series(df).to_numpy(),
        }
    )
    return card_ids, _normalize_l2(vectors), semantic_df


def run_clustering(
    vectors: np.ndarray,
    *,
    use_umap: bool,
    umap_n_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    cluster_vectors = vectors
    n_umap_components = int(vectors.shape[1])

    if use_umap:
        import umap

        projector = umap.UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_state,
        )
        cluster_vectors = np.asarray(projector.fit_transform(vectors), dtype=float)
        n_umap_components = int(cluster_vectors.shape[1])

    import hdbscan

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="best",
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    labels = model.fit_predict(cluster_vectors)
    probabilities = np.asarray(getattr(model, "probabilities_", np.zeros(len(labels), dtype=float)))
    return labels, probabilities, n_umap_components


def compute_cluster_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    tiny_cluster_max_size: int = 3,
) -> dict[str, float | int | str]:
    total = len(labels)
    if total == 0:
        raise ValueError("labels must not be empty")

    if probabilities is None:
        probabilities = np.zeros(total, dtype=float)

    noise_mask = labels == -1
    num_noise_cards = int(np.sum(noise_mask))
    noise_ratio = float(num_noise_cards / total)
    non_noise = labels[~noise_mask]
    non_noise_probabilities = probabilities[~noise_mask]
    num_clusters = int(len(set(non_noise.tolist()))) if len(non_noise) > 0 else 0

    if len(non_noise) > 0:
        counts_series = pd.Series(non_noise).value_counts()
        sizes = counts_series.to_numpy(dtype=float)
        largest_cluster_size = float(sizes.max())
        top_10_sizes = counts_series.head(10).tolist()
        cluster_size_min = float(np.min(sizes))
        cluster_size_max = float(np.max(sizes))
        cluster_size_mean = float(np.mean(sizes))
        cluster_size_median = float(np.median(sizes))
        tiny_cluster_cards = int(np.sum(sizes[sizes <= float(tiny_cluster_max_size)]))
        tiny_cluster_ratio = float(tiny_cluster_cards / total)
        membership_probability_mean = (
            float(np.mean(non_noise_probabilities)) if len(non_noise_probabilities) > 0 else 0.0
        )
    else:
        largest_cluster_size = 0.0
        top_10_sizes = []
        cluster_size_min = 0.0
        cluster_size_max = 0.0
        cluster_size_mean = 0.0
        cluster_size_median = 0.0
        tiny_cluster_ratio = 0.0
        membership_probability_mean = 0.0

    largest_cluster_ratio = float(largest_cluster_size / total)
    return {
        "num_clusters": num_clusters,
        "num_noise_cards": num_noise_cards,
        "noise_ratio": noise_ratio,
        "cluster_size_min": cluster_size_min,
        "cluster_size_max": cluster_size_max,
        "cluster_size_mean": cluster_size_mean,
        "cluster_size_median": cluster_size_median,
        "largest_cluster_ratio": largest_cluster_ratio,
        "tiny_cluster_ratio": tiny_cluster_ratio,
        "membership_probability_mean": membership_probability_mean,
        "top_10_cluster_sizes_json": json.dumps(top_10_sizes),
    }


def compute_structural_score(
    metrics: dict[str, float | int | str], args: argparse.Namespace
) -> dict[str, float]:
    num_clusters = int(metrics["num_clusters"])
    noise_ratio = float(metrics["noise_ratio"])
    largest_cluster_ratio = float(metrics["largest_cluster_ratio"])
    tiny_cluster_ratio = float(metrics["tiny_cluster_ratio"])
    membership_probability_mean = float(metrics["membership_probability_mean"])
    normalized_num_clusters = min(num_clusters / float(max(1, args.num_clusters_normalizer)), 1.0)

    raw_score = (
        args.weight_num_clusters * normalized_num_clusters
        - args.weight_noise_ratio * noise_ratio
        - args.weight_largest_cluster_ratio * largest_cluster_ratio
        - args.weight_tiny_cluster_ratio * tiny_cluster_ratio
        + args.weight_membership_probability * membership_probability_mean
    )
    low_clusters_penalty = (
        args.penalty_low_clusters if num_clusters < args.min_clusters_threshold else 0.0
    )
    high_noise_penalty = (
        args.penalty_high_noise if noise_ratio > args.noise_ratio_threshold else 0.0
    )
    structural_score = float(raw_score - low_clusters_penalty - high_noise_penalty)
    return {
        "normalized_num_clusters": float(normalized_num_clusters),
        "structural_raw_score": float(raw_score),
        "structural_low_clusters_penalty": float(low_clusters_penalty),
        "structural_high_noise_penalty": float(high_noise_penalty),
        "structural_score": structural_score,
    }


def _safe_mean_similarity_to_centroid(vectors: np.ndarray) -> float:
    if len(vectors) <= 1:
        return 0.0
    centroid = np.mean(vectors, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0.0:
        return 0.0
    centroid = centroid / centroid_norm
    sims = np.clip(np.matmul(vectors, centroid), -1.0, 1.0)
    return float(np.mean(sims))


def compute_semantic_metrics(
    labels: np.ndarray,
    vectors: np.ndarray,
    semantic_df: pd.DataFrame,
) -> dict[str, float]:
    df = semantic_df.copy()
    df["cluster_id"] = labels
    non_noise_df = df[df["cluster_id"] != -1].copy()
    unique_clusters = sorted(non_noise_df["cluster_id"].unique().tolist())
    if len(unique_clusters) == 0:
        return {
            "archetype_purity_mean": 0.0,
            "functional_coherence_score": 0.0,
            "label_confidence_mean": 0.0,
            "generic_archetype_balance_score": 0.5,
        }

    purity_values: list[float] = []
    generic_heavy_count = 0
    archetype_heavy_count = 0
    coherence_values: list[float] = []
    labels_arr = np.asarray(labels)
    conf = pd.to_numeric(non_noise_df["label_confidence"], errors="coerce").fillna(0.0)

    for cluster_id in unique_clusters:
        cluster_mask = labels_arr == cluster_id
        cluster_size = int(np.sum(cluster_mask))
        if cluster_size == 0:
            continue

        cluster_slice = non_noise_df[non_noise_df["cluster_id"] == cluster_id]
        counts = cluster_slice["archetype"].value_counts()
        if len(counts) > 0:
            dominant_ratio = float(counts.iloc[0] / len(cluster_slice))
            purity_values.append(dominant_ratio)
            generic_count = int(
                cluster_slice["archetype"]
                .astype("string")
                .str.lower()
                .isin({"generic", "none", "unknown", "other", "misc"})
                .sum()
            )
            generic_ratio = float(generic_count / len(cluster_slice))
            archetype_ratio = float(1.0 - generic_ratio)
            if generic_ratio >= 0.6:
                generic_heavy_count += 1
            if archetype_ratio >= 0.6:
                archetype_heavy_count += 1

        cluster_vectors = vectors[cluster_mask]
        coherence_values.append(_safe_mean_similarity_to_centroid(cluster_vectors))

    archetype_purity_mean = float(np.mean(purity_values)) if purity_values else 0.0
    functional_coherence_score = float(np.mean(coherence_values)) if coherence_values else 0.0
    label_confidence_mean = float(conf.mean()) if len(conf) > 0 else 0.0

    cluster_count = max(1, len(unique_clusters))
    generic_fraction = float(generic_heavy_count / cluster_count)
    archetype_fraction = float(archetype_heavy_count / cluster_count)
    generic_archetype_balance_score = float(
        max(0.0, 1.0 - abs(generic_fraction - archetype_fraction))
    )

    return {
        "archetype_purity_mean": archetype_purity_mean,
        "functional_coherence_score": functional_coherence_score,
        "label_confidence_mean": label_confidence_mean,
        "generic_archetype_balance_score": generic_archetype_balance_score,
    }


def compute_semantic_score(
    semantic_metrics: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, float]:
    purity = float(semantic_metrics["archetype_purity_mean"])
    if purity <= 0.9:
        archetype_component = max(0.0, 1.0 - abs(purity - 0.65) / 0.65)
    else:
        archetype_component = max(0.0, 1.0 - abs(purity - 0.65) / 0.65) - (purity - 0.9) * 2.0

    functional = float(np.clip(semantic_metrics["functional_coherence_score"], 0.0, 1.0))
    label_confidence = float(np.clip(semantic_metrics["label_confidence_mean"], 0.0, 1.0))
    generic_balance = float(np.clip(semantic_metrics["generic_archetype_balance_score"], 0.0, 1.0))

    semantic_score = (
        args.weight_semantic_archetype * archetype_component
        + args.weight_semantic_functional * functional
        + args.weight_semantic_label_confidence * label_confidence
        + args.weight_semantic_generic_balance * generic_balance
    )
    return {
        "archetype_purity_component": float(archetype_component),
        "semantic_functional_component": functional,
        "semantic_label_confidence_component": label_confidence,
        "semantic_generic_balance_component": generic_balance,
        "semantic_score": float(semantic_score),
    }


def compute_final_score(
    structural_score: dict[str, float],
    semantic_score: dict[str, float],
) -> dict[str, float]:
    final_score = float(structural_score["structural_score"] + semantic_score["semantic_score"])
    return {"final_score": final_score}


def is_good_run(
    metrics: dict[str, float | int | str],
    noise_ratio_threshold: float,
    min_clusters_threshold: int,
    largest_cluster_ratio_threshold: float,
) -> bool:
    return (
        float(metrics["noise_ratio"]) < noise_ratio_threshold
        and int(metrics["num_clusters"]) >= min_clusters_threshold
        and float(metrics["largest_cluster_ratio"]) < largest_cluster_ratio_threshold
    )


def compute_guardrail_flags(
    metrics: dict[str, float | int | str],
    *,
    noise_ratio_threshold: float,
    min_clusters_threshold: int,
    largest_cluster_ratio_threshold: float,
    tiny_cluster_ratio_threshold: float,
) -> dict[str, int]:
    noise_flag = int(float(metrics["noise_ratio"]) > noise_ratio_threshold)
    cluster_flag = int(int(metrics["num_clusters"]) < min_clusters_threshold)
    largest_cluster_flag = int(
        float(metrics["largest_cluster_ratio"]) > largest_cluster_ratio_threshold
    )
    tiny_cluster_flag = int(float(metrics["tiny_cluster_ratio"]) > tiny_cluster_ratio_threshold)
    any_flag = int(noise_flag or cluster_flag or largest_cluster_flag or tiny_cluster_flag)
    return {
        "flag_noise_ratio_high": noise_flag,
        "flag_num_clusters_low": cluster_flag,
        "flag_largest_cluster_ratio_high": largest_cluster_flag,
        "flag_tiny_cluster_ratio_excessive": tiny_cluster_flag,
        "flag_any_guardrail": any_flag,
    }


def log_run_to_mlflow(
    tracking_uri: str | None,
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float | int],
    artifact_paths: list[Path],
) -> None:
    if not tracking_uri:
        return
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        numeric_metrics = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, int | float) and k != "top_10_cluster_sizes_json"
        }
        mlflow.log_metrics(numeric_metrics)
        mlflow.set_tag("top_10_cluster_sizes_json", str(metrics["top_10_cluster_sizes_json"]))
        for path in artifact_paths:
            mlflow.log_artifact(str(path))


def _suggest_params(trial: Any, *, use_umap: bool) -> dict[str, Any]:
    params: dict[str, Any] = {
        "use_umap": use_umap,
        "min_cluster_size": trial.suggest_categorical("min_cluster_size", MIN_CLUSTER_SIZE_GRID),
        "min_samples": trial.suggest_categorical("min_samples", MIN_SAMPLES_GRID),
        "cluster_selection_method": trial.suggest_categorical(
            "cluster_selection_method", CLUSTER_SELECTION_METHOD_GRID
        ),
        "cluster_selection_epsilon": trial.suggest_categorical(
            "cluster_selection_epsilon", CLUSTER_SELECTION_EPSILON_GRID
        ),
    }
    if use_umap:
        params["umap_n_components"] = trial.suggest_categorical(
            "umap_n_components", UMAP_N_COMPONENTS_GRID
        )
        params["umap_n_neighbors"] = trial.suggest_categorical(
            "umap_n_neighbors", UMAP_N_NEIGHBORS_GRID
        )
        params["umap_min_dist"] = trial.suggest_categorical("umap_min_dist", UMAP_MIN_DIST_GRID)
        params["umap_metric"] = trial.suggest_categorical("umap_metric", UMAP_METRIC_GRID)
    else:
        params["umap_n_components"] = 0
        params["umap_n_neighbors"] = 0
        params["umap_min_dist"] = 0.0
        params["umap_metric"] = "none"
    return params


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_root = output_dir / "trials"
    trial_root.mkdir(parents=True, exist_ok=True)

    card_ids, vectors, semantic_df = load_embeddings(args.embeddings_file)
    logger.info("Loaded embeddings: cards=%s dim=%s", len(card_ids), vectors.shape[1])
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    study_name = f"hdbscan_optuna_{timestamp}"

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna is required for sweep_hdbscan optimization.") from exc

    sampler = optuna.samplers.TPESampler(seed=args.random_state)
    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler)
    logger.info("Starting Optuna study=%s trials=%s", study_name, args.n_trials)

    def objective(trial: Any) -> float:
        params = _suggest_params(trial, use_umap=bool(args.use_umap))
        labels, probabilities, n_umap_components = run_clustering(
            vectors,
            use_umap=bool(params["use_umap"]),
            umap_n_components=int(params["umap_n_components"] or vectors.shape[1]),
            umap_n_neighbors=int(params["umap_n_neighbors"] or 15),
            umap_min_dist=float(params["umap_min_dist"]),
            umap_metric=str(params["umap_metric"]) if params["use_umap"] else "euclidean",
            min_cluster_size=int(params["min_cluster_size"]),
            min_samples=int(params["min_samples"]),
            cluster_selection_method=str(params["cluster_selection_method"]),
            cluster_selection_epsilon=float(params["cluster_selection_epsilon"]),
            random_state=args.random_state,
        )
        metrics = compute_cluster_metrics(
            labels, probabilities, tiny_cluster_max_size=args.tiny_cluster_max_size
        )
        structural_score = compute_structural_score(metrics, args)
        semantic_metrics = compute_semantic_metrics(labels, vectors, semantic_df)
        semantic_score = compute_semantic_score(semantic_metrics, args)
        final_score = compute_final_score(structural_score, semantic_score)

        guardrails = compute_guardrail_flags(
            metrics,
            noise_ratio_threshold=args.noise_ratio_threshold,
            min_clusters_threshold=args.min_clusters_threshold,
            largest_cluster_ratio_threshold=args.largest_cluster_ratio_threshold,
            tiny_cluster_ratio_threshold=args.tiny_cluster_ratio_threshold,
        )
        good = is_good_run(
            metrics=metrics,
            noise_ratio_threshold=args.noise_ratio_threshold,
            min_clusters_threshold=args.min_clusters_threshold,
            largest_cluster_ratio_threshold=args.largest_cluster_ratio_threshold,
        )

        trial_id = f"trial_{trial.number:04d}"
        cluster_version = f"{timestamp}_{trial_id}"
        run_dir = trial_root / cluster_version
        run_dir.mkdir(parents=True, exist_ok=True)
        labels_df = pd.DataFrame(
            {
                "card_id": card_ids,
                "cluster_id": labels.astype("int64"),
                "cluster_probability": probabilities.astype(float),
            }
        )
        labels_path = run_dir / "labels.parquet"
        metrics_path = run_dir / "metrics.json"
        labels_df.to_parquet(labels_path, index=False)

        payload = {
            "cluster_version": cluster_version,
            "trial_number": int(trial.number),
            "n_umap_components": int(n_umap_components),
            "is_good_run": bool(good),
            **params,
            **metrics,
            **structural_score,
            **semantic_metrics,
            **semantic_score,
            **final_score,
            **guardrails,
        }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        log_run_to_mlflow(
            tracking_uri=args.mlflow_tracking_uri,
            experiment=args.mlflow_experiment,
            run_name=cluster_version,
            params={
                "cluster_version": cluster_version,
                "trial_number": int(trial.number),
                "distance_metric": "euclidean",
                "l2_normalized": True,
                "n_umap_components": int(n_umap_components),
                **params,
            },
            metrics={**payload, "is_good_run": int(good)},
            artifact_paths=[labels_path, metrics_path],
        )
        trial.set_user_attr("metrics_payload", payload)
        logger.info(
            "Trial=%s final=%.4f structural=%.4f clusters=%s noise=%.4f flags=%s",
            trial.number,
            float(final_score["final_score"]),
            float(structural_score["structural_score"]),
            int(metrics["num_clusters"]),
            float(metrics["noise_ratio"]),
            int(guardrails["flag_any_guardrail"]),
        )
        return float(final_score["final_score"])

    study.optimize(objective, n_trials=args.n_trials)

    completed_trials = [trial for trial in study.trials if trial.value is not None]
    summary_rows = [dict(trial.user_attrs["metrics_payload"]) for trial in completed_trials]
    summary_df = pd.DataFrame(summary_rows)
    final_ranked_df = summary_df.sort_values("final_score", ascending=False).reset_index(drop=True)
    structural_ranked_df = summary_df.sort_values("structural_score", ascending=False).reset_index(
        drop=True
    )
    top_final_df = final_ranked_df.head(args.top_n).copy()
    top_structural_df = structural_ranked_df.head(args.top_n).copy()

    summary_parquet_path = output_dir / "sweep_summary.parquet"
    summary_csv_path = output_dir / "sweep_summary.csv"
    top_final_path = output_dir / "top_runs_final_score.csv"
    top_structural_path = output_dir / "top_runs_structural_score.csv"
    best_path = output_dir / "best_run_by_final_score.json"

    summary_df.to_parquet(summary_parquet_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    top_final_df.to_csv(top_final_path, index=False)
    top_structural_df.to_csv(top_structural_path, index=False)
    best_payload = dict(study.best_trial.user_attrs["metrics_payload"])
    best_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    logger.info(
        "Best trial by final_score: trial=%s score=%.4f",
        study.best_trial.number,
        study.best_value,
    )
    logger.info(
        "Top final_score runs:\n%s",
        top_final_df[
            [
                "trial_number",
                "final_score",
                "structural_score",
                "num_clusters",
                "noise_ratio",
                "largest_cluster_ratio",
                "tiny_cluster_ratio",
                "flag_any_guardrail",
            ]
        ].to_string(index=False),
    )
    logger.info(
        "Top structural_score runs:\n%s",
        top_structural_df[
            [
                "trial_number",
                "structural_score",
                "final_score",
                "num_clusters",
                "noise_ratio",
                "largest_cluster_ratio",
                "tiny_cluster_ratio",
                "flag_any_guardrail",
            ]
        ].to_string(index=False),
    )
    logger.info(
        "Promote runs only after manual inspection of top candidates, not purely the #1 score."
    )
    logger.info(
        "Saved outputs: %s %s %s %s %s",
        summary_parquet_path,
        summary_csv_path,
        top_final_path,
        top_structural_path,
        best_path,
    )


if __name__ == "__main__":
    main()
