from __future__ import annotations

import argparse
import itertools
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

MIN_CLUSTER_SIZE_GRID = [5, 10, 15, 25, 50]
MIN_SAMPLES_GRID = [1, 3, 5, 10]
CLUSTER_SELECTION_METHOD_GRID = ["eom", "leaf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HDBSCAN parameter sweep for clustering quality."
    )
    parser.add_argument("--embeddings-file", default="data/features/card_embeddings.parquet")
    parser.add_argument("--output-dir", default="data/features/clusters/hdbscan_sweep")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument(
        "--mlflow-experiment",
        default="yugioh-deck-generator/clustering-embeddings/hdbscan-sweep",
    )
    parser.add_argument("--noise-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--min-clusters-threshold", type=int, default=10)
    parser.add_argument("--largest-cluster-ratio-threshold", type=float, default=0.4)
    return parser.parse_args()


def _normalize_l2(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / safe_norms


def load_embeddings(embeddings_file: str) -> tuple[pd.Series, np.ndarray]:
    df = pd.read_parquet(embeddings_file)
    required = {"card_id", "embedding_vector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    card_ids = df["card_id"].astype("int64")
    vectors = np.asarray(df["embedding_vector"].tolist(), dtype=float)
    if vectors.ndim != 2:
        raise ValueError("Expected 2D vectors")
    return card_ids, _normalize_l2(vectors)


def fit_hdbscan(
    vectors: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
) -> np.ndarray:
    import hdbscan

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="best",
        cluster_selection_method=cluster_selection_method,
    )
    return model.fit_predict(vectors)


def compute_cluster_metrics(labels: np.ndarray) -> dict[str, float | int]:
    total = len(labels)
    if total == 0:
        raise ValueError("labels must not be empty")

    noise_mask = labels == -1
    num_noise_cards = int(np.sum(noise_mask))
    noise_ratio = float(num_noise_cards / total)

    non_noise = labels[~noise_mask]
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
    else:
        largest_cluster_size = 0.0
        top_10_sizes = []
        cluster_size_min = 0.0
        cluster_size_max = 0.0
        cluster_size_mean = 0.0
        cluster_size_median = 0.0

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
        "top_10_cluster_sizes_json": json.dumps(top_10_sizes),
    }


def is_good_run(
    metrics: dict[str, float | int],
    noise_ratio_threshold: float,
    min_clusters_threshold: int,
    largest_cluster_ratio_threshold: float,
) -> bool:
    return (
        float(metrics["noise_ratio"]) < noise_ratio_threshold
        and int(metrics["num_clusters"]) >= min_clusters_threshold
        and float(metrics["largest_cluster_ratio"]) < largest_cluster_ratio_threshold
    )


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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    card_ids, vectors = load_embeddings(args.embeddings_file)
    logger.info("Loaded embeddings: cards=%s dim=%s", len(card_ids), vectors.shape[1])
    logger.info("Running HDBSCAN sweep on %s parameter combinations", 5 * 4 * 2)

    summary_rows: list[dict[str, Any]] = []
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    for min_cluster_size, min_samples, cluster_selection_method in itertools.product(
        MIN_CLUSTER_SIZE_GRID, MIN_SAMPLES_GRID, CLUSTER_SELECTION_METHOD_GRID
    ):
        cluster_version = (
            f"{timestamp}_mcs{min_cluster_size}_ms{min_samples}_csm{cluster_selection_method}"
        )
        logger.info("Sweep run start: %s", cluster_version)
        labels = fit_hdbscan(
            vectors=vectors,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
        )
        metrics = compute_cluster_metrics(labels)

        logger.info("noise_ratio=%.4f", float(metrics["noise_ratio"]))
        logger.info("largest_cluster_ratio=%.4f", float(metrics["largest_cluster_ratio"]))
        logger.info("top_10_cluster_sizes=%s", metrics["top_10_cluster_sizes_json"])

        good = is_good_run(
            metrics=metrics,
            noise_ratio_threshold=args.noise_ratio_threshold,
            min_clusters_threshold=args.min_clusters_threshold,
            largest_cluster_ratio_threshold=args.largest_cluster_ratio_threshold,
        )

        run_dir = output_dir / cluster_version
        run_dir.mkdir(parents=True, exist_ok=True)
        labels_df = pd.DataFrame({"card_id": card_ids, "cluster_id": labels.astype("int64")})
        labels_path = run_dir / "labels.parquet"
        metrics_path = run_dir / "metrics.json"
        labels_df.to_parquet(labels_path, index=False)
        metrics_payload = {
            "cluster_version": cluster_version,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_method": cluster_selection_method,
            **metrics,
            "is_good_run": good,
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        log_run_to_mlflow(
            tracking_uri=args.mlflow_tracking_uri,
            experiment=args.mlflow_experiment,
            run_name=cluster_version,
            params={
                "cluster_version": cluster_version,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_method": cluster_selection_method,
                "distance_metric": "euclidean",
                "l2_normalized": True,
            },
            metrics={**metrics, "is_good_run": int(good)},
            artifact_paths=[labels_path, metrics_path],
        )

        summary_rows.append(metrics_payload)
        logger.info("Sweep run complete: %s is_good_run=%s", cluster_version, good)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "sweep_summary.parquet"
    summary_csv_path = output_dir / "sweep_summary.csv"
    summary_df.to_parquet(summary_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(
        "Sweep finished: total_runs=%s good_runs=%s",
        len(summary_df),
        int(summary_df["is_good_run"].sum()),
    )
    logger.info("Summary saved: %s and %s", summary_path, summary_csv_path)


if __name__ == "__main__":
    main()
