from __future__ import annotations

import argparse
import os

import mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create project and sub-experiments in MLflow.")
    parser.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    parser.add_argument("--project-experiment", default="yugioh-deck-generator")
    parser.add_argument(
        "--clustering-experiment",
        default="yugioh-deck-generator/clustering-embeddings",
    )
    return parser.parse_args()


def ensure_experiment(name: str) -> str:
    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    return mlflow.create_experiment(name)


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)

    project_id = ensure_experiment(args.project_experiment)
    clustering_id = ensure_experiment(args.clustering_experiment)

    mlflow.set_experiment(args.clustering_experiment)
    with mlflow.start_run(run_name="bootstrap-sub-experiment"):
        mlflow.set_tag("project_experiment", args.project_experiment)
        mlflow.set_tag("project_experiment_id", project_id)
        mlflow.set_tag("sub_experiment", args.clustering_experiment)
        mlflow.set_tag("sub_experiment_id", clustering_id)

    print(f"project_experiment={args.project_experiment} id={project_id}")
    print(f"clustering_experiment={args.clustering_experiment} id={clustering_id}")


if __name__ == "__main__":
    main()
