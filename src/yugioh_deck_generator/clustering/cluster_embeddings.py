from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

FUNCTIONAL_ROLES = [
    "starter",
    "extender",
    "searcher",
    "draw",
    "removal",
    "negation",
    "board_breaker",
    "floodgate",
    "graveyard_setup",
    "graveyard_hate",
    "recursion",
    "protection",
    "burn",
    "battle_trick",
    "archetype_core",
    "misc",
]
REVIEW_STATUS_VALUES = {"unreviewed", "accepted", "rejected", "needs_split", "misc"}
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 2.5 embedding cluster artifacts.")
    parser.add_argument("--embeddings-file", default="data/features/card_embeddings.parquet")
    parser.add_argument("--cards-file", default="data/processed/cards.parquet")
    parser.add_argument("--output-dir", default="data/features/clusters")
    parser.add_argument(
        "--cluster-version", default=datetime.now(UTC).strftime("cluster_%Y%m%dT%H%M%SZ")
    )
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--distance-metric", default="euclidean")
    parser.add_argument("--cluster-selection-method", default="eom")
    parser.add_argument("--projection-method", choices=["umap", "tsne"], default="umap")
    parser.add_argument("--include-tsne", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-llm-labeling", action="store_true")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "mistral"))
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument(
        "--mlflow-experiment",
        default=os.getenv(
            "MLFLOW_CLUSTER_EXPERIMENT",
            "yugioh-deck-generator/clustering-embeddings",
        ),
    )
    parser.add_argument("--publish-postgres", action="store_true")
    parser.add_argument("--pg-host", default=os.getenv("POSTGRES_HOST", "localhost"))
    parser.add_argument("--pg-port", type=int, default=int(os.getenv("POSTGRES_PORT", "5432")))
    parser.add_argument("--pg-dbname", default=os.getenv("POSTGRES_DB", "yugioh"))
    parser.add_argument("--pg-user", default=os.getenv("POSTGRES_USER", "yugioh"))
    parser.add_argument("--pg-password", default=os.getenv("POSTGRES_PASSWORD", "yugioh"))
    parser.add_argument("--pg-schema", default=os.getenv("POSTGRES_SCHEMA", "public"))
    return parser.parse_args()


def _extract_embeddings(embeddings_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    columns = {"card_id", "embedding_model", "embedding_version", "embedding_vector"}
    missing = columns - set(embeddings_df.columns)
    if missing:
        raise ValueError(f"Missing embedding columns: {sorted(missing)}")

    records = embeddings_df.copy()
    records["card_id"] = records["card_id"].astype("int64")
    vectors = np.asarray(records["embedding_vector"].tolist(), dtype=float)
    if vectors.ndim != 2:
        raise ValueError("Expected 2D embedding vectors")
    return vectors, records


def cluster_embeddings(
    vectors: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import hdbscan

    metric_lower = metric.lower()
    vectors_for_fit = vectors
    metric_for_fit = metric
    algorithm = "best"

    if metric_lower == "euclidean":
        # Default path: L2-normalize vectors before Euclidean clustering.
        # This keeps comparisons directional and avoids high-memory cosine paths.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        vectors_for_fit = vectors / safe_norms
    elif metric_lower == "cosine":
        # Cosine request is mapped to normalized-euclidean fitting for stability/memory.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        vectors_for_fit = vectors / safe_norms
        metric_for_fit = "euclidean"

    logger.info(
        "HDBSCAN config: requested_metric=%s fit_metric=%s algorithm=%s",
        metric,
        metric_for_fit,
        algorithm,
    )
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric_for_fit,
        cluster_selection_method=cluster_selection_method,
        algorithm=algorithm,
    )
    labels = model.fit_predict(vectors_for_fit)
    probabilities = np.asarray(model.probabilities_, dtype=float)
    outlier_scores = np.asarray(model.outlier_scores_, dtype=float)
    return labels, probabilities, outlier_scores


def _heuristic_role_from_text(text: str) -> tuple[str, str]:
    low = text.lower()
    if "add from deck" in low:
        return "searcher", "heuristic:add_from_deck"
    if "draw" in low:
        return "draw", "heuristic:draw"
    if any(token in low for token in ["destroy", "banish", "target"]):
        return "removal", "heuristic:removal"
    if "negate" in low:
        return "negation", "heuristic:negate"
    if any(token in low for token in ["gy", "graveyard", "send"]):
        return "graveyard_setup", "heuristic:graveyard"
    if any(token in low for token in ["cannot activate", "cannot summon"]):
        return "floodgate", "heuristic:floodgate"
    return "misc", "heuristic:misc"


def _top_terms(texts: list[str], limit: int = 20) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = [tok.lower() for tok in WORD_PATTERN.findall(text) if len(tok) > 2]
        counter.update(tokens)
    return [token for token, _ in counter.most_common(limit)]


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _call_ollama_label(
    summary: dict[str, Any],
    ollama_url: str,
    model: str,
) -> dict[str, Any] | None:
    prompt = {
        "instruction": (
            "Return JSON with keys functional_role,label_hint,label_confidence,reasoning. "
            f"functional_role must be one of: {', '.join(FUNCTIONAL_ROLES)}."
        ),
        "cluster_summary": summary,
    }
    payload = _safe_json(
        {
            "model": model,
            "prompt": _safe_json(prompt),
            "stream": False,
            "format": "json",
        }
    ).encode("utf-8")
    endpoint = f"{ollama_url.rstrip('/')}/api/generate"
    req = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    content = body.get("response")
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    role = str(parsed.get("functional_role", "misc")).strip().lower()
    if role not in FUNCTIONAL_ROLES:
        role = "misc"
    return {
        "functional_role": role,
        "label_hint": str(parsed.get("label_hint", role)).strip()[:120],
        "label_confidence": float(parsed.get("label_confidence", 0.5)),
        "reasoning": str(parsed.get("reasoning", "")).strip()[:400],
    }


def build_cluster_metadata(
    cards_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    vectors: np.ndarray,
    cluster_version: str,
    generated_at: str,
    use_llm_labeling: bool,
    ollama_url: str,
    ollama_model: str,
) -> pd.DataFrame:
    logger.info(
        "Building cluster metadata: use_llm_labeling=%s ollama_model=%s",
        use_llm_labeling,
        ollama_model,
    )
    merged = cluster_df.merge(
        cards_df[["id", "name", "type", "archetype", "desc"]],
        left_on="card_id",
        right_on="id",
        how="left",
    )
    total_clusters = int(merged["cluster_id"].nunique())
    metadata_rows: list[dict[str, Any]] = []
    for idx, (cluster_id, group) in enumerate(merged.groupby("cluster_id", sort=True), start=1):
        cluster_id_int = int(cluster_id)
        card_ids = group["card_id"].astype(int).tolist()
        if not card_ids:
            continue
        if idx == 1 or idx % 10 == 0 or idx == total_clusters:
            logger.info(
                "Metadata progress: cluster=%s (%s/%s)",
                cluster_id_int,
                idx,
                total_clusters,
            )
        vec_idx = group.index.to_numpy(dtype=int)
        cluster_vectors = vectors[vec_idx]
        centroid = np.mean(cluster_vectors, axis=0)
        dists = np.linalg.norm(cluster_vectors - centroid, axis=1)
        medoid_idx = int(np.argmin(dists))
        medoid_card_id = int(group.iloc[medoid_idx]["card_id"])
        top_archetypes = (
            group["archetype"].fillna("unknown").astype(str).value_counts().head(5).to_dict()
        )
        type_distribution = group["type"].fillna("unknown").astype(str).value_counts().to_dict()
        texts = group["desc"].fillna("").astype(str).tolist()
        top_terms = _top_terms(texts, limit=15)
        avg_similarity = float(np.mean(np.clip(1.0 - dists, -1.0, 1.0)))
        heuristic_role, heuristic_source = _heuristic_role_from_text(" ".join(texts))
        label_payload = None
        if use_llm_labeling and cluster_id_int != -1:
            label_payload = _call_ollama_label(
                {
                    "cluster_id": cluster_id_int,
                    "top_terms": top_terms,
                    "top_archetypes": top_archetypes,
                    "type_distribution": type_distribution,
                    "cluster_size": int(len(group)),
                    "avg_similarity": avg_similarity,
                    "representative_cards": group["name"].fillna("").astype(str).head(5).tolist(),
                },
                ollama_url=ollama_url,
                model=ollama_model,
            )
        if label_payload is None:
            label_payload = {
                "functional_role": heuristic_role,
                "label_hint": heuristic_role.replace("_", " "),
                "label_confidence": 0.55,
                "reasoning": heuristic_source,
            }
            label_source = heuristic_source
        else:
            label_source = "llm"

        metadata_rows.append(
            {
                "cluster_id": cluster_id_int,
                "cluster_size": int(len(group)),
                "medoid_card_id": medoid_card_id,
                "functional_role": label_payload["functional_role"],
                "label_hint": label_payload["label_hint"],
                "label_confidence": float(label_payload["label_confidence"]),
                "label_source": label_source,
                "review_status": "unreviewed",
                "metadata": {
                    "top_terms": top_terms,
                    "top_archetypes": top_archetypes,
                    "type_distribution": type_distribution,
                    "avg_similarity": avg_similarity,
                    "noise_cluster": cluster_id_int == -1,
                    "representative_cards": group["name"].fillna("").astype(str).head(5).tolist(),
                    "llm_reasoning": label_payload.get("reasoning", ""),
                },
                "cluster_version": cluster_version,
                "generated_at": generated_at,
            }
        )
    return pd.DataFrame(metadata_rows)


def build_projection_df(
    card_ids: pd.Series,
    vectors: np.ndarray,
    cluster_version: str,
    random_state: int,
    method: str = "umap",
) -> pd.DataFrame:
    logger.info("Building %s projection for %s vectors", method, len(vectors))
    if method == "umap":
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".cache/numba").resolve()))
        os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
        try:
            import umap

            n_components = 2
            logger.info("UMAP reducing embeddings to n_components=%s", n_components)
            projector = umap.UMAP(n_components=n_components, random_state=random_state)
            points = projector.fit_transform(vectors)
            logger.info("UMAP projection complete")
        except Exception as exc:
            logger.warning("UMAP projection failed; falling back to t-SNE for plots: %s", exc)
            from sklearn.manifold import TSNE

            method = "tsne"
            projector = TSNE(n_components=2, random_state=random_state, init="random")
            points = projector.fit_transform(vectors)
            logger.info("t-SNE fallback projection complete")
    else:
        from sklearn.manifold import TSNE

        projector = TSNE(n_components=2, random_state=random_state, init="random")
        points = projector.fit_transform(vectors)
        logger.info("t-SNE projection complete")

    projection = pd.DataFrame(
        {
            "card_id": card_ids.astype("int64"),
            "cluster_version": cluster_version,
            "projection_method": method,
            "projection_x": points[:, 0],
            "projection_y": points[:, 1],
        }
    )
    return projection


def _noise_summary(cluster_df: pd.DataFrame, cards_df: pd.DataFrame) -> dict[str, Any]:
    merged = cluster_df.merge(
        cards_df[["id", "type", "archetype"]], left_on="card_id", right_on="id", how="left"
    )
    total = max(len(merged), 1)
    noise = merged[merged["cluster_id"] == -1]
    return {
        "noise_ratio": len(noise) / total,
        "noise_type_distribution": noise["type"]
        .fillna("unknown")
        .value_counts()
        .head(10)
        .to_dict(),
        "noise_archetype_distribution": noise["archetype"]
        .fillna("unknown")
        .value_counts()
        .head(10)
        .to_dict(),
    }


def _render_scatter_plot(
    merged_plot_df: pd.DataFrame,
    output_path: Path,
    color_col: str,
    annotate_medoids: set[int] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    annotate_medoids = annotate_medoids or set()
    fig, ax = plt.subplots(figsize=(11, 8))
    color_series = merged_plot_df[color_col].astype(str)
    alpha = np.clip(merged_plot_df["cluster_probability"].fillna(0.5).astype(float), 0.1, 1.0)
    size = 20 + (alpha * 40)
    scatter = ax.scatter(
        merged_plot_df["projection_x"],
        merged_plot_df["projection_y"],
        c=pd.factorize(color_series)[0],
        s=size,
        alpha=alpha,
        cmap="tab20",
    )
    ax.set_title(f"Embedding Clusters ({color_col})")
    ax.set_xlabel("projection_x")
    ax.set_ylabel("projection_y")
    for _, row in merged_plot_df.iterrows():
        if int(row["cluster_id"]) == -1:
            ax.scatter(row["projection_x"], row["projection_y"], marker="x", c="black", s=35)
        if int(row["card_id"]) in annotate_medoids:
            ax.annotate(
                str(row.get("name", row["card_id"])),
                (row["projection_x"], row["projection_y"]),
            )
    ax.grid(alpha=0.2)
    fig.colorbar(scatter)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_artifacts(
    output_dir: Path,
    cluster_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    cards_df: pd.DataFrame,
) -> dict[str, Path]:
    logger.info("Writing clustering artifacts to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    clusters_path = output_dir / "card_embedding_clusters.parquet"
    metadata_path = output_dir / "embedding_cluster_metadata.parquet"
    projections_path = output_dir / "embedding_cluster_projections.parquet"
    summary_path = output_dir / "cluster_run_summary.json"

    cluster_df.to_parquet(clusters_path, index=False)
    metadata_df.to_parquet(metadata_path, index=False)
    projections_df.to_parquet(projections_path, index=False)
    noise_summary = _noise_summary(cluster_df, cards_df)
    summary_path.write_text(_safe_json(noise_summary), encoding="utf-8")

    merged_plot_df = cluster_df.merge(
        projections_df, on=["card_id", "cluster_version"], how="left"
    )
    merged_plot_df = merged_plot_df.merge(
        metadata_df[["cluster_id", "functional_role", "medoid_card_id"]],
        on="cluster_id",
        how="left",
    ).merge(cards_df[["id", "name"]], left_on="card_id", right_on="id", how="left")
    medoids = set(metadata_df["medoid_card_id"].dropna().astype(int).tolist())
    by_cluster_path = plots_dir / "clusters_by_cluster_id.png"
    _render_scatter_plot(merged_plot_df, by_cluster_path, "cluster_id", annotate_medoids=medoids)
    by_role_path = plots_dir / "clusters_by_functional_role.png"
    _render_scatter_plot(merged_plot_df, by_role_path, "functional_role", annotate_medoids=medoids)
    logger.info(
        "Artifacts written: clusters=%s metadata=%s projections=%s plots=%s",
        clusters_path,
        metadata_path,
        projections_path,
        plots_dir,
    )

    return {
        "clusters": clusters_path,
        "metadata": metadata_path,
        "projections": projections_path,
        "summary": summary_path,
        "plot_cluster_id": by_cluster_path,
        "plot_role": by_role_path,
    }


def _pg_create_tables(conn: psycopg.Connection[Any], schema: str) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};

    CREATE TABLE IF NOT EXISTS {schema}.card_embedding_clusters (
        card_id BIGINT NOT NULL,
        cluster_id BIGINT NOT NULL,
        cluster_probability DOUBLE PRECISION NOT NULL,
        outlier_score DOUBLE PRECISION NOT NULL,
        embedding_model TEXT NOT NULL,
        embedding_version TEXT NOT NULL,
        cluster_version TEXT NOT NULL,
        generated_at TEXT NOT NULL,
        PRIMARY KEY (card_id, cluster_version)
    );

    CREATE TABLE IF NOT EXISTS {schema}.embedding_cluster_metadata (
        cluster_id BIGINT NOT NULL,
        cluster_size BIGINT NOT NULL,
        medoid_card_id BIGINT,
        functional_role TEXT NOT NULL,
        label_hint TEXT NOT NULL,
        label_confidence DOUBLE PRECISION NOT NULL,
        label_source TEXT NOT NULL,
        review_status TEXT NOT NULL,
        metadata JSONB NOT NULL,
        cluster_version TEXT NOT NULL,
        generated_at TEXT NOT NULL,
        PRIMARY KEY (cluster_id, cluster_version)
    );

    CREATE TABLE IF NOT EXISTS {schema}.embedding_cluster_projections (
        card_id BIGINT NOT NULL,
        cluster_version TEXT NOT NULL,
        projection_method TEXT NOT NULL,
        projection_x DOUBLE PRECISION NOT NULL,
        projection_y DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (card_id, cluster_version, projection_method)
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl)


def _pg_replace_table(conn: psycopg.Connection[Any], schema: str, table: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {schema}.{table};")


def _pg_insert_cluster_df(conn: psycopg.Connection[Any], schema: str, df: pd.DataFrame) -> None:
    records = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {schema}.card_embedding_clusters (
                card_id, cluster_id, cluster_probability, outlier_score,
                embedding_model, embedding_version, cluster_version, generated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            records,
        )


def _pg_insert_metadata_df(conn: psycopg.Connection[Any], schema: str, df: pd.DataFrame) -> None:
    rows: list[tuple[Any, ...]] = []
    for row in df.itertuples(index=False):
        review_status = str(row.review_status)
        if review_status not in REVIEW_STATUS_VALUES:
            raise ValueError(f"Invalid review_status: {review_status}")
        rows.append(
            (
                int(row.cluster_id),
                int(row.cluster_size),
                int(row.medoid_card_id),
                str(row.functional_role),
                str(row.label_hint),
                float(row.label_confidence),
                str(row.label_source),
                review_status,
                _safe_json(row.metadata),
                str(row.cluster_version),
                str(row.generated_at),
            )
        )

    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {schema}.embedding_cluster_metadata (
                cluster_id, cluster_size, medoid_card_id, functional_role, label_hint,
                label_confidence, label_source, review_status, metadata,
                cluster_version, generated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            """,
            rows,
        )


def _pg_insert_projection_df(conn: psycopg.Connection[Any], schema: str, df: pd.DataFrame) -> None:
    records = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {schema}.embedding_cluster_projections (
                card_id, cluster_version, projection_method, projection_x, projection_y
            ) VALUES (%s, %s, %s, %s, %s)
            """,
            records,
        )


def publish_postgres(
    args: argparse.Namespace,
    cluster_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    projections_df: pd.DataFrame,
) -> None:
    logger.info("Publishing clustering outputs to Postgres schema=%s", args.pg_schema)
    conn = psycopg.connect(
        host=args.pg_host,
        port=args.pg_port,
        dbname=args.pg_dbname,
        user=args.pg_user,
        password=args.pg_password,
    )
    try:
        with conn.transaction():
            _pg_create_tables(conn, args.pg_schema)
            _pg_replace_table(conn, args.pg_schema, "card_embedding_clusters")
            _pg_replace_table(conn, args.pg_schema, "embedding_cluster_metadata")
            _pg_replace_table(conn, args.pg_schema, "embedding_cluster_projections")
            _pg_insert_cluster_df(conn, args.pg_schema, cluster_df)
            _pg_insert_metadata_df(conn, args.pg_schema, metadata_df)
            _pg_insert_projection_df(conn, args.pg_schema, projections_df)
    finally:
        conn.close()


def maybe_log_mlflow(
    args: argparse.Namespace,
    cluster_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    artifact_paths: dict[str, Path],
) -> None:
    if not args.mlflow_tracking_uri:
        logger.info("MLflow tracking URI not set; skipping MLflow logging.")
        return

    import mlflow

    logger.info(
        "Logging run to MLflow: tracking_uri=%s experiment=%s",
        args.mlflow_tracking_uri,
        args.mlflow_experiment,
    )
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.cluster_version):
        mlflow.log_params(
            {
                "cluster_version": args.cluster_version,
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
                "distance_metric": args.distance_metric,
                "cluster_selection_method": args.cluster_selection_method,
                "projection_method": args.projection_method,
                "include_tsne": args.include_tsne,
                "use_llm_labeling": args.use_llm_labeling,
            }
        )
        noise_ratio = float((cluster_df["cluster_id"] == -1).mean())
        mlflow.log_metrics(
            {
                "num_cards": float(len(cluster_df)),
                "num_clusters_non_noise": float(len(set(cluster_df["cluster_id"].tolist()) - {-1})),
                "noise_ratio": noise_ratio,
                "avg_cluster_probability": float(cluster_df["cluster_probability"].mean()),
                "avg_label_confidence": float(metadata_df["label_confidence"].mean()),
            }
        )
        for path in artifact_paths.values():
            mlflow.log_artifact(str(path))
        mlflow.log_text(
            cluster_df.head(20).to_json(orient="records"),
            artifact_file="cluster_preview.json",
        )
        mlflow.log_text(
            projections_df.head(20).to_json(orient="records"),
            artifact_file="projection_preview.json",
        )


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    logger.info(
        (
            "Starting clustering run: version=%s embeddings_file=%s cards_file=%s "
            "projection_method=%s include_tsne=%s"
        ),
        args.cluster_version,
        args.embeddings_file,
        args.cards_file,
        args.projection_method,
        args.include_tsne,
    )
    generated_at = datetime.now(UTC).isoformat()
    load_start = time.perf_counter()
    embeddings_df = pd.read_parquet(args.embeddings_file)
    cards_df = pd.read_parquet(args.cards_file)
    vectors, embedding_records = _extract_embeddings(embeddings_df)
    logger.info(
        "Loaded inputs in %.2fs: embeddings=%s cards=%s embedding_dim=%s",
        time.perf_counter() - load_start,
        len(embedding_records),
        len(cards_df),
        vectors.shape[1] if vectors.ndim == 2 else "unknown",
    )

    cluster_start = time.perf_counter()
    labels, probabilities, outlier_scores = cluster_embeddings(
        vectors=vectors,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.distance_metric,
        cluster_selection_method=args.cluster_selection_method,
    )
    logger.info("HDBSCAN clustering complete in %.2fs", time.perf_counter() - cluster_start)
    cluster_df = pd.DataFrame(
        {
            "card_id": embedding_records["card_id"].astype("int64"),
            "cluster_id": labels.astype("int64"),
            "cluster_probability": probabilities.astype(float),
            "outlier_score": outlier_scores.astype(float),
            "embedding_model": embedding_records["embedding_model"].astype(str),
            "embedding_version": embedding_records["embedding_version"].astype(str),
            "cluster_version": args.cluster_version,
            "generated_at": generated_at,
        }
    )
    noise_ratio = float((cluster_df["cluster_id"] == -1).mean())
    non_noise_clusters = len(set(cluster_df["cluster_id"].tolist()) - {-1})
    logger.info(
        "Cluster summary: cards=%s non_noise_clusters=%s noise_ratio=%.4f",
        len(cluster_df),
        non_noise_clusters,
        noise_ratio,
    )

    metadata_start = time.perf_counter()
    metadata_df = build_cluster_metadata(
        cards_df=cards_df,
        cluster_df=cluster_df,
        vectors=vectors,
        cluster_version=args.cluster_version,
        generated_at=generated_at,
        use_llm_labeling=args.use_llm_labeling,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
    )
    logger.info("Metadata build complete in %.2fs", time.perf_counter() - metadata_start)

    projection_start = time.perf_counter()
    projections_df = build_projection_df(
        card_ids=cluster_df["card_id"],
        vectors=vectors,
        cluster_version=args.cluster_version,
        random_state=args.random_state,
        method=args.projection_method,
    )
    if args.include_tsne:
        tsne_projection = build_projection_df(
            card_ids=cluster_df["card_id"],
            vectors=vectors,
            cluster_version=args.cluster_version,
            random_state=args.random_state,
            method="tsne",
        )
        projections_df = pd.concat([projections_df, tsne_projection], ignore_index=True)
    logger.info("Projection build complete in %.2fs", time.perf_counter() - projection_start)

    artifact_start = time.perf_counter()
    artifact_paths = write_artifacts(
        output_dir=Path(args.output_dir),
        cluster_df=cluster_df,
        metadata_df=metadata_df,
        projections_df=projections_df,
        cards_df=cards_df,
    )
    logger.info("Artifact write complete in %.2fs", time.perf_counter() - artifact_start)
    mlflow_start = time.perf_counter()
    maybe_log_mlflow(args, cluster_df, metadata_df, projections_df, artifact_paths)
    logger.info("MLflow logging stage complete in %.2fs", time.perf_counter() - mlflow_start)

    if args.publish_postgres:
        pg_start = time.perf_counter()
        publish_postgres(args, cluster_df, metadata_df, projections_df)
        logger.info("Postgres publish complete in %.2fs", time.perf_counter() - pg_start)
    logger.info(
        (
            "Phase 2.5 clustering complete in %.2fs: assignments=%s metadata=%s "
            "projections=%s output_dir=%s"
        ),
        time.perf_counter() - start_time,
        len(cluster_df),
        len(metadata_df),
        len(projections_df),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
