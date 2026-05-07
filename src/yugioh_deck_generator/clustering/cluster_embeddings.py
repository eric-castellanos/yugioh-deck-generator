from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import random
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from openai import OpenAI

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


def _load_dotenv(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 2.5 embedding cluster artifacts.")
    parser.add_argument("--embeddings-file", default="data/features/card_embeddings.parquet")
    parser.add_argument("--cards-file", default="data/processed/cards.parquet")
    parser.add_argument("--output-dir", default="data/features/clusters")
    parser.add_argument("--cluster-version", default=None)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--distance-metric", default="euclidean")
    parser.add_argument("--cluster-selection-method", default="eom")
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0)

    parser.add_argument(
        "--use-umap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable UMAP before HDBSCAN (default: true). Use --no-use-umap to disable.",
    )
    parser.add_argument("--umap-n-components", type=int, default=10)
    parser.add_argument("--umap-n-neighbors", type=int, default=10)
    parser.add_argument("--umap-min-dist", type=float, default=0.01)
    parser.add_argument("--umap-metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--projection-method", choices=["umap", "tsne"], default="umap")
    parser.add_argument("--include-tsne", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
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
    parser.add_argument("--use-llm-labeling", action="store_true")
    parser.add_argument("--llm-max-retries", type=int, default=6)
    parser.add_argument("--llm-base-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--llm-min-interval-seconds", type=float, default=1.25)
    parser.add_argument(
        "--llm-max-clusters",
        type=int,
        default=0,
        help="Maximum non-noise clusters to label with LLM (0 means no cap).",
    )
    parser.add_argument(
        "--openrouter-url",
        default=os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1"),
    )
    parser.add_argument(
        "--openrouter-model",
        default=os.getenv("OPENROUTER_MODEL", "qwen/qwen3-next-80b-a3b-instruct:free"),
    )
    parser.add_argument("--openrouter-api-key", default=os.getenv("OPENROUTER_API_KEY"))
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


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


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


def _normalize_l2(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / safe_norms


def _build_cluster_version(args: argparse.Namespace, embedding_records: pd.DataFrame) -> str:
    if args.cluster_version:
        return args.cluster_version
    embedding_model = str(embedding_records["embedding_model"].iloc[0])
    embedding_version = str(embedding_records["embedding_version"].iloc[0])
    payload = {
        "embedding_model": embedding_model,
        "embedding_version": embedding_version,
        "distance_metric": args.distance_metric,
        "use_umap": args.use_umap,
        "umap_n_components": args.umap_n_components,
        "umap_n_neighbors": args.umap_n_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "umap_metric": args.umap_metric,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "cluster_selection_method": args.cluster_selection_method,
        "cluster_selection_epsilon": args.cluster_selection_epsilon,
    }
    run_hash = hashlib.sha1(_safe_json(payload).encode("utf-8")).hexdigest()[:10]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"cluster_{embedding_model}_{embedding_version}_{run_hash}_{stamp}".replace(" ", "_")


def cluster_embeddings(
    vectors: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    use_umap: bool,
    umap_n_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    vectors_for_fit = _normalize_l2(vectors)
    n_umap_components = int(vectors_for_fit.shape[1])

    if use_umap:
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".cache/numba").resolve()))
        os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
        import umap

        logger.info(
            "Running UMAP: n_components=%s n_neighbors=%s min_dist=%s metric=%s",
            umap_n_components,
            umap_n_neighbors,
            umap_min_dist,
            umap_metric,
        )
        projector = umap.UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_state,
        )
        vectors_for_fit = np.asarray(projector.fit_transform(vectors_for_fit), dtype=float)
        n_umap_components = int(vectors_for_fit.shape[1])
        # Keep UMAP output scale unchanged to match sweep_hdbscan clustering behavior.

    import hdbscan

    metric_for_fit = metric.lower()
    if metric_for_fit == "cosine":
        metric_for_fit = "euclidean"

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric_for_fit,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        algorithm="best",
    )
    logger.info(
        "HDBSCAN config: metric=%s min_cluster_size=%s min_samples=%s selection=%s epsilon=%s",
        metric_for_fit,
        min_cluster_size,
        min_samples,
        cluster_selection_method,
        cluster_selection_epsilon,
    )
    labels = model.fit_predict(vectors_for_fit)
    probabilities = np.asarray(model.probabilities_, dtype=float)
    outlier_scores = np.asarray(model.outlier_scores_, dtype=float)
    return labels, probabilities, outlier_scores, n_umap_components, vectors_for_fit


def enrich_noise_with_nearest_cluster(
    cluster_df: pd.DataFrame,
    cluster_vectors: np.ndarray,
) -> pd.DataFrame:
    # HDBSCAN is noise-aware by design; points labeled -1 are intentionally left unclustered.
    # nearest_cluster_* fields are auxiliary hints for downstream retrieval, not reassignment.
    enriched = cluster_df.copy()
    enriched["nearest_cluster_id"] = pd.Series([pd.NA] * len(enriched), dtype="Int64")
    enriched["nearest_cluster_distance"] = np.nan
    enriched["nearest_cluster_confidence"] = np.nan
    enriched["nearest_cluster_label_hint"] = ""
    enriched["nearest_cluster_functional_role"] = ""

    non_noise = enriched[enriched["cluster_id"] != -1]
    noise = enriched[enriched["cluster_id"] == -1]
    if len(non_noise) == 0 or len(noise) == 0:
        return enriched

    centroids: dict[int, np.ndarray] = {}
    for cid in sorted(non_noise["cluster_id"].astype(int).unique().tolist()):
        idx = enriched.index[enriched["cluster_id"] == cid].to_numpy(dtype=int)
        if len(idx) == 0:
            continue
        centroids[cid] = np.mean(cluster_vectors[idx], axis=0)
    if not centroids:
        return enriched

    centroid_ids = np.asarray(sorted(centroids.keys()), dtype=int)
    centroid_mat = np.asarray([centroids[c] for c in centroid_ids], dtype=float)
    centroid_norms = np.linalg.norm(centroid_mat, axis=1, keepdims=True)
    centroid_mat = centroid_mat / np.where(centroid_norms == 0.0, 1.0, centroid_norms)

    noise_idx = noise.index.to_numpy(dtype=int)
    noise_mat = np.asarray(cluster_vectors[noise_idx], dtype=float)
    noise_norms = np.linalg.norm(noise_mat, axis=1, keepdims=True)
    noise_mat = noise_mat / np.where(noise_norms == 0.0, 1.0, noise_norms)

    dists = np.linalg.norm(noise_mat[:, None, :] - centroid_mat[None, :, :], axis=2)
    best_pos = np.argmin(dists, axis=1)
    best_ids = centroid_ids[best_pos]
    best_dists = dists[np.arange(len(noise_idx)), best_pos]
    # Inverse-distance confidence, normalized to [0,1] with 0 distance -> 1.0
    confidences = 1.0 / (1.0 + best_dists)

    enriched.loc[noise_idx, "nearest_cluster_id"] = pd.Series(
        best_ids, index=noise_idx, dtype="Int64"
    )
    enriched.loc[noise_idx, "nearest_cluster_distance"] = best_dists.astype(float)
    enriched.loc[noise_idx, "nearest_cluster_confidence"] = confidences.astype(float)
    return enriched


def _compute_noise_summary(
    cluster_df: pd.DataFrame,
    cards_df: pd.DataFrame,
) -> dict[str, Any]:
    merged = cluster_df.merge(
        cards_df[["id", "type", "archetype"]],
        left_on="card_id",
        right_on="id",
        how="left",
    )
    total = max(1, len(merged))
    noise = merged[merged["cluster_id"] == -1]
    nearest_dist = (
        noise["nearest_cluster_id"]
        .dropna()
        .astype("Int64")
        .astype(str)
        .value_counts()
        .head(15)
        .to_dict()
    )
    return {
        "num_noise_cards": int(len(noise)),
        "noise_ratio": float(len(noise) / total),
        "common_noise_card_types": (
            noise["type"].fillna("unknown").astype(str).value_counts().head(10).to_dict()
        ),
        "common_noise_archetypes": noise["archetype"]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .head(10)
        .to_dict(),
        "nearest_cluster_distribution": nearest_dist,
    }


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


def _call_openrouter_label(
    summary: dict[str, Any],
    openrouter_url: str,
    model: str,
    api_key: str | None,
    max_retries: int,
    base_backoff_seconds: float,
) -> tuple[dict[str, Any] | None, str]:
    if not api_key:
        return None, "missing_openrouter_api_key"
    prompt = {
        "instruction": (
            "Return JSON with keys functional_role,label_hint,label_confidence,reasoning. "
            f"functional_role must be one of: {', '.join(FUNCTIONAL_ROLES)}."
        ),
        "cluster_summary": summary,
    }
    client = OpenAI(api_key=api_key, base_url=openrouter_url, max_retries=0, timeout=30.0)
    response = None
    attempts = max(1, int(max_retries) + 1)
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You label Yu-Gi-Oh card clusters. Respond with JSON only.",
                    },
                    {"role": "user", "content": _safe_json(prompt)},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                extra_headers={
                    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
                    "X-OpenRouter-Title": os.getenv("OPENROUTER_TITLE", "yugioh-deck-generator"),
                },
            )
            break
        except Exception as exc:  # noqa: BLE001
            status_code = getattr(exc, "status_code", None)
            if status_code == 429 and attempt < attempts:
                delay = (base_backoff_seconds * (2 ** (attempt - 1))) + random.uniform(0.0, 0.5)
                logger.warning(
                    "OpenRouter rate limited (429). retry=%s/%s sleeping=%.2fs",
                    attempt,
                    attempts - 1,
                    delay,
                )
                time.sleep(delay)
                continue
            if status_code == 429:
                return None, "rate_limited_429"
            return None, f"sdk_error:{type(exc).__name__}"
    if response is None:
        return None, "no_response"

    choices = getattr(response, "choices", None)
    if choices is None or len(choices) == 0:
        return None, "missing_choices"
    message = getattr(choices[0], "message", None)
    if message is None:
        return None, "missing_message"
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return None, "missing_response_text"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None, "response_text_not_json"
    if isinstance(parsed, list):
        if len(parsed) == 0:
            return None, "response_json_empty_list"
        first = parsed[0]
        if not isinstance(first, dict):
            return None, "response_json_list_non_object"
        parsed = first
    elif not isinstance(parsed, dict):
        return None, "response_json_not_object"
    if "result" in parsed and isinstance(parsed["result"], dict):
        parsed = parsed["result"]
    role = str(parsed.get("functional_role", "misc")).strip().lower()
    if role not in FUNCTIONAL_ROLES:
        role = "misc"
    try:
        payload = {
            "functional_role": role,
            "label_hint": str(parsed.get("label_hint", role)).strip()[:120],
            "label_confidence": float(parsed.get("label_confidence", 0.5)),
            "reasoning": str(parsed.get("reasoning", "")).strip()[:400],
        }
    except (TypeError, ValueError):
        return None, "invalid_label_confidence"
    return payload, "ok"


def _compute_cluster_quality_metrics(group: pd.DataFrame) -> dict[str, float]:
    if len(group) == 0:
        return {"cluster_probability_mean": 0.0, "outlier_score_mean": 0.0}
    return {
        "cluster_probability_mean": float(group["cluster_probability"].astype(float).mean()),
        "outlier_score_mean": float(group["outlier_score"].astype(float).mean()),
    }


def build_cluster_metadata(
    cards_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    vectors: np.ndarray,
    cluster_version: str,
    generated_at: str,
    use_llm_labeling: bool,
    openrouter_url: str,
    openrouter_model: str,
    openrouter_api_key: str | None,
    llm_max_retries: int,
    llm_base_backoff_seconds: float,
    llm_min_interval_seconds: float,
    llm_max_clusters: int,
    noise_summary: dict[str, Any] | None = None,
) -> pd.DataFrame:
    merged = cluster_df.merge(
        cards_df[["id", "name", "type", "archetype", "desc"]],
        left_on="card_id",
        right_on="id",
        how="left",
    )
    metadata_rows: list[dict[str, Any]] = []
    llm_attempted = 0
    llm_succeeded = 0
    llm_failed = 0
    heuristic_fallback = 0
    llm_failure_reasons: Counter[str] = Counter()
    llm_failure_samples: list[str] = []
    llm_labeled_clusters = 0
    last_llm_call_ts = 0.0
    for cluster_id, group in merged.groupby("cluster_id", sort=True):
        cluster_id_int = int(cluster_id)
        if len(group) == 0:
            continue
        vec_idx = group.index.to_numpy(dtype=int)
        cluster_vectors = vectors[vec_idx]
        centroid = np.mean(cluster_vectors, axis=0)
        dists = np.linalg.norm(cluster_vectors - centroid, axis=1)
        medoid_idx = int(np.argmin(dists))
        medoid_card_id = int(group.iloc[medoid_idx]["card_id"])

        archetype_dist = (
            group["archetype"].fillna("unknown").astype(str).value_counts().head(15).to_dict()
        )
        type_dist = group["type"].fillna("unknown").astype(str).value_counts().head(15).to_dict()
        texts = group["desc"].fillna("").astype(str).tolist()
        representative_cards = (
            group[["card_id", "name"]]
            .head(10)
            .rename(columns={"name": "card_name"})
            .to_dict(orient="records")
        )
        top_terms = _top_terms(texts, limit=20)
        avg_similarity = float(np.mean(np.clip(1.0 - dists, -1.0, 1.0)))
        heuristic_role, heuristic_source = _heuristic_role_from_text(" ".join(texts))
        metadata_json = {
            "representative_cards": representative_cards,
            "top_terms": top_terms,
            "type_distribution": type_dist,
            "archetype_distribution": archetype_dist,
            "quality_metrics": {
                **_compute_cluster_quality_metrics(group),
                "avg_similarity": avg_similarity,
            },
            "llm_reasoning": "",
        }
        if cluster_id_int == -1:
            metadata_json["noise_summary"] = {
                "noise_cluster": True,
                **(noise_summary or {}),
            }
        label_payload = None
        can_call_llm = (
            use_llm_labeling
            and cluster_id_int != -1
            and (llm_max_clusters <= 0 or llm_labeled_clusters < llm_max_clusters)
        )
        llm_status = "not_attempted"
        if can_call_llm:
            llm_attempted += 1
            elapsed = time.perf_counter() - last_llm_call_ts
            sleep_for = max(0.0, llm_min_interval_seconds - elapsed)
            if sleep_for > 0.0:
                time.sleep(sleep_for)
            label_payload, llm_status = _call_openrouter_label(
                {
                    "cluster_id": cluster_id_int,
                    "cluster_size": int(len(group)),
                    "top_terms": top_terms,
                    "archetype_distribution": archetype_dist,
                    "type_distribution": type_dist,
                    "avg_similarity": avg_similarity,
                    "representative_cards": [x["card_name"] for x in representative_cards[:5]],
                },
                openrouter_url=openrouter_url,
                model=openrouter_model,
                api_key=openrouter_api_key,
                max_retries=llm_max_retries,
                base_backoff_seconds=llm_base_backoff_seconds,
            )
            last_llm_call_ts = time.perf_counter()
        if label_payload is not None and llm_status == "ok":
            llm_succeeded += 1
            llm_labeled_clusters += 1
            label_source = "llm"
        else:
            if llm_status != "not_attempted":
                llm_failed += 1
                llm_failure_reasons[llm_status] += 1
                if len(llm_failure_samples) < 5:
                    llm_failure_samples.append(
                        f"cluster_id={cluster_id_int} status={llm_status} "
                        f"model={openrouter_model} url={openrouter_url}"
                    )
            heuristic_fallback += 1
            label_payload = {
                "functional_role": heuristic_role,
                "label_hint": heuristic_role.replace("_", " "),
                "label_confidence": 0.55,
                "reasoning": heuristic_source,
            }
            label_source = heuristic_source

        metadata_rows.append(
            {
                "cluster_version": cluster_version,
                "cluster_id": cluster_id_int,
                "cluster_size": int(len(group)),
                "functional_role": str(label_payload["functional_role"]),
                "label_hint": str(label_payload["label_hint"]),
                "label_confidence": float(label_payload["label_confidence"]),
                "label_source": str(label_source),
                "review_status": "unreviewed",
                "medoid_card_id": medoid_card_id,
                "metadata": {
                    **metadata_json,
                    "llm_reasoning": str(label_payload.get("reasoning", "")),
                },
                "generated_at": generated_at,
            }
        )
    logger.info(
        (
            "Cluster labeling summary: use_llm_labeling=%s llm_attempted=%s "
            "llm_succeeded=%s llm_failed=%s heuristic_fallback=%s"
        ),
        use_llm_labeling,
        llm_attempted,
        llm_succeeded,
        llm_failed,
        heuristic_fallback,
    )
    if llm_failed > 0:
        logger.warning("LLM labeling failure breakdown: %s", dict(llm_failure_reasons))
        logger.warning("LLM labeling failure samples: %s", llm_failure_samples)
    return pd.DataFrame(metadata_rows)


def build_projection_df(
    card_ids: pd.Series,
    vectors: np.ndarray,
    cluster_version: str,
    random_state: int,
    method: str = "umap",
) -> pd.DataFrame:
    if method == "umap":
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".cache/numba").resolve()))
        os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
        try:
            import umap

            projector = umap.UMAP(n_components=2, random_state=random_state)
            points = projector.fit_transform(vectors)
        except Exception:
            manifold = importlib.import_module("sklearn.manifold")
            TSNE = getattr(manifold, "TSNE")

            method = "tsne"
            points = TSNE(n_components=2, random_state=random_state, init="random").fit_transform(
                vectors
            )
    else:
        manifold = importlib.import_module("sklearn.manifold")
        TSNE = getattr(manifold, "TSNE")

        points = TSNE(n_components=2, random_state=random_state, init="random").fit_transform(
            vectors
        )
    return pd.DataFrame(
        {
            "card_id": card_ids.astype("int64"),
            "cluster_version": cluster_version,
            "projection_method": method,
            "projection_x": points[:, 0],
            "projection_y": points[:, 1],
        }
    )


def export_cluster_plot(
    output_dir: Path,
    cluster_df: pd.DataFrame,
    projections_df: pd.DataFrame,
) -> Path:
    import matplotlib.pyplot as plt

    plot_df = cluster_df.merge(projections_df, on=["card_id", "cluster_version"], how="left")
    plot_df = plot_df.dropna(subset=["projection_x", "projection_y"])
    fig, ax = plt.subplots(figsize=(12, 8))
    encoded = pd.factorize(plot_df["cluster_id"].astype(str))[0]
    alpha = np.clip(plot_df["cluster_probability"].astype(float), 0.1, 1.0)
    ax.scatter(
        plot_df["projection_x"],
        plot_df["projection_y"],
        c=encoded,
        cmap="tab20",
        s=20,
        alpha=alpha,
    )
    noise = plot_df[plot_df["cluster_id"] == -1]
    if len(noise) > 0:
        ax.scatter(noise["projection_x"], noise["projection_y"], marker="x", c="black", s=18)
    ax.set_title("Cluster Projection")
    ax.set_xlabel("projection_x")
    ax.set_ylabel("projection_y")
    ax.grid(alpha=0.2)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "clusters_by_cluster_id.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def compute_final_metrics_and_scores(
    labels: np.ndarray,
    probabilities: np.ndarray,
    tiny_cluster_max_size: int = 3,
) -> dict[str, float | int | str]:
    total = len(labels)
    if total == 0:
        raise ValueError("labels must not be empty")
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


def _compute_semantic_quality_metrics(
    labels: np.ndarray, vectors: np.ndarray, semantic_df: pd.DataFrame
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
    structural_score: dict[str, float], semantic_score: dict[str, float]
) -> dict[str, float]:
    final_score = float(structural_score["structural_score"] + semantic_score["semantic_score"])
    return {"final_score": final_score}


def attach_nearest_cluster_labels(
    cluster_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    enriched = cluster_df.copy()
    label_map = metadata_df.set_index("cluster_id")["label_hint"].to_dict()
    role_map = metadata_df.set_index("cluster_id")["functional_role"].to_dict()

    mask = enriched["cluster_id"] == -1
    nearest_ids = enriched.loc[mask, "nearest_cluster_id"]
    labels: list[str] = []
    roles: list[str] = []
    for v in nearest_ids.tolist():
        if pd.isna(v):
            labels.append("")
            roles.append("")
            continue
        cid = int(v)
        labels.append(str(label_map.get(cid, "")))
        roles.append(str(role_map.get(cid, "")))
    enriched.loc[mask, "nearest_cluster_label_hint"] = labels
    enriched.loc[mask, "nearest_cluster_functional_role"] = roles
    return enriched


def export_final_clusters_json(
    output_dir: Path,
    cluster_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    cards_df: pd.DataFrame,
) -> Path:
    merged = (
        cluster_df.merge(
            cards_df[["id", "name", "type", "archetype", "desc"]],
            left_on="card_id",
            right_on="id",
            how="left",
        )
        .merge(
            metadata_df[["cluster_id", "functional_role", "label_hint"]],
            left_on="cluster_id",
            right_on="cluster_id",
            how="left",
        )
        .rename(columns={"type": "card_type", "desc": "description"})
    )
    final_records = merged[
        [
            "card_id",
            "name",
            "card_type",
            "archetype",
            "description",
            "cluster_id",
            "cluster_probability",
            "outlier_score",
            "nearest_cluster_id",
            "nearest_cluster_distance",
            "nearest_cluster_confidence",
            "nearest_cluster_label_hint",
            "nearest_cluster_functional_role",
            "functional_role",
            "label_hint",
        ]
    ].copy()
    text_cols = [
        "name",
        "card_type",
        "archetype",
        "description",
        "nearest_cluster_label_hint",
        "nearest_cluster_functional_role",
        "functional_role",
        "label_hint",
    ]
    float_cols = [
        "cluster_probability",
        "outlier_score",
        "nearest_cluster_distance",
        "nearest_cluster_confidence",
    ]
    int_cols = ["nearest_cluster_id"]
    final_records[text_cols] = final_records[text_cols].fillna("")
    final_records[float_cols] = final_records[float_cols].fillna(0.0)
    for col in int_cols:
        final_records[col] = final_records[col].astype("Int64")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "final_clusters.json"
    json_path.write_text(final_records.to_json(orient="records"), encoding="utf-8")
    return json_path


def export_cluster_parquet_artifacts(
    output_dir: Path,
    cluster_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "final_cluster_assignments.parquet"
    metadata_path = output_dir / "final_cluster_metadata.parquet"
    cluster_df.to_parquet(assignments_path, index=False)
    metadata_df.to_parquet(metadata_path, index=False)
    return {"final_cluster_assignments": assignments_path, "final_cluster_metadata": metadata_path}


def export_top_clusters_by_size(
    output_dir: Path,
    metadata_df: pd.DataFrame,
) -> Path:
    rows: list[dict[str, Any]] = []
    non_noise = metadata_df[metadata_df["cluster_id"] != -1].copy()
    for _, row in non_noise.sort_values("cluster_size", ascending=False).iterrows():
        metadata = row["metadata"] if isinstance(row["metadata"], dict) else {}
        rows.append(
            {
                "cluster_version": row["cluster_version"],
                "cluster_id": int(row["cluster_id"]),
                "cluster_size": int(row["cluster_size"]),
                "functional_role": str(row["functional_role"]),
                "label_hint": str(row["label_hint"]),
                "medoid_card_id": int(row["medoid_card_id"]),
                "top_archetypes": _safe_json(metadata.get("archetype_distribution", {})),
                "top_card_types": _safe_json(metadata.get("type_distribution", {})),
            }
        )
    df = pd.DataFrame(rows)
    path = output_dir / "top_clusters_by_size.csv"
    df.to_csv(path, index=False)
    return path


def export_noise_cards_report(
    output_dir: Path,
    cluster_df: pd.DataFrame,
    cards_df: pd.DataFrame,
) -> Path:
    merged = (
        cluster_df[cluster_df["cluster_id"] == -1]
        .merge(
            cards_df[["id", "name", "type", "archetype", "desc"]],
            left_on="card_id",
            right_on="id",
            how="left",
        )
        .rename(columns={"type": "card_type", "desc": "description"})
    )
    cols = [
        "card_id",
        "name",
        "card_type",
        "archetype",
        "description",
        "outlier_score",
        "nearest_cluster_id",
        "nearest_cluster_distance",
        "nearest_cluster_confidence",
        "nearest_cluster_label_hint",
        "nearest_cluster_functional_role",
    ]
    path = output_dir / "noise_cards_report.csv"
    merged[cols].to_csv(path, index=False)
    return path


def export_cluster_examples(
    output_dir: Path,
    cluster_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    cards_df: pd.DataFrame,
    vectors: np.ndarray,
    random_state: int,
) -> Path:
    rng = np.random.default_rng(random_state)
    merged = cluster_df.merge(
        cards_df[["id", "name", "type", "archetype", "desc"]],
        left_on="card_id",
        right_on="id",
        how="left",
    )
    meta_map = metadata_df.set_index("cluster_id").to_dict(orient="index")
    labels = cluster_df["cluster_id"].to_numpy(dtype=int)
    normalized_vectors = _normalize_l2(vectors)

    examples: list[dict[str, Any]] = []
    for cid, group in merged[merged["cluster_id"] != -1].groupby("cluster_id", sort=True):
        cid_int = int(cid)
        meta = meta_map.get(cid_int, {})
        idx = np.where(labels == cid_int)[0]
        cluster_vecs = normalized_vectors[idx]
        centroid = np.mean(cluster_vecs, axis=0)
        dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
        closest_idx = np.argsort(dists)[:5]

        g = group.reset_index(drop=True)
        top_5 = []
        for i in closest_idx:
            if i < len(g):
                top_5.append(
                    {
                        "card_id": int(g.iloc[i]["card_id"]),
                        "name": str(g.iloc[i].get("name", "")),
                        "cluster_probability": float(g.iloc[i]["cluster_probability"]),
                    }
                )
        rand_n = min(5, len(g))
        random_idx = rng.choice(len(g), size=rand_n, replace=False).tolist() if rand_n > 0 else []
        random_cards = [
            {
                "card_id": int(g.iloc[i]["card_id"]),
                "name": str(g.iloc[i].get("name", "")),
                "cluster_probability": float(g.iloc[i]["cluster_probability"]),
            }
            for i in random_idx
        ]
        low_conf = g.sort_values("cluster_probability", ascending=True).head(5)
        low_conf_cards = [
            {
                "card_id": int(r["card_id"]),
                "name": str(r.get("name", "")),
                "cluster_probability": float(r["cluster_probability"]),
            }
            for _, r in low_conf.iterrows()
        ]
        medoid_card = {}
        medoid_id = meta.get("medoid_card_id")
        if medoid_id is not None and not pd.isna(medoid_id):
            medoid_match = g[g["card_id"] == int(medoid_id)]
            if len(medoid_match) > 0:
                first = medoid_match.iloc[0]
                medoid_card = {"card_id": int(first["card_id"]), "name": str(first.get("name", ""))}
        examples.append(
            {
                "cluster_id": cid_int,
                "cluster_size": int(len(group)),
                "functional_role": str(meta.get("functional_role", "")),
                "label_hint": str(meta.get("label_hint", "")),
                "medoid_card": medoid_card,
                "top_5_closest_cards": top_5,
                "random_5_cards": random_cards,
                "bottom_5_lowest_confidence_cards": low_conf_cards,
            }
        )
    path = output_dir / "cluster_examples.json"
    path.write_text(json.dumps(examples, ensure_ascii=True), encoding="utf-8")
    return path


def export_cluster_quality_summary(
    output_dir: Path,
    cluster_version: str,
    selected_params: dict[str, Any],
    final_metrics: dict[str, float],
    semantic_metrics: dict[str, float],
) -> Path:
    payload = {
        "cluster_version": cluster_version,
        "selected_params": selected_params,
        "final_score": float(final_metrics.get("final_score", 0.0)),
        "structural_score": float(final_metrics.get("structural_score", 0.0)),
        "semantic_score": float(semantic_metrics.get("semantic_score", 0.0)),
        "num_cards": float(final_metrics.get("num_cards", 0.0)),
        "num_clusters": float(final_metrics.get("num_clusters_non_noise", 0.0)),
        "noise_ratio": float(final_metrics.get("noise_ratio", 0.0)),
        "largest_cluster_ratio": float(final_metrics.get("largest_cluster_ratio", 0.0)),
        "tiny_cluster_ratio": float(final_metrics.get("tiny_cluster_ratio", 0.0)),
        "membership_probability_mean": float(final_metrics.get("membership_probability_mean", 0.0)),
        "archetype_purity_mean": float(semantic_metrics.get("archetype_purity_mean", 0.0)),
        "functional_coherence_score": float(
            semantic_metrics.get("functional_coherence_score", 0.0)
        ),
        "generic_archetype_balance_score": float(
            semantic_metrics.get("generic_archetype_balance_score", 0.0)
        ),
    }
    path = output_dir / "cluster_quality_summary.json"
    path.write_text(_safe_json(payload), encoding="utf-8")
    return path


def _pg_create_tables(conn: psycopg.Connection[Any], schema: str) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};

    CREATE TABLE IF NOT EXISTS {schema}.card_cluster_assignments (
        card_id BIGINT NOT NULL,
        cluster_version TEXT NOT NULL,
        cluster_id BIGINT NOT NULL,
        cluster_probability DOUBLE PRECISION NOT NULL,
        outlier_score DOUBLE PRECISION NOT NULL,
        nearest_cluster_id BIGINT,
        nearest_cluster_distance DOUBLE PRECISION,
        nearest_cluster_confidence DOUBLE PRECISION,
        nearest_cluster_label_hint TEXT,
        nearest_cluster_functional_role TEXT,
        embedding_model TEXT NOT NULL,
        embedding_version TEXT NOT NULL,
        generated_at TEXT NOT NULL,
        PRIMARY KEY (card_id, cluster_version)
    );
    CREATE INDEX IF NOT EXISTS idx_card_cluster_assignments_version_cluster
        ON {schema}.card_cluster_assignments(cluster_version, cluster_id);

    CREATE TABLE IF NOT EXISTS {schema}.embedding_cluster_metadata (
        cluster_version TEXT NOT NULL,
        cluster_id BIGINT NOT NULL,
        cluster_size BIGINT NOT NULL,
        functional_role TEXT NOT NULL,
        label_hint TEXT NOT NULL,
        label_confidence DOUBLE PRECISION NOT NULL,
        label_source TEXT NOT NULL,
        review_status TEXT NOT NULL,
        medoid_card_id BIGINT,
        metadata JSONB NOT NULL,
        generated_at TEXT NOT NULL,
        PRIMARY KEY (cluster_version, cluster_id)
    );
    CREATE INDEX IF NOT EXISTS idx_embedding_cluster_metadata_version_role
        ON {schema}.embedding_cluster_metadata(cluster_version, functional_role, review_status);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
        cur.execute(
            f"ALTER TABLE {schema}.card_cluster_assignments "
            "ADD COLUMN IF NOT EXISTS nearest_cluster_id BIGINT"
        )
        cur.execute(
            f"ALTER TABLE {schema}.card_cluster_assignments "
            "ADD COLUMN IF NOT EXISTS nearest_cluster_distance DOUBLE PRECISION"
        )
        cur.execute(
            f"ALTER TABLE {schema}.card_cluster_assignments "
            "ADD COLUMN IF NOT EXISTS nearest_cluster_confidence DOUBLE PRECISION"
        )
        cur.execute(
            f"ALTER TABLE {schema}.card_cluster_assignments "
            "ADD COLUMN IF NOT EXISTS nearest_cluster_label_hint TEXT"
        )
        cur.execute(
            f"ALTER TABLE {schema}.card_cluster_assignments "
            "ADD COLUMN IF NOT EXISTS nearest_cluster_functional_role TEXT"
        )


def persist_cluster_assignments(
    conn: psycopg.Connection[Any], schema: str, cluster_version: str, cluster_df: pd.DataFrame
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {schema}.card_cluster_assignments WHERE cluster_version = %s",
            (cluster_version,),
        )
        ordered = cluster_df[
            [
                "card_id",
                "cluster_id",
                "cluster_probability",
                "outlier_score",
                "nearest_cluster_id",
                "nearest_cluster_distance",
                "nearest_cluster_confidence",
                "nearest_cluster_label_hint",
                "nearest_cluster_functional_role",
                "embedding_model",
                "embedding_version",
                "cluster_version",
                "generated_at",
            ]
        ].copy()
        rows = [
            tuple(None if pd.isna(v) else v for v in r)
            for r in ordered.itertuples(index=False, name=None)
        ]
        cur.executemany(
            f"""
            INSERT INTO {schema}.card_cluster_assignments (
                card_id, cluster_id, cluster_probability, outlier_score,
                nearest_cluster_id, nearest_cluster_distance, nearest_cluster_confidence,
                nearest_cluster_label_hint, nearest_cluster_functional_role,
                embedding_model, embedding_version, cluster_version, generated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )


def persist_cluster_metadata(
    conn: psycopg.Connection[Any], schema: str, cluster_version: str, metadata_df: pd.DataFrame
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {schema}.embedding_cluster_metadata WHERE cluster_version = %s",
            (cluster_version,),
        )
        rows: list[tuple[Any, ...]] = []
        for row in metadata_df.itertuples(index=False):
            review_status = str(row.review_status)
            if review_status not in REVIEW_STATUS_VALUES:
                raise ValueError(f"Invalid review_status: {review_status}")
            rows.append(
                (
                    str(row.cluster_version),
                    int(row.cluster_id),
                    int(row.cluster_size),
                    str(row.functional_role),
                    str(row.label_hint),
                    float(row.label_confidence),
                    str(row.label_source),
                    review_status,
                    int(row.medoid_card_id),
                    _safe_json(row.metadata),
                    str(row.generated_at),
                )
            )
        cur.executemany(
            f"""
            INSERT INTO {schema}.embedding_cluster_metadata (
                cluster_version, cluster_id, cluster_size, functional_role, label_hint,
                label_confidence, label_source, review_status, medoid_card_id,
                metadata, generated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            """,
            rows,
        )


def _publish_postgres(
    args: argparse.Namespace, cluster_df: pd.DataFrame, metadata_df: pd.DataFrame
) -> None:
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
            persist_cluster_assignments(conn, args.pg_schema, args.cluster_version, cluster_df)
            persist_cluster_metadata(conn, args.pg_schema, args.cluster_version, metadata_df)
    finally:
        conn.close()


def log_final_cluster_artifacts_to_mlflow(
    args: argparse.Namespace,
    artifact_paths: dict[str, Path],
    final_metrics: dict[str, float],
    n_umap_components: int,
) -> None:
    if not args.mlflow_tracking_uri:
        logger.info("MLflow tracking URI not set; skipping MLflow logging.")
        return
    import mlflow

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.cluster_version):
        mlflow.log_params(
            {
                "cluster_version": args.cluster_version,
                "distance_metric": args.distance_metric,
                "l2_normalized": True,
                "n_umap_components": n_umap_components,
                "use_umap": args.use_umap,
                "min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
                "cluster_selection_method": args.cluster_selection_method,
                "cluster_selection_epsilon": args.cluster_selection_epsilon,
                "umap_n_components": args.umap_n_components,
                "umap_n_neighbors": args.umap_n_neighbors,
                "umap_min_dist": args.umap_min_dist,
                "umap_metric": args.umap_metric,
                "pg_schema": args.pg_schema,
                "pg_assignments_table": "card_cluster_assignments",
                "pg_metadata_table": "embedding_cluster_metadata",
            }
        )
        mlflow.log_metrics(final_metrics)
        for path in artifact_paths.values():
            mlflow.log_artifact(str(path))


def main() -> None:
    start_time = time.perf_counter()
    _load_dotenv()
    args = parse_args()
    embeddings_df = pd.read_parquet(args.embeddings_file)
    cards_df = pd.read_parquet(args.cards_file)
    vectors, embedding_records = _extract_embeddings(embeddings_df)
    args.cluster_version = _build_cluster_version(args, embedding_records)
    generated_at = datetime.now(UTC).isoformat()

    labels, probabilities, outlier_scores, n_umap_components, clustering_space_vectors = (
        cluster_embeddings(
            vectors=vectors,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric=args.distance_metric,
            cluster_selection_method=args.cluster_selection_method,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            use_umap=args.use_umap,
            umap_n_components=args.umap_n_components,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_metric=args.umap_metric,
            random_state=args.random_state,
        )
    )

    cluster_df = pd.DataFrame(
        {
            "card_id": embedding_records["card_id"].astype("int64"),
            "cluster_version": args.cluster_version,
            # Keep HDBSCAN output untouched; -1 marks low-density/outlier noise points.
            "cluster_id": labels.astype("int64"),
            "cluster_probability": probabilities.astype(float),
            "outlier_score": outlier_scores.astype(float),
            "embedding_model": embedding_records["embedding_model"].astype(str),
            "embedding_version": embedding_records["embedding_version"].astype(str),
            "generated_at": generated_at,
        }
    )
    initial_noise_count = int((cluster_df["cluster_id"] == -1).sum())
    logger.info(
        "Initial HDBSCAN labels: noise_cards=%s noise_ratio=%.4f",
        initial_noise_count,
        float(initial_noise_count / max(1, len(cluster_df))),
    )
    cluster_df = enrich_noise_with_nearest_cluster(cluster_df, clustering_space_vectors)
    noise_summary = _compute_noise_summary(cluster_df, cards_df)
    metadata_df = build_cluster_metadata(
        cards_df=cards_df,
        cluster_df=cluster_df,
        vectors=_normalize_l2(vectors),
        cluster_version=args.cluster_version,
        generated_at=generated_at,
        use_llm_labeling=args.use_llm_labeling,
        openrouter_url=args.openrouter_url,
        openrouter_model=args.openrouter_model,
        openrouter_api_key=args.openrouter_api_key,
        llm_max_retries=args.llm_max_retries,
        llm_base_backoff_seconds=args.llm_base_backoff_seconds,
        llm_min_interval_seconds=args.llm_min_interval_seconds,
        llm_max_clusters=args.llm_max_clusters,
        noise_summary=noise_summary,
    )
    cluster_df = attach_nearest_cluster_labels(cluster_df, metadata_df)
    final_noise_count = int((cluster_df["cluster_id"] == -1).sum())
    if final_noise_count != initial_noise_count:
        raise RuntimeError(
            "Invariant violation: cluster_id changed after nearest-cluster enrichment. "
            f"before={initial_noise_count} after={final_noise_count}"
        )
    logger.info(
        "Post-enrichment labels unchanged: noise_cards=%s noise_ratio=%.4f",
        final_noise_count,
        float(final_noise_count / max(1, len(cluster_df))),
    )

    output_dir = Path(args.output_dir) / args.cluster_version
    parquet_paths = export_cluster_parquet_artifacts(output_dir, cluster_df, metadata_df)
    final_json_path = export_final_clusters_json(output_dir, cluster_df, metadata_df, cards_df)
    projections_df = build_projection_df(
        card_ids=cluster_df["card_id"],
        vectors=_normalize_l2(vectors),
        cluster_version=args.cluster_version,
        random_state=args.random_state,
        method=args.projection_method,
    )
    if args.include_tsne:
        tsne_df = build_projection_df(
            card_ids=cluster_df["card_id"],
            vectors=_normalize_l2(vectors),
            cluster_version=args.cluster_version,
            random_state=args.random_state,
            method="tsne",
        )
        projections_df = pd.concat([projections_df, tsne_df], ignore_index=True)
    projections_path = output_dir / "embedding_cluster_projections.parquet"
    projections_df.to_parquet(projections_path, index=False)
    plot_path = export_cluster_plot(output_dir, cluster_df, projections_df)

    clustering_metrics = compute_final_metrics_and_scores(
        labels=labels,
        probabilities=probabilities,
        tiny_cluster_max_size=args.tiny_cluster_max_size,
    )
    structural_score = compute_structural_score(clustering_metrics, args)
    semantic_df = cluster_df[["card_id", "cluster_id"]].merge(
        cards_df[["id", "archetype"]],
        left_on="card_id",
        right_on="id",
        how="left",
    )
    semantic_df = semantic_df.merge(
        metadata_df[["cluster_id", "label_confidence"]],
        on="cluster_id",
        how="left",
    )
    semantic_df["archetype"] = semantic_df["archetype"].fillna("unknown").astype("string")
    semantic_df["label_confidence"] = (
        pd.to_numeric(semantic_df["label_confidence"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    semantic_metrics = _compute_semantic_quality_metrics(labels, vectors, semantic_df)
    semantic_score = compute_semantic_score(semantic_metrics, args)
    final_score = compute_final_score(structural_score, semantic_score)

    final_metrics: dict[str, float] = {
        **{k: float(v) for k, v in clustering_metrics.items() if isinstance(v, int | float)},
        **structural_score,
        **semantic_metrics,
        **semantic_score,
        **final_score,
        "num_cards": float(len(cluster_df)),
        "num_clusters_non_noise": float(clustering_metrics["num_clusters"]),
    }
    summary_path = output_dir / "cluster_run_summary.json"
    summary_payload = {
        "cluster_version": args.cluster_version,
        "final_metrics": final_metrics,
        "postgres_tables": {
            "card_cluster_assignments": f"{args.pg_schema}.card_cluster_assignments",
            "embedding_cluster_metadata": f"{args.pg_schema}.embedding_cluster_metadata",
        },
    }
    summary_path.write_text(_safe_json(summary_payload), encoding="utf-8")

    top_clusters_path = export_top_clusters_by_size(output_dir, metadata_df)
    noise_report_path = export_noise_cards_report(output_dir, cluster_df, cards_df)
    examples_path = export_cluster_examples(
        output_dir=output_dir,
        cluster_df=cluster_df,
        metadata_df=metadata_df,
        cards_df=cards_df,
        vectors=vectors,
        random_state=args.random_state,
    )
    quality_summary_path = export_cluster_quality_summary(
        output_dir=output_dir,
        cluster_version=args.cluster_version,
        selected_params={
            "distance_metric": args.distance_metric,
            "l2_normalized": True,
            "n_umap_components": n_umap_components,
            "use_umap": args.use_umap,
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "cluster_selection_method": args.cluster_selection_method,
            "cluster_selection_epsilon": args.cluster_selection_epsilon,
            "umap_n_components": args.umap_n_components,
            "umap_n_neighbors": args.umap_n_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "umap_metric": args.umap_metric,
        },
        final_metrics=final_metrics,
        semantic_metrics=semantic_metrics,
    )

    if args.publish_postgres:
        _publish_postgres(args, cluster_df, metadata_df)

    artifact_paths = {
        **parquet_paths,
        "final_clusters_json": final_json_path,
        "projections": projections_path,
        "cluster_plot": plot_path,
        "run_summary": summary_path,
        "top_clusters_by_size": top_clusters_path,
        "noise_cards_report": noise_report_path,
        "cluster_examples": examples_path,
        "cluster_quality_summary": quality_summary_path,
    }
    log_final_cluster_artifacts_to_mlflow(
        args=args,
        artifact_paths=artifact_paths,
        final_metrics=final_metrics,
        n_umap_components=n_umap_components,
    )
    logger.info(
        "Clustering complete in %.2fs: version=%s output_dir=%s",
        time.perf_counter() - start_time,
        args.cluster_version,
        output_dir,
    )


if __name__ == "__main__":
    main()
