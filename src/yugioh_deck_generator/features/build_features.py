import argparse
import json
import logging
import math
import os
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import psycopg

Decorator = Callable[..., Any]
step: Decorator
pipeline: Decorator

try:
    from zenml import pipeline as _zenml_pipeline
    from zenml import step as _zenml_step
    step = _zenml_step
    pipeline = _zenml_pipeline
except ImportError:  # pragma: no cover

    def _identity_decorator(func: Any | None = None, **_: Any) -> Any:
        if func is None:
            return lambda inner: inner
        return func

    step = _identity_decorator
    pipeline = _identity_decorator


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
ROLE_TAGS = ["starter", "extender", "searcher", "interruption", "brick"]
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
_EMBEDDING_MODEL_INSTANCE: Any | None = None
_EMBEDDING_MODEL_CACHE_DIR: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 2 feature artifacts from cards parquet."
    )
    parser.add_argument("--cards-file", default="data/processed/cards.parquet")
    parser.add_argument("--output-dir", default="data/features")

    parser.add_argument("--publish-postgres", action="store_true")
    parser.add_argument("--pg-host", default=os.getenv("POSTGRES_HOST", "localhost"))
    parser.add_argument("--pg-port", type=int, default=int(os.getenv("POSTGRES_PORT", "5432")))
    parser.add_argument("--pg-dbname", default=os.getenv("POSTGRES_DB", "yugioh"))
    parser.add_argument("--pg-user", default=os.getenv("POSTGRES_USER", "yugioh"))
    parser.add_argument("--pg-password", default=os.getenv("POSTGRES_PASSWORD", "yugioh"))
    parser.add_argument("--pg-schema", default=os.getenv("POSTGRES_SCHEMA", "public"))

    parser.add_argument("--publish-feature-store", action="store_true")
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))

    parser.add_argument("--publish-vector-store", action="store_true")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument(
        "--qdrant-collection", default=os.getenv("QDRANT_COLLECTION", "card_embeddings")
    )

    parser.add_argument("--feature-version", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument(
        "--embedding-cache-dir",
        default=os.getenv("EMBEDDING_CACHE_DIR", ".cache/models/sentence_transformers"),
    )
    parser.add_argument("--disable-zenml", action="store_true")
    return parser.parse_args()


def _safe_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_combat_stat(value: Any) -> int | None:
    parsed = _safe_int(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _text_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    tokens = _text_tokens(text)
    if not tokens:
        return vector

    for token in tokens:
        index = hash(token) % dim
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _load_sentence_transformer(cache_dir: str | None = None) -> Any:
    global _EMBEDDING_MODEL_INSTANCE, _EMBEDDING_MODEL_CACHE_DIR
    if _EMBEDDING_MODEL_INSTANCE is not None:
        if cache_dir != _EMBEDDING_MODEL_CACHE_DIR:
            logger.info(
                "Embedding model already loaded with cache_dir=%s; ignoring new cache_dir=%s",
                _EMBEDDING_MODEL_CACHE_DIR,
                cache_dir,
            )
        return _EMBEDDING_MODEL_INSTANCE

    from sentence_transformers import SentenceTransformer

    cache_path = None
    if cache_dir:
        cache_path = str(Path(cache_dir).expanduser())
        Path(cache_path).mkdir(parents=True, exist_ok=True)
    _EMBEDDING_MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL, cache_folder=cache_path)
    _EMBEDDING_MODEL_CACHE_DIR = cache_path
    return _EMBEDDING_MODEL_INSTANCE


def _model_embeddings(
    texts: list[str], embedding_cache_dir: str | None = None
) -> tuple[list[list[float]], int]:
    try:
        model = _load_sentence_transformer(cache_dir=embedding_cache_dir)
        vectors = model.encode(texts, normalize_embeddings=True).tolist()
        dim = int(model.get_sentence_embedding_dimension())
        return vectors, dim
    except Exception as exc:  # pragma: no cover - fallback for offline/test envs
        logger.warning(
            "Falling back to hashing embeddings because transformer model load/encode failed: %s",
            exc,
        )
        vectors = [_hash_embedding(text, dim=EMBEDDING_DIM) for text in texts]
        return vectors, EMBEDDING_DIM


def _role_rows(card_row: pd.Series, role_version: str, generated_at: str) -> list[dict[str, Any]]:
    card_id = int(card_row["id"])
    desc = str(card_row.get("desc", "") or "").lower()
    card_type = str(card_row.get("type", "") or "").lower()
    attack = _safe_int(card_row.get("atk"))
    defense = _safe_int(card_row.get("def"))

    roles: list[tuple[str, float, str]] = []

    if any(token in desc for token in ["add", "search", "from your deck"]):
        roles.append(("searcher", 0.9, "desc_keyword:search"))
        roles.append(("starter", 0.75, "desc_keyword:starter"))

    if any(token in desc for token in ["special summon", "summon"]):
        roles.append(("extender", 0.8, "desc_keyword:summon"))

    if any(token in desc for token in ["negate", "banish", "destroy", "cannot"]):
        roles.append(("interruption", 0.8, "desc_keyword:interaction"))

    if "monster" in card_type and attack is not None and attack >= 2500:
        roles.append(("brick", 0.55, "stat_high_attack"))

    if (
        "monster" in card_type
        and attack is not None
        and defense is not None
        and attack <= 500
        and defense <= 500
    ):
        roles.append(("starter", 0.5, "stat_low_stats"))

    dedup: dict[str, tuple[float, str]] = {}
    for role, confidence, rule_source in roles:
        existing = dedup.get(role)
        if existing is None or confidence > existing[0]:
            dedup[role] = (confidence, rule_source)

    return [
        {
            "card_id": card_id,
            "role_tag": role,
            "confidence": confidence,
            "rule_source": rule_source,
            "role_version": role_version,
            "generated_at": generated_at,
        }
        for role, (confidence, rule_source) in dedup.items()
        if role in ROLE_TAGS
    ]


def build_card_features(
    cards_df: pd.DataFrame, feature_version: str, generated_at: str
) -> pd.DataFrame:
    features = pd.DataFrame()
    features["card_id"] = cards_df["id"].astype("int64")
    features["type"] = cards_df["type"]
    features["race"] = cards_df["race"]
    features["attribute"] = cards_df["attribute"]
    features["level"] = cards_df["level"].apply(_safe_int)
    features["atk"] = cards_df["atk"].apply(_clean_combat_stat)
    features["def"] = cards_df["def"].apply(_clean_combat_stat)
    features["scale"] = cards_df["scale"].apply(_safe_int)
    features["linkval"] = cards_df["linkval"].apply(_safe_int)
    features["archetype"] = cards_df["archetype"]

    type_lower = cards_df["type"].fillna("").astype(str).str.lower()
    features["is_effect_monster"] = type_lower.str.contains("effect")
    features["is_extra_deck"] = type_lower.str.contains("fusion|synchro|xyz|link")
    features["is_spell"] = type_lower.str.contains("spell")
    features["is_trap"] = type_lower.str.contains("trap")

    features["feature_version"] = feature_version
    features["generated_at"] = generated_at
    return features


def build_card_embeddings(
    cards_df: pd.DataFrame,
    embedding_version: str,
    generated_at: str,
    embedding_cache_dir: str | None = None,
) -> pd.DataFrame:
    text_series = (
        cards_df["name"].fillna("").astype(str) + " " + cards_df["desc"].fillna("").astype(str)
    ).str.strip()
    texts = text_series.tolist()
    vectors, embedding_dim = _model_embeddings(texts, embedding_cache_dir=embedding_cache_dir)

    embeddings = pd.DataFrame()
    embeddings["card_id"] = cards_df["id"].astype("int64")
    embeddings["embedding_model"] = EMBEDDING_MODEL
    embeddings["embedding_dim"] = embedding_dim
    embeddings["embedding_vector"] = vectors
    embeddings["embedding_version"] = embedding_version
    embeddings["generated_at"] = generated_at
    return embeddings


def build_card_role_tags(
    cards_df: pd.DataFrame, role_version: str, generated_at: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in cards_df.iterrows():
        rows.extend(_role_rows(row, role_version=role_version, generated_at=generated_at))
    return pd.DataFrame(rows)


def validate_features(card_features: pd.DataFrame, card_embeddings: pd.DataFrame) -> None:
    if card_features["card_id"].duplicated().any():
        raise ValueError("card_features has duplicate card_id values")
    if card_embeddings[["card_id", "embedding_model"]].duplicated().any():
        raise ValueError("card_embeddings has duplicate (card_id, embedding_model) values")


def write_feature_artifacts(
    output_dir: Path,
    card_features: pd.DataFrame,
    card_embeddings: pd.DataFrame,
    card_role_tags: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    card_features_path = output_dir / "card_features.parquet"
    card_embeddings_path = output_dir / "card_embeddings.parquet"
    card_role_tags_path = output_dir / "card_role_tags.parquet"

    card_features.to_parquet(card_features_path, index=False)
    card_embeddings.to_parquet(card_embeddings_path, index=False)
    card_role_tags.to_parquet(card_role_tags_path, index=False)

    logger.info("Wrote %s rows to %s", len(card_features), card_features_path)
    logger.info("Wrote %s rows to %s", len(card_embeddings), card_embeddings_path)
    logger.info("Wrote %s rows to %s", len(card_role_tags), card_role_tags_path)


def _pg_create_tables(conn: psycopg.Connection[Any], schema: str) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};

    CREATE TABLE IF NOT EXISTS {schema}.card_features (
        card_id BIGINT PRIMARY KEY,
        type TEXT,
        race TEXT,
        attribute TEXT,
        level BIGINT,
        atk BIGINT,
        def BIGINT,
        scale BIGINT,
        linkval BIGINT,
        archetype TEXT,
        is_effect_monster BOOLEAN NOT NULL,
        is_extra_deck BOOLEAN NOT NULL,
        is_spell BOOLEAN NOT NULL,
        is_trap BOOLEAN NOT NULL,
        feature_version TEXT NOT NULL,
        generated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS {schema}.card_embeddings (
        card_id BIGINT NOT NULL,
        embedding_model TEXT NOT NULL,
        embedding_dim BIGINT NOT NULL,
        embedding_vector JSONB NOT NULL,
        embedding_version TEXT NOT NULL,
        generated_at TEXT NOT NULL,
        PRIMARY KEY (card_id, embedding_model)
    );

    CREATE TABLE IF NOT EXISTS {schema}.card_role_tags (
        card_id BIGINT NOT NULL,
        role_tag TEXT NOT NULL,
        confidence DOUBLE PRECISION NOT NULL,
        rule_source TEXT NOT NULL,
        role_version TEXT NOT NULL,
        generated_at TEXT NOT NULL,
        PRIMARY KEY (card_id, role_tag, role_version)
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl)


def _pg_replace_table(conn: psycopg.Connection[Any], schema: str, table: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {schema}.{table};")


def _pg_insert_card_features(conn: psycopg.Connection[Any], schema: str, df: pd.DataFrame) -> None:
    records = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {schema}.card_features (
                card_id, type, race, attribute, level, atk, def, scale, linkval, archetype,
                is_effect_monster, is_extra_deck, is_spell, is_trap, feature_version, generated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            records,
        )


def _pg_insert_card_embeddings(
    conn: psycopg.Connection[Any], schema: str, df: pd.DataFrame
) -> None:
    rows: list[tuple[Any, ...]] = []
    for row in df.itertuples(index=False, name=None):
        card_id, model, dim, vector, version, generated_at = row
        rows.append((card_id, model, dim, json.dumps(vector), version, generated_at))

    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {schema}.card_embeddings (
                card_id, embedding_model, embedding_dim, embedding_vector,
                embedding_version, generated_at
            ) VALUES (%s, %s, %s, %s::jsonb, %s, %s)
            """,
            rows,
        )


def _pg_insert_card_role_tags(conn: psycopg.Connection[Any], schema: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    records = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {schema}.card_role_tags (
                card_id, role_tag, confidence, rule_source, role_version, generated_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            records,
        )


def publish_postgres(
    args: argparse.Namespace,
    card_features: pd.DataFrame,
    card_embeddings: pd.DataFrame,
    card_role_tags: pd.DataFrame,
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
            _pg_replace_table(conn, args.pg_schema, "card_features")
            _pg_replace_table(conn, args.pg_schema, "card_embeddings")
            _pg_replace_table(conn, args.pg_schema, "card_role_tags")
            _pg_insert_card_features(conn, args.pg_schema, card_features)
            _pg_insert_card_embeddings(conn, args.pg_schema, card_embeddings)
            _pg_insert_card_role_tags(conn, args.pg_schema, card_role_tags)
    finally:
        conn.close()

    logger.info(
        "Published to Postgres schema=%s: features=%s embeddings=%s role_tags=%s",
        args.pg_schema,
        len(card_features),
        len(card_embeddings),
        len(card_role_tags),
    )


def publish_feature_store_redis(
    redis_host: str,
    redis_port: int,
    card_features: pd.DataFrame,
    card_role_tags: pd.DataFrame,
) -> None:
    import redis

    client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    if card_features.empty:
        return

    for _, row in card_features.iterrows():
        card_id = int(row["card_id"])
        key = f"card:features:{card_id}"
        feature_payload = {
            "type": row.get("type"),
            "race": row.get("race"),
            "attribute": row.get("attribute"),
            "level": row.get("level"),
            "atk": row.get("atk"),
            "def": row.get("def"),
            "archetype": row.get("archetype"),
            "is_effect_monster": bool(row.get("is_effect_monster")),
            "is_extra_deck": bool(row.get("is_extra_deck")),
            "is_spell": bool(row.get("is_spell")),
            "is_trap": bool(row.get("is_trap")),
        }
        client.set(key, json.dumps(feature_payload))

    if not card_role_tags.empty:
        grouped = card_role_tags.groupby("card_id")
        for card_id, group in grouped:
            key = f"card:roles:{int(card_id)}"
            role_payload = [
                {
                    "role_tag": str(row["role_tag"]),
                    "confidence": float(row["confidence"]),
                    "rule_source": str(row["rule_source"]),
                }
                for _, row in group.iterrows()
            ]
            client.set(key, json.dumps(role_payload))

    logger.info(
        "Published feature payloads to feature-store endpoint %s:%s", redis_host, redis_port
    )


def publish_vector_store_qdrant(
    qdrant_url: str,
    collection: str,
    card_embeddings: pd.DataFrame,
) -> None:
    collection_url = f"{qdrant_url.rstrip('/')}/collections/{collection}"
    create_payload = json.dumps({"vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}}).encode(
        "utf-8"
    )
    create_request = urllib.request.Request(
        collection_url,
        data=create_payload,
        method="PUT",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(create_request) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        if exc.code not in (200, 409):
            raise

    points: list[dict[str, Any]] = []
    for _, row in card_embeddings.iterrows():
        points.append(
            {
                "id": int(row["card_id"]),
                "vector": row["embedding_vector"],
                "payload": {
                    "embedding_model": row["embedding_model"],
                    "embedding_version": row["embedding_version"],
                },
            }
        )

    if not points:
        logger.info("No embeddings to upsert for Qdrant collection=%s", collection)
        return

    upsert_url = f"{collection_url}/points?wait=true"
    batch_size = 500
    total = len(points)
    for offset in range(0, total, batch_size):
        batch = points[offset : offset + batch_size]
        upsert_payload = json.dumps({"points": batch}).encode("utf-8")
        upsert_request = urllib.request.Request(
            upsert_url,
            data=upsert_payload,
            method="PUT",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(upsert_request) as response:
                response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed Qdrant upsert for batch starting at index {offset} "
                f"(batch_size={len(batch)}): {exc}"
            ) from exc

    logger.info("Upserted %s embeddings into Qdrant collection=%s", total, collection)


@step
def load_cards_step(cards_file: str) -> pd.DataFrame:
    return pd.read_parquet(cards_file)


@step
def build_feature_tables_step(
    cards_df: pd.DataFrame,
    feature_version: str,
    generated_at: str,
    embedding_cache_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    card_features = build_card_features(
        cards_df, feature_version=feature_version, generated_at=generated_at
    )
    card_embeddings = build_card_embeddings(
        cards_df,
        embedding_version=feature_version,
        generated_at=generated_at,
        embedding_cache_dir=embedding_cache_dir,
    )
    card_role_tags = build_card_role_tags(
        cards_df, role_version=feature_version, generated_at=generated_at
    )
    validate_features(card_features=card_features, card_embeddings=card_embeddings)
    return card_features, card_embeddings, card_role_tags


@step
def publish_outputs_step(
    cards_file: str,
    output_dir: str,
    feature_version: str,
    generated_at: str,
    publish_postgres_flag: bool,
    pg_host: str,
    pg_port: int,
    pg_dbname: str,
    pg_user: str,
    pg_password: str,
    pg_schema: str,
    publish_feature_store_flag: bool,
    redis_host: str,
    redis_port: int,
    publish_vector_store_flag: bool,
    qdrant_url: str,
    qdrant_collection: str,
    embedding_cache_dir: str,
) -> None:
    cards_df = pd.read_parquet(cards_file)
    card_features = build_card_features(
        cards_df, feature_version=feature_version, generated_at=generated_at
    )
    card_embeddings = build_card_embeddings(
        cards_df,
        embedding_version=feature_version,
        generated_at=generated_at,
        embedding_cache_dir=embedding_cache_dir,
    )
    card_role_tags = build_card_role_tags(
        cards_df, role_version=feature_version, generated_at=generated_at
    )
    validate_features(card_features=card_features, card_embeddings=card_embeddings)

    write_feature_artifacts(
        output_dir=Path(output_dir),
        card_features=card_features,
        card_embeddings=card_embeddings,
        card_role_tags=card_role_tags,
    )

    ns = argparse.Namespace(
        pg_host=pg_host,
        pg_port=pg_port,
        pg_dbname=pg_dbname,
        pg_user=pg_user,
        pg_password=pg_password,
        pg_schema=pg_schema,
    )
    if publish_postgres_flag:
        publish_postgres(ns, card_features, card_embeddings, card_role_tags)
    if publish_feature_store_flag:
        publish_feature_store_redis(redis_host, redis_port, card_features, card_role_tags)
    if publish_vector_store_flag:
        publish_vector_store_qdrant(qdrant_url, qdrant_collection, card_embeddings)


@pipeline
def feature_pipeline(
    cards_file: str,
    output_dir: str,
    feature_version: str,
    generated_at: str,
    publish_postgres_flag: bool,
    pg_host: str,
    pg_port: int,
    pg_dbname: str,
    pg_user: str,
    pg_password: str,
    pg_schema: str,
    publish_feature_store_flag: bool,
    redis_host: str,
    redis_port: int,
    publish_vector_store_flag: bool,
    qdrant_url: str,
    qdrant_collection: str,
    embedding_cache_dir: str,
) -> None:
    # Minimal lineage path
    cards_df = load_cards_step(cards_file=cards_file)
    build_feature_tables_step(
        cards_df=cards_df,
        feature_version=feature_version,
        generated_at=generated_at,
        embedding_cache_dir=embedding_cache_dir,
    )
    publish_outputs_step(
        cards_file=cards_file,
        output_dir=output_dir,
        feature_version=feature_version,
        generated_at=generated_at,
        publish_postgres_flag=publish_postgres_flag,
        pg_host=pg_host,
        pg_port=pg_port,
        pg_dbname=pg_dbname,
        pg_user=pg_user,
        pg_password=pg_password,
        pg_schema=pg_schema,
        publish_feature_store_flag=publish_feature_store_flag,
        redis_host=redis_host,
        redis_port=redis_port,
        publish_vector_store_flag=publish_vector_store_flag,
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
        embedding_cache_dir=embedding_cache_dir,
    )


def main() -> None:
    args = parse_args()
    cards_path = Path(args.cards_file)
    output_dir = Path(args.output_dir)

    generated_at = datetime.now(UTC).isoformat()
    feature_version = args.feature_version

    logger.info("Building phase-2 features from %s", cards_path)
    if args.disable_zenml:
        cards_df = pd.read_parquet(cards_path)
        card_features = build_card_features(
            cards_df, feature_version=feature_version, generated_at=generated_at
        )
        card_embeddings = build_card_embeddings(
            cards_df,
            embedding_version=feature_version,
            generated_at=generated_at,
            embedding_cache_dir=args.embedding_cache_dir,
        )
        card_role_tags = build_card_role_tags(
            cards_df, role_version=feature_version, generated_at=generated_at
        )
        validate_features(card_features=card_features, card_embeddings=card_embeddings)
        write_feature_artifacts(
            output_dir=output_dir,
            card_features=card_features,
            card_embeddings=card_embeddings,
            card_role_tags=card_role_tags,
        )
        if args.publish_postgres:
            publish_postgres(args, card_features, card_embeddings, card_role_tags)
        if args.publish_feature_store:
            publish_feature_store_redis(
                args.redis_host, args.redis_port, card_features, card_role_tags
            )
        if args.publish_vector_store:
            publish_vector_store_qdrant(args.qdrant_url, args.qdrant_collection, card_embeddings)
    else:
        feature_pipeline(
            cards_file=str(cards_path),
            output_dir=str(output_dir),
            feature_version=feature_version,
            generated_at=generated_at,
            publish_postgres_flag=args.publish_postgres,
            pg_host=args.pg_host,
            pg_port=args.pg_port,
            pg_dbname=args.pg_dbname,
            pg_user=args.pg_user,
            pg_password=args.pg_password,
            pg_schema=args.pg_schema,
            publish_feature_store_flag=args.publish_feature_store,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            publish_vector_store_flag=args.publish_vector_store,
            qdrant_url=args.qdrant_url,
            qdrant_collection=args.qdrant_collection,
            embedding_cache_dir=args.embedding_cache_dir,
        )

    logger.info("Phase 2 feature build complete")


if __name__ == "__main__":
    main()
