from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from psycopg import sql

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

PHASE1_TABLES = [
    "cards",
    "card_images",
    "card_sets",
    "card_prices",
    "card_archetypes",
    "banlist_info",
]

PRIMARY_KEYS: dict[str, list[str]] = {
    "cards": ["id"],
    "card_images": ["card_id", "image_id"],
    "card_sets": ["card_id", "set_code"],
    "card_prices": ["card_id"],
    "card_archetypes": ["card_id", "archetype"],
    "banlist_info": ["card_id"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load normalized Yu-Gi-Oh parquet tables into Postgres."
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing normalized parquet files.",
    )
    parser.add_argument(
        "--target-file",
        default=None,
        help=(
            "Optional path to a specific parquet table file (for example "
            "data/processed/cards.parquet). If set, only that table is loaded."
        ),
    )
    parser.add_argument("--host", default=os.getenv("POSTGRES_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("POSTGRES_PORT", "5432")))
    parser.add_argument("--dbname", default=os.getenv("POSTGRES_DB", "yugioh"))
    parser.add_argument("--user", default=os.getenv("POSTGRES_USER", "yugioh"))
    parser.add_argument("--password", default=os.getenv("POSTGRES_PASSWORD", "yugioh"))
    parser.add_argument(
        "--schema",
        default=os.getenv("POSTGRES_SCHEMA", "public"),
        help="Target postgres schema.",
    )
    parser.add_argument(
        "--mode",
        choices=["replace", "append"],
        default="replace",
        help="replace truncates target tables first; append only inserts.",
    )
    return parser.parse_args()


def _df_from_parquet(processed_dir: Path, table_name: str) -> pd.DataFrame:
    table_path = processed_dir / f"{table_name}.parquet"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing parquet table: {table_path}")

    df = pd.read_parquet(table_path)
    # Convert pandas NaN/NaT values into DB-friendly None.
    return df.where(pd.notnull(df), None)


def _df_from_target_file(target_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(target_path)
    return df.where(pd.notnull(df), None)


def create_tables(conn: psycopg.Connection[Any], schema: str) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};

    CREATE TABLE IF NOT EXISTS {schema}.cards (
        id BIGINT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        "frameType" TEXT,
        "desc" TEXT NOT NULL,
        race TEXT,
        archetype TEXT,
        atk INTEGER,
        "def" INTEGER,
        level INTEGER,
        attribute TEXT,
        scale INTEGER,
        linkval INTEGER,
        beta_name TEXT,
        views INTEGER,
        viewsweek INTEGER,
        upvotes INTEGER,
        downvotes INTEGER,
        formats TEXT,
        tcg_date TEXT,
        ocg_date TEXT,
        konami_id INTEGER,
        has_effect INTEGER,
        md_rarity TEXT
    );

    CREATE TABLE IF NOT EXISTS {schema}.card_images (
        card_id BIGINT NOT NULL REFERENCES {schema}.cards(id),
        image_id BIGINT NOT NULL,
        image_url TEXT NOT NULL,
        image_url_small TEXT NOT NULL,
        image_url_cropped TEXT NOT NULL,
        PRIMARY KEY (card_id, image_id)
    );

    CREATE TABLE IF NOT EXISTS {schema}.card_sets (
        card_id BIGINT NOT NULL REFERENCES {schema}.cards(id),
        set_name TEXT NOT NULL,
        set_code TEXT NOT NULL,
        set_rarity TEXT NOT NULL,
        set_rarity_code TEXT NOT NULL,
        set_price DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (card_id, set_code)
    );

    CREATE TABLE IF NOT EXISTS {schema}.card_prices (
        card_id BIGINT NOT NULL REFERENCES {schema}.cards(id),
        cardmarket_price DOUBLE PRECISION,
        tcgplayer_price DOUBLE PRECISION,
        ebay_price DOUBLE PRECISION,
        amazon_price DOUBLE PRECISION,
        coolstuffinc_price DOUBLE PRECISION,
        PRIMARY KEY (card_id)
    );

    CREATE TABLE IF NOT EXISTS {schema}.card_archetypes (
        card_id BIGINT NOT NULL REFERENCES {schema}.cards(id),
        archetype TEXT NOT NULL,
        PRIMARY KEY (card_id, archetype)
    );

    CREATE TABLE IF NOT EXISTS {schema}.banlist_info (
        card_id BIGINT NOT NULL REFERENCES {schema}.cards(id),
        ban_tcg TEXT,
        ban_ocg TEXT,
        ban_goat TEXT,
        ban_edison TEXT,
        PRIMARY KEY (card_id)
    );

    CREATE INDEX IF NOT EXISTS idx_cards_name ON {schema}.cards(name);
    CREATE INDEX IF NOT EXISTS idx_cards_archetype ON {schema}.cards(archetype);
    CREATE INDEX IF NOT EXISTS idx_card_images_card_id ON {schema}.card_images(card_id);
    CREATE INDEX IF NOT EXISTS idx_card_sets_card_id ON {schema}.card_sets(card_id);
    CREATE INDEX IF NOT EXISTS idx_card_archetypes_card_id ON {schema}.card_archetypes(card_id);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
        # Ensure legacy integer columns are widened for larger upstream ids/metrics.
        cur.execute(
            f"""
            ALTER TABLE {schema}.cards
                ALTER COLUMN id TYPE BIGINT,
                ALTER COLUMN atk TYPE BIGINT,
                ALTER COLUMN "def" TYPE BIGINT,
                ALTER COLUMN level TYPE BIGINT,
                ALTER COLUMN scale TYPE BIGINT,
                ALTER COLUMN linkval TYPE BIGINT,
                ALTER COLUMN views TYPE BIGINT,
                ALTER COLUMN viewsweek TYPE BIGINT,
                ALTER COLUMN upvotes TYPE BIGINT,
                ALTER COLUMN downvotes TYPE BIGINT,
                ALTER COLUMN konami_id TYPE BIGINT,
                ALTER COLUMN has_effect TYPE BIGINT;

            ALTER TABLE {schema}.card_images
                ALTER COLUMN card_id TYPE BIGINT,
                ALTER COLUMN image_id TYPE BIGINT;

            ALTER TABLE {schema}.card_sets
                ALTER COLUMN card_id TYPE BIGINT;

            ALTER TABLE {schema}.card_prices
                ALTER COLUMN card_id TYPE BIGINT;

            ALTER TABLE {schema}.card_archetypes
                ALTER COLUMN card_id TYPE BIGINT;

            ALTER TABLE {schema}.banlist_info
                ALTER COLUMN card_id TYPE BIGINT;
            """
        )


def truncate_tables(
    conn: psycopg.Connection[Any], schema: str, tables_to_truncate: list[str]
) -> None:
    ordered_tables = [
        "card_images",
        "card_sets",
        "card_prices",
        "card_archetypes",
        "banlist_info",
        "cards",
    ]
    requested = set(tables_to_truncate)
    # If cards is replaced, child tables must be truncated in the same statement
    # due to FK constraints.
    if "cards" in requested:
        requested.update(
            {"card_images", "card_sets", "card_prices", "card_archetypes", "banlist_info"}
        )

    tables = [table for table in ordered_tables if table in requested]
    if not tables:
        return

    table_refs = [
        sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table)) for table in tables
    ]
    with conn.cursor() as cur:
        cur.execute(sql.SQL("TRUNCATE TABLE {}").format(sql.SQL(", ").join(table_refs)))


def insert_dataframe(
    conn: psycopg.Connection[Any], schema: str, table: str, df: pd.DataFrame
) -> None:
    if df.empty:
        logger.info("Skipping %s: no rows", table)
        return

    columns = list(df.columns)
    placeholders = sql.SQL(", ").join([sql.Placeholder() for _ in columns])
    column_sql = sql.SQL(", ").join([sql.Identifier(col) for col in columns])
    insert_sql = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({})").format(
        sql.Identifier(schema),
        sql.Identifier(table),
        column_sql,
        placeholders,
    )

    records: list[tuple[Any, ...]] = []
    for row in df.itertuples(index=False, name=None):
        converted = tuple(None if pd.isna(value) else value for value in row)
        records.append(converted)

    with conn.cursor() as cur:
        cur.executemany(insert_sql, records)

    logger.info("Inserted %s rows into %s.%s", len(records), schema, table)


def _assert_numeric_ranges_fit_table(
    conn: psycopg.Connection[Any], schema: str, table: str, df: pd.DataFrame
) -> None:
    pg_ranges: dict[str, tuple[int, int]] = {
        "smallint": (-32768, 32767),
        "integer": (-2147483648, 2147483647),
        "bigint": (-9223372036854775808, 9223372036854775807),
    }

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        column_types = {row[0]: row[1] for row in cur.fetchall()}

    for column_name in df.columns:
        data_type = column_types.get(column_name)
        if data_type not in pg_ranges:
            continue

        series = pd.to_numeric(df[column_name], errors="coerce").dropna()
        if series.empty:
            continue

        observed_min = int(series.min())
        observed_max = int(series.max())
        allowed_min, allowed_max = pg_ranges[data_type]
        if observed_min < allowed_min or observed_max > allowed_max:
            raise ValueError(
                "Numeric range mismatch before insert for "
                f"{schema}.{table}.{column_name}: "
                f"postgres type={data_type} supports [{allowed_min}, {allowed_max}] "
                f"but data has [{observed_min}, {observed_max}]"
            )


def deduplicate_for_primary_key(table: str, df: pd.DataFrame) -> pd.DataFrame:
    pk_columns = PRIMARY_KEYS.get(table)
    if not pk_columns:
        return df

    before_count = len(df)
    deduped = df.drop_duplicates(subset=pk_columns, keep="first")
    dropped = before_count - len(deduped)
    if dropped > 0:
        logger.warning(
            "Dropped %s duplicate rows for %s using primary key columns %s",
            dropped,
            table,
            pk_columns,
        )
    return deduped


def validate_integrity(conn: psycopg.Connection[Any], schema: str) -> None:
    checks = [
        ("card_images", "card_id"),
        ("card_sets", "card_id"),
        ("card_prices", "card_id"),
        ("card_archetypes", "card_id"),
        ("banlist_info", "card_id"),
    ]

    with conn.cursor() as cur:
        for table, fk_col in checks:
            query = sql.SQL(
                """
                SELECT COUNT(*)
                FROM {}.{} child
                LEFT JOIN {}.cards parent
                    ON child.{} = parent.id
                WHERE parent.id IS NULL
                """
            ).format(
                sql.Identifier(schema),
                sql.Identifier(table),
                sql.Identifier(schema),
                sql.Identifier(fk_col),
            )
            cur.execute(query)
            orphan_count = cur.fetchone()[0]
            if orphan_count != 0:
                raise ValueError(f"Integrity check failed for {table}: {orphan_count} orphan rows")


def log_row_counts(conn: psycopg.Connection[Any], schema: str) -> None:
    with conn.cursor() as cur:
        for table in PHASE1_TABLES:
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
            count = cur.fetchone()[0]
            logger.info("Row count %s.%s = %s", schema, table, count)


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    if args.target_file:
        target_path = Path(args.target_file)
        if not target_path.exists():
            raise FileNotFoundError(f"Target parquet file does not exist: {target_path}")
        table_name = target_path.stem
        if table_name not in PHASE1_TABLES:
            raise ValueError(
                f"Unsupported table '{table_name}' from target file. "
                f"Expected one of: {', '.join(PHASE1_TABLES)}"
            )
        tables_to_load = [table_name]
        tables = {table_name: _df_from_target_file(target_path)}
    else:
        tables_to_load = PHASE1_TABLES
        tables = {name: _df_from_parquet(processed_dir, name) for name in tables_to_load}

    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )

    try:
        with conn.transaction():
            create_tables(conn, args.schema)
            if args.mode == "replace":
                truncate_tables(conn, args.schema, tables_to_load)
            for table_name in tables_to_load:
                deduped_df = deduplicate_for_primary_key(table_name, tables[table_name])
                _assert_numeric_ranges_fit_table(conn, args.schema, table_name, deduped_df)
                insert_dataframe(conn, args.schema, table_name, deduped_df)
            validate_integrity(conn, args.schema)
            log_row_counts(conn, args.schema)
        logger.info("Postgres load complete")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
