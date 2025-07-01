import boto3
import io
import polars as pl
from pathlib import Path
import json
import uuid

from src.utils.s3_utils import read_parquet_from_s3

# ----------- CONFIGURATION ------------
S3_BUCKET = "yugioh-data"
S3_KEY = "processed/feature_engineered/deck_scoring/2025-06/feature_engineered.parquet"
YDK_OUTPUT_DIR = Path("decks/novel")
YDK_OUTPUT_DIR.mkdir(parents=True ,exist_ok=True)
# --------------------------------------

def extract_card_ids(deck: list[dict]) -> list[int]:
    """Extract card IDs from a list of card dictionaries."""
    if not deck:
        return []
    return [card["id"] for card in deck if isinstance(card, dict) and "id" in card]


def write_ydk(main_ids: list[int], extra_ids: list[int], filename: str):
    """Write a YDK file with main and extra deck card IDs."""
    ydk_path = YDK_OUTPUT_DIR / f"{filename}.ydk"
    with open(ydk_path, "w") as f:
        f.write("#created by edopro_duel_runner\n#main\n")
        for cid in main_ids:
            f.write(f"{cid}\n")
        f.write("#extra\n")
        for cid in extra_ids:
            f.write(f"{cid}\n")
        f.write("#side\n")
    print(f"Wrote: {ydk_path}")


def main():
    df = read_parquet_from_s3(S3_BUCKET, S3_KEY)

    if "main_deck" not in df.columns or "extra_deck" not in df.columns:
        raise ValueError("Expected columns 'main_deck' and 'extra_deck' in the DataFrame.")

    # Convert JSON strings to lists if needed
    if df["main_deck"].dtype == pl.Utf8:
        df = df.with_columns([
            pl.col("main_deck").apply(json.loads).alias("main_deck"),
            pl.col("extra_deck").apply(json.loads).alias("extra_deck")
        ])

    for i, row in enumerate(df.iter_rows(named=True)):
        main_ids = extract_card_ids(row["main_deck"])
        extra_ids = extract_card_ids(row["extra_deck"])
        filename = f"deck_{i:03d}_{uuid.uuid4().hex[:6]}"
        write_ydk(main_ids, extra_ids, filename)


if __name__ == "__main__":
    main()
