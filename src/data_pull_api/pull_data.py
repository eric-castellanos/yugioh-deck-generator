from datetime import datetime
import logging
import sys
import requests

import pygo_API
import polars as pl
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def flatten_yugioh_cards(cards: list[dict]) -> pl.DataFrame:
    logging.info("Flattening card data into DataFrame.")

    return pl.from_dicts([
        {
            "id": c.get("id"),
            "name": c.get("name"),
            "type": c.get("type"),
            "desc": c.get("desc"),
            "atk": c.get("atk"),
            "def": c.get("def"),
            "level": c.get("level"),
            "attribute": c.get("attribute"),
            "archetype": c.get("archetype"),
            "tcgplayer_price": float(c.get("card_prices", [{}])[0].get("tcgplayer_price", 0)),
            "ebay_price": float(c.get("card_prices", [{}])[0].get("ebay_price", 0)),
            "amazon_price": float(c.get("card_prices", [{}])[0].get("amazon_price", 0)),
            "coolstuffinc_price": float(c.get("card_prices", [{}])[0].get("coolstuffinc_price", 0)),
            "image_url": c.get("card_images", [{}])[0].get("image_url"),
            "image_url_small": c.get("card_images", [{}])[0].get("image_url_small"),
            "image_url_cropped": c.get("card_images", [{}])[0].get("image_url_cropped"),
        }
        for c in cards
    ])

def pull_data():
    try:
        logging.info("Fetching Yu-Gi-Oh card data...")
        cards = pygo_API.Card().getData()
        logging.info("Data fetched. Flattening...")
        return flatten_yugioh_cards(cards)
    except requests.RequestException as e:
        logging.error(f"Request failed: {type(e).__name__} - {e}")
        raise

def upload_to_s3(df, bucket: str, prefix: str):
    filename = f"yugioh_raw_{datetime.today().strftime('%Y-%m-%d')}.parquet"
    filepath = f"/tmp/{filename}"
    s3_key = f"{prefix}/{datetime.today().strftime('%Y-%m')}/{filename}"

    # Save locally
    df.write_parquet(filepath)

    # Upload to S3
    try:
        logging.info(f"Uploading {filepath} to s3://{bucket}/{s3_key}")
        s3 = boto3.client("s3")
        s3.upload_file(filepath, bucket, s3_key)
        logging.info(f"Successfully uploaded to s3://{bucket}/{s3_key}")
    except (BotoCoreError, ClientError) as e:
        logging.exception(f"Failed to upload {filepath} to S3 bucket '{bucket}' with key '{s3_key}'")
        raise

if __name__ == "__main__":
    tcg_df = pull_data()
    upload_to_s3(tcg_df, "yugioh-data", "raw")
