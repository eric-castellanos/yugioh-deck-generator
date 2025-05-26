from datetime import datetime

import pygo_API
import polars as pl
import boto3

def flatten_yugioh_cards(cards: list[dict]) -> pl.DataFrame:
    return pl.from_dicts(
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
    )

def pull_data():
    # For all cards or a subset
    cards = pygo_API.Card().getData()
    df = flatten_yugioh_cards(cards)
    return df

def upload_to_s3(df, bucket: str, prefix: str):
    filename = f"yugioh_raw_{datetime.today().strftime('%Y-%m-%d')}.parquet"
    filepath = f"/tmp/{filename}"
    s3_key = f"{prefix}/{datetime.today().strftime('%Y-%m')}/{filename}"

    # Save locally
    df.write_parquet(filepath)

    # Upload to S3
    s3 = boto3.client("s3")
    s3.upload_file(filepath, bucket, s3_key)
    print(f"Uploaded to s3://{bucket}/{s3_key}")

if __name__ == "__main__":
    tcg_df = pull_data()
    upload_to_s3(tcg_df, "yugioh-data", "raw")
