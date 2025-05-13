import pygo_API
import polars as pl

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

tcg_df = pull_data()
print(tcg_df.head(10))
print(len(tcg_df))
