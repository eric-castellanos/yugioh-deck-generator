from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from yugioh_deck_generator.cleaning.schemas import (
    BanlistInfoRow,
    CardArchetypeRow,
    CardImageRow,
    CardMiscInfoRow,
    CardPriceRow,
    CardRow,
    CardSetRow,
)

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize YGOPRODeck raw JSON into structured Parquet tables."
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing raw JSON files.")
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional path to a specific raw JSON file. If omitted, latest file in raw-dir is used.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where normalized parquet files are written.",
    )
    return parser.parse_args()


def latest_raw_file(raw_dir: Path) -> Path:
    files = list(raw_dir.glob("ygoprodeck_cards_*.json"))
    if not files:
        raise FileNotFoundError(f"No raw YGOPRODeck files found in {raw_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def normalize_payload(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cards = payload.get("data")
    if not isinstance(cards, list):
        raise ValueError("Raw payload missing `data` list")

    card_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    set_rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []
    archetype_rows: list[dict[str, Any]] = []
    banlist_rows: list[dict[str, Any]] = []
    misc_info_rows: list[dict[str, Any]] = []

    for card in cards:
        card_id = int(card["id"])

        card_row = CardRow(
            id=card_id,
            name=card["name"],
            type=card["type"],
            frameType=card.get("frameType"),
            desc=card.get("desc", ""),
            race=card.get("race"),
            archetype=card.get("archetype"),
            atk=card.get("atk"),
            def_=card.get("def"),
            level=card.get("level"),
            attribute=card.get("attribute"),
            scale=card.get("scale"),
            linkval=card.get("linkval"),
        )
        card_rows.append(
            {
                "id": card_row.id,
                "name": card_row.name,
                "type": card_row.type,
                "frameType": card_row.frameType,
                "desc": card_row.desc,
                "race": card_row.race,
                "archetype": card_row.archetype,
                "atk": card_row.atk,
                "def": card_row.def_,
                "level": card_row.level,
                "attribute": card_row.attribute,
                "scale": card_row.scale,
                "linkval": card_row.linkval,
            }
        )

        if card.get("archetype"):
            archetype_rows.append(CardArchetypeRow(card_id=card_id, archetype=card["archetype"]).model_dump())

        for image in card.get("card_images", []):
            image_rows.append(
                CardImageRow(
                    card_id=card_id,
                    image_id=image["id"],
                    image_url=image["image_url"],
                    image_url_small=image["image_url_small"],
                    image_url_cropped=image["image_url_cropped"],
                ).model_dump()
            )

        for card_set in card.get("card_sets", []):
            set_rows.append(
                CardSetRow(
                    card_id=card_id,
                    set_name=card_set["set_name"],
                    set_code=card_set["set_code"],
                    set_rarity=card_set["set_rarity"],
                    set_rarity_code=card_set["set_rarity_code"],
                    set_price=float(card_set["set_price"]),
                ).model_dump()
            )

        for price in card.get("card_prices", []):
            price_rows.append(
                CardPriceRow(
                    card_id=card_id,
                    cardmarket_price=_to_float(price.get("cardmarket_price")),
                    tcgplayer_price=_to_float(price.get("tcgplayer_price")),
                    ebay_price=_to_float(price.get("ebay_price")),
                    amazon_price=_to_float(price.get("amazon_price")),
                    coolstuffinc_price=_to_float(price.get("coolstuffinc_price")),
                ).model_dump()
            )

        if card.get("banlist_info"):
            banlist_rows.append(
                BanlistInfoRow(
                    card_id=card_id,
                    ban_tcg=card["banlist_info"].get("ban_tcg"),
                    ban_ocg=card["banlist_info"].get("ban_ocg"),
                    ban_goat=card["banlist_info"].get("ban_goat"),
                    ban_edison=card["banlist_info"].get("ban_edison"),
                ).model_dump()
            )

        for misc in card.get("misc_info", []):
            formats = misc.get("formats")
            misc_info_rows.append(
                CardMiscInfoRow(
                    card_id=card_id,
                    beta_name=misc.get("beta_name"),
                    views=misc.get("views"),
                    viewsweek=misc.get("viewsweek"),
                    upvotes=misc.get("upvotes"),
                    downvotes=misc.get("downvotes"),
                    formats="|".join(formats) if isinstance(formats, list) else None,
                    tcg_date=misc.get("tcg_date"),
                    ocg_date=misc.get("ocg_date"),
                    konami_id=misc.get("konami_id"),
                    has_effect=misc.get("has_effect"),
                    md_rarity=misc.get("md_rarity"),
                ).model_dump()
            )

    return {
        "cards": card_rows,
        "card_images": image_rows,
        "card_sets": set_rows,
        "card_prices": price_rows,
        "card_archetypes": archetype_rows,
        "banlist_info": banlist_rows,
        "card_misc_info": misc_info_rows,
    }


def write_parquet_tables(tables: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for table_name, rows in tables.items():
        table_path = output_dir / f"{table_name}.parquet"
        df = pd.DataFrame(rows)
        df.to_parquet(table_path, index=False)
        logger.info("Wrote %s rows to %s", len(df), table_path)


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    input_path = Path(args.input_file) if args.input_file else latest_raw_file(raw_dir)
    output_dir = Path(args.output_dir)

    logger.info("Normalizing source file: %s", input_path)
    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    tables = normalize_payload(payload)
    write_parquet_tables(tables=tables, output_dir=output_dir)
    logger.info("Normalization complete")


if __name__ == "__main__":
    main()
