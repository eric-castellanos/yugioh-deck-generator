from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CardRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    type: str
    frameType: str | None = None
    desc: str
    race: str | None = None
    archetype: str | None = None
    atk: int | None = None
    def_: int | None = None
    level: int | None = None
    attribute: str | None = None
    scale: int | None = None
    linkval: int | None = None


class CardImageRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    card_id: int
    image_id: int
    image_url: str
    image_url_small: str
    image_url_cropped: str


class CardSetRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    card_id: int
    set_name: str
    set_code: str
    set_rarity: str
    set_rarity_code: str
    set_price: float


class CardPriceRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    card_id: int
    cardmarket_price: float | None = None
    tcgplayer_price: float | None = None
    ebay_price: float | None = None
    amazon_price: float | None = None
    coolstuffinc_price: float | None = None


class CardArchetypeRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    card_id: int
    archetype: str


class BanlistInfoRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    card_id: int
    ban_tcg: str | None = None
    ban_ocg: str | None = None
    ban_goat: str | None = None
    ban_edison: str | None = None


class CardMiscInfoRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    card_id: int
    beta_name: str | None = None
    views: int | None = None
    viewsweek: int | None = None
    upvotes: int | None = None
    downvotes: int | None = None
    formats: str | None = None
    tcg_date: str | None = None
    ocg_date: str | None = None
    konami_id: int | None = None
    has_effect: int | None = None
    md_rarity: str | None = None
