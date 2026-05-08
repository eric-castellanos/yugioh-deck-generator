from __future__ import annotations

import logging
from typing import Any

import pandas as pd

EXTRA_TYPE_TOKENS = ("fusion", "synchro", "xyz", "link")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _type_text(value: Any) -> str:
    return str(value or "").lower()


def build_candidate_pool(
    cards_df: pd.DataFrame,
    *,
    allow_card_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Building candidate pools from %d input cards", len(cards_df))
    df = cards_df.copy()
    if "id" not in df.columns:
        logger.error("Cannot build candidate pools: missing required `id` column")
        raise ValueError("cards dataframe requires `id` column")
    df["id"] = df["id"].astype(int)

    if allow_card_ids is not None:
        before = len(df)
        df = df[df["id"].isin(allow_card_ids)]
        logger.info(
            "Applied allowlist filter: %d -> %d cards (allowlist size=%d)",
            before,
            len(df),
            len(allow_card_ids),
        )

    type_text = df["type"].fillna("").astype(str).str.lower()
    extra_mask = type_text.str.contains("|".join(EXTRA_TYPE_TOKENS))
    main_mask = ~type_text.str.contains("token") & ~extra_mask

    main_df = df[main_mask].copy()
    extra_df = df[extra_mask].copy()
    logger.info(
        "Candidate pools built: main_candidates=%d extra_candidates=%d",
        len(main_df),
        len(extra_df),
    )
    return main_df, extra_df
