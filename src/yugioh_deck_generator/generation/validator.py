from __future__ import annotations

import pandas as pd

from yugioh_deck_generator.generation.constraints import validate_deck


def validate_generated_deck(
    *,
    cards_df: pd.DataFrame,
    main_ids: list[int],
    extra_ids: list[int],
    side_ids: list[int],
    main_size: int,
    extra_min: int,
    extra_max: int,
    banlist: dict[str, list[int]] | None,
    ratios: dict[str, float],
    tolerance_count: int,
    card_lookup: pd.DataFrame | None = None,
    known_ids: set[int] | None = None,
) -> tuple[bool, dict[str, bool], list[str]]:
    return validate_deck(
        main_ids=main_ids,
        extra_ids=extra_ids,
        side_ids=side_ids,
        cards_df=cards_df,
        main_size=main_size,
        extra_min=extra_min,
        extra_max=extra_max,
        banlist=banlist,
        ratio_targets=ratios,
        tolerance_count=tolerance_count,
        card_lookup=card_lookup,
        known_ids=known_ids,
    )
