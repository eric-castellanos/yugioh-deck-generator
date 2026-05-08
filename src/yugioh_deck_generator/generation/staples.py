from __future__ import annotations

import logging
from random import Random

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def choose_staple_ids(
    *,
    rng: Random,
    p_staple: float,
    pool: list[dict],
    eligible_ids: set[int],
) -> list[int]:
    logger.info(
        "Selecting staples from pool_size=%d eligible_ids=%d p_staple=%.3f",
        len(pool),
        len(eligible_ids),
        p_staple,
    )
    if not pool or p_staple <= 0:
        logger.info("Skipping staple selection: empty pool or non-positive p_staple")
        return []

    picks: list[int] = []
    considered = 0
    for entry in pool:
        cid = int(entry.get("card_id", -1))
        if cid not in eligible_ids:
            continue
        considered += 1
        weight = float(entry.get("weight", 1.0))
        prob = max(0.0, min(1.0, p_staple * weight))
        if rng.random() < prob:
            picks.append(cid)
    logger.info(
        "Staple selection complete: considered=%d selected=%d",
        considered,
        len(picks),
    )
    return picks
