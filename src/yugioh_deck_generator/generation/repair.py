from __future__ import annotations

import logging
from random import Random

import pandas as pd

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def repair_deck(
    *,
    rng: Random,
    main_ids: list[int],
    extra_ids: list[int],
    main_size: int,
    extra_size: int,
    main_pool: pd.DataFrame,
    extra_pool: pd.DataFrame,
) -> tuple[list[int], list[int], int]:
    logger.info(
        "Repair start main=%d extra=%d target_main=%d target_extra=%d",
        len(main_ids),
        len(extra_ids),
        main_size,
        extra_size,
    )
    repairs = 0
    main = list(main_ids)
    extra = list(extra_ids)

    candidates = main_pool["id"].astype(int).tolist()
    rng.shuffle(candidates)
    main_filled = 0
    while len(main) < main_size and candidates:
        main.append(candidates.pop())
        repairs += 1
        main_filled += 1
    if main_filled > 0:
        logger.info("Main deck fill added %d cards", main_filled)
    if len(main) > main_size:
        removed = len(main) - main_size
        main = main[:main_size]
        repairs += 1
        logger.info("Main deck truncation removed %d cards", removed)

    extra_candidates = extra_pool["id"].astype(int).tolist()
    rng.shuffle(extra_candidates)
    extra_filled = 0
    while len(extra) < extra_size and extra_candidates:
        extra.append(extra_candidates.pop())
        repairs += 1
        extra_filled += 1
    if extra_filled > 0:
        logger.info("Extra deck fill added %d cards", extra_filled)
    if len(extra) > extra_size:
        removed = len(extra) - extra_size
        extra = extra[:extra_size]
        repairs += 1
        logger.info("Extra deck truncation removed %d cards", removed)

    logger.info("Repair complete main=%d extra=%d repairs=%d", len(main), len(extra), repairs)
    return main, extra, repairs
