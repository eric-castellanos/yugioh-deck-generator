from __future__ import annotations

import logging
from random import Random

import pandas as pd

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def pick_novelty_ids(
    *,
    rng: Random,
    joined_df: pd.DataFrame,
    novelty_ratio: float,
    main_size: int,
    allowed_clusters: set[int],
    near_share: float = 0.67,
    far_share: float = 0.33,
    novelty_cache: dict[tuple[int, ...], dict[str, list[int]]] | None = None,
) -> list[int]:
    logger.info(
        "Selecting novelty ids novelty_ratio=%.3f main_size=%d allowed_clusters=%d",
        novelty_ratio,
        main_size,
        len(allowed_clusters),
    )
    target = max(0, round(main_size * novelty_ratio))
    if target == 0 or joined_df.empty:
        logger.info(
            "Skipping novelty selection: target=%d joined_df_empty=%s", target, joined_df.empty
        )
        return []
    if "nearest_cluster_id" not in joined_df.columns:
        logger.warning("Skipping novelty selection: missing `nearest_cluster_id` column")
        return []

    total_share = near_share + far_share
    if total_share <= 0:
        near_share = 1.0
        far_share = 0.0
    else:
        near_share = near_share / total_share
        far_share = far_share / total_share

    near_target = round(target * near_share)
    far_target = max(0, target - near_target)

    cache_key = tuple(sorted(int(x) for x in allowed_clusters))
    cached = novelty_cache.get(cache_key) if novelty_cache is not None else None
    if cached is None:
        near_pool = joined_df[
            joined_df["nearest_cluster_id"].fillna(-999).astype(int).isin(allowed_clusters)
        ]
        near_ids_base = near_pool["id"].astype(int).drop_duplicates().tolist()

        far_pool = joined_df[
            ~joined_df["nearest_cluster_id"].fillna(-999).astype(int).isin(allowed_clusters)
        ].copy()
        if not far_pool.empty and "nearest_cluster_confidence" in far_pool.columns:
            # Lower confidence implies weaker nearest-cluster affinity => semantically farther.
            far_pool["nearest_cluster_confidence"] = (
                far_pool["nearest_cluster_confidence"].fillna(1.0).astype(float)
            )
            far_pool = far_pool.sort_values("nearest_cluster_confidence", ascending=True)
            far_ids_ranked_base = far_pool["id"].astype(int).drop_duplicates().tolist()
        else:
            far_ids_ranked_base = far_pool["id"].astype(int).drop_duplicates().tolist()
        cached = {
            "near_ids_base": near_ids_base,
            "far_ids_ranked_base": far_ids_ranked_base,
        }
        if novelty_cache is not None:
            novelty_cache[cache_key] = cached

    near_ids = cached["near_ids_base"][:]
    rng.shuffle(near_ids)
    selected_near = near_ids[:near_target]

    # Far novelty: candidates that are not near-cluster matches.
    far_ids_ranked = cached["far_ids_ranked_base"][:]
    rng.shuffle(far_ids_ranked)
    selected_far: list[int] = []
    for cid in far_ids_ranked:
        if len(selected_far) >= far_target:
            break
        if cid in selected_near:
            continue
        selected_far.append(cid)

    selected = selected_near + selected_far
    logger.info(
        "Novelty selection complete: near_target=%d near_selected=%d "
        "far_target=%d far_selected=%d total_selected=%d",
        near_target,
        len(selected_near),
        far_target,
        len(selected_far),
        len(selected),
    )
    return selected
