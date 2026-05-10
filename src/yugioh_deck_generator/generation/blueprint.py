from __future__ import annotations

import logging
from datetime import UTC, datetime
from random import Random
from uuid import uuid4

import pandas as pd

from yugioh_deck_generator.generation.schemas import DeckSpec, FormatConfig, RatioConfig

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _jitter_ratios(base: RatioConfig, rng: Random, jitter: float) -> RatioConfig:
    if jitter <= 0:
        return RatioConfig(
            monster=float(base.monster),
            spell=float(base.spell),
            trap=float(base.trap),
            tolerance_count=int(base.tolerance_count),
        )

    for _ in range(200):
        monster = float(base.monster) + rng.uniform(-jitter, jitter)
        spell = float(base.spell) + rng.uniform(-jitter, jitter)
        trap = 1.0 - monster - spell
        if not (0.0 <= monster <= 1.0 and 0.0 <= spell <= 1.0 and 0.0 <= trap <= 1.0):
            continue
        if abs(monster - float(base.monster)) > jitter:
            continue
        if abs(spell - float(base.spell)) > jitter:
            continue
        if abs(trap - float(base.trap)) > jitter:
            continue
        return RatioConfig(
            monster=monster,
            spell=spell,
            trap=trap,
            tolerance_count=int(base.tolerance_count),
        )

    logger.warning(
        "Unable to apply ratio jitter=%.3f within constraints; using base ratios", jitter
    )
    return RatioConfig(
        monster=float(base.monster),
        spell=float(base.spell),
        trap=float(base.trap),
        tolerance_count=int(base.tolerance_count),
    )


def build_blueprint(
    *,
    fmt: FormatConfig,
    strategy_type: str,
    seed: int,
    p_staple: float,
    novelty_ratio: float,
    clusters_df: pd.DataFrame | None = None,
) -> DeckSpec:
    logger.info(
        "Building blueprint format=%s strategy=%s seed=%d p_staple=%.3f novelty_ratio=%.3f",
        fmt.name,
        strategy_type,
        seed,
        p_staple,
        novelty_ratio,
    )
    rng = Random(seed)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    primary_cluster: int | None = None
    if clusters_df is not None and "cluster_id" in clusters_df.columns and not clusters_df.empty:
        counts = (
            clusters_df["cluster_id"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_values(ascending=False)
        )
        if not counts.empty:
            primary_cluster = int(counts.index[0])
            logger.info("Primary cluster selected: %d", primary_cluster)
    else:
        logger.info("No usable cluster dataframe provided; primary cluster remains unset")

    secondary_clusters: list[int] = []
    if primary_cluster is not None and clusters_df is not None:
        all_clusters = sorted(
            c for c in clusters_df["cluster_id"].dropna().astype(int).unique().tolist() if c >= 0
        )
        pool = [c for c in all_clusters if c != primary_cluster]
        rng.shuffle(pool)
        secondary_clusters = pool[:2]
        logger.info(
            "Secondary clusters selected: %s (pool_size=%d)",
            secondary_clusters,
            len(pool),
        )

    base_ratios = RatioConfig(
        monster=float(fmt.card_type_ratios.monster),
        spell=float(fmt.card_type_ratios.spell),
        trap=float(fmt.card_type_ratios.trap),
        tolerance_count=int(fmt.card_type_ratios.tolerance_count),
    )
    generation_cfg = getattr(fmt, "generation", {}) or {}
    if not isinstance(generation_cfg, dict):
        generation_cfg = {}
    ratio_jitter = float(generation_cfg.get("ratio_jitter", 0.0))
    ratios = _jitter_ratios(base_ratios, rng, ratio_jitter)
    logger.info(
        "Blueprint ratios monster=%.3f spell=%.3f trap=%.3f tolerance=%d jitter=%.3f",
        ratios.monster,
        ratios.spell,
        ratios.trap,
        ratios.tolerance_count,
        ratio_jitter,
    )

    spec = DeckSpec(
        deck_spec_id=str(uuid4()),
        run_id=run_id,
        format=fmt.name,
        strategy_type=strategy_type,
        seed=seed,
        main_deck_size=fmt.main_deck_size,
        extra_deck_size=fmt.extra_deck_size,
        p_staple=p_staple,
        novelty_ratio=novelty_ratio,
        card_type_ratios=ratios,
        primary_cluster=primary_cluster,
        secondary_clusters=secondary_clusters,
    )
    logger.info(
        "Blueprint built deck_spec_id=%s run_id=%s main_size=%d extra_size=%d",
        spec.deck_spec_id,
        spec.run_id,
        spec.main_deck_size,
        spec.extra_deck_size,
    )
    return spec
