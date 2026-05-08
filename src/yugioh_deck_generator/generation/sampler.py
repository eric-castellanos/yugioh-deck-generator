from __future__ import annotations

import logging
from collections import Counter
from random import Random
from typing import Any

import pandas as pd

from yugioh_deck_generator.generation.material_requirements import (
    extract_required_monster_tag_counts,
    main_monster_matches_tag,
)
from yugioh_deck_generator.generation.novelty import pick_novelty_ids
from yugioh_deck_generator.generation.staples import choose_staple_ids

MAX_DEFAULT_COPIES = 3
PREFERRED_MIN_COPIES = 2
PREFERRED_MAX_COPIES = 3
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
FUSION_ENABLER_NAMES = {
    "polymerization",
    "fusion gate",
    "miracle fusion",
    "future fusion",
    "metamorphosis",
}


def _type_bucket(card_type: Any) -> str:
    t = str(card_type or "").lower()
    if "spell" in t:
        return "spell"
    if "trap" in t:
        return "trap"
    return "monster"


def _desired_main_copies(rng: Random) -> int:
    return rng.randint(PREFERRED_MIN_COPIES, PREFERRED_MAX_COPIES)


def _add_main_card(
    *,
    main_ids: list[int],
    counts: Counter[int],
    cid: int,
    main_size: int,
    desired_copies: int,
    card_copy_limits: dict[int, int] | None = None,
) -> int:
    max_copies = (
        int(card_copy_limits.get(cid, MAX_DEFAULT_COPIES))
        if card_copy_limits
        else MAX_DEFAULT_COPIES
    )
    if max_copies <= 0:
        return 0
    if counts[cid] >= max_copies:
        return 0
    to_add = min(
        desired_copies,
        max_copies - counts[cid],
        main_size - len(main_ids),
    )
    if to_add <= 0:
        return 0
    main_ids.extend([cid] * to_add)
    counts[cid] += to_add
    return to_add


def _sample_extra_first(
    *,
    rng: Random,
    extra_df: pd.DataFrame,
    extra_size: int,
    card_copy_limits: dict[int, int] | None,
) -> list[int]:
    extra_ids: list[int] = []
    extra_pool = extra_df["id"].astype(int).drop_duplicates().tolist()
    rng.shuffle(extra_pool)
    while len(extra_ids) < extra_size and extra_pool:
        cid = extra_pool.pop()
        max_copies = (
            int(card_copy_limits.get(cid, MAX_DEFAULT_COPIES))
            if card_copy_limits
            else MAX_DEFAULT_COPIES
        )
        if max_copies <= 0:
            continue
        if extra_ids.count(cid) >= max_copies:
            continue
        extra_ids.append(cid)
    return extra_ids


def _max_supported_tag_copies(
    *,
    main_monsters_df: pd.DataFrame,
    tag: str,
    card_copy_limits: dict[int, int] | None,
) -> int:
    total = 0
    for _, row in main_monsters_df.iterrows():
        if not main_monster_matches_tag(row, tag):
            continue
        cid = int(row["id"])
        total += (
            int(card_copy_limits.get(cid, MAX_DEFAULT_COPIES))
            if card_copy_limits
            else MAX_DEFAULT_COPIES
        )
    return total


def _sample_extra_with_material_feasibility(
    *,
    rng: Random,
    extra_df: pd.DataFrame,
    extra_size: int,
    main_monsters_df: pd.DataFrame,
    card_copy_limits: dict[int, int] | None,
    support_capacity_cache: dict[str, int] | None = None,
    max_attempts: int = 12,
) -> tuple[list[int], dict[str, int]]:
    best_extra: list[int] = []
    best_requirements: dict[str, int] = {}
    if support_capacity_cache is None:
        support_capacity_cache = {}
    extra_lookup = extra_df.set_index("id", drop=False)
    for _ in range(max_attempts):
        candidate_extra = _sample_extra_first(
            rng=rng,
            extra_df=extra_df,
            extra_size=extra_size,
            card_copy_limits=card_copy_limits,
        )
        extra_rows = [extra_lookup.loc[cid] for cid in candidate_extra if cid in extra_lookup.index]
        requirements = extract_required_monster_tag_counts(extra_rows)
        feasible = True
        for tag, required_count in requirements.items():
            if tag not in support_capacity_cache:
                support_capacity_cache[tag] = _max_supported_tag_copies(
                    main_monsters_df=main_monsters_df,
                    tag=tag,
                    card_copy_limits=card_copy_limits,
                )
            if support_capacity_cache[tag] < required_count:
                feasible = False
                break
        if feasible:
            return candidate_extra, requirements
        best_extra = candidate_extra
        best_requirements = requirements
    return best_extra, best_requirements


def prepare_sampling_inputs(
    *,
    main_df: pd.DataFrame,
    extra_df: pd.DataFrame,
) -> dict[str, Any]:
    main_monsters_df = main_df[
        main_df["type"].fillna("").astype(str).str.lower().str.contains("monster")
    ]
    main_by_type_base: dict[str, list[int]] = {"monster": [], "spell": [], "trap": []}
    id_to_bucket: dict[int, str] = {}
    for _, row in main_df.iterrows():
        bucket = _type_bucket(row.get("type"))
        cid = int(row["id"])
        main_by_type_base[bucket].append(cid)
        id_to_bucket[cid] = bucket
    all_main_pool_base = main_df["id"].astype(int).tolist()
    return {
        "eligible_main_ids": set(main_df["id"].astype(int).tolist()),
        "main_monsters_df": main_monsters_df,
        "main_effect_monsters_df": main_monsters_df[
            main_monsters_df["type"].fillna("").astype(str).str.lower().str.contains("effect")
        ],
        "main_tuners_df": main_monsters_df[
            main_monsters_df["type"].fillna("").astype(str).str.lower().str.contains("tuner")
        ],
        "main_non_tuners_df": main_monsters_df[
            ~main_monsters_df["type"].fillna("").astype(str).str.lower().str.contains("tuner")
        ],
        "main_pendulums_df": main_df[
            main_df["type"].fillna("").astype(str).str.lower().str.contains("pendulum")
        ],
        "spell_fusion_enablers": [
            int(cid)
            for cid, row in main_df.set_index("id", drop=False).iterrows()
            if str(row.get("name", "")).strip().lower() in FUSION_ENABLER_NAMES
        ],
        "main_by_type_base": main_by_type_base,
        "id_to_bucket": id_to_bucket,
        "all_main_pool_base": all_main_pool_base,
        "tag_candidate_cache": {},
        "support_capacity_cache": {},
        "novelty_cache": {},
        "extra_lookup": extra_df.set_index("id", drop=False),
        "extra_index_set": set(extra_df["id"].astype(int).tolist()),
    }


def sample_deck(
    *,
    rng: Random,
    main_df: pd.DataFrame,
    extra_df: pd.DataFrame,
    joined_df: pd.DataFrame,
    main_size: int,
    extra_size: int,
    ratios: dict[str, float],
    staple_pool: list[dict],
    p_staple: float,
    novelty_ratio: float,
    allowed_novelty_clusters: set[int],
    card_copy_limits: dict[int, int] | None = None,
    near_novelty_share: float = 0.67,
    far_novelty_share: float = 0.33,
    sampling_inputs: dict[str, Any] | None = None,
) -> tuple[list[int], list[int], dict[str, int]]:
    logger.info(
        "Sampling deck main_size=%d extra_size=%d p_staple=%.3f novelty_ratio=%.3f",
        main_size,
        extra_size,
        p_staple,
        novelty_ratio,
    )
    counts: Counter[int] = Counter()
    main_ids: list[int] = []
    prepared = sampling_inputs or prepare_sampling_inputs(main_df=main_df, extra_df=extra_df)
    eligible_main_ids = prepared["eligible_main_ids"]
    main_monsters_df = prepared["main_monsters_df"]
    main_effect_monsters_df = prepared["main_effect_monsters_df"]
    main_tuners_df = prepared["main_tuners_df"]
    main_non_tuners_df = prepared["main_non_tuners_df"]
    main_pendulums_df = prepared["main_pendulums_df"]
    spell_fusion_enablers = prepared["spell_fusion_enablers"]
    main_by_type_base = prepared["main_by_type_base"]
    id_to_bucket: dict[int, str] = prepared["id_to_bucket"]
    all_main_pool_base = prepared["all_main_pool_base"]
    tag_candidate_cache: dict[str, list[int]] = prepared["tag_candidate_cache"]
    support_capacity_cache: dict[str, int] = prepared["support_capacity_cache"]
    novelty_cache = prepared["novelty_cache"]
    extra_lookup = prepared["extra_lookup"]
    extra_index_set = prepared["extra_index_set"]
    type_counts = {"monster": 0, "spell": 0, "trap": 0}

    def _remaining_slots() -> int:
        return main_size - len(main_ids)

    def _add_card_with_tracking(cid: int, desired_copies: int) -> int:
        added = _add_main_card(
            main_ids=main_ids,
            counts=counts,
            cid=cid,
            main_size=main_size,
            desired_copies=desired_copies,
            card_copy_limits=card_copy_limits,
        )
        if added > 0:
            bucket = id_to_bucket.get(cid, "monster")
            type_counts[bucket] += added
        return added

    extra_ids, required_monster_tag_counts = _sample_extra_with_material_feasibility(
        rng=rng,
        extra_df=extra_df,
        extra_size=extra_size,
        main_monsters_df=main_monsters_df,
        card_copy_limits=card_copy_limits,
        support_capacity_cache=support_capacity_cache,
    )
    extra_types = [
        str(extra_lookup.at[cid, "type"]).lower() for cid in extra_ids if cid in extra_index_set
    ]
    needs_fusion = any("fusion" in t for t in extra_types)
    needs_synchro = any("synchro" in t for t in extra_types)
    needs_xyz = any("xyz" in t for t in extra_types)
    needs_link = any("link" in t for t in extra_types)
    needs_pendulum = any("pendulum" in t for t in extra_types)

    support_added = 0
    if needs_fusion and spell_fusion_enablers:
        cid = rng.choice(spell_fusion_enablers)
        support_added += _add_card_with_tracking(cid, 1)

    if needs_synchro:
        if not main_tuners_df.empty:
            cid = int(main_tuners_df.sample(n=1, random_state=rng.randint(0, 10**9))["id"].iloc[0])
            support_added += _add_card_with_tracking(cid, 1)
        if not main_non_tuners_df.empty:
            cid = int(
                main_non_tuners_df.sample(n=1, random_state=rng.randint(0, 10**9))["id"].iloc[0]
            )
            support_added += _add_card_with_tracking(cid, 1)

    if needs_xyz:
        by_level = main_monsters_df.copy()
        if "level" in by_level.columns:
            by_level["level"] = pd.to_numeric(by_level["level"], errors="coerce")
            level_counts = (
                by_level.dropna(subset=["level"])
                .groupby("level")["id"]
                .nunique()
                .sort_values(ascending=False)
            )
            if not level_counts.empty:
                chosen_level = level_counts.index[0]
                level_pool = (
                    by_level[by_level["level"] == chosen_level]["id"]
                    .astype(int)
                    .drop_duplicates()
                    .tolist()
                )
                rng.shuffle(level_pool)
                for cid in level_pool[:2]:
                    support_added += _add_card_with_tracking(cid, 1)

    if needs_link and not main_effect_monsters_df.empty:
        link_pool = main_effect_monsters_df["id"].astype(int).drop_duplicates().tolist()
        rng.shuffle(link_pool)
        for cid in link_pool[:2]:
            support_added += _add_card_with_tracking(cid, 1)

    if required_monster_tag_counts:
        for tag, required_count in sorted(required_monster_tag_counts.items()):
            candidates = tag_candidate_cache.get(tag)
            if candidates is None:
                candidates = [
                    int(row["id"])
                    for _, row in main_monsters_df.iterrows()
                    if main_monster_matches_tag(row, tag)
                ]
                tag_candidate_cache[tag] = candidates
            if not candidates:
                continue
            cid = rng.choice(candidates)
            support_added += _add_card_with_tracking(cid, max(1, int(required_count)))

    if needs_pendulum and not main_pendulums_df.empty:
        pend = main_pendulums_df.copy()
        pend["scale"] = pd.to_numeric(pend.get("scale"), errors="coerce")
        rows = pend.dropna(subset=["scale"])[["id", "scale"]].drop_duplicates()
        picked_pair: list[int] = []
        if len(rows) >= 2:
            vals = rows.to_dict("records")
            rng.shuffle(vals)
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    a = int(vals[i]["scale"])
                    b = int(vals[j]["scale"])
                    if abs(a - b) >= 2:
                        picked_pair = [int(vals[i]["id"]), int(vals[j]["id"])]
                        break
                if picked_pair:
                    break
        for cid in picked_pair[:2]:
            support_added += _add_card_with_tracking(cid, 1)
    logger.info(
        "Support injection based on extra deck added=%d fusion=%s synchro=%s "
        "xyz=%s link=%s pendulum=%s required_tags=%d",
        support_added,
        needs_fusion,
        needs_synchro,
        needs_xyz,
        needs_link,
        needs_pendulum,
        len(required_monster_tag_counts),
    )

    target_counts = {
        "monster": round(main_size * float(ratios.get("monster", 0.5))),
        "spell": round(main_size * float(ratios.get("spell", 0.3))),
    }
    target_counts["trap"] = main_size - target_counts["monster"] - target_counts["spell"]

    def _required_slots_to_meet_ratio() -> int:
        return sum(max(0, target_counts[b] - type_counts[b]) for b in ("monster", "spell", "trap"))

    logger.info(
        "Main type slot targets: monster=%d spell=%d trap=%d",
        target_counts["monster"],
        target_counts["spell"],
        target_counts["trap"],
    )

    main_by_type = {bucket: ids[:] for bucket, ids in main_by_type_base.items()}
    for bucket in ("monster", "spell", "trap"):
        rng.shuffle(main_by_type[bucket])

    staples = choose_staple_ids(
        rng=rng,
        p_staple=p_staple,
        pool=staple_pool,
        eligible_ids=eligible_main_ids,
    )
    staple_added = 0
    for cid in staples:
        if len(main_ids) >= main_size:
            break
        if _remaining_slots() <= _required_slots_to_meet_ratio():
            break
        staple_added += _add_card_with_tracking(cid, _desired_main_copies(rng))
    logger.info("Staple stage selected=%d copies_added=%d", len(staples), staple_added)

    novelty_ids = pick_novelty_ids(
        rng=rng,
        joined_df=joined_df,
        novelty_ratio=novelty_ratio,
        main_size=main_size,
        allowed_clusters=allowed_novelty_clusters,
        near_share=near_novelty_share,
        far_share=far_novelty_share,
        novelty_cache=novelty_cache,
    )
    novelty_added = 0
    for cid in novelty_ids:
        if len(main_ids) >= main_size:
            break
        if cid not in eligible_main_ids:
            continue
        if _remaining_slots() <= _required_slots_to_meet_ratio():
            break
        novelty_added += _add_card_with_tracking(cid, _desired_main_copies(rng))
    logger.info("Novelty stage selected=%d copies_added=%d", len(novelty_ids), novelty_added)

    for bucket in ("monster", "spell", "trap"):
        needed = max(0, target_counts[bucket] - type_counts[bucket])
        pool = main_by_type[bucket][:]
        before_bucket = len(main_ids)
        while needed > 0 and pool and len(main_ids) < main_size:
            cid = pool.pop()
            added = _add_card_with_tracking(cid, _desired_main_copies(rng))
            needed = max(0, needed - added)
        logger.info("Bucket fill `%s` added %d cards", bucket, len(main_ids) - before_bucket)

    all_main_pool = all_main_pool_base[:]
    rng.shuffle(all_main_pool)
    tail_added = 0
    while len(main_ids) < main_size and all_main_pool:
        cid = all_main_pool.pop()
        tail_added += _add_card_with_tracking(cid, _desired_main_copies(rng))
    logger.info("Tail fill added %d cards", tail_added)

    unique_main = len(set(main_ids))
    avg_main_copies = (len(main_ids) / unique_main) if unique_main else 0.0
    logger.info(
        "Sampling complete: main=%d unique_main=%d avg_copies=%.2f extra=%d",
        len(main_ids),
        unique_main,
        avg_main_copies,
        len(extra_ids),
    )

    diagnostics = {
        "num_staples_injected": len(staples),
        "num_novelty_cards": len(novelty_ids),
        "num_unique_main_cards": unique_main,
    }
    return main_ids, extra_ids, diagnostics
