from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

import pandas as pd

from yugioh_deck_generator.generation.material_requirements import (
    extract_required_monster_tag_counts,
    format_requirement_label,
    main_monster_matches_tag,
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

EXTRA_ALLOWED_TYPES = ("fusion", "synchro", "xyz", "link", "pendulum")
EXTRA_ONLY_TYPES = ("fusion", "synchro", "xyz", "link")
FUSION_ENABLERS = {
    "polymerization",
    "fusion gate",
    "miracle fusion",
    "future fusion",
    "metamorphosis",
}
TUNER_REQUIREMENT_PATTERN = re.compile(r"(?<!non[\s-])\btuners?\b", re.IGNORECASE)


def _type_text(card_type: Any) -> str:
    return str(card_type or "").lower()


def _name_text(name: Any) -> str:
    return str(name or "").strip().lower()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _merged_limits_from_banlist(
    banlist: dict[str, list[int]] | None,
    default_max: int = 3,
) -> dict[int, int]:
    limits: dict[int, int] = {}
    if not banlist:
        return limits
    for cid in banlist.get("forbidden", []):
        limits[int(cid)] = 0
    for cid in banlist.get("limited", []):
        limits[int(cid)] = 1
    for cid in banlist.get("semi_limited", []):
        limits[int(cid)] = 2
    return limits


def validate_deck(
    *,
    main_ids: list[int],
    extra_ids: list[int],
    side_ids: list[int],
    cards_df: pd.DataFrame,
    main_size: int,
    extra_min: int,
    extra_max: int,
    max_default_copies: int = 3,
    banlist: dict[str, list[int]] | None = None,
    ratio_targets: dict[str, float] | None = None,
    tolerance_count: int = 2,
    card_lookup: pd.DataFrame | None = None,
    known_ids: set[int] | None = None,
    tcg_release_cutoff: str | None = None,
) -> tuple[bool, dict[str, bool], list[str]]:
    logger.info(
        "Validating deck main=%d extra=%d side=%d main_size=%d extra_range=[%d,%d]",
        len(main_ids),
        len(extra_ids),
        len(side_ids),
        main_size,
        extra_min,
        extra_max,
    )
    errors: list[str] = []
    flags: dict[str, bool] = {}

    flags["main_size_ok"] = len(main_ids) == main_size
    if not flags["main_size_ok"]:
        errors.append(f"main size {len(main_ids)} != {main_size}")
        logger.info("Main size validation failed: %s", errors[-1])

    flags["extra_size_ok"] = extra_min <= len(extra_ids) <= extra_max
    if not flags["extra_size_ok"]:
        errors.append(f"extra size {len(extra_ids)} outside [{extra_min},{extra_max}]")
        logger.info("Extra size validation failed: %s", errors[-1])

    all_ids = main_ids + extra_ids + side_ids
    lookup = card_lookup if card_lookup is not None else cards_df.set_index("id", drop=False)
    known = known_ids if known_ids is not None else set(int(x) for x in lookup.index.tolist())
    flags["cards_exist_ok"] = all(cid in known for cid in all_ids)
    if not flags["cards_exist_ok"]:
        errors.append("unknown card id present")
        logger.info("Card existence validation failed")

    format_release_ok = True
    if tcg_release_cutoff:
        cutoff_ts = pd.to_datetime(tcg_release_cutoff, errors="coerce")
        if pd.isna(cutoff_ts):
            logger.warning("Invalid tcg_release_cutoff=%s; skipping date validation", tcg_release_cutoff)
        elif "tcg_date" not in lookup.columns:
            format_release_ok = False
            errors.append("format tcg cutoff set but cards data has no tcg_date column")
        else:
            for cid in all_ids:
                if cid not in known:
                    continue
                release_date = pd.to_datetime(lookup.at[cid, "tcg_date"], errors="coerce")
                if pd.isna(release_date):
                    format_release_ok = False
                    errors.append(f"card {cid} has missing/invalid tcg_date for cutoff {tcg_release_cutoff}")
                    continue
                if release_date > cutoff_ts:
                    format_release_ok = False
                    errors.append(
                        f"card {cid} tcg_date {release_date.date().isoformat()} exceeds cutoff {tcg_release_cutoff}"
                    )
    flags["format_release_ok"] = format_release_ok
    if not format_release_ok:
        logger.info("Format release date validation failed")

    limits = _merged_limits_from_banlist(banlist, max_default_copies)
    merged_counts = Counter(all_ids)
    copy_ok = True
    for cid, count in merged_counts.items():
        allowed = limits.get(cid, max_default_copies)
        if count > allowed:
            copy_ok = False
            errors.append(f"card {cid} has {count} copies > allowed {allowed}")
    flags["max_copies_ok"] = copy_ok
    if not copy_ok:
        logger.info("Copy limit validation failed")

    extra_ok = True
    fusion_present = False
    synchro_present = False
    xyz_present = False
    link_present = False
    pendulum_present = False
    for cid in extra_ids:
        if cid not in known:
            continue
        t = _type_text(lookup.at[cid, "type"])
        if not any(token in t for token in EXTRA_ALLOWED_TYPES):
            extra_ok = False
            errors.append(f"card {cid} in extra deck has invalid type `{t}`")
        if "fusion" in t:
            fusion_present = True
        if "synchro" in t:
            synchro_present = True
        if "xyz" in t:
            xyz_present = True
        if "link" in t:
            link_present = True
        if "pendulum" in t:
            pendulum_present = True
    flags["extra_type_ok"] = extra_ok
    if not extra_ok:
        logger.info("Extra deck type validation failed")

    main_rows = [lookup.loc[cid] for cid in main_ids if cid in known]
    extra_rows = [lookup.loc[cid] for cid in extra_ids if cid in known]
    main_extra_type_ok = True
    for row in main_rows:
        t = _type_text(row.get("type"))
        if any(token in t for token in EXTRA_ONLY_TYPES):
            main_extra_type_ok = False
            errors.append(f"main deck contains extra-deck-only type `{t}`")
            break
    flags["main_extra_type_ok"] = main_extra_type_ok
    if not main_extra_type_ok:
        logger.info("Main deck extra-only type validation failed")

    required_monster_tag_counts = extract_required_monster_tag_counts(extra_rows)
    material_requirements_ok = True
    for tag, required_count in sorted(required_monster_tag_counts.items()):
        found_count = sum(1 for row in main_rows if main_monster_matches_tag(row, tag))
        if found_count < required_count:
            material_requirements_ok = False
            errors.append(
                "missing required material monster tag "
                f"`{format_requirement_label(tag)}` copies: "
                f"required={required_count} found={found_count}"
            )
    flags["material_requirements_ok"] = material_requirements_ok
    if not material_requirements_ok:
        logger.info("Material requirement validation failed")

    fusion_path_ok = True
    if fusion_present:
        names = [_name_text(row.get("name")) for row in main_rows]
        if not any(name in FUSION_ENABLERS for name in names):
            fusion_path_ok = False
            errors.append("fusion monsters present without fusion enabler")
    flags["fusion_enabler_ok"] = fusion_path_ok
    if not fusion_path_ok:
        logger.info("Fusion path validation failed")

    synchro_path_ok = True
    if synchro_present:
        synchro_requires_tuner = False
        for row in extra_rows:
            if "synchro" not in _type_text(row.get("type")):
                continue
            desc_text = str(row.get("desc") or "")
            if TUNER_REQUIREMENT_PATTERN.search(desc_text):
                synchro_requires_tuner = True
                break
        main_monster_count = 0
        tuner_count = 0
        non_tuner_count = 0
        for row in main_rows:
            t = _type_text(row.get("type"))
            if "monster" not in t:
                continue
            main_monster_count += 1
            if "tuner" in t:
                tuner_count += 1
            else:
                non_tuner_count += 1
        if main_monster_count < 2 or (
            synchro_requires_tuner and (tuner_count < 1 or non_tuner_count < 1)
        ):
            synchro_path_ok = False
            errors.append("synchro monsters present without sufficient tuner/non-tuner support")
    flags["synchro_path_ok"] = synchro_path_ok
    if not synchro_path_ok:
        logger.info("Synchro path validation failed")

    xyz_path_ok = True
    if xyz_present:
        level_counts: Counter[int] = Counter()
        for row in main_rows:
            t = _type_text(row.get("type"))
            if "monster" not in t:
                continue
            level = _safe_int(row.get("level"))
            if level is not None and level > 0:
                level_counts[level] += 1
        if not any(count >= 2 for count in level_counts.values()):
            xyz_path_ok = False
            errors.append(
                "xyz monsters present without at least one duplicate main-deck level pair"
            )
    flags["xyz_path_ok"] = xyz_path_ok
    if not xyz_path_ok:
        logger.info("Xyz path validation failed")

    link_path_ok = True
    if link_present:
        effect_monster_count = sum(
            1
            for row in main_rows
            if "monster" in _type_text(row.get("type")) and "effect" in _type_text(row.get("type"))
        )
        if effect_monster_count < 2:
            link_path_ok = False
            errors.append("link monsters present without enough effect monsters for link material")
    flags["link_path_ok"] = link_path_ok
    if not link_path_ok:
        logger.info("Link path validation failed")

    pendulum_path_ok = True
    if pendulum_present:
        pendulum_scales: list[int] = []
        pendulum_count = 0
        for row in main_rows:
            t = _type_text(row.get("type"))
            if "pendulum" not in t:
                continue
            pendulum_count += 1
            scale = _safe_int(row.get("scale"))
            if scale is not None:
                pendulum_scales.append(scale)
        valid_scale_pair = False
        for left in pendulum_scales:
            for right in pendulum_scales:
                if left < right and (right - left) >= 2:
                    valid_scale_pair = True
                    break
            if valid_scale_pair:
                break
        if pendulum_count < 2 or not valid_scale_pair:
            pendulum_path_ok = False
            errors.append("pendulum monsters present without viable pendulum scale setup")
    flags["pendulum_path_ok"] = pendulum_path_ok
    if not pendulum_path_ok:
        logger.info("Pendulum path validation failed")

    ratio_ok = True
    if ratio_targets:
        monster = 0
        spell = 0
        trap = 0
        for cid in main_ids:
            if cid not in known:
                continue
            t = _type_text(lookup.at[cid, "type"])
            if "spell" in t:
                spell += 1
            elif "trap" in t:
                trap += 1
            else:
                monster += 1
        total = len(main_ids) if main_ids else 1
        actual = {
            "monster": monster / total,
            "spell": spell / total,
            "trap": trap / total,
        }
        for key in ("monster", "spell", "trap"):
            target = float(ratio_targets.get(key, actual[key]))
            if abs(round(actual[key] * total) - round(target * total)) > tolerance_count:
                ratio_ok = False
                errors.append(f"{key} ratio mismatch target={target:.3f} actual={actual[key]:.3f}")
    flags["type_ratio_ok"] = ratio_ok
    if not ratio_ok:
        logger.info("Type ratio validation failed")

    ok = all(flags.values())
    logger.info(
        "Deck validation complete status=%s failed_flags=%s",
        "pass" if ok else "fail",
        [name for name, passed in flags.items() if not passed],
    )
    return ok, flags, errors
