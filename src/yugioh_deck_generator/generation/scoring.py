from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd

from yugioh_deck_generator.generation.io import read_table, write_jsonl, write_summary

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


_WORKER_STATE: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score generated decks with heuristic and evaluator metrics."
    )
    parser.add_argument("--decks-file", required=True)
    parser.add_argument("--cards-file", required=True)
    parser.add_argument("--embeddings-file", default=None)
    parser.add_argument("--card-tags-file", default=None)
    parser.add_argument("--sim-trials", type=int, default=2000)
    parser.add_argument("--sim-seed", type=int, default=12345)
    parser.add_argument("--score-version", choices=("v1", "v2", "both"), default="both")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument(
        "--mlflow-experiment",
        default="yugioh-deck-generator/generation-heuristic-scoring",
    )
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def _is_monster(card_type: str) -> bool:
    return "monster" in card_type.lower()


def _is_spell(card_type: str) -> bool:
    return "spell" in card_type.lower()


def _is_trap(card_type: str) -> bool:
    return "trap" in card_type.lower()


def _main_type_counts(main_ids: list[int], card_type_map: dict[int, str]) -> tuple[int, int, int]:
    monster = 0
    spell = 0
    trap = 0
    for card_id in main_ids:
        card_type = card_type_map.get(int(card_id), "")
        if _is_monster(card_type):
            monster += 1
        elif _is_spell(card_type):
            spell += 1
        elif _is_trap(card_type):
            trap += 1
    return monster, spell, trap


def _extract_int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except json.JSONDecodeError:
            return []
    return []


def _extract_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    parsed = value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, list):
        return None
    try:
        vec = [float(x) for x in parsed]
    except (TypeError, ValueError):
        return None
    if not vec:
        return None
    return vec


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_0_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _ratio_targets_for_format(fmt: Any) -> dict[str, float]:
    name = str(fmt or "").strip().lower()
    if name == "goat":
        return {"monster": 0.50, "spell": 0.35, "trap": 0.15}
    if name == "hat":
        return {"monster": 0.50, "spell": 0.25, "trap": 0.25}
    return {"monster": 0.50, "spell": 0.30, "trap": 0.20}


def _compute_balance_metrics(
    *,
    main_ids: list[int],
    card_type_map: dict[int, str],
    fmt: Any,
) -> dict[str, float]:
    monster_count, spell_count, trap_count = _main_type_counts(main_ids, card_type_map)
    known_type_count = monster_count + spell_count + trap_count
    monster_ratio = (monster_count / float(known_type_count)) if known_type_count else 0.0
    spell_ratio = (spell_count / float(known_type_count)) if known_type_count else 0.0
    trap_ratio = (trap_count / float(known_type_count)) if known_type_count else 0.0

    targets = _ratio_targets_for_format(fmt)
    l1_dist = (
        abs(monster_ratio - targets["monster"])
        + abs(spell_ratio - targets["spell"])
        + abs(trap_ratio - targets["trap"])
    )
    max_l1 = 2.0
    balance_score = _clamp_0_100(100.0 * (1.0 - (l1_dist / max_l1)))

    return {
        "monster_count": float(monster_count),
        "spell_count": float(spell_count),
        "trap_count": float(trap_count),
        "monster_ratio": float(monster_ratio),
        "spell_ratio": float(spell_ratio),
        "trap_ratio": float(trap_ratio),
        "type_ratio_distance_l1": float(l1_dist),
        "balance_score": float(balance_score),
    }


def _normalize_vector(vec: list[float]) -> list[float] | None:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0.0:
        return None
    return [x / norm for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b, strict=False)))


def _compute_synergy_metrics(
    *,
    main_ids: list[int],
    embedding_map: dict[int, list[float]],
) -> dict[str, float]:
    vectors: list[list[float]] = []
    for cid in main_ids:
        vec = embedding_map.get(int(cid))
        if vec is not None:
            vectors.append(vec)

    if len(vectors) < 2:
        return {
            "main_embedding_coverage": float(len(vectors) / float(len(main_ids) or 1)),
            "main_avg_pairwise_similarity": 0.0,
            "main_centroid_similarity": 0.0,
            "synergy_score": 50.0,
        }

    coverage = len(vectors) / float(len(main_ids) or 1)
    dim = len(vectors[0])
    centroid = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            centroid[i] += x
    centroid = [x / float(len(vectors)) for x in centroid]
    centroid = _normalize_vector(centroid) or centroid

    centroid_sims = [_dot(v, centroid) for v in vectors]
    centroid_mean = sum(centroid_sims) / float(len(centroid_sims))

    pair_sum = 0.0
    pair_count = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            pair_sum += _dot(vectors[i], vectors[j])
            pair_count += 1
    pairwise_mean = pair_sum / float(pair_count) if pair_count else centroid_mean

    pairwise_norm = max(0.0, min(1.0, (pairwise_mean + 1.0) / 2.0))
    centroid_norm = max(0.0, min(1.0, (centroid_mean + 1.0) / 2.0))
    synergy_score = _clamp_0_100(100.0 * ((0.7 * pairwise_norm) + (0.3 * centroid_norm)))

    return {
        "main_embedding_coverage": float(coverage),
        "main_avg_pairwise_similarity": float(pairwise_mean),
        "main_centroid_similarity": float(centroid_mean),
        "synergy_score": float(synergy_score),
    }


def _starter_like_from_text(desc: str, card_type: str) -> bool:
    d = desc.lower()
    t = card_type.lower()
    if "monster" in t and "add" in d and "deck" in d and "hand" in d:
        return True
    if "monster" in t and "special summon" in d and "from your hand" not in d:
        return True
    if "spell" in t and "add" in d and "deck" in d:
        return True
    return False


def _enabler_like_from_text(desc: str, card_type: str) -> bool:
    d = desc.lower()
    t = card_type.lower()
    if "special summon" in d:
        return True
    if "spell" in t and ("draw" in d or "add" in d):
        return True
    if "monster" in t and ("extender" in d or "summon" in d):
        return True
    return False


def _is_brick_card(
    *,
    cid: int,
    card_type: str,
    card_level: int | None,
    explicit_brick_ids: set[int],
) -> bool:
    if cid in explicit_brick_ids:
        return True
    t = card_type.lower()
    return "monster" in t and (card_level or 0) >= 7


def _simulate_opening_hand_metrics(
    *,
    main_ids: list[int],
    starter_ids: set[int],
    brick_ids: set[int],
    enabler_ids: set[int],
    card_type_map: dict[int, str],
    card_level_map: dict[int, int | None],
    sim_trials: int,
    sim_seed: int,
) -> dict[str, float]:
    if not main_ids:
        return {
            "n_trials": float(sim_trials),
            "hand_size": 5.0,
            "starter_hit_rate": 0.0,
            "brick_hand_rate": 1.0,
        }

    n = len(main_ids)
    hand_size = min(5, n)
    if hand_size <= 0:
        return {
            "n_trials": float(sim_trials),
            "hand_size": 5.0,
            "starter_hit_rate": 0.0,
            "brick_hand_rate": 1.0,
        }

    rng = random.Random(sim_seed)
    starter_hits = 0
    brick_hands = 0

    for _ in range(sim_trials):
        hand_idx = rng.sample(range(n), hand_size)
        hand = [int(main_ids[i]) for i in hand_idx]

        has_starter = any(cid in starter_ids for cid in hand)
        if has_starter:
            starter_hits += 1

        brick_cards_in_hand = 0
        enablers_in_hand = 0
        for cid in hand:
            card_type = card_type_map.get(cid, "")
            level = card_level_map.get(cid)
            if cid in enabler_ids:
                enablers_in_hand += 1
            if _is_brick_card(
                cid=cid,
                card_type=card_type,
                card_level=level,
                explicit_brick_ids=brick_ids,
            ):
                brick_cards_in_hand += 1

        is_brick_hand = (not has_starter) or (brick_cards_in_hand >= 3 and enablers_in_hand == 0)
        if is_brick_hand:
            brick_hands += 1

    starter_hit_rate = starter_hits / float(sim_trials)
    brick_hand_rate = brick_hands / float(sim_trials)
    return {
        "n_trials": float(sim_trials),
        "hand_size": float(hand_size),
        "starter_hit_rate": float(starter_hit_rate),
        "brick_hand_rate": float(brick_hand_rate),
    }


def _compute_consistency_and_brick_metrics(
    *,
    main_ids: list[int],
    explicit_starter_ids: set[int],
    explicit_brick_ids: set[int],
    card_type_map: dict[int, str],
    card_desc_map: dict[int, str],
    card_level_map: dict[int, int | None],
    sim_trials: int,
    sim_seed: int,
) -> dict[str, float]:
    starter_ids: set[int] = set(explicit_starter_ids)
    enabler_ids: set[int] = set()

    for cid in set(main_ids):
        t = card_type_map.get(cid, "")
        d = card_desc_map.get(cid, "")
        if cid not in starter_ids and _starter_like_from_text(d, t):
            starter_ids.add(cid)
        if _enabler_like_from_text(d, t) or cid in starter_ids:
            enabler_ids.add(cid)

    sim = _simulate_opening_hand_metrics(
        main_ids=main_ids,
        starter_ids=starter_ids,
        brick_ids=explicit_brick_ids,
        enabler_ids=enabler_ids,
        card_type_map=card_type_map,
        card_level_map=card_level_map,
        sim_trials=max(1, int(sim_trials)),
        sim_seed=sim_seed,
    )

    consistency_score = _clamp_0_100(100.0 * sim["starter_hit_rate"])
    brick_risk_score = _clamp_0_100(100.0 * (1.0 - sim["brick_hand_rate"]))

    return {
        "starter_count": float(sum(1 for cid in main_ids if cid in starter_ids)),
        "starter_unique_count": float(len(starter_ids.intersection(set(main_ids)))),
        "starter_hit_rate": float(sim["starter_hit_rate"]),
        "brick_hand_rate": float(sim["brick_hand_rate"]),
        "consistency_score": float(consistency_score),
        "brick_risk_score": float(brick_risk_score),
        "sim_trials": float(sim["n_trials"]),
        "hand_size": float(sim["hand_size"]),
    }


def _compose_overall_score_v2(
    *,
    consistency_score: float,
    synergy_score: float,
    brick_risk_score: float,
    balance_score: float,
) -> tuple[float, dict[str, float]]:
    weights = {
        "consistency": 0.30,
        "synergy": 0.30,
        "brick_risk": 0.20,
        "balance": 0.20,
    }
    overall = (
        (weights["consistency"] * consistency_score)
        + (weights["synergy"] * synergy_score)
        + (weights["brick_risk"] * brick_risk_score)
        + (weights["balance"] * balance_score)
    )
    return float(_clamp_0_100(overall)), weights


def _score_one_deck(
    *,
    deck: dict[str, Any],
    card_type_map: dict[int, str],
    card_desc_map: dict[int, str],
    card_level_map: dict[int, int | None],
    embedding_map: dict[int, list[float]],
    explicit_starter_ids: set[int],
    explicit_brick_ids: set[int],
    sim_trials: int,
    sim_seed: int,
) -> dict[str, Any]:
    main_ids = _extract_int_list(deck.get("main"))
    extra_ids = _extract_int_list(deck.get("extra"))
    diagnostics = deck.get("diagnostics") or {}
    constraint_flags = deck.get("constraint_flags") or {}
    errors = deck.get("errors") or []

    main_size = len(main_ids)
    extra_size = len(extra_ids)
    unique_main = len(set(main_ids))
    unique_ratio = (unique_main / float(main_size)) if main_size else 0.0

    balance = _compute_balance_metrics(
        main_ids=main_ids,
        card_type_map=card_type_map,
        fmt=deck.get("format"),
    )
    ratio_penalty = balance["type_ratio_distance_l1"]

    staples = int(diagnostics.get("num_staples_injected", 0) or 0)
    novelty = int(diagnostics.get("num_novelty_cards", 0) or 0)
    repairs = int(diagnostics.get("num_repairs", 0) or 0)
    resamples = int(diagnostics.get("num_resamples", 0) or 0)
    accepted_flags = sum(1 for _, passed in dict(constraint_flags).items() if bool(passed))
    total_flags = len(dict(constraint_flags))

    heuristic_score_v1 = (
        100.0 + (15.0 * novelty) + (1.5 * staples) + (12.0 * unique_ratio) - (30.0 * ratio_penalty)
    )

    consistency_brick = _compute_consistency_and_brick_metrics(
        main_ids=main_ids,
        explicit_starter_ids=explicit_starter_ids,
        explicit_brick_ids=explicit_brick_ids,
        card_type_map=card_type_map,
        card_desc_map=card_desc_map,
        card_level_map=card_level_map,
        sim_trials=sim_trials,
        sim_seed=sim_seed,
    )
    synergy = _compute_synergy_metrics(main_ids=main_ids, embedding_map=embedding_map)
    heuristic_score_v2, component_weights = _compose_overall_score_v2(
        consistency_score=consistency_brick["consistency_score"],
        synergy_score=synergy["synergy_score"],
        brick_risk_score=consistency_brick["brick_risk_score"],
        balance_score=balance["balance_score"],
    )

    evaluation_v2 = {
        "overall_score": float(heuristic_score_v2),
        "consistency_score": float(consistency_brick["consistency_score"]),
        "synergy_score": float(synergy["synergy_score"]),
        "brick_risk_score": float(consistency_brick["brick_risk_score"]),
        "balance_score": float(balance["balance_score"]),
        "component_weights": component_weights,
        "simulator": {
            "n_trials": int(consistency_brick["sim_trials"]),
            "hand_size": int(consistency_brick["hand_size"]),
            "starter_hit_rate": float(consistency_brick["starter_hit_rate"]),
            "brick_hand_rate": float(consistency_brick["brick_hand_rate"]),
            "seed": int(sim_seed),
        },
        "features": {
            "starter_count": int(consistency_brick["starter_count"]),
            "starter_unique_count": int(consistency_brick["starter_unique_count"]),
            "main_embedding_coverage": float(synergy["main_embedding_coverage"]),
            "main_avg_pairwise_similarity": float(synergy["main_avg_pairwise_similarity"]),
            "main_centroid_similarity": float(synergy["main_centroid_similarity"]),
            "type_ratio_distance_l1": float(balance["type_ratio_distance_l1"]),
        },
    }

    return {
        "deck_id": deck.get("deck_id"),
        "deck_spec_id": deck.get("deck_spec_id"),
        "run_id": deck.get("run_id"),
        "format": deck.get("format"),
        "status": deck.get("status"),
        "heuristic_score_v1": float(heuristic_score_v1),
        "heuristic_score_v2": float(heuristic_score_v2),
        "main_size": main_size,
        "extra_size": extra_size,
        "unique_main_count": unique_main,
        "unique_main_ratio": float(unique_ratio),
        "monster_ratio": float(balance["monster_ratio"]),
        "spell_ratio": float(balance["spell_ratio"]),
        "trap_ratio": float(balance["trap_ratio"]),
        "num_staples_injected": staples,
        "num_novelty_cards": novelty,
        "num_repairs": repairs,
        "num_resamples": resamples,
        "constraint_flags_passed": int(accepted_flags),
        "constraint_flags_total": int(total_flags),
        "error_count": int(len(errors) if isinstance(errors, list) else 0),
        "consistency_score": float(consistency_brick["consistency_score"]),
        "synergy_score": float(synergy["synergy_score"]),
        "brick_risk_score": float(consistency_brick["brick_risk_score"]),
        "balance_score": float(balance["balance_score"]),
        "starter_hit_rate": float(consistency_brick["starter_hit_rate"]),
        "brick_hand_rate": float(consistency_brick["brick_hand_rate"]),
        "main_embedding_coverage": float(synergy["main_embedding_coverage"]),
        "main_avg_pairwise_similarity": float(synergy["main_avg_pairwise_similarity"]),
        "main_centroid_similarity": float(synergy["main_centroid_similarity"]),
        "type_ratio_distance_l1": float(balance["type_ratio_distance_l1"]),
        "evaluation_v2": evaluation_v2,
    }


def _init_worker(
    card_type_map: dict[int, str],
    card_desc_map: dict[int, str],
    card_level_map: dict[int, int | None],
    embedding_map: dict[int, list[float]],
    explicit_starter_ids: set[int],
    explicit_brick_ids: set[int],
    sim_trials: int,
    sim_seed: int,
) -> None:
    _WORKER_STATE["card_type_map"] = card_type_map
    _WORKER_STATE["card_desc_map"] = card_desc_map
    _WORKER_STATE["card_level_map"] = card_level_map
    _WORKER_STATE["embedding_map"] = embedding_map
    _WORKER_STATE["explicit_starter_ids"] = explicit_starter_ids
    _WORKER_STATE["explicit_brick_ids"] = explicit_brick_ids
    _WORKER_STATE["sim_trials"] = sim_trials
    _WORKER_STATE["sim_seed"] = sim_seed


def _score_one_deck_worker(payload: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    idx, deck = payload
    row = _score_one_deck(
        deck=deck,
        card_type_map=_WORKER_STATE["card_type_map"],
        card_desc_map=_WORKER_STATE["card_desc_map"],
        card_level_map=_WORKER_STATE["card_level_map"],
        embedding_map=_WORKER_STATE["embedding_map"],
        explicit_starter_ids=_WORKER_STATE["explicit_starter_ids"],
        explicit_brick_ids=_WORKER_STATE["explicit_brick_ids"],
        sim_trials=_WORKER_STATE["sim_trials"],
        sim_seed=int(_WORKER_STATE["sim_seed"]) + idx,
    )
    row["source_row_index"] = int(idx)
    return idx, row


def _load_checkpoint_rows(checkpoint_path: Path) -> dict[int, dict[str, Any]]:
    if not checkpoint_path.exists():
        return {}
    loaded: dict[int, dict[str, Any]] = {}
    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = _safe_int(row.get("source_row_index"))
            if idx is None:
                continue
            loaded[int(idx)] = row
    return loaded


def _extract_role_ids(role_tags_df: pd.DataFrame) -> tuple[set[int], set[int]]:
    if (
        role_tags_df.empty
        or "card_id" not in role_tags_df.columns
        or "role_tag" not in role_tags_df.columns
    ):
        return set(), set()
    starter: set[int] = set()
    brick: set[int] = set()
    for row in role_tags_df.to_dict(orient="records"):
        cid = _safe_int(row.get("card_id"))
        if cid is None:
            continue
        role = str(row.get("role_tag") or "").strip().lower()
        confidence = _safe_float(row.get("confidence"))
        if confidence is not None and confidence <= 0.0:
            continue
        if role == "starter":
            starter.add(cid)
        elif role == "brick":
            brick.add(cid)
    return starter, brick


def _extract_embedding_map(embeddings_df: pd.DataFrame) -> dict[int, list[float]]:
    if embeddings_df.empty or "card_id" not in embeddings_df.columns:
        return {}

    vector_col = None
    for candidate in ("embedding_vector", "vector", "embedding"):
        if candidate in embeddings_df.columns:
            vector_col = candidate
            break
    if vector_col is None:
        return {}

    mapping: dict[int, list[float]] = {}
    for row in embeddings_df.to_dict(orient="records"):
        cid = _safe_int(row.get("card_id"))
        if cid is None:
            continue
        vec = _extract_float_list(row.get(vector_col))
        if vec is None:
            continue
        nvec = _normalize_vector(vec)
        if nvec is None:
            continue
        mapping[cid] = nvec
    return mapping


def _score_column_name(score_version: str) -> str:
    if score_version == "v1":
        return "heuristic_score_v1"
    return "heuristic_score_v2"


def score_decks(
    *,
    decks_df: pd.DataFrame,
    cards_df: pd.DataFrame,
    output_dir: str | Path,
    source_summary_path: Path | None = None,
    embeddings_df: pd.DataFrame | None = None,
    role_tags_df: pd.DataFrame | None = None,
    sim_trials: int = 2000,
    sim_seed: int = 12345,
    score_version: str = "both",
    num_workers: int = 1,
    checkpoint_every: int = 200,
    resume: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cards_unique = cards_df.drop_duplicates(subset=["id"]).copy()
    card_type_map = {
        int(row["id"]): str(row.get("type", ""))
        for _, row in cards_unique[["id", "type"]].iterrows()
    }
    card_desc_map = {
        int(row["id"]): str(row.get("desc", ""))
        for _, row in cards_unique[["id", "desc"]].iterrows()
    }
    card_level_map = {
        int(row["id"]): _safe_int(row.get("level"))
        for _, row in cards_unique[["id", "level"]].iterrows()
    }

    embedding_source = embeddings_df if embeddings_df is not None else pd.DataFrame()
    tags_source = role_tags_df if role_tags_df is not None else pd.DataFrame()
    embedding_map = _extract_embedding_map(embedding_source)
    starter_ids, brick_ids = _extract_role_ids(tags_source)

    deck_records = decks_df.to_dict(orient="records")
    checkpoint_path = out_root / "deck_heuristic_scores.checkpoint.jsonl"
    if not resume and checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed existing checkpoint file for fresh run: %s", checkpoint_path)
    scored_by_idx = _load_checkpoint_rows(checkpoint_path) if resume else {}
    pending_payloads = [
        (idx, row) for idx, row in enumerate(deck_records) if idx not in scored_by_idx
    ]
    logger.info(
        "Starting scoring run total_decks=%d resume=%s recovered=%d pending=%d "
        "workers=%d trials=%d",
        len(deck_records),
        str(resume).lower(),
        len(scored_by_idx),
        len(pending_payloads),
        int(num_workers),
        int(sim_trials),
    )

    _init_worker(
        card_type_map=card_type_map,
        card_desc_map=card_desc_map,
        card_level_map=card_level_map,
        embedding_map=embedding_map,
        explicit_starter_ids=starter_ids,
        explicit_brick_ids=brick_ids,
        sim_trials=sim_trials,
        sim_seed=sim_seed,
    )

    append_since_flush = 0
    processed_since_log = 0
    next_log_target = max(500, int(checkpoint_every))
    processed_total = len(scored_by_idx)
    if pending_payloads:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("a", encoding="utf-8") as ckpt:
            if int(num_workers) <= 1:
                for payload in pending_payloads:
                    idx, row = _score_one_deck_worker(payload)
                    scored_by_idx[idx] = row
                    ckpt.write(json.dumps(row, ensure_ascii=True) + "\n")
                    append_since_flush += 1
                    processed_since_log += 1
                    processed_total += 1
                    if append_since_flush >= max(1, int(checkpoint_every)):
                        ckpt.flush()
                        append_since_flush = 0
                    if processed_since_log >= next_log_target:
                        logger.info(
                            "Scoring progress processed=%d/%d (%.2f%%)",
                            processed_total,
                            len(deck_records),
                            (100.0 * processed_total / float(len(deck_records) or 1)),
                        )
                        processed_since_log = 0
            else:
                worker_count = max(1, int(num_workers))
                logger.info("Scoring with multiprocessing workers=%d", worker_count)
                with mp.Pool(
                    processes=worker_count,
                    initializer=_init_worker,
                    initargs=(
                        card_type_map,
                        card_desc_map,
                        card_level_map,
                        embedding_map,
                        starter_ids,
                        brick_ids,
                        sim_trials,
                        sim_seed,
                    ),
                ) as pool:
                    chunk_size = max(
                        1,
                        min(200, len(pending_payloads) // (worker_count * 4) or 1),
                    )
                    for idx, row in pool.imap_unordered(
                        _score_one_deck_worker,
                        pending_payloads,
                        chunksize=chunk_size,
                    ):
                        scored_by_idx[idx] = row
                        ckpt.write(json.dumps(row, ensure_ascii=True) + "\n")
                        append_since_flush += 1
                        processed_since_log += 1
                        processed_total += 1
                        if append_since_flush >= max(1, int(checkpoint_every)):
                            ckpt.flush()
                            append_since_flush = 0
                        if processed_since_log >= next_log_target:
                            logger.info(
                                "Scoring progress processed=%d/%d (%.2f%%)",
                                processed_total,
                                len(deck_records),
                                (100.0 * processed_total / float(len(deck_records) or 1)),
                            )
                            processed_since_log = 0
            ckpt.flush()

    if len(scored_by_idx) != len(deck_records):
        missing = len(deck_records) - len(scored_by_idx)
        raise RuntimeError(f"Scoring incomplete; missing {missing} deck rows.")
    logger.info("Scoring complete processed=%d/%d", len(scored_by_idx), len(deck_records))

    scored_rows = [scored_by_idx[idx] for idx in sorted(scored_by_idx.keys())]
    scored_df = pd.DataFrame(scored_rows)

    scored_jsonl = out_root / "deck_heuristic_scores.jsonl"
    scored_parquet = out_root / "deck_heuristic_scores.parquet"
    summary_json = out_root / "scoring_summary.json"
    per_format_json = out_root / "deck_scores_by_format.json"

    write_jsonl(scored_jsonl, scored_rows)
    scored_df.to_parquet(scored_parquet, index=False)

    score_col = _score_column_name(score_version)
    top_row = (
        scored_df.sort_values(score_col, ascending=False).iloc[0].to_dict()
        if not scored_df.empty
        else {}
    )
    summary = {
        "score_version": score_version,
        "scored_deck_count": int(len(scored_df)),
        "heuristic_score_v1_mean": (
            float(scored_df["heuristic_score_v1"].mean()) if len(scored_df) else 0.0
        ),
        "heuristic_score_v1_std": (
            float(scored_df["heuristic_score_v1"].std(ddof=0)) if len(scored_df) else 0.0
        ),
        "heuristic_score_v1_min": (
            float(scored_df["heuristic_score_v1"].min()) if len(scored_df) else 0.0
        ),
        "heuristic_score_v1_max": (
            float(scored_df["heuristic_score_v1"].max()) if len(scored_df) else 0.0
        ),
        "heuristic_score_v2_mean": (
            float(scored_df["heuristic_score_v2"].mean()) if len(scored_df) else 0.0
        ),
        "heuristic_score_v2_std": (
            float(scored_df["heuristic_score_v2"].std(ddof=0)) if len(scored_df) else 0.0
        ),
        "heuristic_score_v2_min": (
            float(scored_df["heuristic_score_v2"].min()) if len(scored_df) else 0.0
        ),
        "heuristic_score_v2_max": (
            float(scored_df["heuristic_score_v2"].max()) if len(scored_df) else 0.0
        ),
        "consistency_score_mean": (
            float(scored_df["consistency_score"].mean()) if len(scored_df) else 0.0
        ),
        "synergy_score_mean": float(scored_df["synergy_score"].mean()) if len(scored_df) else 0.0,
        "brick_risk_score_mean": (
            float(scored_df["brick_risk_score"].mean()) if len(scored_df) else 0.0
        ),
        "balance_score_mean": float(scored_df["balance_score"].mean()) if len(scored_df) else 0.0,
        "top_deck": top_row,
    }
    write_summary(summary_json, summary)

    by_format: dict[str, Any] = {}
    if len(scored_df):
        grouped = scored_df.groupby("format", dropna=False)[score_col]
        for fmt, series in grouped:
            by_format[str(fmt)] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
    write_summary(per_format_json, by_format)

    artifacts = [scored_jsonl, scored_parquet, summary_json, per_format_json]
    if source_summary_path and source_summary_path.exists():
        artifacts.append(source_summary_path)
    return scored_df, summary, artifacts


def log_scoring_to_mlflow(
    *,
    tracking_uri: str | None,
    experiment: str,
    run_name: str,
    run_tags: dict[str, str],
    summary_metrics: dict[str, Any],
    artifacts: list[Path],
    scored_df: pd.DataFrame,
    score_version: str,
) -> None:
    if not tracking_uri:
        return

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.set_tag("scoring_method", f"heuristic_score_{score_version}")
        for key, value in run_tags.items():
            mlflow.set_tag(key, value)

        scalar_keys = [
            "scored_deck_count",
            "heuristic_score_v1_mean",
            "heuristic_score_v1_std",
            "heuristic_score_v1_min",
            "heuristic_score_v1_max",
            "heuristic_score_v2_mean",
            "heuristic_score_v2_std",
            "heuristic_score_v2_min",
            "heuristic_score_v2_max",
            "consistency_score_mean",
            "synergy_score_mean",
            "brick_risk_score_mean",
            "balance_score_mean",
        ]
        mlflow.log_metrics(
            {k: float(summary_metrics[k]) for k in scalar_keys if k in summary_metrics}
        )

        from mlflow.entities import Metric
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        run_id = active_run.info.run_id
        metric_specs: list[tuple[str, str]] = [
            ("heuristic_score_v1", "deck_heuristic_score_v1"),
            ("heuristic_score_v2", "deck_heuristic_score_v2"),
            ("consistency_score", "deck_consistency_score"),
            ("synergy_score", "deck_synergy_score"),
            ("brick_risk_score", "deck_brick_risk_score"),
            ("balance_score", "deck_balance_score"),
            ("starter_hit_rate", "deck_starter_hit_rate"),
            ("brick_hand_rate", "deck_brick_hand_rate"),
            ("unique_main_ratio", "deck_unique_main_ratio"),
            ("monster_ratio", "deck_monster_ratio"),
            ("spell_ratio", "deck_spell_ratio"),
            ("trap_ratio", "deck_trap_ratio"),
            ("num_staples_injected", "deck_num_staples_injected"),
            ("num_novelty_cards", "deck_num_novelty_cards"),
            ("num_repairs", "deck_num_repairs"),
            ("num_resamples", "deck_num_resamples"),
            ("constraint_flags_passed", "deck_constraint_flags_passed"),
            ("constraint_flags_total", "deck_constraint_flags_total"),
            ("error_count", "deck_error_count"),
        ]

        batch_metrics: list[Metric] = []
        chunk_size = 1000
        for step, row in enumerate(scored_df.to_dict(orient="records")):
            ts_ms = int(time.time() * 1000)
            for row_key, metric_name in metric_specs:
                batch_metrics.append(
                    Metric(key=metric_name, value=float(row[row_key]), step=step, timestamp=ts_ms)
                )
            if len(batch_metrics) >= chunk_size:
                client.log_batch(run_id=run_id, metrics=batch_metrics)
                batch_metrics = []
        if batch_metrics:
            client.log_batch(run_id=run_id, metrics=batch_metrics)

        mlflow.log_table(scored_df, artifact_file="per_deck_scoring_table.json")
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))


def run(args: argparse.Namespace) -> int:
    decks_df = read_table(args.decks_file)
    cards_df = read_table(args.cards_file)

    embeddings_df: pd.DataFrame | None = None
    if args.embeddings_file:
        embeddings_df = read_table(args.embeddings_file)

    role_tags_df: pd.DataFrame | None = None
    if args.card_tags_file:
        role_tags_df = read_table(args.card_tags_file)

    decks_path = Path(args.decks_file)
    source_summary_path = decks_path.parent / "summary.json"
    out_root = Path(args.output_dir) if args.output_dir else decks_path.parent / "scoring"

    scored_df, summary, artifacts = score_decks(
        decks_df=decks_df,
        cards_df=cards_df,
        output_dir=out_root,
        source_summary_path=source_summary_path,
        embeddings_df=embeddings_df,
        role_tags_df=role_tags_df,
        sim_trials=args.sim_trials,
        sim_seed=args.sim_seed,
        score_version=args.score_version,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )

    run_name = args.mlflow_run_name or f"deck-scoring-{args.run_id or 'adhoc'}"
    tags = {
        "source_decks_file": str(decks_path),
        "source_cards_file": str(Path(args.cards_file)),
        "scoring_output_dir": str(out_root),
        "score_version": str(args.score_version),
        "sim_trials": str(args.sim_trials),
        "num_workers": str(args.num_workers),
        "resume_enabled": str(bool(args.resume)).lower(),
    }
    if args.embeddings_file:
        tags["source_embeddings_file"] = str(Path(args.embeddings_file))
    if args.card_tags_file:
        tags["source_card_tags_file"] = str(Path(args.card_tags_file))
    if args.run_id:
        tags["generation_run_id"] = args.run_id
    if not scored_df.empty and "run_id" in scored_df.columns:
        first_run_id = (
            str(scored_df["run_id"].dropna().iloc[0]) if scored_df["run_id"].notna().any() else ""
        )
        if first_run_id:
            tags.setdefault("generation_run_id", first_run_id)
    log_scoring_to_mlflow(
        tracking_uri=args.mlflow_tracking_uri,
        experiment=args.mlflow_experiment,
        run_name=run_name,
        run_tags=tags,
        summary_metrics=summary,
        artifacts=artifacts,
        scored_df=scored_df,
        score_version=args.score_version,
    )
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
