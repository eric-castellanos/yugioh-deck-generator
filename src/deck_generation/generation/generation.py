"""
Generation script for creating and analyzing large sets of Yu-Gi-Oh! decks.

This script generates a large number of decks, calculates various metrics,
and creates a comprehensive dataframe for analysis with both novel and meta-aware decks.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import tempfile
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from src.deck_generation.generation.deck_generator import (
    SimpleDeckGenerator,
    DeckMetadata,
    get_card_data_with_clusters,
)
from src.utils.mlflow.mlflow_utils import (
    log_deck_artifacts,
    log_deck_generation_params,
    log_deck_generation_tags,
    setup_experiment,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Weight configs (now passed directly into generate_deck instead of mutating a
# global inside the generator module)
# -----------------------------------------------------------------------------
DEFAULT_WEIGHTS: Dict[str, float] = {
    # Higher entropy = more diverse clusters
    "cluster_entropy": 0.35,
    # Higher distance = more surprising combinations
    "cluster_distance": 0.40,
    # Higher rarity = more novel combinations
    "rarity": 0.15,
    # Higher noise percentage = less cohesive deck (penalize)
    "noise_penalty": 0.10,
}

WEIGHT_GRIDS: Dict[str, List[float]] = {
    "cluster_entropy": [0.2, 0.3, 0.35, 0.4, 0.5],
    "cluster_distance": [0.2, 0.3, 0.4, 0.45, 0.5],
    "rarity": [0.1, 0.15, 0.2, 0.25],
    "noise_penalty": [0.05, 0.1, 0.15, 0.2],
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def clean_decks_directory() -> None:
    """
    Clean the decks/generated directory to ensure a fresh run.
    """
    decks_dir = Path("decks/generated")
    if decks_dir.exists():
        existing_files = list(decks_dir.rglob("*.ydk"))
        if existing_files:
            logger.info(f"🗑️  Cleaning {len(existing_files)} existing deck files from {decks_dir}")
            shutil.rmtree(decks_dir)
            logger.info(f"✅ Cleaned decks directory: {decks_dir}")
    decks_dir.mkdir(parents=True, exist_ok=True)


def pad_main_deck_to_minimum(
    main_deck: List[Dict[str, Any]], 
    min_size: int = 40,
    card_pool: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Ensure main deck has at least min_size cards by adding fallback cards.
    
    Args:
        main_deck: Current main deck list
        min_size: Minimum number of cards required (default 40)
        card_pool: Available card pool to choose from (if None, will get from clusters)
        
    Returns:
        Updated main deck with padding cards if needed
    """
    if len(main_deck) >= min_size:
        return main_deck
        
    cards_needed = min_size - len(main_deck)
    logger.warning(f"⚠️ Main deck has only {len(main_deck)} cards, padding with {cards_needed} additional cards")
    
    # Get card pool if not provided
    if card_pool is None:
        try:
            card_pool = get_card_data_with_clusters()
        except Exception as e:
            logger.error(f"Failed to get card pool for padding: {e}")
            return main_deck
    
    # Count existing cards to respect copy limits
    existing_cards = {}
    for card in main_deck:
        card_id = card.get("id")
        if card_id:
            existing_cards[card_id] = existing_cards.get(card_id, 0) + 1
    
    # Define copy limits (Yu-Gi-Oh rules: max 3 copies of most cards)
    max_copies = 3
    
    # Create fallback pool prioritizing generic staples
    generic_staples = [
        "Mystical Space Typhoon", "Mirror Force", "Torrential Tribute", 
        "Book of Moon", "Compulsory Evacuation Device", "Bottomless Trap Hole",
        "Solemn Warning", "Dark Hole", "Raigeki", "Pot of Desires",
        "Twin Twisters", "Ash Blossom & Joyous Spring"
    ]
    
    # First try to add generic staples
    fallback_cards = []
    for card in card_pool:
        card_name = card.get("name", "")
        card_id = card.get("id")
        
        # Skip if we don't have essential info
        if not card_id or not card_name:
            continue
            
        # Prioritize generic staples
        is_staple = any(staple.lower() in card_name.lower() for staple in generic_staples)
        current_count = existing_cards.get(card_id, 0)
        
        if is_staple and current_count < max_copies:
            fallback_cards.append(card)
        elif not is_staple and current_count < max_copies:
            # Add other cards as secondary options
            fallback_cards.append(card)
    
    # If we still don't have enough fallback cards, use any available cards
    if not fallback_cards:
        fallback_cards = [card for card in card_pool if card.get("id") and 
                         existing_cards.get(card.get("id"), 0) < max_copies]
    
    # Add cards until we reach minimum size
    padded_deck = main_deck.copy()
    cards_added = 0
    
    while len(padded_deck) < min_size and cards_added < cards_needed * 2:  # Safety limit
        if not fallback_cards:
            logger.warning("⚠️ No more cards available for padding")
            break
            
        # Pick a random card from fallback pool
        card_to_add = random.choice(fallback_cards)
        card_id = card_to_add.get("id")
        
        # Check if we can still add this card (copy limit)
        current_count = existing_cards.get(card_id, 0)
        if current_count < max_copies:
            padded_deck.append(card_to_add)
            existing_cards[card_id] = current_count + 1
            cards_added += 1
        
        # Remove card from fallback pool if we've reached its limit
        if existing_cards.get(card_id, 0) >= max_copies:
            fallback_cards = [c for c in fallback_cards if c.get("id") != card_id]
    
    final_count = len(padded_deck)
    if final_count >= min_size:
        logger.info(f"✅ Successfully padded deck to {final_count} cards")
    else:
        logger.warning(f"⚠️ Could only pad deck to {final_count} cards (target: {min_size})")
    
    return padded_deck


def write_ydk(main_ids: List[int], extra_ids: List[int], path: Path) -> None:
    """
    Write a .ydk file with standard sections and no side deck.
    NOTE: switched to '#side' (EDOPro-friendly) instead of '!side'.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("#created by yugioh-deck-generator\n")
        f.write("#main\n")
        for cid in main_ids:
            f.write(f"{cid}\n")
        f.write("#extra\n")
        for cid in extra_ids:
            f.write(f"{cid}\n")
        f.write("#side\n")


def generate_weight_combinations(grid_search_mode: str = "full") -> List[Dict[str, float]]:
    """
    Build weight combinations for parameter search.
    """
    if grid_search_mode == "full":
        combos = list(
            product(
                WEIGHT_GRIDS["cluster_entropy"],
                WEIGHT_GRIDS["cluster_distance"],
                WEIGHT_GRIDS["rarity"],
                WEIGHT_GRIDS["noise_penalty"],
            )
        )
        valid: List[Dict[str, float]] = []
        for ce, cd, r, np_ in combos:
            total = ce + cd + r + np_
            if 0.9 <= total <= 1.1:
                valid.append(
                    {
                        "cluster_entropy": ce,
                        "cluster_distance": cd,
                        "rarity": r,
                        "noise_penalty": np_,
                    }
                )
        return valid

    if grid_search_mode == "random":
        rng = random.Random(42)
        out: List[Dict[str, float]] = []
        for _ in range(50):
            weights = [
                rng.choice(WEIGHT_GRIDS["cluster_entropy"]),
                rng.choice(WEIGHT_GRIDS["cluster_distance"]),
                rng.choice(WEIGHT_GRIDS["rarity"]),
                rng.choice(WEIGHT_GRIDS["noise_penalty"]),
            ]
            total = sum(weights)
            norm = [w / total for w in weights]
            out.append(
                {
                    "cluster_entropy": norm[0],
                    "cluster_distance": norm[1],
                    "rarity": norm[2],
                    "noise_penalty": norm[3],
                }
            )
        return out

    if grid_search_mode == "targeted":
        return [
            # Entropy-focused
            {"cluster_entropy": 0.5, "cluster_distance": 0.3, "rarity": 0.15, "noise_penalty": 0.05},
            {"cluster_entropy": 0.45, "cluster_distance": 0.35, "rarity": 0.15, "noise_penalty": 0.05},
            # Distance-focused
            {"cluster_entropy": 0.3, "cluster_distance": 0.5, "rarity": 0.15, "noise_penalty": 0.05},
            {"cluster_entropy": 0.35, "cluster_distance": 0.45, "rarity": 0.15, "noise_penalty": 0.05},
            # Balanced
            {"cluster_entropy": 0.35, "cluster_distance": 0.4, "rarity": 0.15, "noise_penalty": 0.1},
            {"cluster_entropy": 0.3, "cluster_distance": 0.4, "rarity": 0.2, "noise_penalty": 0.1},
            {"cluster_entropy": 0.4, "cluster_distance": 0.35, "rarity": 0.2, "noise_penalty": 0.05},
            # Rarity-focused
            {"cluster_entropy": 0.3, "cluster_distance": 0.35, "rarity": 0.25, "noise_penalty": 0.1},
            {"cluster_entropy": 0.35, "cluster_distance": 0.3, "rarity": 0.25, "noise_penalty": 0.1},
            # Low noise penalty
            {"cluster_entropy": 0.4, "cluster_distance": 0.4, "rarity": 0.15, "noise_penalty": 0.05},
            {"cluster_entropy": 0.35, "cluster_distance": 0.45, "rarity": 0.15, "noise_penalty": 0.05},
        ]

    return [DEFAULT_WEIGHTS]


# -----------------------------------------------------------------------------
# Parameter search
# -----------------------------------------------------------------------------
def run_parameter_search(
    total_decks: int = 1000,
    grid_search_mode: str = "targeted",
    novel_ratio: float = 0.65,
    meta_archetypes: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Run parameter search by generating decks with different weight configurations.
    Each deck cycles through weight configurations.
    """
    logger.info(f"Starting parameter search with mode: {grid_search_mode}")
    logger.info(f"Will generate {total_decks} decks, cycling weight configurations")

    clean_decks_directory()

    logger.info("Loading clustered cards from MLflow model registry...")
    clustered_cards = get_card_data_with_clusters()
    logger.info(f"✅ Successfully loaded {len(clustered_cards)} card clusters")

    weight_combinations = generate_weight_combinations(grid_search_mode)
    logger.info(f"Cycling through {len(weight_combinations)} weight configurations")

    experiment_id = setup_experiment("yugioh_deck_generation_parameter_search")

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("grid_search_mode", grid_search_mode)
        mlflow.log_param("total_decks", total_decks)
        mlflow.log_param("num_weight_configs", len(weight_combinations))
        mlflow.log_param("novel_ratio", novel_ratio)

        df_result = generate_decks_with_parameter_search(
            total_decks=total_decks,
            weight_combinations=weight_combinations,
            novel_ratio=novel_ratio,
            meta_archetypes=meta_archetypes,
            clustered_cards=clustered_cards,
        )

        mlflow.log_metric("generated_decks", len(df_result))
        mlflow.log_metric("avg_cluster_entropy", float(df_result["cluster_entropy"].mean()))
        mlflow.log_metric("avg_cluster_distance", float(df_result["intra_deck_cluster_distance"].mean()))
        mlflow.log_metric("avg_rarity", float(df_result["cluster_co_occurrence_rarity"].mean()))
        mlflow.log_metric("avg_noise_penalty", float(df_result["noise_card_percentage"].mean()))

        config_summary = (
            df_result.group_by("weight_config_id")
            .agg(
                [
                    pl.count("deck_id").alias("deck_count"),
                    pl.mean("cluster_entropy").alias("avg_entropy"),
                    pl.mean("intra_deck_cluster_distance").alias("avg_distance"),
                    pl.mean("cluster_co_occurrence_rarity").alias("avg_rarity"),
                    pl.mean("noise_card_percentage").alias("avg_noise"),
                    pl.first("weight_cluster_entropy").alias("weight_cluster_entropy"),
                    pl.first("weight_cluster_distance").alias("weight_cluster_distance"),
                    pl.first("weight_rarity").alias("weight_rarity"),
                    pl.first("weight_noise_penalty").alias("weight_noise_penalty"),
                ]
            )
            .sort("avg_entropy", descending=True)
        )

        logger.info("\n=== Weight Configuration Analysis ===")
        for row in config_summary.to_dicts():
            logger.info(
                f"Config {row['weight_config_id']}: {row['deck_count']} decks, "
                f"Avg_entropy={row['avg_entropy']:.4f} | "
                f"Weights(ent={row['weight_cluster_entropy']:.3f}, "
                f"dist={row['weight_cluster_distance']:.3f}, "
                f"rar={row['weight_rarity']:.3f}, "
                f"noise={row['weight_noise_penalty']:.3f})"
            )

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("outputs/parameter_search")
        output_dir.mkdir(parents=True, exist_ok=True)

        combined_results_file = output_dir / f"parameter_search_results_{timestamp}.csv"
        df_result.to_pandas().to_csv(combined_results_file, index=False)

        config_summary_file = output_dir / f"parameter_search_summary_{timestamp}.csv"
        config_summary.to_pandas().to_csv(config_summary_file, index=False)

        logger.info(f"\n✅ Parameter search complete!")
        logger.info(f"📊 Combined results saved: {combined_results_file}")
        logger.info(f"📈 Configuration summary saved: {config_summary_file}")
        logger.info(f"🔬 Total decks generated: {len(df_result)}")
        logger.info(f"⚙️ Configurations tested: {len(weight_combinations)}")

    return df_result


def _prepare_generator_with_clusters(
    clustered_cards: Dict[str, List[Dict[str, Any]]],
    num_cards: int = 40,
) -> SimpleDeckGenerator:
    """
    Create a SimpleDeckGenerator and populate its cluster maps from clustered_cards.
    (Keeps compatibility even if deck_generator.py now loads clusters itself.)
    """
    generator = SimpleDeckGenerator(None, None, num_cards=num_cards)
    generator.clustered_cards = clustered_cards

    # Initialize the dictionary attributes
    generator.main_deck_clusters = {}
    generator.extra_deck_clusters = {}
    generator.monster_clusters = {}
    generator.spell_clusters = {}
    generator.trap_clusters = {}

    logger.info("Processing clusters into card type categories...")
    for cluster_id, cards in clustered_cards.items():
        main_cards: List[Dict[str, Any]] = []
        extra_cards: List[Dict[str, Any]] = []
        monsters: List[Dict[str, Any]] = []
        spells: List[Dict[str, Any]] = []
        traps: List[Dict[str, Any]] = []

        for card in cards:
            ctype = card.get("type", "")
            if not ctype or not card.get("name"):
                continue

            # Extra deck types
            if any(et in ctype for et in ["Fusion", "Synchro", "XYZ", "Link"]):
                extra_cards.append(card)
            else:
                main_cards.append(card)
                if "Monster" in ctype:
                    monsters.append(card)
                elif "Spell" in ctype:
                    spells.append(card)
                elif "Trap" in ctype:
                    traps.append(card)

        if main_cards:
            generator.main_deck_clusters[cluster_id] = main_cards
            if monsters:
                generator.monster_clusters[cluster_id] = monsters
            if spells:
                generator.spell_clusters[cluster_id] = spells
            if traps:
                generator.trap_clusters[cluster_id] = traps

        if extra_cards:
            generator.extra_deck_clusters[cluster_id] = extra_cards

    logger.info(
        "✅ Processed %d monster clusters, %d spell clusters, %d trap clusters, %d extra deck clusters",
        len(generator.monster_clusters),
        len(generator.spell_clusters),
        len(generator.trap_clusters),
        len(generator.extra_deck_clusters),
    )

    return generator


def generate_decks_with_parameter_search(
    total_decks: int,
    weight_combinations: List[Dict[str, float]],
    novel_ratio: float = 0.65,
    meta_archetypes: Optional[List[str]] = None,
    clustered_cards: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
    """
    Generate decks cycling through different weight configurations.
    Each deck is generated with a different weight configuration.
    """
    assert clustered_cards is not None, "clustered_cards must be provided"
    generator = _prepare_generator_with_clusters(clustered_cards, num_cards=40)

    # Count how many of each mode to generate
    num_novel = int(total_decks * novel_ratio)
    num_meta = total_decks - num_novel

    # Build archetype list if not provided
    if not meta_archetypes:
        archetype_counts = defaultdict(int)
        for cluster_cards in clustered_cards.values():
            for card in cluster_cards:
                if card.get("archetypes"):
                    for arch in card["archetypes"]:
                        archetype_counts[arch] += 1
        meta_archetypes = [a for a, _ in sorted(archetype_counts.items(), key=lambda x: x[1], reverse=True)]

    logger.info(f"Using meta archetypes: {', '.join(meta_archetypes[:20])}{' ...' if len(meta_archetypes) > 20 else ''}")

    # Output directory
    deck_output_dir = Path("decks/generated")
    deck_output_dir.mkdir(parents=True, exist_ok=True)

    all_decks_data: List[Dict[str, Any]] = []
    successful_decks = 0
    start_time = time.time()

    logger.info(f"Generating {total_decks} decks with cycling weight configurations...")
    pbar = tqdm(total=total_decks, desc="Generating decks")

    for i in range(total_decks):
        deck_generated = False
        attempts = 0
        max_attempts = 3

        config_index = i % len(weight_combinations)
        current_weights = weight_combinations[config_index]

        while not deck_generated and attempts < max_attempts:
            try:
                deck_id = f"deck_{successful_decks:03d}"

                if i < num_novel:
                    generation_mode = "novel"
                    target_archetype = None
                    main_deck, extra_deck, metadata = generator.generate_deck(
                        mode="novel",
                        use_mlflow=False,
                    )
                else:
                    generation_mode = "meta_aware"
                    target_archetype = random.choice(meta_archetypes)
                    main_deck, extra_deck, metadata = generator.generate_deck(
                        mode="meta_aware",
                        target_archetype=target_archetype,
                        use_mlflow=False,
                    )

                if not main_deck:
                    raise ValueError("Generated deck is empty")

                # Ensure main deck has at least 40 cards
                card_pool = []
                for cluster_cards in clustered_cards.values():
                    card_pool.extend(cluster_cards)
                main_deck = pad_main_deck_to_minimum(main_deck, min_size=40, card_pool=card_pool)

                # Write .ydk
                main_ids = [c.get("id") for c in main_deck if isinstance(c, dict) and c.get("id") is not None]
                extra_ids = [c.get("id") for c in extra_deck if isinstance(c, dict) and c.get("id") is not None]
                write_ydk(main_ids, extra_ids, deck_output_dir / f"{deck_id}.ydk")

                # Metrics
                entropy = metadata.cluster_entropy
                distance = metadata.intra_deck_cluster_distance
                rarity = metadata.cluster_co_occurrence_rarity
                noise_pct = metadata.noise_card_percentage

                deck_data: Dict[str, Any] = {
                    "deck_id": deck_id,
                    "generation_mode": generation_mode,
                    "weight_config_id": config_index,
                    "main_deck_size": len(main_deck),
                    "extra_deck_size": len(extra_deck),
                    "monster_count": metadata.monster_count,
                    "spell_count": metadata.spell_count,
                    "trap_count": metadata.trap_count,
                    "monster_ratio": metadata.monster_count / len(main_deck) if main_deck else 0.0,
                    "spell_ratio": metadata.spell_count / len(main_deck) if main_deck else 0.0,
                    "trap_ratio": metadata.trap_count / len(main_deck) if main_deck else 0.0,
                    "dominant_archetype": metadata.dominant_archetype,
                    "cluster_entropy": entropy,
                    "intra_deck_cluster_distance": distance,
                    "cluster_co_occurrence_rarity": rarity,
                    "noise_card_percentage": noise_pct,
                    # raw copies
                    "raw_entropy": entropy,
                    "raw_distance": distance,
                    "raw_rarity": rarity,
                    "raw_noise": noise_pct,
                    # weights used
                    "weight_cluster_entropy": current_weights["cluster_entropy"],
                    "weight_cluster_distance": current_weights["cluster_distance"],
                    "weight_rarity": current_weights["rarity"],
                    "weight_noise_penalty": current_weights["noise_penalty"],
                    "weight_config_json": json.dumps(current_weights),
                    # keep cards
                    "main_deck": main_deck,
                    "extra_deck": extra_deck,
                }

                if generation_mode == "meta_aware":
                    deck_data["target_archetype"] = target_archetype

                all_decks_data.append(deck_data)
                successful_decks += 1
                deck_generated = True

                if successful_decks % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = successful_decks / elapsed if elapsed > 0 else 0
                    logger.info(f"Generated {successful_decks}/{total_decks} decks ({rate:.1f} decks/sec)")

            except Exception as e:
                attempts += 1
                if attempts >= max_attempts:
                    logger.warning(f"Failed to generate deck after {max_attempts} attempts (position {i+1}): {e}")
                else:
                    logger.debug(f"Deck generation attempt {attempts} failed for position {i+1}: {e}")

        pbar.update(1)

    pbar.close()

    if not all_decks_data:
        logger.error("❌ No deck data was collected!")
        raise ValueError("No successful deck generation occurred")

    df = pl.DataFrame(all_decks_data)
    logger.info(
        f"✅ Generated {successful_decks} of {total_decks} decks successfully "
        f"({successful_decks/total_decks:.1%} success rate)"
    )
    logger.info(f"📁 Deck files saved to: {deck_output_dir}")
    return df


# -----------------------------------------------------------------------------
# Bulk generation (single config)
# -----------------------------------------------------------------------------
def generate_decks(
    total_decks: int = 10000,
    novel_ratio: float = 0.65,
    meta_archetypes: Optional[List[str]] = None,
    log_individual_decks: bool = False,
    clustered_cards: Optional[Dict[str, Any]] = None,
    skip_mlflow_setup: bool = False,
    config_id: Optional[int] = None,
    seed: Optional[int] = 1337,
    weights: Optional[Dict[str, float]] = None,
) -> pl.DataFrame:
    """
    Generate a specified number of decks, with a mix of novel and meta-aware decks.
    """
    # Reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

    if clustered_cards is None:
        logger.info("Loading clustered cards from MLflow model registry...")
        clustered_cards = get_card_data_with_clusters()
        logger.info(f"✅ Successfully loaded {len(clustered_cards)} card clusters")
    else:
        logger.info(f"✅ Using pre-loaded {len(clustered_cards)} card clusters")

    experiment_id = None
    if not skip_mlflow_setup:
        experiment_id = setup_experiment("yugioh_deck_generation_bulk")

    num_novel = int(total_decks * novel_ratio)
    num_meta = total_decks - num_novel

    if not meta_archetypes:
        archetype_counts = defaultdict(int)
        for cluster_cards in clustered_cards.values():
            for card in cluster_cards:
                if card.get("archetypes"):
                    for arch in card["archetypes"]:
                        archetype_counts[arch] += 1
        meta_archetypes = [a for a, _ in sorted(archetype_counts.items(), key=lambda x: x[1], reverse=True)]

    logger.info(f"Using meta archetypes: {', '.join(meta_archetypes[:20])}{' ...' if len(meta_archetypes) > 20 else ''}")

    generator = _prepare_generator_with_clusters(clustered_cards, num_cards=40)
    use_weights = weights or DEFAULT_WEIGHTS

    def generate_all_decks() -> pl.DataFrame:
        if not skip_mlflow_setup:
            log_deck_generation_tags(
                generation_mode="bulk",
                stage="analysis",
                version="v1.0",
                batch_generation=True,
                num_decks=total_decks,
            )
            log_deck_generation_params(
                generation_mode="bulk",
                deck_count=total_decks,
                novel_ratio=novel_ratio,
                meta_archetypes=meta_archetypes[:50] if meta_archetypes else None,
                game_constraint_enabled=True,
            )

        all_decks_data: List[Dict[str, Any]] = []
        successful_decks = 0
        start_time = time.time()

        pbar = tqdm(total=total_decks, desc="Generating decks")

        # Output directory
        deck_output_dir = (
            Path(f"decks/generated/config_{config_id:02d}") if config_id is not None else Path("decks/generated")
        )
        deck_output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(total_decks):
            deck_generated = False
            attempts = 0
            max_attempts = 3

            while not deck_generated and attempts < max_attempts:
                try:
                    deck_id = f"deck_{successful_decks:03d}"

                    if i < num_novel:
                        generation_mode = "novel"
                        target_archetype = None
                        main_deck, extra_deck, metadata = generator.generate_deck(
                            mode="novel",
                            use_mlflow=False,
                        )
                    else:
                        generation_mode = "meta_aware"
                        target_archetype = random.choice(meta_archetypes)
                        main_deck, extra_deck, metadata = generator.generate_deck(
                            mode="meta_aware",
                            target_archetype=target_archetype,
                            use_mlflow=False,
                        )

                    if not main_deck:
                        raise ValueError("Generated deck is empty")

                    # Ensure main deck has at least 40 cards
                    card_pool = []
                    for cluster_cards in clustered_cards.values():
                        card_pool.extend(cluster_cards)
                    main_deck = pad_main_deck_to_minimum(main_deck, min_size=40, card_pool=card_pool)

                    # Write .ydk
                    main_ids = [c.get("id") for c in main_deck if isinstance(c, dict) and c.get("id") is not None]
                    extra_ids = [c.get("id") for c in extra_deck if isinstance(c, dict) and c.get("id") is not None]
                    write_ydk(main_ids, extra_ids, deck_output_dir / f"{deck_id}.ydk")

                    # Log artifacts per deck if requested
                    if log_individual_decks and not skip_mlflow_setup:
                        try:
                            log_deck_artifacts(
                                main_deck=main_deck,
                                extra_deck=extra_deck,
                                metadata=metadata,
                                prefix=f"deck_{deck_id}",
                            )
                        except Exception as log_err:
                            logger.warning(f"Failed to log deck artifacts for {deck_id}: {log_err}")

                    # Metrics and row
                    entropy = metadata.cluster_entropy
                    distance = metadata.intra_deck_cluster_distance
                    rarity = metadata.cluster_co_occurrence_rarity
                    noise_pct = metadata.noise_card_percentage

                    deck_data: Dict[str, Any] = {
                        "deck_id": deck_id,
                        "generation_mode": generation_mode,
                        "main_deck_size": len(main_deck),
                        "extra_deck_size": len(extra_deck),
                        "monster_count": metadata.monster_count,
                        "spell_count": metadata.spell_count,
                        "trap_count": metadata.trap_count,
                        "monster_ratio": metadata.monster_count / len(main_deck) if main_deck else 0.0,
                        "spell_ratio": metadata.spell_count / len(main_deck) if main_deck else 0.0,
                        "trap_ratio": metadata.trap_count / len(main_deck) if main_deck else 0.0,
                        "dominant_archetype": metadata.dominant_archetype,
                        "cluster_entropy": entropy,
                        "intra_deck_cluster_distance": distance,
                        "cluster_co_occurrence_rarity": rarity,
                        "noise_card_percentage": noise_pct,
                        "raw_entropy": entropy,
                        "raw_distance": distance,
                        "raw_rarity": rarity,
                        "raw_noise": noise_pct,
                        "main_deck": main_deck,
                        "extra_deck": extra_deck,
                    }
                    if generation_mode == "meta_aware":
                        deck_data["target_archetype"] = target_archetype

                    all_decks_data.append(deck_data)
                    successful_decks += 1
                    deck_generated = True

                    if successful_decks % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = successful_decks / elapsed if elapsed > 0 else 0
                        logger.info(f"Generated {successful_decks}/{total_decks} decks ({rate:.1f} decks/sec)")

                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.warning(f"Failed to generate deck after {max_attempts} attempts (position {i+1}): {e}")
                    else:
                        logger.debug(f"Deck generation attempt {attempts} failed for position {i+1}: {e}")

            pbar.update(1)

        pbar.close()

        if not all_decks_data:
            logger.error("❌ No deck data was collected!")
            raise ValueError("No successful deck generation occurred")

        if len(all_decks_data) < total_decks * 0.8:
            logger.warning(
                f"⚠️  Only generated {len(all_decks_data)}/{total_decks} decks "
                f"({len(all_decks_data)/total_decks:.1%} success rate)"
            )

        df = pl.DataFrame(all_decks_data)

        if not skip_mlflow_setup:
            mlflow.log_metric("generated_decks", successful_decks)
            mlflow.log_metric("success_rate", successful_decks / total_decks)
            mlflow.log_metric("avg_cluster_entropy", float(df["cluster_entropy"].mean()))
            mlflow.log_metric("max_cluster_entropy", float(df["cluster_entropy"].max()))
            mlflow.log_metric("min_cluster_entropy", float(df["cluster_entropy"].min()))
            mlflow.log_metric("avg_cluster_distance", float(df["intra_deck_cluster_distance"].mean()))
            mlflow.log_metric("avg_rarity", float(df["cluster_co_occurrence_rarity"].mean()))
            mlflow.log_metric("avg_noise_penalty", float(df["noise_card_percentage"].mean()))

        mode_stats = df.group_by("generation_mode").agg(
            [
                pl.count("deck_id").alias("count"),
                pl.mean("cluster_entropy").alias("avg_entropy"),
                pl.mean("intra_deck_cluster_distance").alias("avg_distance"),
                pl.mean("cluster_co_occurrence_rarity").alias("avg_rarity"),
                pl.mean("noise_card_percentage").alias("avg_noise"),
            ]
        )

        if not skip_mlflow_setup:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
                pd_df = df.to_pandas()
                pd_df = convert_numpy_to_list_in_df(pd_df)
                pd_df.to_csv(tmp_file.name, index=False)
                mlflow.log_artifact(tmp_file.name, "all_decks_data.csv")

            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp_file:
                pd_df.to_html(tmp_file, index=False)
                mlflow.log_artifact(tmp_file.name, "all_decks_data.html")

            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
                pd_mode_stats = mode_stats.to_pandas()
                pd_mode_stats = convert_numpy_to_list_in_df(pd_mode_stats)
                pd_mode_stats.to_csv(tmp_file.name, index=False)
                mlflow.log_artifact(tmp_file.name, "generation_mode_stats.csv")

        if not skip_mlflow_setup:
            logger.info(f"✅ Generation complete. Run ID: {mlflow.active_run().info.run_id if mlflow.active_run() else 'N/A'}")
            if experiment_id:
                logger.info(f"✅ MLflow UI: http://localhost:5000/#/experiments/{experiment_id}")

        logger.info(f"📁 Deck files saved to: {deck_output_dir}")
        return df

    if skip_mlflow_setup:
        return generate_all_decks()
    else:
        with mlflow.start_run(experiment_id=experiment_id):
            return generate_all_decks()


# -----------------------------------------------------------------------------
# DataFrame cleaning helpers
# -----------------------------------------------------------------------------
def convert_numpy_to_list_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any NumPy arrays in a pandas DataFrame to Python lists for CSV safety.
    """
    result_df = df.copy()
    for col in result_df.columns:
        if result_df[col].dtype == "object":
            result_df[col] = result_df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

            if result_df[col].apply(lambda x: isinstance(x, dict)).any():
                result_df[col] = result_df[col].apply(
                    lambda d: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
                    if isinstance(d, dict)
                    else d
                )

            if result_df[col].apply(lambda x: isinstance(x, list)).any():
                result_df[col] = result_df[col].apply(
                    lambda lst: [
                        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in item.items()}
                        if isinstance(item, dict)
                        else item
                        for item in lst
                    ]
                    if isinstance(lst, list)
                    else lst
                )
    return result_df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> pd.DataFrame:
    """Main function with parameter search option."""
    import argparse

    parser = argparse.ArgumentParser(description="Yu-Gi-Oh! Deck Generation with Parameter Search")
    parser.add_argument("--mode", choices=["single", "parameter_search"], default="single", help="single or parameter_search")
    parser.add_argument("--total-decks", type=int, default=10000, help="Total decks to generate (per configuration)")
    parser.add_argument("--grid-search-mode", choices=["full", "random", "targeted"], default="targeted", help="Grid search mode")
    parser.add_argument("--novel-ratio", type=float, default=0.65, help="Ratio of novel vs meta-aware decks")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.mode == "parameter_search":
        logger.info("🔬 Starting parameter search mode")
        logger.info(f"📊 Total decks: {args.total_decks}")
        logger.info(f"🎯 Grid search mode: {args.grid_search_mode}")

        combined_df = run_parameter_search(
            total_decks=args.total_decks,
            grid_search_mode=args.grid_search_mode,
            novel_ratio=args.novel_ratio,
        )
        return combined_df.to_pandas()

    # Single generation mode
    logger.info("🎮 Starting single generation mode")
    logger.info(f"📊 Total decks: {args.total_decks}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    df_pl = generate_decks(
        total_decks=args.total_decks,
        novel_ratio=args.novel_ratio,
        log_individual_decks=False,
        seed=args.seed,
        weights=DEFAULT_WEIGHTS,  # pass weights explicitly
    )

    df_pd = df_pl.to_pandas()
    df_pd = convert_numpy_to_list_in_df(df_pd)

    output_dir = Path("outputs/deck_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / f"deck_generation_{timestamp}.csv"
    df_pd.to_csv(output_file, index=False)

    # Basic stats
    monster_ratio = float(df_pd["monster_ratio"].mean())
    spell_ratio = float(df_pd["spell_ratio"].mean())
    trap_ratio = float(df_pd["trap_ratio"].mean())

    logger.info(f"Generation complete! Shape of final DataFrame: {df_pd.shape}")
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Average deck composition: {monster_ratio:.1%} monsters, {spell_ratio:.1%} spells, {trap_ratio:.1%} traps")
    logger.info("Game constraints have been applied to all decks")

    top_novel = df_pd[df_pd["generation_mode"] == "novel"].nlargest(5, "cluster_entropy")
    top_meta = df_pd[df_pd["generation_mode"] == "meta_aware"].nlargest(5, "cluster_entropy")

    logger.info("\n=== Top 5 Novel Decks by Cluster Entropy ===")
    for i, (_, row) in enumerate(top_novel.iterrows(), 1):
        logger.info(
            f"{i}. Deck {row['deck_id']} - Entropy: {row['cluster_entropy']:.4f} - "
            f"Archetype: {row['dominant_archetype']}"
        )

    logger.info("\n=== Top 5 Meta-Aware Decks by Cluster Entropy ===")
    for i, (_, row) in enumerate(top_meta.iterrows(), 1):
        logger.info(
            f"{i}. Deck {row['deck_id']} - Entropy: {row['cluster_entropy']:.4f} - "
            f"Archetype: {row['dominant_archetype']}"
        )

    return df_pd


if __name__ == "__main__":
    main()

