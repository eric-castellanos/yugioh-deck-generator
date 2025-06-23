"""
Generation script for creating and analyzing large sets of Yu-Gi-Oh! decks.

This script generates a large number of decks, calculates various metrics,
and creates a comprehensive dataframe for analysis with both novel and meta-aware decks.
"""

import polars as pl
import pandas as pd
import numpy as np
import random
import os
import tempfile
import math
import time
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from tqdm import tqdm
import mlflow
from collections import defaultdict, Counter

from src.deck_generation.deck_generator import (
    DeckGenerator, 
    DeckMetadata,
    get_card_data_with_clusters
)
from src.utils.mlflow.mlflow_utils import (
    setup_deck_generation_experiment,
    log_deck_generation_tags,
    log_deck_generation_params,
    log_deck_artifacts
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Weights for composite score calculation
WEIGHTS = {
    'cluster_entropy': 0.35,        # Higher entropy = more diverse clusters
    'cluster_distance': 0.25,       # Higher distance = more surprising combinations
    'rarity': 0.30,                 # Higher rarity = more novel combinations
    'noise_penalty': 0.10           # Higher noise percentage = less cohesive deck
}

def normalize_metric(value: float, min_val: float, max_val: float) -> float:
    """Normalize a metric to a 0-1 scale based on min/max values."""
    if max_val == min_val:
        return 0.5  # Default middle value if all values are the same
    return (value - min_val) / (max_val - min_val)

def calculate_composite_score(
    entropy: float, 
    distance: float, 
    rarity: float, 
    noise_pct: float,
    min_max_values: Dict[str, Tuple[float, float]]
) -> float:
    """
    Calculate a composite novelty score.
    
    Args:
        entropy: Cluster entropy score
        distance: Average cluster distance
        rarity: Cluster co-occurrence rarity score
        noise_pct: Percentage of noise cards
        min_max_values: Dictionary with min and max values for each metric for normalization
        
    Returns:
        Composite score (0-1 scale, higher = more novel)
    """
    # Normalize each metric to 0-1 scale
    norm_entropy = normalize_metric(
        entropy, min_max_values['entropy'][0], min_max_values['entropy'][1]
    )
    norm_distance = normalize_metric(
        distance, min_max_values['distance'][0], min_max_values['distance'][1]
    )
    norm_rarity = normalize_metric(
        rarity, min_max_values['rarity'][0], min_max_values['rarity'][1]
    )
    norm_noise = normalize_metric(
        noise_pct, min_max_values['noise'][0], min_max_values['noise'][1]
    )
    
    # Calculate composite score (noise is a negative factor)
    composite_score = (
        WEIGHTS['cluster_entropy'] * norm_entropy +
        WEIGHTS['cluster_distance'] * norm_distance +
        WEIGHTS['rarity'] * norm_rarity -
        WEIGHTS['noise_penalty'] * norm_noise
    )
    
    # Normalize to 0-1 range
    return max(0.0, min(1.0, composite_score))

def generate_decks(
    total_decks: int = 1000, 
    novel_ratio: float = 0.7,
    meta_archetypes: Optional[List[str]] = None,
    log_individual_decks: bool = False  # New parameter to control individual deck logging
) -> pl.DataFrame:
    """
    Generate a specified number of decks, with a mix of novel and meta-aware decks.
    
    Args:
        total_decks: Total number of decks to generate
        novel_ratio: Ratio of novel decks (vs meta-aware)
        meta_archetypes: List of archetypes to use for meta-aware generation
        log_individual_decks: Whether to log each deck to MLflow individually
        
    Returns:
        DataFrame with deck generation metrics
    """
    # Load card clustering model and data
    logger.info("Loading clustered cards from MLflow model registry...")
    clustered_cards = get_card_data_with_clusters()
    logger.info(f"✅ Successfully loaded {len(clustered_cards)} card clusters")
    
    # Initialize MLflow experiment for bulk deck generation
    setup_deck_generation_experiment(experiment_name="yugioh_deck_generation_bulk")
    
    # Initialize deck generator with clustering model
    generator = DeckGenerator(clustering_model=None, card_data=clustered_cards)
    
    # Select meta archetypes if not provided 
    if meta_archetypes is None:
        all_archetypes = []
        for cluster_cards in clustered_cards.values():
            for card in cluster_cards:
                if 'archetype' in card and card['archetype']:
                    all_archetypes.append(card['archetype'])
        
        # Count most common archetypes
        archetype_counts = Counter(all_archetypes)
        meta_archetypes = ["Unknown"] + [arch for arch, _ in archetype_counts.most_common(9)]
    
    logger.info(f"Using meta archetypes: {', '.join(meta_archetypes)}")
    
    # Calculate number of novel vs meta-aware decks
    num_novel = int(total_decks * novel_ratio)
    num_meta = total_decks - num_novel
    
    # Create empty lists to store deck data
    all_decks = []
    
    logger.info(f"Generating {total_decks} decks ({num_novel} novel, {num_meta} meta-aware)...")
    
    # Generate decks with progress bar
    with tqdm(total=total_decks, desc="Generating decks") as pbar:
        # Generate novel decks
        for i in range(num_novel):
            try:
                main_deck, extra_deck, metadata = generator.generate_deck(
                    mode="novel",
                    use_mlflow=log_individual_decks
                )
                
                # Get metrics
                metrics = metadata.get_metrics() 
                
                # Add to our collection
                deck_info = {
                    'deck_id': f"novel_{i+1}",
                    'generation_type': 'novel',
                    'main_deck_size': len(main_deck),
                    'extra_deck_size': len(extra_deck),
                    'monster_count': metrics['monster_count'],
                    'spell_count': metrics['spell_count'],
                    'trap_count': metrics['trap_count'],
                    'monster_ratio': metrics['monster_ratio'],
                    'spell_ratio': metrics['spell_ratio'],
                    'trap_ratio': metrics['trap_ratio']
                }
                
                # Calculate entropy as a measure of card copy distribution
                copy_counts = defaultdict(int)
                for card in main_deck:
                    name = card.get('name', '')
                    if name:
                        copy_counts[name] += 1
                
                # Entropy calculation
                counts = list(copy_counts.values())
                if counts:
                    total_cards = sum(counts)
                    probabilities = [count / total_cards for count in counts]
                    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
                    normalized_entropy = entropy / math.log(len(counts)) if len(counts) > 1 else 0
                else:
                    entropy = 0
                    normalized_entropy = 0
                
                # Additional metrics
                deck_info.update({
                    'unique_cards': len(copy_counts),
                    'raw_entropy': entropy,
                    'normalized_entropy': normalized_entropy,
                    'fusion_count': metrics['fusion_count'],
                    'synchro_count': metrics['synchro_count'],
                    'xyz_count': metrics['xyz_count'],
                    'link_count': metrics['link_count'],
                    'cards_as_1_ofs': metrics.get('cards_as_1_ofs', 0),
                    'cards_as_2_ofs': metrics.get('cards_as_2_ofs', 0),
                    'cards_as_3_ofs': metrics.get('cards_as_3_ofs', 0),
                    'has_tuners': metrics.get('has_tuners', 0),
                    'has_pendulums': metrics.get('has_pendulums', 0),
                })
                
                all_decks.append(deck_info)
                
            except Exception as e:
                logger.warning(f"Failed to generate novel deck {i+1}: {e}")
            finally:
                pbar.update(1)
        
        # Generate meta-aware decks 
        for i in range(num_meta):
            # Randomly select an archetype
            archetype = random.choice(meta_archetypes)
            
            try:
                main_deck, extra_deck, metadata = generator.generate_deck(
                    mode="meta_aware",
                    target_archetype=archetype,
                    use_mlflow=log_individual_decks
                )
                
                # Get metrics
                metrics = metadata.get_metrics()
                
                # Add to our collection
                deck_info = {
                    'deck_id': f"meta_{i+1}",
                    'generation_type': 'meta_aware',
                    'target_archetype': archetype,
                    'main_deck_size': len(main_deck),
                    'extra_deck_size': len(extra_deck),
                    'monster_count': metrics['monster_count'],
                    'spell_count': metrics['spell_count'],
                    'trap_count': metrics['trap_count'],
                    'monster_ratio': metrics['monster_ratio'],
                    'spell_ratio': metrics['spell_ratio'],
                    'trap_ratio': metrics['trap_ratio']
                }
                
                # Calculate entropy as a measure of card copy distribution
                copy_counts = defaultdict(int)
                for card in main_deck:
                    name = card.get('name', '')
                    if name:
                        copy_counts[name] += 1
                
                # Entropy calculation
                counts = list(copy_counts.values())
                if counts:
                    total_cards = sum(counts)
                    probabilities = [count / total_cards for count in counts]
                    try:
                        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
                        normalized_entropy = entropy / math.log(len(counts)) if len(counts) > 1 else 0
                    except (ZeroDivisionError, ValueError):
                        entropy = 0
                        normalized_entropy = 0
                else:
                    entropy = 0
                    normalized_entropy = 0
                
                # Additional metrics
                deck_info.update({
                    'unique_cards': len(copy_counts),
                    'raw_entropy': entropy,
                    'normalized_entropy': normalized_entropy,
                    'fusion_count': metrics['fusion_count'],
                    'synchro_count': metrics['synchro_count'],
                    'xyz_count': metrics['xyz_count'],
                    'link_count': metrics['link_count'],
                    'cards_as_1_ofs': metrics.get('cards_as_1_ofs', 0),
                    'cards_as_2_ofs': metrics.get('cards_as_2_ofs', 0),
                    'cards_as_3_ofs': metrics.get('cards_as_3_ofs', 0),
                    'has_tuners': metrics.get('has_tuners', 0),
                    'has_pendulums': metrics.get('has_pendulums', 0),
                })
                
                all_decks.append(deck_info)
                
            except Exception as e:
                logger.warning(f"Failed to generate meta-aware deck for {archetype}: {e}")
            finally:
                pbar.update(1)
    
    # Create a dataframe with all deck data
    df = pl.DataFrame(all_decks) if all_decks else pl.DataFrame()
    
    if df.height == 0:
        logger.warning("No decks were successfully generated!")
        return df
    
    # Add calculated columns for balance metrics
    
    # Fix entropy calculation issues
    if "raw_entropy" not in df.columns:
        df = df.with_columns([
            pl.lit(0.5).alias("raw_entropy"),
            pl.lit(0.5).alias("normalized_entropy")
        ])
    
    # Ensure all required columns exist with defaults
    required_metrics = [
        "monster_ratio", "spell_ratio", "trap_ratio", 
        "raw_entropy", "normalized_entropy"
    ]
    
    for col in required_metrics:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.5).alias(col))
    
    try:
        # Calculate composite scores
        logger.info("Calculating composite scores...")
        
        # Simple composite score using weighted average of metrics
        df = df.with_columns([
            (pl.col("normalized_entropy") * 0.5 + 
             ((pl.col("monster_ratio") + pl.col("spell_ratio") + pl.col("trap_ratio")) / 3) * 0.5)
            .alias("composite_score")
        ])
        
    except Exception as e:
        logger.error(f"Error calculating composite scores: {e}")
        df = df.with_columns([pl.lit(0.5).alias("composite_score")])
    
    return df
