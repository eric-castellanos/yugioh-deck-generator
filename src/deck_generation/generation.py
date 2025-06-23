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
from collections import defaultdict

from src.deck_generation.deck_generator import (
    SimpleDeckGenerator, 
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
        log_individual_decks: Whether to log each deck as an artifact (can be resource intensive)
        
    Returns:
        Polars DataFrame with all deck data and metrics
    """
    logger.info(f"Loading clustered cards from MLflow model registry...")
    
    # Load clustered card data
    try:
        clustered_cards = get_card_data_with_clusters()
        logger.info(f"✅ Successfully loaded {len(clustered_cards)} card clusters")
    except Exception as e:
        logger.error(f"❌ Failed to load clusters: {e}")
        raise
    
    # Set up the MLflow experiment
    experiment_id = setup_deck_generation_experiment("yugioh_deck_generation_bulk")
    
    # Calculate number of decks per type
    num_novel = int(total_decks * novel_ratio)
    num_meta = total_decks - num_novel
    
    # If no archetypes provided, identify common archetypes from the clusters
    if not meta_archetypes:
        # Find archetypes with sufficient cards
        archetype_counts = defaultdict(int)
        for cluster_cards in clustered_cards.values():
            for card in cluster_cards:
                if 'archetypes' in card and card['archetypes']:
                    for arch in card['archetypes']:
                        archetype_counts[arch] += 1
        
        # Get top 10 archetypes with at least 20 cards
        meta_archetypes = [arch for arch, count in 
                          sorted(archetype_counts.items(), key=lambda x: x[1], reverse=True)
                          if count >= 20][:10]
    
    logger.info(f"Using meta archetypes: {', '.join(meta_archetypes)}")
    
    # Initialize deck generator with the clustered cards
    # Use DeckGenerator directly with the cards instead of the SimpleDeckGenerator alias
    generator = SimpleDeckGenerator(None, None)  # Create instance
    generator.clustered_cards = clustered_cards  # Set the clustered cards
    
    # Now load the card data properly
    logger.info(f"Processing clusters into card type categories...")
    # Manually organize cards by type and cluster to populate the necessary dictionaries
    for cluster_id, cards in clustered_cards.items():
        main_cards = []
        extra_cards = []
        monsters = []
        spells = []
        traps = []
        
        for card in cards:
            card_type = card.get('type', '')
            
            # Skip if no type or name
            if not card_type or not card.get('name', ''):
                continue
            
            # Categorize by main/extra deck
            if any(et in card_type for et in ['Fusion', 'Synchro', 'Xyz', 'Link']):
                extra_cards.append(card)
            else:
                main_cards.append(card)
                # Further categorize by type for main deck
                if 'Monster' in card_type:
                    monsters.append(card)
                elif 'Spell' in card_type:
                    spells.append(card)
                elif 'Trap' in card_type:
                    traps.append(card)
        
        # Store in appropriate clusters
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
    
    logger.info(f"✅ Processed {len(generator.monster_clusters)} monster clusters, {len(generator.spell_clusters)} spell clusters, {len(generator.trap_clusters)} trap clusters, {len(generator.extra_deck_clusters)} extra deck clusters")
    
    # Start MLflow run for the bulk generation
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Log generation parameters
        log_deck_generation_tags(
            generation_mode="bulk",
            stage="analysis",
            version="v1.0",
            batch_generation=True,
            num_decks=total_decks
        )
        
        log_deck_generation_params(
            generation_mode="bulk",
            deck_count=total_decks,
            novel_ratio=novel_ratio,
            meta_archetypes=meta_archetypes,
            game_constraint_enabled=True  # Log that we're using game constraints
        )
        
        # Keep track of min/max values for normalization
        min_max_values = {
            'entropy': [float('inf'), float('-inf')],
            'distance': [float('inf'), float('-inf')],
            'rarity': [float('inf'), float('-inf')],
            'noise': [float('inf'), float('-inf')]
        }
        
        # Create a list to store all deck data
        all_decks_data = []
        successful_decks = 0
        
        # Progress bar for better visibility
        logger.info(f"Generating {total_decks} decks ({num_novel} novel, {num_meta} meta-aware)...")
        start_time = time.time()
        
        # First pass to generate all decks and calculate min/max values for normalization
        pbar = tqdm(total=total_decks, desc="Generating decks")
        
        # Generate novel decks
        for i in range(num_novel):
            try:
                deck_id = f"novel_{i+1}"
                
                # Generate a novel deck
                main_deck, extra_deck, metadata = generator.generate_deck(
                    mode="novel", 
                    use_mlflow=False  # We'll log manually below
                )
                
                # Extract metrics
                entropy = metadata.cluster_entropy
                distance = metadata.intra_deck_cluster_distance
                rarity = metadata.cluster_co_occurrence_rarity
                noise_pct = metadata.noise_card_percentage
                
                # Update min/max values
                min_max_values['entropy'][0] = min(min_max_values['entropy'][0], entropy)
                min_max_values['entropy'][1] = max(min_max_values['entropy'][1], entropy)
                min_max_values['distance'][0] = min(min_max_values['distance'][0], distance)
                min_max_values['distance'][1] = max(min_max_values['distance'][1], distance)
                min_max_values['rarity'][0] = min(min_max_values['rarity'][0], rarity)
                min_max_values['rarity'][1] = max(min_max_values['rarity'][1], rarity)
                min_max_values['noise'][0] = min(min_max_values['noise'][0], noise_pct)
                min_max_values['noise'][1] = max(min_max_values['noise'][1], noise_pct)
                
                # Log this deck as an artifact if enabled
                if log_individual_decks:
                    try:
                        log_deck_artifacts(
                            main_deck=main_deck,
                            extra_deck=extra_deck,
                            metadata=metadata,
                            prefix=f"deck_{deck_id}"
                        )
                    except Exception as log_err:
                        logger.warning(f"Failed to log deck artifacts for {deck_id}: {log_err}")
                
                # Store deck data
                all_decks_data.append({
                    'deck_id': deck_id,
                    'generation_mode': 'novel',
                    'main_deck_size': len(main_deck),
                    'extra_deck_size': len(extra_deck),
                    'monster_count': metadata.monster_count,
                    'spell_count': metadata.spell_count,
                    'trap_count': metadata.trap_count,
                    'monster_ratio': metadata.monster_count / len(main_deck) if main_deck else 0,
                    'spell_ratio': metadata.spell_count / len(main_deck) if main_deck else 0,
                    'trap_ratio': metadata.trap_count / len(main_deck) if main_deck else 0,
                    'dominant_archetype': metadata.dominant_archetype,
                    'cluster_entropy': entropy,
                    'intra_deck_cluster_distance': distance,
                    'cluster_co_occurrence_rarity': rarity,
                    'noise_card_percentage': noise_pct,
                    'raw_entropy': entropy,
                    'raw_distance': distance,
                    'raw_rarity': rarity,
                    'raw_noise': noise_pct,
                    'main_deck': main_deck,  # Store the full list of main deck cards
                    'extra_deck': extra_deck,  # Store the full list of extra deck cards
                    # Composite score will be calculated in second pass
                })
                
                successful_decks += 1
                pbar.update(1)
                
                # Log progress periodically
                if i % 50 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    logger.info(f"Generated {i}/{total_decks} decks ({rate:.1f} decks/sec)")
                
            except Exception as e:
                logger.warning(f"Failed to generate novel deck {i+1}: {e}")
                pbar.update(1)
        
        # Generate meta-aware decks
        for i in range(num_meta):
            try:
                deck_id = f"meta_{i+1}"
                
                # Choose a random archetype from the list
                target_archetype = random.choice(meta_archetypes)
                
                # Generate a meta-aware deck
                main_deck, extra_deck, metadata = generator.generate_deck(
                    mode="meta_aware", 
                    target_archetype=target_archetype,
                    use_mlflow=False  # We'll log manually below
                )
                
                # Extract metrics
                entropy = metadata.cluster_entropy
                distance = metadata.intra_deck_cluster_distance
                rarity = metadata.cluster_co_occurrence_rarity
                noise_pct = metadata.noise_card_percentage
                
                # Update min/max values
                min_max_values['entropy'][0] = min(min_max_values['entropy'][0], entropy)
                min_max_values['entropy'][1] = max(min_max_values['entropy'][1], entropy)
                min_max_values['distance'][0] = min(min_max_values['distance'][0], distance)
                min_max_values['distance'][1] = max(min_max_values['distance'][1], distance)
                min_max_values['rarity'][0] = min(min_max_values['rarity'][0], rarity)
                min_max_values['rarity'][1] = max(min_max_values['rarity'][1], rarity)
                min_max_values['noise'][0] = min(min_max_values['noise'][0], noise_pct)
                min_max_values['noise'][1] = max(min_max_values['noise'][1], noise_pct)
                
                # Log this deck as an artifact if enabled
                if log_individual_decks:
                    try:
                        log_deck_artifacts(
                            main_deck=main_deck,
                            extra_deck=extra_deck,
                            metadata=metadata,
                            prefix=f"deck_{deck_id}_{target_archetype}"
                        )
                    except Exception as log_err:
                        logger.warning(f"Failed to log deck artifacts for {deck_id} ({target_archetype}): {log_err}")
                
                # Store deck data
                all_decks_data.append({
                    'deck_id': deck_id,
                    'generation_mode': 'meta_aware',
                    'target_archetype': target_archetype,
                    'main_deck_size': len(main_deck),
                    'extra_deck_size': len(extra_deck),
                    'monster_count': metadata.monster_count,
                    'spell_count': metadata.spell_count,
                    'trap_count': metadata.trap_count,
                    'monster_ratio': metadata.monster_count / len(main_deck) if main_deck else 0,
                    'spell_ratio': metadata.spell_count / len(main_deck) if main_deck else 0,
                    'trap_ratio': metadata.trap_count / len(main_deck) if main_deck else 0,
                    'dominant_archetype': metadata.dominant_archetype,
                    'cluster_entropy': entropy,
                    'intra_deck_cluster_distance': distance,
                    'cluster_co_occurrence_rarity': rarity,
                    'noise_card_percentage': noise_pct,
                    'raw_entropy': entropy,
                    'raw_distance': distance,
                    'raw_rarity': rarity,
                    'raw_noise': noise_pct,
                    'main_deck': main_deck,  # Store the full list of main deck cards
                    'extra_deck': extra_deck,  # Store the full list of extra deck cards
                    # Composite score will be calculated in second pass
                })
                
                successful_decks += 1
                pbar.update(1)
                
            except Exception as e:
                logger.warning(f"Failed to generate meta-aware deck for {target_archetype}: {e}")
                pbar.update(1)
        
        pbar.close()
        
        # Create polars DataFrame
        df = pl.DataFrame(all_decks_data)
        
        # Calculate composite scores using min/max values
        # We need to do this in a second pass now that we know the min/max values
        logger.info("Calculating composite scores...")
        
        df = df.with_columns([
            pl.struct([
                pl.col("cluster_entropy").alias("raw_entropy"),
                pl.col("intra_deck_cluster_distance").alias("raw_distance"),
                pl.col("cluster_co_occurrence_rarity").alias("raw_rarity"),
                pl.col("noise_card_percentage").alias("raw_noise"),
            ]).map_elements(lambda row: calculate_composite_score(
                row["raw_entropy"],
                row["raw_distance"],
                row["raw_rarity"],
                row["raw_noise"],
                min_max_values
            )).alias("composite_score")
        ])
        
        # Log summary metrics
        mlflow.log_metric("generated_decks", successful_decks)
        mlflow.log_metric("success_rate", successful_decks / total_decks)
        mlflow.log_metric("avg_composite_score", df["composite_score"].mean())
        mlflow.log_metric("max_composite_score", df["composite_score"].max())
        mlflow.log_metric("min_composite_score", df["composite_score"].min())
        
        # Group by generation mode and calculate statistics
        mode_stats = df.group_by("generation_mode").agg([
            pl.count("deck_id").alias("count"),
            pl.mean("composite_score").alias("avg_score"),
            pl.mean("cluster_entropy").alias("avg_entropy"),
            pl.mean("intra_deck_cluster_distance").alias("avg_distance"),
            pl.mean("cluster_co_occurrence_rarity").alias("avg_rarity"),
            pl.mean("noise_card_percentage").alias("avg_noise")
        ])
        
        # Log the data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            # Convert to pandas and save CSV
            pd_df = df.to_pandas()
            pd_df.to_csv(tmp_file.name, index=False)
            mlflow.log_artifact(tmp_file.name, "all_decks_data.csv")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
            pd_df.to_html(tmp_file, index=False)
            mlflow.log_artifact(tmp_file.name, "all_decks_data.html")
        
        # Log summary statistics
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            # Convert to pandas and save CSV
            pd_mode_stats = mode_stats.to_pandas()
            pd_mode_stats.to_csv(tmp_file.name, index=False)
            mlflow.log_artifact(tmp_file.name, "generation_mode_stats.csv")
        
        # Log information about the run
        logger.info(f"✅ Generated {successful_decks} of {total_decks} decks successfully")
        logger.info(f"✅ Generation complete. Run ID: {run.info.run_id}")
        logger.info(f"✅ MLflow UI: http://localhost:5000/#/experiments/{experiment_id}")
        
        # Return the Polars DataFrame
        return df

def main():
    """Main function to execute the generation process."""
    # Define generation parameters
    num_decks = 1000
    novel_ratio = 0.7
    
    # Generate decks and get results as polars DataFrame
    logger.info(f"Starting generation of {num_decks} decks with game-based constraints...")
    
    # Save the timestamp for the output files
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Generate decks with game constraints enabled
    df_pl = generate_decks(
        total_decks=num_decks, 
        novel_ratio=novel_ratio,
        log_individual_decks=False  # Log each deck to MLflow
    )
    
    # Convert to pandas at the end (for compatibility with other tools)
    df_pd = df_pl.to_pandas()
    
    # Save the output to a CSV file for further analysis
    output_dir = Path("outputs/deck_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / f"deck_generation_{timestamp}.csv"
    df_pd.to_csv(output_file, index=False)
    
    logger.info(f"Generation complete! Shape of final DataFrame: {df_pd.shape}")
    logger.info(f"Results saved to {output_file}")
    
    # Print some basic statistics
    monster_ratios = df_pd['monster_ratio'].mean()
    spell_ratios = df_pd['spell_ratio'].mean()
    trap_ratios = df_pd['trap_ratio'].mean()
    
    logger.info(f"Average deck composition: {monster_ratios:.1%} monsters, {spell_ratios:.1%} spells, {trap_ratios:.1%} traps")
    logger.info(f"Game constraints have been applied to all decks")
    
    # Display summary statistics
    top_novel = df_pd[df_pd['generation_mode'] == 'novel'].nlargest(5, 'composite_score')
    top_meta = df_pd[df_pd['generation_mode'] == 'meta_aware'].nlargest(5, 'composite_score')
    
    logger.info("\n=== Top 5 Novel Decks by Composite Score ===")
    for i, (_, row) in enumerate(top_novel.iterrows()):
        logger.info(f"{i+1}. Deck {row['deck_id']} - Score: {row['composite_score']:.4f} - "
                   f"Archetype: {row['dominant_archetype']}")
    
    logger.info("\n=== Top 5 Meta-Aware Decks by Composite Score ===")
    for i, (_, row) in enumerate(top_meta.iterrows()):
        logger.info(f"{i+1}. Deck {row['deck_id']} - Score: {row['composite_score']:.4f} - "
                   f"Archetype: {row['dominant_archetype']}")
    
    # Save the full dataframe locally
    output_dir = Path("outputs/deck_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"deck_generation_{timestamp}.csv"
    df_pd.to_csv(output_file, index=False)
    
    logger.info(f"Full dataset saved to: {output_file}")
    
    return df_pd

if __name__ == "__main__":
    main()
