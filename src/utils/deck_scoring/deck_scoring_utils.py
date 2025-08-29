"""
Deck Scoring Utilities for Yu-Gi-Oh! Deck Generator.

This module provides simplified scoring functions for calculating various metrics:
- Cluster entropy
- Intra-deck cluster distance
- Cluster co-occurrence rarity
- Noise card percentage
- Cluster distribution
- Archetype distribution
"""

import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional


def calculate_cluster_entropy(cluster_distribution: Dict[str, int]) -> float:
    """
    Calculate Shannon entropy of cluster distribution in a deck.
    Higher entropy means more diverse clusters.
    
    Args:
        cluster_distribution: Dictionary mapping cluster IDs to counts in deck
        
    Returns:
        Entropy score (higher = more diverse)
    """
    if not cluster_distribution:
        return 0.0
        
    total_cards = sum(cluster_distribution.values())
    if total_cards == 0:
        return 0.0
    
    # Calculate normalized probabilities
    probabilities = [count / total_cards for count in cluster_distribution.values()]
    
    # Shannon entropy formula: -sum(p * log2(p))
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    return entropy


def calculate_intra_deck_cluster_distance(
    cluster_distribution: Dict[str, int],
    cluster_embeddings: Optional[Dict[str, List[float]]] = None
) -> float:
    """
    Calculate average distance between clusters in a deck.
    Higher distance means more diverse/surprising combinations.
    
    Args:
        cluster_distribution: Dictionary mapping cluster IDs to counts in deck
        cluster_embeddings: Dictionary mapping cluster IDs to vector embeddings
        
    Returns:
        Average distance score (higher = more surprising combinations)
    """
    if not cluster_distribution or len(cluster_distribution) <= 1 or not cluster_embeddings:
        return 0.0
    
    # Get clusters that are both in the deck and have embeddings
    valid_clusters = [
        cluster_id for cluster_id in cluster_distribution
        if cluster_id in cluster_embeddings
    ]
    
    if len(valid_clusters) <= 1:
        return 0.0
    
    # Calculate pairwise distances between all clusters
    total_distance = 0.0
    pair_count = 0
    
    for i, cluster1 in enumerate(valid_clusters):
        emb1 = cluster_embeddings.get(cluster1)
        if not emb1:
            continue
            
        for cluster2 in valid_clusters[i+1:]:
            emb2 = cluster_embeddings.get(cluster2)
            if not emb2:
                continue
                
            # Calculate Euclidean distance
            distance = np.linalg.norm(np.array(emb1) - np.array(emb2))
            total_distance += distance
            pair_count += 1
    
    # Return average distance
    if pair_count > 0:
        return total_distance / pair_count
    return 0.0


def calculate_cluster_cooccurrence_rarity(
    cluster_distribution: Dict[str, int],
    global_cooccurrence: Optional[Dict[Tuple[str, str], float]] = None
) -> float:
    """
    Calculate how rare the co-occurrence of clusters in this deck is.
    Higher score means more novel combinations.
    
    Args:
        cluster_distribution: Dictionary mapping cluster IDs to counts in deck
        global_cooccurrence: Dictionary mapping cluster pairs to global frequency
        
    Returns:
        Rarity score (higher = more novel combinations)
    """
    if not cluster_distribution or len(cluster_distribution) <= 1 or not global_cooccurrence:
        return 0.0
    
    clusters = list(cluster_distribution.keys())
    
    # If no global data available, just return 0
    if not global_cooccurrence:
        return 0.0
    
    # Calculate average rarity (1 - frequency) of all cluster pairs
    total_rarity = 0.0
    pair_count = 0
    
    for i, cluster1 in enumerate(clusters):
        for cluster2 in clusters[i+1:]:
            # Try both orders of the pair
            pair1 = (cluster1, cluster2)
            pair2 = (cluster2, cluster1)
            
            frequency = global_cooccurrence.get(pair1, global_cooccurrence.get(pair2, 0.0))
            
            # Inverse frequency is rarity
            rarity = 1.0 - min(1.0, frequency)
            total_rarity += rarity
            pair_count += 1
    
    # Return average rarity
    if pair_count > 0:
        return total_rarity / pair_count
    return 0.0


def calculate_noise_card_percentage(
    main_deck: List[Dict], 
    clustered_cards: Dict[str, List[Dict]],
    card_to_cluster: Optional[Dict[str, str]] = None
) -> float:
    """
    Calculate percentage of cards labeled as noise (cluster = -1).
    Lower is generally better for deck coherence.
    
    Args:
        main_deck: List of card dictionaries in the main deck
        clustered_cards: Dictionary of all clustered cards
        card_to_cluster: Dictionary mapping card names to cluster IDs
        
    Returns:
        Percentage of noise cards (0.0 to 1.0)
    """
    if not main_deck:
        return 0.0
    
    # If we have a mapping of cards to clusters
    if card_to_cluster:
        # Count cards with cluster label -1 (may be stored as integer or string)
        noise_cards = sum(1 for card in main_deck if 
                          card.get('name', '') in card_to_cluster and 
                          (card_to_cluster[card.get('name', '')] == '-1' or 
                           card_to_cluster[card.get('name', '')] == -1))
    else:
        # Create a set of all card names in clusters
        clustered_card_names = set()
        for cluster_id, cards in clustered_cards.items():
            # Skip the noise cluster (labeled as -1)
            if cluster_id == '-1':
                continue
            
            for card in cards:
                name = card.get('name', '')
                if name:
                    clustered_card_names.add(name)
        
        # Count cards not in any valid cluster
        noise_cards = sum(1 for card in main_deck if card.get('name', '') not in clustered_card_names)
    
    return noise_cards / len(main_deck) if main_deck else 0.0


def get_cluster_distribution(
    all_cards: List[Dict], 
    card_to_cluster: Optional[Dict[str, str]] = None
) -> Dict[str, int]:
    """
    Get distribution of cards across clusters.
    
    Args:
        all_cards: List of all cards in the deck
        card_to_cluster: Dictionary mapping card names to cluster IDs
        
    Returns:
        Dictionary mapping cluster IDs to counts
    """
    if not all_cards or not card_to_cluster:
        return {}
    
    distribution = Counter()
    
    for card in all_cards:
        card_name = card.get('name', '')
        if card_name and card_name in card_to_cluster:
            cluster_id = str(card_to_cluster[card_name])  # Convert to string for consistency
            distribution[cluster_id] += 1
    
    return dict(distribution)


def get_archetype_distribution(all_cards: List[Dict]) -> Dict[str, int]:
    """
    Get distribution of archetypes in the deck.
    
    Args:
        all_cards: List of all cards in the deck
        
    Returns:
        Dictionary mapping archetypes to counts
    """
    archetypes = Counter()
    
    for card in all_cards:
        # Check for archetypes list
        if 'archetypes' in card and isinstance(card['archetypes'], list):
            for archetype in card['archetypes']:
                archetypes[archetype] += 1
        
        # Check for single archetype field
        elif 'archetype' in card and card['archetype']:
            archetype = card['archetype']
            archetypes[archetype] += 1
    
    return dict(archetypes)


def get_dominant_archetype(archetype_distribution: Dict[str, int]) -> str:
    """
    Get the dominant archetype in the deck based on card count.
    
    Args:
        archetype_distribution: Dictionary mapping archetypes to counts
        
    Returns:
        Name of dominant archetype or "Mixed" if none dominant
    """
    if not archetype_distribution:
        return "Unknown"
    
    total_cards = sum(archetype_distribution.values())
    
    # Get the archetype with highest count
    dominant_archetype, highest_count = max(
        archetype_distribution.items(), 
        key=lambda x: x[1], 
        default=("Unknown", 0)
    )
    
    # If the highest archetype has at least 30% of cards, consider it dominant
    if highest_count >= 0.3 * total_cards and highest_count >= 3:
        return dominant_archetype
    
    return "Mixed"


def create_card_to_cluster_mapping(clustered_cards: Dict[str, List[Dict]]) -> Dict[str, str]:
    """
    Create a mapping from card name to its cluster ID.
    
    Args:
        clustered_cards: Dictionary mapping cluster IDs to lists of cards
        
    Returns:
        Dictionary mapping card names to cluster IDs
    """
    card_to_cluster = {}
    
    for cluster_id, cards in clustered_cards.items():
        # Convert cluster_id to string for consistency
        str_cluster_id = str(cluster_id)
        for card in cards:
            name = card.get('name', '')
            if name:
                card_to_cluster[name] = str_cluster_id
    
    return card_to_cluster


def calculate_prior_probability(battle_results_df) -> float:
    """
    Calculate prior probability (p) from battle results data.
    
    This estimates the overall win rate across all decks and battles,
    used as a Bayesian prior for win rate adjustment.
    
    Args:
        battle_results_df: DataFrame with columns: deck_id, win_rate, successful_battles, total_battles
        
    Returns:
        Prior probability (overall win rate across all data)
    """
    # Filter out failed battles
    valid_results = battle_results_df[battle_results_df['win_rate'] != 'FAILED'].copy()
    
    if len(valid_results) == 0:
        return 0.5  # Default to neutral prior if no valid data
    
    # Convert win_rate to numeric if it's not already
    valid_results['win_rate'] = pd.to_numeric(valid_results['win_rate'], errors='coerce')
    
    # Remove any rows where conversion failed
    valid_results = valid_results.dropna(subset=['win_rate'])
    
    if len(valid_results) == 0:
        return 0.5
    
    # Calculate total duels won and total duels played
    # Each successful battle consists of 6 duels
    # duels_won = win_rate * 6 * successful_battles
    # duels_played = 6 * successful_battles
    valid_results['duels_won'] = valid_results['win_rate'] * 6 * valid_results['successful_battles']
    valid_results['duels_played'] = 6 * valid_results['successful_battles']
    
    total_duels_won = valid_results['duels_won'].sum()
    total_duels_played = valid_results['duels_played'].sum()
    
    if total_duels_played == 0:
        return 0.5
    
    prior_probability = total_duels_won / total_duels_played
    return prior_probability


def add_bayesian_adjusted_win_rate(battle_results_df) -> None:
    """
    Add Bayesian adjusted win rate column to battle results DataFrame.
    
    Uses the formula:
    adjusted_win_rate = (successful_battles * win_rate + p * total_battles) / (successful_battles + total_battles)
    
    Where p is the prior probability calculated from the entire dataset.
    
    Args:
        battle_results_df: DataFrame with columns: deck_id, win_rate, successful_battles, total_battles
                          Will be modified in place to add 'adjusted_win_rate' column
    """
    import pandas as pd
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Calculate prior probability from the dataset
    p = calculate_prior_probability(battle_results_df)
    logger.info(f"Calculated prior probability (p) = {p:.4f}")
    
    # Filter out failed battles for adjustment calculation
    valid_mask = battle_results_df['win_rate'] != 'FAILED'
    
    # Initialize adjusted_win_rate column with original win_rate
    battle_results_df['adjusted_win_rate'] = battle_results_df['win_rate']
    
    # Convert win_rate to numeric for valid rows
    valid_results = battle_results_df[valid_mask].copy()
    valid_results['win_rate_numeric'] = pd.to_numeric(valid_results['win_rate'], errors='coerce')
    
    # Remove rows where conversion failed
    valid_numeric_mask = valid_results['win_rate_numeric'].notna()
    valid_results = valid_results[valid_numeric_mask]
    
    if len(valid_results) == 0:
        logger.warning("No valid data to adjust")
        return  # No valid data to adjust
    
    # Calculate Bayesian adjusted win rate
    # Formula: (successful_battles * win_rate + p * total_battles) / (successful_battles + total_battles)
    numerator = (valid_results['successful_battles'] * valid_results['win_rate_numeric'] + 
                p * valid_results['total_battles'])
    denominator = valid_results['successful_battles'] + valid_results['total_battles']
    
    # Avoid division by zero
    denominator = denominator.replace(0, 1)
    
    adjusted_win_rates = numerator / denominator
    
    # Update the original DataFrame
    battle_results_df.loc[valid_results.index, 'adjusted_win_rate'] = adjusted_win_rates
    logger.info(f"Applied Bayesian adjustment to {len(valid_results)} rows")
