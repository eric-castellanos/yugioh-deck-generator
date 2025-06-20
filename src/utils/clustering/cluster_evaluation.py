"""
Cluster evaluation utilities for Yu-Gi-Oh card clustering analysis.
"""

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Union, Optional


def analyze_cluster_top_features(
    X: np.ndarray, 
    labels: np.ndarray, 
    top_n: int = 10, 
    feature_names: Optional[List[str]] = None
) -> Dict[int, List[Union[str, int]]]:
    """
    Analyze top features for each cluster based on mean feature values.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        top_n: Number of top features to return per cluster
        feature_names: Optional list of feature names for interpretation
        
    Returns:
        Dictionary mapping cluster_id to list of top features
    """
    cluster_features = {}
    
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Skip noise points in HDBSCAN
            continue
            
        cluster_mask = labels == cluster_id
        cluster_vectors = X[cluster_mask]
        
        if len(cluster_vectors) == 0:
            continue
            
        try:
            mean_vector = cluster_vectors.mean(axis=0)
            
            # Get indices of top features
            top_feature_indices = np.argsort(mean_vector)[::-1][:top_n]
            
            if feature_names is not None and len(feature_names) > 0:
                top_features = [feature_names[i] for i in top_feature_indices if i < len(feature_names)]
            else:
                top_features = top_feature_indices.tolist()
                
            cluster_features[cluster_id] = top_features
        except (IndexError, ValueError) as e:
            # Skip this cluster if there's an issue
            continue
        
    return cluster_features


def find_cluster_representative_cards(
    X: np.ndarray, 
    labels: np.ndarray, 
    card_data: Union[pd.DataFrame, pl.DataFrame],
    top_n: int = 5
) -> Dict[int, List[Dict]]:
    """
    Find representative cards for each cluster (closest to centroid).
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        card_data: DataFrame with card information (must include 'name' column)
        top_n: Number of representative cards per cluster
        
    Returns:
        Dictionary mapping cluster_id to list of representative card info
    """
    cluster_representatives = {}
    
    # Convert polars to pandas if needed
    if isinstance(card_data, pl.DataFrame):
        card_df = card_data.to_pandas()
    else:
        card_df = card_data.copy()
    
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Skip noise points
            continue
            
        cluster_mask = labels == cluster_id
        cluster_vectors = X[cluster_mask]
        
        if len(cluster_vectors) == 0:
            continue
            
        # Calculate centroid
        centroid = cluster_vectors.mean(axis=0).reshape(1, -1)
        
        # Calculate similarities to centroid
        similarities = cosine_similarity(cluster_vectors, centroid).flatten()
        
        # Get indices of most representative cards
        top_indices = similarities.argsort()[::-1][:top_n]
        
        # Map back to original dataframe indices
        original_indices = np.where(cluster_mask)[0][top_indices]
        
        representative_cards = []
        for i, idx in enumerate(original_indices):
            try:
                # Find the similarity score for this card
                sim_score = similarities[top_indices[i]]
                
                card_info = {
                    'index': int(idx),
                    'name': card_df.iloc[idx]['name'] if 'name' in card_df.columns else f'Card_{idx}',
                    'similarity_to_centroid': float(sim_score)
                }
                
                # Add additional card info if available
                for col in ['archetype', 'type', 'attribute', 'atk', 'def', 'level']:
                    if col in card_df.columns:
                        card_info[col] = card_df.iloc[idx][col]
                        
                representative_cards.append(card_info)
            except (IndexError, ValueError) as e:
                # Skip this card if there's an indexing issue
                continue
            
        cluster_representatives[cluster_id] = representative_cards
        
    return cluster_representatives


def analyze_archetype_distribution(
    labels: np.ndarray, 
    card_data: Union[pd.DataFrame, pl.DataFrame],
    normalize: bool = True
) -> Dict[int, Dict[str, float]]:
    """
    Analyze archetype distribution within each cluster.
    
    Args:
        labels: Cluster labels
        card_data: DataFrame with card information (must include 'archetype' column)
        normalize: Whether to return normalized proportions
        
    Returns:
        Dictionary mapping cluster_id to archetype distribution
    """
    # Convert polars to pandas if needed
    if isinstance(card_data, pl.DataFrame):
        card_df = card_data.to_pandas()
    else:
        card_df = card_data.copy()
        
    card_df['cluster'] = labels
    
    archetype_distributions = {}
    
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Skip noise points
            continue
            
        try:
            cluster_data = card_df[card_df['cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
                
            if 'archetype' in cluster_data.columns:
                archetype_counts = cluster_data['archetype'].value_counts(normalize=normalize)
                archetype_distributions[cluster_id] = archetype_counts.to_dict()
            else:
                archetype_distributions[cluster_id] = {}
        except (KeyError, ValueError) as e:
            # Skip this cluster if there's an issue
            archetype_distributions[cluster_id] = {}
            
    return archetype_distributions


def analyze_cluster_stats(
    labels: np.ndarray, 
    card_data: Union[pd.DataFrame, pl.DataFrame],
    numeric_columns: List[str] = ['atk', 'def', 'level']
) -> Dict[int, Dict[str, float]]:
    """
    Analyze statistical properties of clusters for numeric columns.
    
    Args:
        labels: Cluster labels
        card_data: DataFrame with card information
        numeric_columns: List of numeric columns to analyze
        
    Returns:
        Dictionary mapping cluster_id to statistical summaries
    """
    # Convert polars to pandas if needed
    if isinstance(card_data, pl.DataFrame):
        card_df = card_data.to_pandas()
    else:
        card_df = card_data.copy()
        
    card_df['cluster'] = labels
    
    # Filter to available numeric columns
    available_columns = [col for col in numeric_columns if col in card_df.columns]
    
    cluster_stats = {}
    
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Skip noise points
            continue
            
        try:
            cluster_data = card_df[card_df['cluster'] == cluster_id]
            
            if len(cluster_data) > 0 and available_columns:
                stats = cluster_data[available_columns].describe()
                cluster_stats[cluster_id] = stats.loc['mean'].to_dict()
            else:
                cluster_stats[cluster_id] = {}
        except (KeyError, ValueError) as e:
            # Skip this cluster if there's an issue
            cluster_stats[cluster_id] = {}
            
    return cluster_stats


def print_cluster_analysis_summary(
    X: np.ndarray,
    labels: np.ndarray,
    card_data: Union[pd.DataFrame, pl.DataFrame],
    feature_names: Optional[List[str]] = None,
    top_features: int = 10,
    top_cards: int = 5
) -> None:
    """
    Print a comprehensive cluster analysis summary.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        card_data: DataFrame with card information
        feature_names: Optional list of feature names
        top_features: Number of top features to show per cluster
        top_cards: Number of representative cards to show per cluster
    """
    print("=" * 60)
    print("CLUSTER ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Get analysis results
    top_features_dict = analyze_cluster_top_features(X, labels, top_features, feature_names)
    representative_cards = find_cluster_representative_cards(X, labels, card_data, top_cards)
    archetype_dist = analyze_archetype_distribution(labels, card_data)
    cluster_stats = analyze_cluster_stats(labels, card_data)
    
    # Overall statistics
    unique_labels = np.unique(labels)
    n_clusters = len([label for label in unique_labels if label != -1])
    n_noise = np.sum(labels == -1) if -1 in labels else 0
    
    print(f"Total clusters: {n_clusters}")
    print(f"Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    print()
    
    # Per-cluster analysis
    for cluster_id in sorted([label for label in unique_labels if label != -1]):
        print(f"--- CLUSTER {cluster_id} ---")
        cluster_size = np.sum(labels == cluster_id)
        print(f"Size: {cluster_size} cards ({cluster_size/len(labels)*100:.1f}%)")
        
        # Top features
        if cluster_id in top_features_dict:
            features = top_features_dict[cluster_id][:5]  # Show top 5
            print(f"Top features: {features}")
        
        # Representative cards
        if cluster_id in representative_cards:
            print("Representative cards:")
            for i, card in enumerate(representative_cards[cluster_id][:3], 1):
                print(f"  {i}. {card['name']} (similarity: {card['similarity_to_centroid']:.3f})")
        
        # Archetype distribution
        if cluster_id in archetype_dist and archetype_dist[cluster_id]:
            top_archetypes = sorted(archetype_dist[cluster_id].items(), 
                                  key=lambda x: x[1], reverse=True)[:3]
            print(f"Top archetypes: {[f'{arch} ({prop:.2%})' for arch, prop in top_archetypes]}")
        
        # Stats
        if cluster_id in cluster_stats and cluster_stats[cluster_id]:
            stats_str = ", ".join([f"{col}: {val:.1f}" for col, val in cluster_stats[cluster_id].items()])
            print(f"Avg stats: {stats_str}")
    
    if n_noise > 0:
        print(f"\n--- NOISE POINTS ---")
        print(f"Count: {n_noise} cards ({n_noise/len(labels)*100:.1f}%)")


# Example usage function
def evaluate_clustering_results(
    X: np.ndarray,
    labels: np.ndarray, 
    card_data: Union[pd.DataFrame, pl.DataFrame],
    feature_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Comprehensive evaluation of clustering results.
    
    Args:
        X: Feature matrix used for clustering
        labels: Cluster labels from clustering algorithm
        card_data: DataFrame with card metadata
        feature_names: Optional feature names for interpretation
        
    Returns:
        Dictionary with all analysis results
    """
    results = {
        'top_features': analyze_cluster_top_features(X, labels, 10, feature_names),
        'representative_cards': find_cluster_representative_cards(X, labels, card_data, 5),
        'archetype_distribution': analyze_archetype_distribution(labels, card_data),
        'cluster_stats': analyze_cluster_stats(labels, card_data),
        'summary_stats': {
            'n_clusters': len(np.unique(labels)) - (1 if -1 in labels else 0),
            'n_noise': np.sum(labels == -1) if -1 in labels else 0,
            'total_points': len(labels)
        }
    }
    
    return results