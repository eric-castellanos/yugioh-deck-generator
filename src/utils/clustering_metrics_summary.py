"""
Enhanced clustering metrics for Yu-Gi-Oh card analysis.
Focused on summary statistics to avoid overwhelming MLflow with thousands of metrics.
"""

import logging
import numpy as np
import polars as pl
from collections import Counter
from scipy import stats
from typing import Dict, Any, List, Tuple


def calculate_cluster_entropy_summary(meta_df: pl.DataFrame, column: str = "archetype") -> Dict[str, float]:
    """
    Calculate entropy summary statistics for clusters.
    
    Args:
        meta_df: DataFrame with cluster assignments and categorical columns
        column: Column name to calculate entropy for (default: "archetype")
        
    Returns:
        Dictionary with entropy summary metrics (not per-cluster to avoid overwhelming MLflow)
    """
    entropies = {}
    cluster_entropies = []
    cluster_sizes = []
    overall_entropy = 0.0
    total_samples = 0
    
    # Get unique clusters (excluding noise points if present)
    unique_clusters = sorted([c for c in meta_df["cluster"].unique() if c != -1])
    
    for cluster_id in unique_clusters:
        cluster_data = meta_df.filter(pl.col("cluster") == cluster_id)
        
        if len(cluster_data) == 0:
            continue
            
        # Get value counts for the column
        value_counts = cluster_data[column].value_counts()
        total_in_cluster = len(cluster_data)
        
        # Calculate entropy for this cluster
        probabilities = (value_counts["count"] / total_in_cluster).to_numpy()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small epsilon to avoid log(0)
        
        cluster_entropies.append(entropy)
        cluster_sizes.append(total_in_cluster)
        
        # Contribute to overall weighted entropy
        overall_entropy += entropy * total_in_cluster
        total_samples += total_in_cluster
    
    # Calculate summary entropy statistics
    if cluster_entropies:
        entropies[f"{column}_entropy_mean"] = float(np.mean(cluster_entropies))
        entropies[f"{column}_entropy_std"] = float(np.std(cluster_entropies))
        entropies[f"{column}_entropy_min"] = float(np.min(cluster_entropies))
        entropies[f"{column}_entropy_max"] = float(np.max(cluster_entropies))
        entropies[f"{column}_entropy_median"] = float(np.median(cluster_entropies))
        
        # Weighted entropy
        if total_samples > 0:
            entropies[f"{column}_entropy_weighted"] = float(overall_entropy / total_samples)
            
        # Entropy quartiles
        entropies[f"{column}_entropy_q25"] = float(np.percentile(cluster_entropies, 25))
        entropies[f"{column}_entropy_q75"] = float(np.percentile(cluster_entropies, 75))
    
    return entropies


def get_cluster_diversity_summary(meta_df: pl.DataFrame, column: str = "type") -> Dict[str, Any]:
    """
    Get diversity summary statistics across clusters.
    
    Args:
        meta_df: DataFrame with cluster assignments and categorical columns
        column: Column name to analyze (default: "type")
        
    Returns:
        Dictionary with diversity summary metrics
    """
    diversity_metrics = {}
    
    # Get unique clusters (excluding noise points if present)
    unique_clusters = sorted([c for c in meta_df["cluster"].unique() if c != -1])
    
    if not unique_clusters:
        return diversity_metrics
        
    # Overall unique items
    overall_unique_items = len(meta_df[column].unique()) if column in meta_df.columns else 0
    diversity_metrics[f"{column}_total_unique_items"] = overall_unique_items
    
    # Per-cluster diversity metrics
    cluster_unique_counts = []
    cluster_dominant_percentages = []
    
    for cluster_id in unique_clusters:
        cluster_data = meta_df.filter(pl.col("cluster") == cluster_id)
        
        if len(cluster_data) == 0 or column not in cluster_data.columns:
            continue
        
        # Unique items in this cluster
        unique_items_in_cluster = len(cluster_data[column].unique())
        cluster_unique_counts.append(unique_items_in_cluster)
        
        # Dominance: percentage of most common item
        value_counts = cluster_data[column].value_counts()
        if len(value_counts) > 0:
            most_common_count = value_counts["count"][0]
            dominance_percentage = (most_common_count / len(cluster_data)) * 100
            cluster_dominant_percentages.append(dominance_percentage)
    
    # Summary statistics for diversity
    if cluster_unique_counts:
        diversity_metrics[f"{column}_unique_items_mean"] = float(np.mean(cluster_unique_counts))
        diversity_metrics[f"{column}_unique_items_std"] = float(np.std(cluster_unique_counts))
        diversity_metrics[f"{column}_unique_items_min"] = int(np.min(cluster_unique_counts))
        diversity_metrics[f"{column}_unique_items_max"] = int(np.max(cluster_unique_counts))
        diversity_metrics[f"{column}_unique_items_median"] = float(np.median(cluster_unique_counts))
    
    # Summary statistics for dominance
    if cluster_dominant_percentages:
        diversity_metrics[f"{column}_dominance_mean"] = float(np.mean(cluster_dominant_percentages))
        diversity_metrics[f"{column}_dominance_std"] = float(np.std(cluster_dominant_percentages))
        diversity_metrics[f"{column}_dominance_min"] = float(np.min(cluster_dominant_percentages))
        diversity_metrics[f"{column}_dominance_max"] = float(np.max(cluster_dominant_percentages))
        diversity_metrics[f"{column}_dominance_median"] = float(np.median(cluster_dominant_percentages))
    
    return diversity_metrics


def analyze_archetype_distribution_summary(meta_df: pl.DataFrame) -> Dict[str, Any]:
    """
    Analyze archetype distribution summary across clusters.
    
    Args:
        meta_df: DataFrame with cluster assignments and archetype column
        
    Returns:
        Dictionary with archetype distribution summary metrics
    """
    archetype_metrics = {}
    
    if "archetype" not in meta_df.columns:
        return {"archetype_analysis": "archetype column not available"}
    
    # Get unique clusters (excluding noise points if present)
    unique_clusters = sorted([c for c in meta_df["cluster"].unique() if c != -1])
    
    # Overall archetype counts
    overall_archetype_counts = meta_df["archetype"].value_counts()
    total_unique_archetypes = len(overall_archetype_counts)
    
    archetype_metrics["total_unique_archetypes"] = total_unique_archetypes
    archetype_metrics["total_clusters"] = len(unique_clusters)
    
    # Per-cluster archetype diversity
    cluster_archetype_counts = []
    cluster_diversity_ratios = []
    
    for cluster_id in unique_clusters:
        cluster_data = meta_df.filter(pl.col("cluster") == cluster_id)
        
        if len(cluster_data) == 0:
            continue
            
        cluster_archetypes = cluster_data["archetype"].value_counts()
        unique_archetypes_in_cluster = len(cluster_archetypes)
        
        cluster_archetype_counts.append(unique_archetypes_in_cluster)
        diversity_ratio = unique_archetypes_in_cluster / total_unique_archetypes
        cluster_diversity_ratios.append(diversity_ratio)
    
    # Summary statistics
    if cluster_archetype_counts:
        archetype_metrics["archetype_count_per_cluster_mean"] = float(np.mean(cluster_archetype_counts))
        archetype_metrics["archetype_count_per_cluster_std"] = float(np.std(cluster_archetype_counts))
        archetype_metrics["archetype_count_per_cluster_min"] = int(np.min(cluster_archetype_counts))
        archetype_metrics["archetype_count_per_cluster_max"] = int(np.max(cluster_archetype_counts))
        archetype_metrics["archetype_count_per_cluster_median"] = float(np.median(cluster_archetype_counts))
    
    if cluster_diversity_ratios:
        archetype_metrics["archetype_diversity_ratio_mean"] = float(np.mean(cluster_diversity_ratios))
        archetype_metrics["archetype_diversity_ratio_std"] = float(np.std(cluster_diversity_ratios))
        archetype_metrics["archetype_diversity_ratio_min"] = float(np.min(cluster_diversity_ratios))
        archetype_metrics["archetype_diversity_ratio_max"] = float(np.max(cluster_diversity_ratios))
        archetype_metrics["archetype_diversity_ratio_median"] = float(np.median(cluster_diversity_ratios))
    
    return archetype_metrics


def analyze_singleton_vs_multicard_summary(meta_df: pl.DataFrame) -> Dict[str, Any]:
    """
    Analyze singleton vs. multi-card archetype summary across clusters.
    
    Args:
        meta_df: DataFrame with cluster assignments and archetype column
        
    Returns:
        Dictionary with singleton vs multicard summary metrics
    """
    singleton_metrics = {}
    
    if "archetype" not in meta_df.columns:
        return {"singleton_analysis": "archetype column not available"}
    
    # Get overall archetype card counts
    overall_archetype_counts = meta_df["archetype"].value_counts()
    singleton_archetypes = set()
    multicard_archetypes = set()
    
    for row in overall_archetype_counts.iter_rows():
        archetype, count = row
        if count == 1:
            singleton_archetypes.add(archetype)
        else:
            multicard_archetypes.add(archetype)
    
    singleton_metrics["total_singleton_archetypes"] = len(singleton_archetypes)
    singleton_metrics["total_multicard_archetypes"] = len(multicard_archetypes)
    singleton_metrics["singleton_archetype_ratio"] = float(len(singleton_archetypes) / len(overall_archetype_counts))
    
    # Get unique clusters (excluding noise points if present)
    unique_clusters = sorted([c for c in meta_df["cluster"].unique() if c != -1])
    
    cluster_singleton_ratios = []
    cluster_singleton_card_ratios = []
    
    for cluster_id in unique_clusters:
        cluster_data = meta_df.filter(pl.col("cluster") == cluster_id)
        
        if len(cluster_data) == 0:
            continue
            
        cluster_archetypes = set(cluster_data["archetype"].unique())
        
        # Count singletons vs multicards in this cluster
        cluster_singletons = cluster_archetypes.intersection(singleton_archetypes)
        
        # Singleton archetype ratio in cluster
        if len(cluster_archetypes) > 0:
            singleton_ratio = len(cluster_singletons) / len(cluster_archetypes)
            cluster_singleton_ratios.append(singleton_ratio)
        
        # Singleton card ratio in cluster
        singleton_cards = len(cluster_data.filter(pl.col("archetype").is_in(list(cluster_singletons))))
        if len(cluster_data) > 0:
            singleton_card_ratio = singleton_cards / len(cluster_data)
            cluster_singleton_card_ratios.append(singleton_card_ratio)
    
    # Summary statistics for singleton archetype ratios
    if cluster_singleton_ratios:
        singleton_metrics["singleton_archetype_ratio_per_cluster_mean"] = float(np.mean(cluster_singleton_ratios))
        singleton_metrics["singleton_archetype_ratio_per_cluster_std"] = float(np.std(cluster_singleton_ratios))
        singleton_metrics["singleton_archetype_ratio_per_cluster_min"] = float(np.min(cluster_singleton_ratios))
        singleton_metrics["singleton_archetype_ratio_per_cluster_max"] = float(np.max(cluster_singleton_ratios))
        singleton_metrics["singleton_archetype_ratio_per_cluster_median"] = float(np.median(cluster_singleton_ratios))
    
    # Summary statistics for singleton card ratios
    if cluster_singleton_card_ratios:
        singleton_metrics["singleton_card_ratio_per_cluster_mean"] = float(np.mean(cluster_singleton_card_ratios))
        singleton_metrics["singleton_card_ratio_per_cluster_std"] = float(np.std(cluster_singleton_card_ratios))
        singleton_metrics["singleton_card_ratio_per_cluster_min"] = float(np.min(cluster_singleton_card_ratios))
        singleton_metrics["singleton_card_ratio_per_cluster_max"] = float(np.max(cluster_singleton_card_ratios))
        singleton_metrics["singleton_card_ratio_per_cluster_median"] = float(np.median(cluster_singleton_card_ratios))
    
    return singleton_metrics


def calculate_enhanced_clustering_metrics(meta_df: pl.DataFrame) -> Dict[str, Any]:
    """
    Calculate all enhanced clustering metrics as summary statistics.
    
    Args:
        meta_df: DataFrame with cluster assignments and metadata columns
        
    Returns:
        Dictionary with summary clustering metrics (avoiding thousands of individual cluster metrics)
    """
    all_metrics = {}
    
    logging.info("Calculating enhanced clustering summary metrics...")
    
    # Basic cluster statistics
    unique_clusters = sorted([c for c in meta_df["cluster"].unique() if c != -1])
    cluster_sizes = []
    
    for cluster_id in unique_clusters:
        cluster_size = len(meta_df.filter(pl.col("cluster") == cluster_id))
        cluster_sizes.append(cluster_size)
    
    if cluster_sizes:
        all_metrics["num_clusters"] = len(unique_clusters)
        all_metrics["cluster_size_mean"] = float(np.mean(cluster_sizes))
        all_metrics["cluster_size_std"] = float(np.std(cluster_sizes))
        all_metrics["cluster_size_min"] = int(np.min(cluster_sizes))
        all_metrics["cluster_size_max"] = int(np.max(cluster_sizes))
        all_metrics["cluster_size_median"] = float(np.median(cluster_sizes))
        all_metrics["cluster_size_q25"] = float(np.percentile(cluster_sizes, 25))
        all_metrics["cluster_size_q75"] = float(np.percentile(cluster_sizes, 75))
        
        # Cluster size distribution
        all_metrics["cluster_size_skewness"] = float(stats.skew(cluster_sizes))
        all_metrics["cluster_size_kurtosis"] = float(stats.kurtosis(cluster_sizes))
    
    # Noise point analysis (for HDBSCAN)
    noise_points = len(meta_df.filter(pl.col("cluster") == -1))
    if noise_points >= 0:  # Include even if 0
        all_metrics["noise_points"] = noise_points
        all_metrics["noise_percentage"] = float(noise_points / len(meta_df) * 100)
    
    # Entropy summary analysis
    if "archetype" in meta_df.columns:
        archetype_entropy = calculate_cluster_entropy_summary(meta_df, "archetype")
        all_metrics.update(archetype_entropy)
    
    if "type" in meta_df.columns:
        type_entropy = calculate_cluster_entropy_summary(meta_df, "type")
        all_metrics.update(type_entropy)
    
    # Diversity summary analysis
    if "type" in meta_df.columns:
        type_diversity = get_cluster_diversity_summary(meta_df, "type")
        all_metrics.update(type_diversity)
    
    if "attribute" in meta_df.columns:
        attribute_diversity = get_cluster_diversity_summary(meta_df, "attribute")
        all_metrics.update(attribute_diversity)
    
    # Archetype-specific summary analysis
    if "archetype" in meta_df.columns:
        archetype_dist = analyze_archetype_distribution_summary(meta_df)
        all_metrics.update(archetype_dist)
        
        singleton_analysis = analyze_singleton_vs_multicard_summary(meta_df)
        all_metrics.update(singleton_analysis)
    
    logging.info(f"Enhanced summary metrics calculated for {len(unique_clusters)} clusters")
    logging.info(f"Total metrics generated: {len(all_metrics)}")
    
    return all_metrics
