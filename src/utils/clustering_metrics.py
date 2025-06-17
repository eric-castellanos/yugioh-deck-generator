"""
Enhanced clustering metrics for Yu-Gi-Oh card analysis.
"""

import logging
import numpy as np
import polars as pl
from collections import Counter
from scipy import stats
from typing import Dict, Any, List, Tuple


def calculate_cluster_entropy(meta_df: pl.DataFrame, column: str = "archetype") -> Dict[str, float]:
    """
    Calculate entropy per cluster for a given categorical column.
    
    Args:
        meta_df: DataFrame with cluster assignments and categorical columns
        column: Column name to calculate entropy for (default: "archetype")
        
    Returns:
        Dictionary with cluster-wise entropy metrics
    """
    entropies = {}
    overall_entropy = 0.0
    total_samples = 0
    
    # Get unique clusters (excluding noise points if present)
    unique_clusters = sorted([c for c in meta_df["cluster"].unique() if c != -1])
    
    for cluster_id in unique_clusters:
        cluster_data = meta_df.filter(pl.col("cluster") == cluster_id)
        
        if len(cluster_data) == 0:
            entropies[f"cluster_{cluster_id}_entropy"] = 0.0
            continue
            
        # Get value counts for the column
        value_counts = cluster_data[column].value_counts()
        total_in_cluster = len(cluster_data)
        
        # Calculate entropy for this cluster
        probabilities = (value_counts["count"] / total_in_cluster).to_numpy()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small epsilon to avoid log(0)
        
        entropies[f"cluster_{cluster_id}_entropy"] = float(entropy)
        entropies[f"cluster_{cluster_id}_size"] = int(total_in_cluster)
        
        # Contribute to overall weighted entropy
        overall_entropy += entropy * total_in_cluster
        total_samples += total_in_cluster
    
    # Calculate overall weighted entropy
    if total_samples > 0:
        entropies["overall_weighted_entropy"] = float(overall_entropy / total_samples)
        entropies["avg_cluster_entropy"] = float(np.mean([entropies[k] for k in entropies.keys() if k.endswith("_entropy") and not k.startswith("overall")]))
    else:
        entropies["overall_weighted_entropy"] = 0.0
        entropies["avg_cluster_entropy"] = 0.0
    
    return entropies


def get_top_items_per_cluster(meta_df: pl.DataFrame, column: str = "type", top_n: int = 3) -> Dict[str, Any]:
    """
    Get top items (e.g., card types, archetypes) per cluster.
    
    Args:
        meta_df: DataFrame with cluster assignments and categorical columns
        column: Column name to analyze (default: "type")
        top_n: Number of top items to return per cluster
        
    Returns:
        Dictionary with top items per cluster
    """
    cluster_analysis = {}
    
    # Get unique clusters (excluding noise points if present)
    unique_clusters = sorted([c for c in meta_df["cluster"].unique() if c != -1])
    
    for cluster_id in unique_clusters:
        cluster_data = meta_df.filter(pl.col("cluster") == cluster_id)
        
        if len(cluster_data) == 0:
            cluster_analysis[f"cluster_{cluster_id}_top_{column}"] = []
            continue
        
        # Get value counts and top N
        if column in cluster_data.columns:
            value_counts = cluster_data[column].value_counts().head(top_n)
            
            top_items = []
            for row in value_counts.iter_rows():
                item_name, count = row
                percentage = (count / len(cluster_data)) * 100
                top_items.append({
                    "item": str(item_name),
                    "count": int(count), 
                    "percentage": float(percentage)
                })
            
            cluster_analysis[f"cluster_{cluster_id}_top_{column}"] = top_items
            cluster_analysis[f"cluster_{cluster_id}_unique_{column}_count"] = len(cluster_data[column].unique())
        else:
            cluster_analysis[f"cluster_{cluster_id}_top_{column}"] = []
            cluster_analysis[f"cluster_{cluster_id}_unique_{column}_count"] = 0
    
    return cluster_analysis


def analyze_archetype_distribution(meta_df: pl.DataFrame) -> Dict[str, Any]:
    """
    Analyze archetype distribution across clusters.
    
    Args:
        meta_df: DataFrame with cluster assignments and archetype column
        
    Returns:
        Dictionary with archetype distribution metrics
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
    
    for cluster_id in unique_clusters:
        cluster_data = meta_df.filter(pl.col("cluster") == cluster_id)
        
        if len(cluster_data) == 0:
            continue
            
        cluster_archetypes = cluster_data["archetype"].value_counts()
        unique_archetypes_in_cluster = len(cluster_archetypes)
        
        archetype_metrics[f"cluster_{cluster_id}_unique_archetypes"] = unique_archetypes_in_cluster
        archetype_metrics[f"cluster_{cluster_id}_archetype_diversity"] = float(unique_archetypes_in_cluster / total_unique_archetypes)
        
        # Find dominant archetype in cluster
        if len(cluster_archetypes) > 0:
            dominant_archetype = cluster_archetypes.head(1)
            for row in dominant_archetype.iter_rows():
                archetype_name, count = row
                dominance_percentage = (count / len(cluster_data)) * 100
                archetype_metrics[f"cluster_{cluster_id}_dominant_archetype"] = str(archetype_name)
                archetype_metrics[f"cluster_{cluster_id}_dominance_percentage"] = float(dominance_percentage)
                break
    
    return archetype_metrics


def analyze_singleton_vs_multicard(meta_df: pl.DataFrame) -> Dict[str, Any]:
    """
    Analyze singleton archetypes vs. multi-card archetypes per cluster.
    
    Args:
        meta_df: DataFrame with cluster assignments and archetype column
        
    Returns:
        Dictionary with singleton vs multicard analysis
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
    
    # Get unique clusters (excluding noise points if present)
    unique_clusters = sorted([c for c in meta_df["cluster"].unique() if c != -1])
    
    for cluster_id in unique_clusters:
        cluster_data = meta_df.filter(pl.col("cluster") == cluster_id)
        
        if len(cluster_data) == 0:
            continue
            
        cluster_archetypes = set(cluster_data["archetype"].unique())
        
        # Count singletons vs multicards in this cluster
        cluster_singletons = cluster_archetypes.intersection(singleton_archetypes)
        cluster_multicards = cluster_archetypes.intersection(multicard_archetypes)
        
        singleton_metrics[f"cluster_{cluster_id}_singleton_archetypes"] = len(cluster_singletons)
        singleton_metrics[f"cluster_{cluster_id}_multicard_archetypes"] = len(cluster_multicards)
        singleton_metrics[f"cluster_{cluster_id}_singleton_ratio"] = float(len(cluster_singletons) / len(cluster_archetypes)) if len(cluster_archetypes) > 0 else 0.0
        
        # Count actual cards from singletons vs multicards
        singleton_cards = len(cluster_data.filter(pl.col("archetype").is_in(list(cluster_singletons))))
        multicard_cards = len(cluster_data.filter(pl.col("archetype").is_in(list(cluster_multicards))))
        
        singleton_metrics[f"cluster_{cluster_id}_singleton_cards"] = singleton_cards
        singleton_metrics[f"cluster_{cluster_id}_multicard_cards"] = multicard_cards
        singleton_metrics[f"cluster_{cluster_id}_singleton_card_ratio"] = float(singleton_cards / len(cluster_data)) if len(cluster_data) > 0 else 0.0
    
    return singleton_metrics


def calculate_enhanced_clustering_metrics(meta_df: pl.DataFrame) -> Dict[str, Any]:
    """
    Calculate all enhanced clustering metrics.
    
    Args:
        meta_df: DataFrame with cluster assignments and metadata columns
        
    Returns:
        Dictionary with all enhanced metrics
    """
    all_metrics = {}
    
    logging.info("Calculating enhanced clustering metrics...")
    
    # Entropy analysis
    if "archetype" in meta_df.columns:
        archetype_entropy = calculate_cluster_entropy(meta_df, "archetype")
        all_metrics.update({f"archetype_{k}": v for k, v in archetype_entropy.items()})
    
    if "type" in meta_df.columns:
        type_entropy = calculate_cluster_entropy(meta_df, "type")
        all_metrics.update({f"type_{k}": v for k, v in type_entropy.items()})
    
    # Top items analysis
    if "type" in meta_df.columns:
        type_analysis = get_top_items_per_cluster(meta_df, "type", top_n=3)
        all_metrics.update(type_analysis)
    
    if "attribute" in meta_df.columns:
        attribute_analysis = get_top_items_per_cluster(meta_df, "attribute", top_n=3)
        all_metrics.update(attribute_analysis)
    
    # Archetype-specific analysis
    if "archetype" in meta_df.columns:
        archetype_dist = analyze_archetype_distribution(meta_df)
        all_metrics.update(archetype_dist)
        
        singleton_analysis = analyze_singleton_vs_multicard(meta_df)
        all_metrics.update(singleton_analysis)
    
    # Basic cluster size statistics
    unique_clusters = sorted([c for c in meta_df["cluster"].unique() if c != -1])
    cluster_sizes = []
    
    for cluster_id in unique_clusters:
        cluster_size = len(meta_df.filter(pl.col("cluster") == cluster_id))
        cluster_sizes.append(cluster_size)
        all_metrics[f"cluster_{cluster_id}_size"] = cluster_size
    
    if cluster_sizes:
        all_metrics["avg_cluster_size"] = float(np.mean(cluster_sizes))
        all_metrics["median_cluster_size"] = float(np.median(cluster_sizes))
        all_metrics["min_cluster_size"] = int(np.min(cluster_sizes))
        all_metrics["max_cluster_size"] = int(np.max(cluster_sizes))
        all_metrics["cluster_size_std"] = float(np.std(cluster_sizes))
    
    # Noise point analysis (for HDBSCAN)
    noise_points = len(meta_df.filter(pl.col("cluster") == -1))
    if noise_points > 0:
        all_metrics["noise_points"] = noise_points
        all_metrics["noise_percentage"] = float(noise_points / len(meta_df) * 100)
    
    logging.info(f"Enhanced metrics calculated for {len(unique_clusters)} clusters")
    
    return all_metrics
