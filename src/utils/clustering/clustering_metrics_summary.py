"""
Enhanced clustering metrics summary for Yu-Gi-Oh card data analysis.
"""

import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Union, Optional


def calculate_enhanced_clustering_metrics(
    card_data: Union[pd.DataFrame, pl.DataFrame]
) -> Dict[str, float]:
    """
    Calculate enhanced clustering metrics for card data.
    
    Args:
        card_data: DataFrame with card information and cluster labels
        
    Returns:
        Dictionary with enhanced metrics
    """
    # Convert polars to pandas if needed
    if isinstance(card_data, pl.DataFrame):
        df = card_data.to_pandas()
    else:
        df = card_data.copy()
    
    metrics = {}
    
    if 'cluster' not in df.columns:
        return metrics
    
    # Basic cluster statistics
    unique_clusters = df['cluster'].unique()
    n_clusters = len([c for c in unique_clusters if c != -1])
    n_noise = sum(df['cluster'] == -1) if -1 in unique_clusters else 0
    
    metrics['n_clusters'] = int(n_clusters)
    metrics['n_noise_points'] = int(n_noise)
    metrics['noise_ratio'] = float(n_noise / len(df)) if len(df) > 0 else 0.0
    
    # Cluster size statistics
    cluster_sizes = []
    for cluster_id in unique_clusters:
        if cluster_id != -1:
            size = sum(df['cluster'] == cluster_id)
            cluster_sizes.append(size)
    
    if cluster_sizes:
        metrics['avg_cluster_size'] = float(np.mean(cluster_sizes))
        metrics['std_cluster_size'] = float(np.std(cluster_sizes))
        metrics['min_cluster_size'] = int(np.min(cluster_sizes))
        metrics['max_cluster_size'] = int(np.max(cluster_sizes))
        metrics['cluster_size_cv'] = float(np.std(cluster_sizes) / np.mean(cluster_sizes)) if np.mean(cluster_sizes) > 0 else 0.0
    
    # Archetype diversity per cluster
    if 'archetype' in df.columns:
        archetype_diversities = []
        for cluster_id in unique_clusters:
            if cluster_id != -1:
                cluster_df = df[df['cluster'] == cluster_id]
                if len(cluster_df) > 0:
                    unique_archetypes = cluster_df['archetype'].nunique()
                    cluster_size = len(cluster_df)
                    diversity = unique_archetypes / cluster_size if cluster_size > 0 else 0
                    archetype_diversities.append(diversity)
        
        if archetype_diversities:
            metrics['avg_archetype_diversity'] = float(np.mean(archetype_diversities))
            metrics['std_archetype_diversity'] = float(np.std(archetype_diversities))
    
    # Type distribution analysis
    if 'type' in df.columns:
        type_diversities = []
        for cluster_id in unique_clusters:
            if cluster_id != -1:
                cluster_df = df[df['cluster'] == cluster_id]
                if len(cluster_df) > 0:
                    unique_types = cluster_df['type'].nunique()
                    cluster_size = len(cluster_df)
                    diversity = unique_types / cluster_size if cluster_size > 0 else 0
                    type_diversities.append(diversity)
        
        if type_diversities:
            metrics['avg_type_diversity'] = float(np.mean(type_diversities))
            metrics['std_type_diversity'] = float(np.std(type_diversities))
    
    # Numerical attribute statistics
    numeric_cols = ['atk', 'def', 'level']
    available_numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    for col in available_numeric_cols:
        cluster_means = []
        for cluster_id in unique_clusters:
            if cluster_id != -1:
                cluster_df = df[df['cluster'] == cluster_id]
                if len(cluster_df) > 0 and not cluster_df[col].isna().all():
                    mean_val = cluster_df[col].mean()
                    if not pd.isna(mean_val):
                        cluster_means.append(mean_val)
        
        if cluster_means and len(cluster_means) > 1:
            metrics[f'{col}_cluster_variance'] = float(np.var(cluster_means))
            metrics[f'{col}_cluster_range'] = float(np.max(cluster_means) - np.min(cluster_means))
    
    return metrics


def calculate_cluster_quality_metrics(
    cluster_labels: np.ndarray,
    feature_matrix: np.ndarray,
    card_metadata: Optional[Union[pd.DataFrame, pl.DataFrame]] = None
) -> Dict[str, float]:
    """
    Calculate cluster quality metrics based on feature matrix and labels.
    
    Args:
        cluster_labels: Array of cluster labels
        feature_matrix: Feature matrix used for clustering
        card_metadata: Optional metadata for enhanced metrics
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {}
    
    # Basic cluster statistics
    unique_labels = np.unique(cluster_labels)
    n_clusters = len([label for label in unique_labels if label != -1])
    n_noise = np.sum(cluster_labels == -1) if -1 in unique_labels else 0
    
    metrics['n_clusters'] = int(n_clusters)
    metrics['n_noise_points'] = int(n_noise)
    metrics['noise_percentage'] = float(n_noise / len(cluster_labels) * 100) if len(cluster_labels) > 0 else 0.0
    
    # Intra-cluster distances
    if n_clusters > 0:
        intra_cluster_distances = []
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_mask = cluster_labels == cluster_id
            cluster_points = feature_matrix[cluster_mask]
            
            if len(cluster_points) > 1:
                # Calculate centroid
                centroid = cluster_points.mean(axis=0)
                # Calculate distances to centroid
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                intra_cluster_distances.extend(distances)
        
        if intra_cluster_distances:
            metrics['avg_intra_cluster_distance'] = float(np.mean(intra_cluster_distances))
            metrics['std_intra_cluster_distance'] = float(np.std(intra_cluster_distances))
    
    # Inter-cluster distances (centroids)
    if n_clusters > 1:
        centroids = []
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue
            cluster_mask = cluster_labels == cluster_id
            cluster_points = feature_matrix[cluster_mask]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                centroids.append(centroid)
        
        if len(centroids) > 1:
            inter_distances = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    distance = np.linalg.norm(centroids[i] - centroids[j])
                    inter_distances.append(distance)
            
            metrics['avg_inter_cluster_distance'] = float(np.mean(inter_distances))
            metrics['min_inter_cluster_distance'] = float(np.min(inter_distances))
    
    return metrics
