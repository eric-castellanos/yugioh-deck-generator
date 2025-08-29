#!/usr/bin/env python3
"""
Feature Analysis Script for MLP Model

This script loads the MLP dataset and analyzes all features by their correlation
to the target variable (adjusted_win_rate).
"""

import sys
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model.deck_scoring.pytorch_mlp.Dataset import create_cv_splits_no_leakage


def analyze_features():
    """Analyze all features used in the MLP and rank by correlation to win_rate"""
    print("🔍 FEATURE ANALYSIS FOR MLP MODEL")
    print("=" * 50)
    
    print("Loading dataset...")
    # Get the first split to analyze features (need at least 2 splits for GroupKFold)
    splits = create_cv_splits_no_leakage(use_pca=False, n_splits=2)
    train_dataset, val_dataset, preprocessor = next(iter(splits))
    
    X_train = train_dataset.X
    y_train = train_dataset.y
    feature_names = preprocessor.feature_names_
    
    print(f"Dataset loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Target variable: adjusted_win_rate")
    print(f"Target range: {np.min(y_train):.4f} to {np.max(y_train):.4f}")
    print()
    
    # Debug: Check for interaction features
    interaction_features = ['interact_low_level_xyz', 'interact_monster_p_two', 'interact_non_tuner_unique']
    found_interactions = [f for f in interaction_features if f in feature_names]
    print(f"🔧 Interaction features found: {found_interactions}")
    if len(found_interactions) != 3:
        missing = [f for f in interaction_features if f not in feature_names]
        print(f"⚠️  Missing interaction features: {missing}")
    print()
    
    # Debug: Show dense features from preprocessor
    print(f"📋 Dense features in preprocessor: {len(preprocessor.dense_features)}")
    dense_interactions = [f for f in preprocessor.dense_features if 'interact_' in f]
    print(f"   Interaction features in dense list: {dense_interactions}")
    print()
    
    # Calculate correlations for each feature
    correlations = []
    
    print("Calculating correlations...")
    for i, feature_name in enumerate(feature_names):
        if i < X_train.shape[1]:  # Safety check
            feature_values = X_train[:, i]
            
            # Skip features with zero variance
            if np.std(feature_values) < 1e-8:
                correlations.append((feature_name, 0.0, 0.0, "zero_variance"))
                continue
            
            # Calculate both Pearson and Spearman correlations
            try:
                pearson_corr = pearsonr(feature_values, y_train)[0]
                spearman_corr = spearmanr(feature_values, y_train)[0]
                
                # Handle NaN values
                if np.isnan(pearson_corr):
                    pearson_corr = 0.0
                if np.isnan(spearman_corr):
                    spearman_corr = 0.0
                    
                correlations.append((feature_name, pearson_corr, spearman_corr, "normal"))
                
            except Exception as e:
                correlations.append((feature_name, 0.0, 0.0, f"error: {str(e)[:20]}"))
    
    # Sort by absolute Spearman correlation (more robust to outliers)
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\n📊 FEATURE CORRELATION ANALYSIS")
    print("-" * 80)
    print(f"{'Rank':<4} {'Feature Name':<35} {'Pearson':<8} {'Spearman':<8} {'Status'}")
    print("-" * 80)
    
    # Print all features
    for rank, (feature_name, pearson, spearman, status) in enumerate(correlations, 1):
        # Truncate long feature names
        display_name = feature_name[:32] + "..." if len(feature_name) > 35 else feature_name
        
        print(f"{rank:<4} {display_name:<35} {pearson:>7.4f} {spearman:>7.4f} {status}")
    
    print("-" * 80)
    
    # Summary statistics
    valid_correlations = [(p, s) for _, p, s, status in correlations if status == "normal"]
    if valid_correlations:
        pearson_vals = [p for p, s in valid_correlations]
        spearman_vals = [s for p, s in valid_correlations]
        
        print(f"\n📈 CORRELATION SUMMARY:")
        print(f"Valid features: {len(valid_correlations)}")
        print(f"Pearson  - Mean: {np.mean(np.abs(pearson_vals)):.4f}, Max: {np.max(np.abs(pearson_vals)):.4f}")
        print(f"Spearman - Mean: {np.mean(np.abs(spearman_vals)):.4f}, Max: {np.max(np.abs(spearman_vals)):.4f}")
    
    # Highlight potentially problematic features
    print(f"\n⚠️  POTENTIAL LEAKAGE FEATURES (|correlation| > 0.8):")
    leakage_features = [(name, p, s) for name, p, s, status in correlations 
                       if status == "normal" and (abs(p) > 0.8 or abs(s) > 0.8)]
    
    if leakage_features:
        for feature_name, pearson, spearman in leakage_features:
            print(f"   {feature_name}: Pearson={pearson:.4f}, Spearman={spearman:.4f}")
    else:
        print("   ✅ No features with suspiciously high correlation found")
    
    # Show zero variance features
    zero_var_features = [name for name, _, _, status in correlations if status == "zero_variance"]
    if zero_var_features:
        print(f"\n⚠️  ZERO VARIANCE FEATURES ({len(zero_var_features)}):")
        for feature_name in zero_var_features:
            print(f"   {feature_name}")
    
    # Show features by category
    print(f"\n🏷️  FEATURES BY CATEGORY:")
    categories = {
        'Interaction': [],
        'Cluster': [],
        'Mechanic': [],
        'NLP/Text': [],
        'Basic Stats': [],
        'Ranking': [],
        'Other': []
    }
    
    for feature_name, pearson, spearman, status in correlations:
        if 'interact_' in feature_name.lower():
            categories['Interaction'].append((feature_name, spearman))
        elif any(word in feature_name.lower() for word in ['cluster', 'entropy', 'distance', 'rarity', 'noise']):
            categories['Cluster'].append((feature_name, spearman))
        elif any(word in feature_name.lower() for word in ['tuner', 'synchro', 'xyz', 'fusion', 'link', 'pendulum', 'can_']):
            categories['Mechanic'].append((feature_name, spearman))
        elif any(word in feature_name.lower() for word in ['bow_', 'tfidf', 'embedding', 'banish', 'graveyard', 'draw', 'search', 'summon', 'negate', 'destroy', 'shuffle']):
            categories['NLP/Text'].append((feature_name, spearman))
        elif any(word in feature_name.lower() for word in ['confidence', 'percentile', 'combo_potential']):
            categories['Ranking'].append((feature_name, spearman))
        elif any(word in feature_name.lower() for word in ['level', 'copies', 'unique', 'count', 'ratio']):
            categories['Basic Stats'].append((feature_name, spearman))
        else:
            categories['Other'].append((feature_name, spearman))
    
    for category, features in categories.items():
        if features:
            # Sort by absolute correlation within category
            features.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"\n   {category} ({len(features)} features):")
            for feature_name, spearman in features[:5]:  # Show top 5 in each category
                print(f"     {feature_name:<30} {spearman:>7.4f}")
            if len(features) > 5:
                print(f"     ... and {len(features) - 5} more")


if __name__ == "__main__":
    analyze_features()
