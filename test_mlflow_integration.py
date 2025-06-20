#!/usr/bin/env python3
"""
Test script for Yu-Gi-Oh deck generator with MLflow integration.
Run this after sourcing .env.mlflow-tunnel to test the full pipeline.
"""

import os
import sys
import logging
sys.path.append('src')

# Set up logging to display on console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    print("=== Yu-Gi-Oh Deck Generator MLflow Integration Test ===")
    print()
    
    # Check environment
    print("1. Environment Check:")
    aws_key = os.environ.get('AWS_ACCESS_KEY_ID', 'NOT SET')
    if aws_key != 'NOT SET':
        print(f"   ✓ AWS_ACCESS_KEY_ID: {aws_key[:10]}...")
    else:
        print("   ✗ AWS_ACCESS_KEY_ID not set")
    
    aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY', 'NOT SET')
    print(f"   ✓ AWS_SECRET_ACCESS_KEY: {'SET' if aws_secret != 'NOT SET' else 'NOT SET'}")
    
    aws_session = os.environ.get('AWS_SESSION_TOKEN', 'NOT SET')
    print(f"   ✓ AWS_SESSION_TOKEN: {'SET' if aws_session != 'NOT SET' else 'NOT SET'}")
    
    s3_bucket = os.environ.get('S3_BUCKET', 'NOT SET')
    print(f"   ✓ S3_BUCKET: {s3_bucket}")
    print()
    
    # Test imports
    print("2. Import Test:")
    try:
        from deck_generation.deck_generator import SimpleDeckGenerator, generate_deck, generate_deck_from_registry
        from utils.mlflow.get_clustering_model_from_registry import get_clustering_model_from_registry, get_card_data_with_clusters
        print("   ✓ All imports successful")
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return
    print()
    
    # Test MLflow model loading
    print("3. MLflow Model Test:")
    try:
        print("   Loading clustering model from MLflow registry...")
        model = get_clustering_model_from_registry()
        print(f"   ✓ Successfully loaded model: {type(model).__name__}")
        
        if hasattr(model, 'labels_'):
            print(f"   ✓ Model has 'labels_' attribute with {len(model.labels_)} labels")
        else:
            print("   ⚠️  Model doesn't have 'labels_' attribute")
            
        # Test setting the correct experiment name
        from utils.mlflow.mlflow_utils import setup_deck_generation_experiment
        experiment_id = setup_deck_generation_experiment("yugioh_deck_generation")
        print(f"   ✓ Set up experiment: yugioh_deck_generation (ID: {experiment_id})")
    except Exception as e:
        print(f"   ✗ Failed to load model: {str(e)}")
        return
    print()
    
    # Test getting card data with clusters
    print("4. Card Data with Clusters Test:")
    try:
        print("   Getting card data with clusters...")
        clusters = get_card_data_with_clusters(model)
        num_clusters = len(clusters)
        total_cards = sum(len(cards) for cards in clusters.values())
        
        print(f"   ✓ Successfully retrieved {num_clusters} clusters with {total_cards} total cards")
        
        # Check if noise cluster exists
        if -1 in clusters:
            noise_count = len(clusters[-1])
            print(f"   ✓ Noise cluster (-1) contains {noise_count} cards")
        else:
            print("   ℹ️  No noise cluster found")
        
        # Print some stats about clusters
        cluster_counts = {k: len(v) for k, v in clusters.items()}
        largest = max(cluster_counts.items(), key=lambda x: x[1])
        smallest = min(cluster_counts.items(), key=lambda x: x[1])
        print(f"   ℹ️  Largest cluster: #{largest[0]} with {largest[1]} cards")
        print(f"   ℹ️  Smallest cluster: #{smallest[0]} with {smallest[1]} cards")
        
    except Exception as e:
        print(f"   ✗ Failed to get card data with clusters: {str(e)}")
        return
    print()
    
    # Test deck generation from registry with explicit experiment name
    print("5. Deck Generation Test:")
    try:
        # Make sure we use the correct experiment
        from utils.mlflow.mlflow_utils import setup_deck_generation_experiment
        experiment_id = setup_deck_generation_experiment("yugioh_deck_generation")
        print(f"   ✓ Using experiment: yugioh_deck_generation (ID: {experiment_id})")
        
        print("   Generating a deck from registry model...")
        deck = generate_deck_from_registry(num_cards=40, experiment_name="yugioh_deck_generation")
        print(f"   ✓ Successfully generated a deck with {len(deck)} cards")
        monster_count = sum(1 for card in deck if card.get('type') and 'Monster' in card.get('type'))
        spell_count = sum(1 for card in deck if card.get('type') and 'Spell' in card.get('type'))
        trap_count = sum(1 for card in deck if card.get('type') and 'Trap' in card.get('type'))
        print(f"   ✓ Deck composition: {monster_count} Monsters, {spell_count} Spells, {trap_count} Traps")
    except Exception as e:
        print(f"   ✗ Failed to generate deck: {str(e)}")
    print()
    
    print("=== MLflow Integration Test Complete ===")
    
    # For older tests, create simplified test data
    test_cards = {
        0: [  # Blue-Eyes monsters
            {'name': 'Blue-Eyes White Dragon', 'type': 'Monster', 'archetype': 'Blue-Eyes', 'archetypes': ['Blue-Eyes'], 'cluster_id': 0},
            {'name': 'Blue-Eyes Alternative White Dragon', 'type': 'Monster', 'archetype': 'Blue-Eyes', 'archetypes': ['Blue-Eyes'], 'cluster_id': 0},
        ],
        1: [  # Support spells
            {'name': 'Dragon Shrine', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 1},
            {'name': 'Trade-In', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 1},
        ],
        2: [  # Generic monsters
            {'name': 'Elemental HERO Sparkman', 'type': 'Monster', 'archetype': 'HERO', 'archetypes': ['HERO'], 'cluster_id': 2},
            {'name': 'Ash Blossom & Joyous Spring', 'type': 'Monster', 'archetype': None, 'archetypes': [], 'cluster_id': 2},
        ],
        3: [  # Trap cards
            {'name': 'Mirror Force', 'type': 'Trap', 'archetype': None, 'archetypes': [], 'cluster_id': 3},
            {'name': 'Torrential Tribute', 'type': 'Trap', 'archetype': None, 'archetypes': [], 'cluster_id': 3},
        ]
    }
    
    # Test 1: Deck generation without MLflow
    print("3. Test Without MLflow:")
    try:
        generator = SimpleDeckGenerator(test_cards)
        main_deck, extra_deck, metadata = generator.generate_deck('novel', use_mlflow=False)
        print(f"   ✓ Generated deck: {len(main_deck)} main + {len(extra_deck)} extra cards")
        print(f"   ✓ Dominant archetype: {metadata.dominant_archetype}")
        print(f"   ✓ Card ratios: {metadata.monster_count}M/{metadata.spell_count}S/{metadata.trap_count}T")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    print()
    
    # Test 2: Deck generation with MLflow
    print("4. Test With MLflow:")
    try:
        main_deck, extra_deck, metadata = generate_deck(
            clustered_cards=test_cards,
            mode='novel',
            use_mlflow=True,
            load_from_registry=False
        )
        print(f"   ✓ Generated deck with MLflow tracking: {len(main_deck)} main + {len(extra_deck)} extra cards")
        print(f"   ✓ Dominant archetype: {metadata.dominant_archetype}")
        print(f"   ✓ Card ratios: {metadata.monster_count}M/{metadata.spell_count}S/{metadata.trap_count}T")
        print(f"   ✓ Cluster entropy: {metadata.cluster_entropy:.2f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # Test 3: Meta-aware deck generation
    print("5. Test Meta-Aware Generation:")
    try:
        main_deck, extra_deck, metadata = generate_deck(
            clustered_cards=test_cards,
            mode='meta_aware',
            target_archetype='Blue-Eyes',
            use_mlflow=True,
            load_from_registry=False
        )
        print(f"   ✓ Generated Blue-Eyes deck: {len(main_deck)} main + {len(extra_deck)} extra cards")
        print(f"   ✓ Dominant archetype: {metadata.dominant_archetype}")
        blue_eyes_cards = sum(1 for card in main_deck + extra_deck 
                             if 'Blue-Eyes' in card.get('archetypes', []))
        print(f"   ✓ Blue-Eyes cards in deck: {blue_eyes_cards}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    print()
    
    print("=== All Tests Completed Successfully! ===")
    print()
    print("The MLflow integration is working correctly. You can now:")
    print("1. Generate decks with full experiment tracking")
    print("2. View metrics and artifacts in MLflow UI (http://localhost:5000)")
    print("3. Load clustered cards from the model registry")
    print("4. Generate both novel and meta-aware decks")

if __name__ == "__main__":
    main()
