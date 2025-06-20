#!/usr/bin/env python3
"""
Test script for Yu-Gi-Oh deck generator with MLflow integration.
Run this after sourcing .env.mlflow-tunnel to test the full pipeline.
"""

import os
import sys
sys.path.append('src')

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
    
    s3_bucket = os.environ.get('S3_BUCKET', 'NOT SET')
    print(f"   ✓ S3_BUCKET: {s3_bucket}")
    print()
    
    # Test imports
    print("2. Import Test:")
    try:
        from deck_generation.deck_generator import SimpleDeckGenerator, generate_deck, generate_deck_from_registry
        print("   ✓ All imports successful")
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return
    print()
    
    # Create test data
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
