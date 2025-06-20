"""
Simple Yu-Gi-Oh! Deck Generator

A streamlined deck generation system that creates decks from clustered card data.
Supports both meta-aware and novel deck generation modes.
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import random
import math
import logging
import numpy as np
from enum import Enum

# MLflow imports
import mlflow
from src.utils.mlflow.mlflow_utils import (
    setup_deck_generation_experiment,
    log_deck_generation_tags,
    log_deck_generation_params,
    log_deck_metrics,
    log_deck_artifacts
)

from src.utils.mlflow.get_clustering_model_from_registry import (
    get_clustering_model_from_registry,
    get_card_data_with_clusters
)

logger = logging.getLogger(__name__)


class CardType(Enum):
    """Yu-Gi-Oh! card types."""
    MONSTER = "Monster"
    SPELL = "Spell"
    TRAP = "Trap"
    FUSION = "Fusion"
    SYNCHRO = "Synchro"
    XYZ = "Xyz"
    LINK = "Link"
    RITUAL = "Ritual Monster"

    @classmethod
    def main_deck_types(cls):
        """Return card types that go in the main deck."""
        return [cls.MONSTER, cls.SPELL, cls.TRAP, cls.RITUAL]
    
    @classmethod
    def extra_deck_types(cls):
        """Return card types that go in the extra deck."""
        return [cls.FUSION, cls.SYNCHRO, cls.XYZ, cls.LINK]


class DeckMetadata:
    """Metadata for a generated deck."""
    
    def __init__(self, main_deck: List[Dict], extra_deck: List[Dict], clustered_cards: Dict[int, List[Dict]]):
        self.main_deck = main_deck
        self.extra_deck = extra_deck
        self.clustered_cards = clustered_cards
        
        # Calculate basic stats
        self.monster_count = sum(1 for card in main_deck if card.get('type') and 'Monster' in card.get('type'))
        self.spell_count = sum(1 for card in main_deck if card.get('type') and 'Spell' in card.get('type'))
        self.trap_count = sum(1 for card in main_deck if card.get('type') and 'Trap' in card.get('type'))
        
        # Calculate archetype distribution
        self.archetype_distribution = self._calculate_archetype_distribution()
        self.dominant_archetype = self._get_dominant_archetype()
        
        # Calculate cluster distribution
        self.cluster_distribution = self._calculate_cluster_distribution()
        self.cluster_entropy = self._calculate_cluster_entropy()
        
        # Calculate novelty metrics
        self.intra_deck_cluster_distance = calculate_intra_deck_cluster_distance(main_deck, clustered_cards)
        self.cluster_co_occurrence_rarity = calculate_cluster_co_occurrence_rarity(main_deck, clustered_cards)
        self.noise_card_percentage = calculate_noise_card_percentage(main_deck)
        
    def _calculate_archetype_distribution(self) -> Dict[str, int]:
        """Calculate archetype distribution in the deck."""
        archetype_counts = Counter()
        
        for card in self.main_deck + self.extra_deck:
            archetypes = card.get('archetypes', [])
            if archetypes:
                for archetype in archetypes:
                    archetype_counts[archetype] += 1
            else:
                archetype_counts['Generic'] += 1
        
        return dict(archetype_counts)
    
    def _get_dominant_archetype(self) -> str:
        """Get the most common archetype in the deck."""
        if self.archetype_distribution:
            return max(self.archetype_distribution.items(), key=lambda x: x[1])[0]
        return "Unknown"
    
    def _calculate_cluster_distribution(self) -> Dict[int, int]:
        """Calculate cluster distribution in the deck."""
        cluster_counts = Counter()
        
        for card in self.main_deck + self.extra_deck:
            cluster_id = card.get('cluster_id', -1)
            cluster_counts[cluster_id] += 1
        
        return dict(cluster_counts)
    
    def _calculate_cluster_entropy(self) -> float:
        """Calculate cluster entropy (diversity measure)."""
        if not self.cluster_distribution:
            return 0.0
        
        total_cards = sum(self.cluster_distribution.values())
        entropy = 0.0
        
        for count in self.cluster_distribution.values():
            if count > 0:
                probability = count / total_cards
                # Use proper Shannon entropy formula
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'monster_count': self.monster_count,
            'spell_count': self.spell_count,
            'trap_count': self.trap_count,
            'archetype_distribution': self.archetype_distribution,
            'dominant_archetype': self.dominant_archetype,
            'cluster_distribution': self.cluster_distribution,
            'cluster_entropy': self.cluster_entropy,
            'intra_deck_cluster_distance': self.intra_deck_cluster_distance,
            'cluster_co_occurrence_rarity': self.cluster_co_occurrence_rarity,
            'noise_card_percentage': self.noise_card_percentage,
            'main_deck_size': len(self.main_deck),
            'extra_deck_size': len(self.extra_deck)
        }
    
    def __str__(self) -> str:
        return (f"Deck: {len(self.main_deck)} main, {len(self.extra_deck)} extra | "
                f"{self.monster_count}M/{self.spell_count}S/{self.trap_count}T | "
                f"Primary: {self.dominant_archetype}")


class SimpleDeckGenerator:
    """
    Simple Yu-Gi-Oh! deck generator that creates decks from clustered card data.
    
    Supports two generation modes:
    - meta_aware: Builds decks anchored on existing archetypes
    - novel: Creates diverse decks by combining clusters creatively
    """
    
    def __init__(self, clustered_cards: Dict[int, List[Dict]], num_cards: int = 40):
        """
        Initialize the deck generator.
        
        Args:
            clustered_cards: Dictionary mapping cluster IDs to lists of card dictionaries
            num_cards: Number of cards in the main deck (default: 40)
        """
        self.num_cards = num_cards
        self.clustered_cards = clustered_cards
        
        # Separate cards by deck placement
        self.main_deck_clusters = {}
        self.extra_deck_clusters = {}
        
        # Separate cards by type within main deck clusters
        self.monster_clusters = {}
        self.spell_clusters = {}
        self.trap_clusters = {}
        
        # Define card types for main and extra deck
        main_deck_monster_types = ['Monster', 'Effect Monster', 'Normal Monster', 'Ritual Monster', 'Pendulum Effect Monster', 'Pendulum Normal Monster', 'Gemini Monster', 'Spirit Monster', 'Toon Monster', 'Tuner Monster', 'Union Monster', 'Flip Effect Monster']
        main_deck_spell_types = ['Spell Card', 'Spell']
        main_deck_trap_types = ['Trap Card', 'Trap']
        extra_deck_types = ['Fusion Monster', 'Synchro Monster', 'Xyz Monster', 'Link Monster', 'Pendulum Monster']
        
        # Define combined list of all main deck types
        main_deck_types = main_deck_monster_types + main_deck_spell_types + main_deck_trap_types
        
        for cluster_id, cards in clustered_cards.items():
            main_cards = [c for c in cards if c.get('type') in main_deck_types]
            extra_cards = [c for c in cards if c.get('type') in extra_deck_types]
            
            if main_cards:
                self.main_deck_clusters[cluster_id] = main_cards
                
                # Separate by card type for ratio enforcement
                monsters = [c for c in main_cards if c.get('type') in main_deck_monster_types]
                spells = [c for c in main_cards if c.get('type') in main_deck_spell_types]
                traps = [c for c in main_cards if c.get('type') in main_deck_trap_types]
                
                if monsters:
                    self.monster_clusters[cluster_id] = monsters
                if spells:
                    self.spell_clusters[cluster_id] = spells
                if traps:
                    self.trap_clusters[cluster_id] = traps
                    
            if extra_cards:
                self.extra_deck_clusters[cluster_id] = extra_cards
        
        # Define main deck ratios with fluctuation from 60/20/20 baseline (±15% tolerance)
        # Apply a random fluctuation within reasonable bounds
        fluctuation = random.uniform(-0.1, 0.1)  # Up to ±10% fluctuation
        
        # Base ratios
        base_monster = 0.6
        base_spell = 0.2
        base_trap = 0.2
        
        # Apply fluctuation primarily to monster ratio, then adjust others proportionally
        self.target_monster_ratio = max(0.45, min(0.75, base_monster + fluctuation))
        
        # Distribute remaining percentage between spells and traps proportionally
        remaining = 1.0 - self.target_monster_ratio
        spell_weight = base_spell / (base_spell + base_trap)
        trap_weight = base_trap / (base_spell + base_trap)
        
        self.target_spell_ratio = remaining * spell_weight
        self.target_trap_ratio = remaining * trap_weight
        
        # Keep the standard tolerance
        self.ratio_tolerance = 0.15
        
        logger.info(f"Deck ratios set to: {self.target_monster_ratio:.1%}M/{self.target_spell_ratio:.1%}S/{self.target_trap_ratio:.1%}T (±{self.ratio_tolerance:.0%} tolerance)")
    
    def _calculate_target_counts(self, deck_size: int = 40) -> Tuple[int, int, int]:
        """Calculate target counts for each card type based on ratios."""
        target_monsters = int(deck_size * self.target_monster_ratio)
        target_spells = int(deck_size * self.target_spell_ratio)
        target_traps = int(deck_size * self.target_trap_ratio)
        
        # Adjust to ensure total equals deck_size
        total = target_monsters + target_spells + target_traps
        if total < deck_size:
            # Add remainder to monsters (most flexible category)
            target_monsters += deck_size - total
        elif total > deck_size:
            # Remove excess from monsters
            target_monsters -= total - deck_size
            
        return target_monsters, target_spells, target_traps
    
    def _check_ratios(self, deck: List[Dict]) -> Dict[str, float]:
        """Check current ratios in a deck."""
        deck_size = len(deck)
        if deck_size == 0:
            return {'monster_ratio': 0.0, 'spell_ratio': 0.0, 'trap_ratio': 0.0}
            
        monster_count = sum(1 for card in deck if card.get('type') and 'Monster' in card.get('type'))
        spell_count = sum(1 for card in deck if card.get('type') and 'Spell' in card.get('type'))
        trap_count = sum(1 for card in deck if card.get('type') and 'Trap' in card.get('type'))
        
        return {
            'monster_ratio': monster_count / deck_size,
            'spell_ratio': spell_count / deck_size,
            'trap_ratio': trap_count / deck_size
        }
    
    def _is_ratio_acceptable(self, deck: List[Dict]) -> bool:
        """Check if deck ratios are within acceptable tolerance."""
        ratios = self._check_ratios(deck)
        
        monster_ok = abs(ratios['monster_ratio'] - self.target_monster_ratio) <= self.ratio_tolerance
        spell_ok = abs(ratios['spell_ratio'] - self.target_spell_ratio) <= self.ratio_tolerance
        trap_ok = abs(ratios['trap_ratio'] - self.target_trap_ratio) <= self.ratio_tolerance
        
        return monster_ok and spell_ok and trap_ok
    
    def _randomize_ratios(self):
        """Randomize the deck ratios for each deck generation to ensure variety."""
        # Apply a stronger random fluctuation for more variation between decks
        fluctuation = random.uniform(-0.15, 0.15)  # Up to ±15% fluctuation
        
        # Base ratios
        base_monster = 0.6
        base_spell = 0.2
        base_trap = 0.2
        
        # Apply fluctuation primarily to monster ratio, with wider bounds
        # Min 40% monsters, max 80% monsters for greater variety
        self.target_monster_ratio = max(0.4, min(0.8, base_monster + fluctuation))
        
        # Distribute remaining percentage between spells and traps
        # Apply additional variation to their distribution
        remaining = 1.0 - self.target_monster_ratio
        spell_trap_variation = random.uniform(-0.2, 0.2)  # This shifts the balance between spells and traps
        
        # Baseline weights
        spell_weight = 0.5 + spell_trap_variation  # Can shift between 0.3 and 0.7 of remaining
        spell_weight = max(0.3, min(0.7, spell_weight))  # Cap between 30% and 70% of remaining
        trap_weight = 1.0 - spell_weight
        
        self.target_spell_ratio = remaining * spell_weight
        self.target_trap_ratio = remaining * trap_weight
        
        # Keep the standard tolerance
        self.ratio_tolerance = 0.15
        
        logger.info(f"Deck ratios randomized to: {self.target_monster_ratio:.1%}M/{self.target_spell_ratio:.1%}S/{self.target_trap_ratio:.1%}T (±{self.ratio_tolerance:.0%} tolerance)")
    
    def generate_deck(self, mode: str = "novel", target_archetype: Optional[str] = None, 
                     use_mlflow: bool = True) -> Tuple[List[Dict], List[Dict], DeckMetadata]:
        """
        Generate a complete Yu-Gi-Oh! deck with MLflow experiment tracking.
        
        Args:
            mode: Generation mode ("meta_aware" or "novel")
            target_archetype: Target archetype for meta_aware mode
            use_mlflow: Whether to use MLflow experiment tracking
            
        Returns:
            Tuple of (main_deck, extra_deck, metadata)
        """
        if use_mlflow:
            # Setup MLflow experiment
            setup_deck_generation_experiment()
            
            with mlflow.start_run():
                # Log tags and parameters
                log_deck_generation_tags(generation_mode=mode, target_archetype=target_archetype)
                log_deck_generation_params(
                    generation_mode=mode,
                    target_archetype=target_archetype,
                    target_monster_ratio=self.target_monster_ratio,
                    target_spell_ratio=self.target_spell_ratio, 
                    target_trap_ratio=self.target_trap_ratio,
                    ratio_tolerance=self.ratio_tolerance
                )
                
                # Generate the deck
                main_deck, extra_deck, metadata = self._generate_deck_internal(mode, target_archetype)
                
                # Log metrics
                self._log_deck_metrics(main_deck, extra_deck, metadata)
                
                # Log deck artifacts (with error handling for S3 issues)
                try:
                    log_deck_artifacts(main_deck, extra_deck, metadata)
                    logger.info("Successfully logged deck artifacts")
                except Exception as e:
                    logger.warning(f"Failed to log deck artifacts (likely S3 credentials issue): {e}")
                    # Continue without artifacts - the experiment tracking still works
                
                return main_deck, extra_deck, metadata
        else:
            # Generate without MLflow tracking
            return self._generate_deck_internal(mode, target_archetype)
    
    def _generate_deck_internal(self, mode: str, target_archetype: Optional[str] = None) -> Tuple[List[Dict], List[Dict], DeckMetadata]:
        """Internal deck generation logic without MLflow tracking."""
        # Randomize ratios for each deck generation
        self._randomize_ratios()
        
        if mode == "meta_aware":
            if not target_archetype:
                raise ValueError("target_archetype is required for meta_aware mode")
            return self._generate_meta_aware_deck(target_archetype)
        elif mode == "novel":
            return self._generate_novel_deck()
        else:
            raise ValueError(f"Unknown generation mode: {mode}")
    
    def _log_deck_metrics(self, main_deck: List[Dict], extra_deck: List[Dict], metadata: DeckMetadata):
        """Log deck generation metrics to MLflow using utility function."""
        log_deck_metrics(main_deck, extra_deck, metadata)
    
    def _generate_meta_aware_deck(self, target_archetype: str) -> Tuple[List[Dict], List[Dict], DeckMetadata]:
        """Generate a deck anchored on a target archetype with proper card type ratios."""
        
        # Calculate target counts for each card type
        target_monsters, target_spells, target_traps = self._calculate_target_counts(self.num_cards)
        
        # Find clusters containing the target archetype
        archetype_monster_clusters = []
        archetype_spell_clusters = []
        archetype_trap_clusters = []
        
        for cluster_id in self.monster_clusters.keys():
            for card in self.monster_clusters[cluster_id]:
                if target_archetype in card.get('archetypes', []) or card.get('archetype') == target_archetype:
                    archetype_monster_clusters.append(cluster_id)
                    break
                    
        for cluster_id in self.spell_clusters.keys():
            for card in self.spell_clusters[cluster_id]:
                if target_archetype in card.get('archetypes', []) or card.get('archetype') == target_archetype:
                    archetype_spell_clusters.append(cluster_id)
                    break
                    
        for cluster_id in self.trap_clusters.keys():
            for card in self.trap_clusters[cluster_id]:
                if target_archetype in card.get('archetypes', []) or card.get('archetype') == target_archetype:
                    archetype_trap_clusters.append(cluster_id)
                    break
        
        # If no archetype clusters found, fall back to novel generation
        if not any([archetype_monster_clusters, archetype_spell_clusters, archetype_trap_clusters]):
            logger.warning(f"No clusters found for archetype {target_archetype}, falling back to novel generation")
            return self._generate_novel_deck()
        
        main_deck = []
        
        # Add monsters (60% of deck, ~24 cards)
        monsters_added = 0
        # First, add archetype monsters
        for cluster_id in archetype_monster_clusters[:2]:  # Use up to 2 archetype clusters
            if monsters_added >= target_monsters:
                break
            cluster_cards = self.monster_clusters[cluster_id]
            archetype_cards = [c for c in cluster_cards 
                             if target_archetype in c.get('archetypes', []) or c.get('archetype') == target_archetype]
            
            cards_to_add = min(target_monsters // 2, len(archetype_cards), target_monsters - monsters_added)
            if cards_to_add > 0:
                selected = random.sample(archetype_cards, cards_to_add)
                main_deck.extend(selected)
                monsters_added += len(selected)
        
        # Fill remaining monster slots with generic monsters
        if monsters_added < target_monsters:
            remaining_monster_slots = target_monsters - monsters_added
            other_monster_clusters = [cid for cid in self.monster_clusters.keys() 
                                    if cid not in archetype_monster_clusters]
            
            for cluster_id in random.sample(other_monster_clusters, min(2, len(other_monster_clusters))):
                if monsters_added >= target_monsters:
                    break
                cluster_cards = self.monster_clusters[cluster_id]
                cards_to_add = min(remaining_monster_slots, len(cluster_cards))
                selected = random.sample(cluster_cards, cards_to_add)
                main_deck.extend(selected)
                monsters_added += len(selected)
                remaining_monster_slots -= len(selected)
        
        # Add spells (20% of deck, ~8 cards)
        spells_added = 0
        # Archetype spells first
        for cluster_id in archetype_spell_clusters[:2]:
            if spells_added >= target_spells:
                break
            cluster_cards = self.spell_clusters[cluster_id]
            archetype_cards = [c for c in cluster_cards 
                             if target_archetype in c.get('archetypes', []) or c.get('archetype') == target_archetype]
            
            cards_to_add = min(target_spells // 2, len(archetype_cards), target_spells - spells_added)
            if cards_to_add > 0:
                selected = random.sample(archetype_cards, cards_to_add)
                main_deck.extend(selected)
                spells_added += len(selected)
        
        # Fill remaining spell slots with generic spells
        if spells_added < target_spells:
            remaining_spell_slots = target_spells - spells_added
            other_spell_clusters = [cid for cid in self.spell_clusters.keys() 
                                  if cid not in archetype_spell_clusters]
            
            for cluster_id in random.sample(other_spell_clusters, min(2, len(other_spell_clusters))):
                if spells_added >= target_spells:
                    break
                cluster_cards = self.spell_clusters[cluster_id]
                cards_to_add = min(remaining_spell_slots, len(cluster_cards))
                selected = random.sample(cluster_cards, cards_to_add)
                main_deck.extend(selected)
                spells_added += len(selected)
                remaining_spell_slots -= len(selected)
        
        # Add traps (20% of deck, ~8 cards)
        traps_added = 0
        # Archetype traps first
        for cluster_id in archetype_trap_clusters[:2]:
            if traps_added >= target_traps:
                break
            cluster_cards = self.trap_clusters[cluster_id]
            archetype_cards = [c for c in cluster_cards 
                             if target_archetype in c.get('archetypes', []) or c.get('archetype') == target_archetype]
            
            cards_to_add = min(target_traps // 2, len(archetype_cards), target_traps - traps_added)
            if cards_to_add > 0:
                selected = random.sample(archetype_cards, cards_to_add)
                main_deck.extend(selected)
                traps_added += len(selected)
        
        # Fill remaining trap slots with generic traps
        if traps_added < target_traps:
            remaining_trap_slots = target_traps - traps_added
            other_trap_clusters = [cid for cid in self.trap_clusters.keys() 
                                 if cid not in archetype_trap_clusters]
            
            for cluster_id in random.sample(other_trap_clusters, min(2, len(other_trap_clusters))):
                if traps_added >= target_traps:
                    break
                cluster_cards = self.trap_clusters[cluster_id]
                cards_to_add = min(remaining_trap_slots, len(cluster_cards))
                selected = random.sample(cluster_cards, cards_to_add)
                main_deck.extend(selected)
                traps_added += len(selected)
                remaining_trap_slots -= len(selected)
        
        # Ensure exactly 40 cards and fill any remaining slots
        while len(main_deck) < 40:
            # Determine which type we need more of based on current ratios
            ratios = self._check_ratios(main_deck)
            
            if ratios['monster_ratio'] < self.target_monster_ratio - 0.05 and self.monster_clusters:
                # Need more monsters
                all_monster_cards = [card for cards in self.monster_clusters.values() for card in cards]
                if all_monster_cards:
                    main_deck.append(random.choice(all_monster_cards))
            elif ratios['spell_ratio'] < self.target_spell_ratio - 0.05 and self.spell_clusters:
                # Need more spells
                all_spell_cards = [card for cards in self.spell_clusters.values() for card in cards]
                if all_spell_cards:
                    main_deck.append(random.choice(all_spell_cards))
            elif ratios['trap_ratio'] < self.target_trap_ratio - 0.05 and self.trap_clusters:
                # Need more traps
                all_trap_cards = [card for cards in self.trap_clusters.values() for card in cards]
                if all_trap_cards:
                    main_deck.append(random.choice(all_trap_cards))
            else:
                # Fill with any available card
                all_main_cards = [card for cards in self.main_deck_clusters.values() for card in cards]
                if all_main_cards:
                    main_deck.append(random.choice(all_main_cards))
                else:
                    break
        
        # Trim to exactly 40 cards if needed
        main_deck = main_deck[:40]
        
        # Generate extra deck
        extra_deck = self._generate_extra_deck(target_archetype)
        
        # Create metadata
        metadata = DeckMetadata(main_deck, extra_deck, self.clustered_cards)
        
        # Log ratio results
        final_ratios = self._check_ratios(main_deck)
        logger.info(f"Generated meta-aware deck with ratios: "
                   f"Monsters: {final_ratios['monster_ratio']:.1%}, "
                   f"Spells: {final_ratios['spell_ratio']:.1%}, "
                   f"Traps: {final_ratios['trap_ratio']:.1%}")
        
        return main_deck, extra_deck, metadata
    
    def _generate_novel_deck(self) -> Tuple[List[Dict], List[Dict], DeckMetadata]:
        """Generate a novel deck by combining diverse clusters with proper card type ratios."""
        logger.info("Generating novel deck with diverse cluster combination and proper ratios")
        
        # Calculate target counts for each card type
        target_monsters, target_spells, target_traps = self._calculate_target_counts(self.num_cards)
        
        main_deck = []
        
        # Select diverse clusters for each card type
        monster_cluster_ids = list(self.monster_clusters.keys())
        spell_cluster_ids = list(self.spell_clusters.keys())
        trap_cluster_ids = list(self.trap_clusters.keys())
        
        # Add monsters (60% of deck, ~24 cards)
        if monster_cluster_ids:
            selected_monster_clusters = random.sample(monster_cluster_ids, min(3, len(monster_cluster_ids)))
            monsters_per_cluster = target_monsters // len(selected_monster_clusters)
            monsters_added = 0
            
            for cluster_id in selected_monster_clusters:
                if monsters_added >= target_monsters:
                    break
                cluster_cards = self.monster_clusters[cluster_id]
                cards_to_add = min(monsters_per_cluster, len(cluster_cards), target_monsters - monsters_added)
                if cards_to_add > 0:
                    selected = random.sample(cluster_cards, cards_to_add)
                    main_deck.extend(selected)
                    monsters_added += len(selected)
            
            # Fill remaining monster slots if needed
            while monsters_added < target_monsters and monster_cluster_ids:
                remaining_slots = target_monsters - monsters_added
                all_monster_cards = [card for cards in self.monster_clusters.values() for card in cards]
                if all_monster_cards:
                    additional = random.sample(all_monster_cards, min(remaining_slots, len(all_monster_cards)))
                    main_deck.extend(additional)
                    monsters_added += len(additional)
                else:
                    break
        
        # Add spells (20% of deck, ~8 cards)
        if spell_cluster_ids:
            selected_spell_clusters = random.sample(spell_cluster_ids, min(2, len(spell_cluster_ids)))
            spells_per_cluster = max(1, target_spells // len(selected_spell_clusters))
            spells_added = 0
            
            for cluster_id in selected_spell_clusters:
                if spells_added >= target_spells:
                    break
                cluster_cards = self.spell_clusters[cluster_id]
                cards_to_add = min(spells_per_cluster, len(cluster_cards), target_spells - spells_added)
                if cards_to_add > 0:
                    selected = random.sample(cluster_cards, cards_to_add)
                    main_deck.extend(selected)
                    spells_added += len(selected)
            
            # Fill remaining spell slots if needed
            while spells_added < target_spells and spell_cluster_ids:
                remaining_slots = target_spells - spells_added
                all_spell_cards = [card for cards in self.spell_clusters.values() for card in cards]
                if all_spell_cards:
                    additional = random.sample(all_spell_cards, min(remaining_slots, len(all_spell_cards)))
                    main_deck.extend(additional)
                    spells_added += len(additional)
                else:
                    break
        
        # Add traps (20% of deck, ~8 cards)
        if trap_cluster_ids:
            selected_trap_clusters = random.sample(trap_cluster_ids, min(2, len(trap_cluster_ids)))
            traps_per_cluster = max(1, target_traps // len(selected_trap_clusters))
            traps_added = 0
            
            for cluster_id in selected_trap_clusters:
                if traps_added >= target_traps:
                    break
                cluster_cards = self.trap_clusters[cluster_id]
                cards_to_add = min(traps_per_cluster, len(cluster_cards), target_traps - traps_added)
                if cards_to_add > 0:
                    selected = random.sample(cluster_cards, cards_to_add)
                    main_deck.extend(selected)
                    traps_added += len(selected)
            
            # Fill remaining trap slots if needed
            while traps_added < target_traps and trap_cluster_ids:
                remaining_slots = target_traps - traps_added
                all_trap_cards = [card for cards in self.trap_clusters.values() for card in cards]
                if all_trap_cards:
                    additional = random.sample(all_trap_cards, min(remaining_slots, len(all_trap_cards)))
                    main_deck.extend(additional)
                    traps_added += len(additional)
                else:
                    break
        
        # Ensure exactly 40 cards and fill any remaining slots
        while len(main_deck) < 40:
            # Determine which type we need more of based on current ratios
            ratios = self._check_ratios(main_deck)
            
            if ratios['monster_ratio'] < self.target_monster_ratio - 0.05 and self.monster_clusters:
                # Need more monsters
                all_monster_cards = [card for cards in self.monster_clusters.values() for card in cards]
                if all_monster_cards:
                    main_deck.append(random.choice(all_monster_cards))
            elif ratios['spell_ratio'] < self.target_spell_ratio - 0.05 and self.spell_clusters:
                # Need more spells
                all_spell_cards = [card for cards in self.spell_clusters.values() for card in cards]
                if all_spell_cards:
                    main_deck.append(random.choice(all_spell_cards))
            elif ratios['trap_ratio'] < self.target_trap_ratio - 0.05 and self.trap_clusters:
                # Need more traps
                all_trap_cards = [card for cards in self.trap_clusters.values() for card in cards]
                if all_trap_cards:
                    main_deck.append(random.choice(all_trap_cards))
            else:
                # Fill with any available card
                all_main_cards = [card for cards in self.main_deck_clusters.values() for card in cards]
                if all_main_cards:
                    main_deck.append(random.choice(all_main_cards))
                else:
                    break
        
        # Trim to exactly 40 cards if needed
        main_deck = main_deck[:40]
        
        # Generate extra deck
        extra_deck = self._generate_extra_deck()
        
        # Create metadata
        metadata = DeckMetadata(main_deck, extra_deck, self.clustered_cards)
        
        # Log ratio results
        final_ratios = self._check_ratios(main_deck)
        logger.info(f"Generated novel deck with ratios: "
                   f"Monsters: {final_ratios['monster_ratio']:.1%}, "
                   f"Spells: {final_ratios['spell_ratio']:.1%}, "
                   f"Traps: {final_ratios['trap_ratio']:.1%}")
        
        return main_deck, extra_deck, metadata
    
    def _generate_extra_deck(self, target_archetype: Optional[str] = None) -> List[Dict]:
        """Generate extra deck cards. Always returns exactly 15 cards."""
        extra_deck = []
        target_size = 15
        
        if not self.extra_deck_clusters:
            # If no extra deck cards available, return empty list
            return extra_deck
        
        # Get all available extra deck cards
        all_extra_cards = [card for cards in self.extra_deck_clusters.values() for card in cards]
        
        if not all_extra_cards:
            return extra_deck
        
        if target_archetype:
            # Prioritize archetype extra deck cards
            archetype_cards = [c for c in all_extra_cards if target_archetype in c.get('archetypes', [])]
            if archetype_cards:
                # Add up to 8 archetype cards (majority but not all)
                selected = random.sample(archetype_cards, min(8, len(archetype_cards)))
                extra_deck.extend(selected)
        
        # Fill remaining slots with random extra deck cards
        remaining_slots = target_size - len(extra_deck)
        if remaining_slots > 0:
            # Remove already selected cards to avoid duplicates
            available_cards = [c for c in all_extra_cards if c not in extra_deck]
            
            if len(available_cards) >= remaining_slots:
                # Enough cards available, sample exactly what we need
                additional = random.sample(available_cards, remaining_slots)
                extra_deck.extend(additional)
            else:
                # Not enough unique cards, fill with what we have and allow duplicates if needed
                extra_deck.extend(available_cards)
                still_needed = target_size - len(extra_deck)
                
                if still_needed > 0:
                    # Add duplicates to reach exactly 15 cards
                    for _ in range(still_needed):
                        extra_deck.append(random.choice(all_extra_cards))
        
        # Ensure exactly 15 cards
        return extra_deck[:target_size]


# Novelty Metrics Functions
def calculate_cluster_entropy(deck: List[Dict]) -> float:
    """
    Calculate the Shannon entropy of cluster distribution in a deck.
    Higher values indicate a more diverse set of clusters.

    Args:
        deck: List of card dictionaries representing a deck

    Returns:
        Entropy score measuring cluster diversity
    """
    # Count occurrences of each cluster in the deck
    cluster_counts = Counter([card.get('cluster_id', -1) for card in deck])
    
    # Calculate Shannon entropy
    total_cards = len(deck)
    entropy = 0.0
    
    for count in cluster_counts.values():
        if count > 0:
            probability = count / total_cards
            entropy -= probability * math.log2(probability)
    
    return entropy

def calculate_intra_deck_cluster_distance(deck: List[Dict], clustered_cards: Dict[int, List[Dict]]) -> float:
    """
    Calculate the average distance between clusters used in the deck.
    Higher values indicate more surprising combinations.

    Args:
        deck: List of card dictionaries representing a deck
        clustered_cards: Dictionary mapping cluster IDs to lists of card dictionaries

    Returns:
        Average distance between clusters in the deck
    """
    # Get unique clusters in the deck
    deck_clusters = list(set([card.get('cluster_id', -1) for card in deck]))
    
    # Need at least 2 clusters to measure distance
    if len(deck_clusters) < 2:
        return 0.0
    
    # Extract feature vectors for cards by cluster
    cluster_vectors = {}
    
    # For each cluster in the deck, calculate its average feature vector
    for cluster_id in deck_clusters:
        # Get all cards for this cluster from the deck
        cluster_cards = [card for card in deck if card.get('cluster_id', -1) == cluster_id]
        
        # Skip if no cards found for this cluster (shouldn't happen)
        if not cluster_cards:
            continue
        
        # For each card, extract its represented features
        # Since we might not have feature vectors directly, we'll use a card's properties as a proxy
        # using numerical values + one-hot encoding of categorical fields
        card_features = []
        for card in cluster_cards:
            features = []
            # Add level/rank if it exists (for monsters)
            if 'level' in card:
                features.append(card.get('level', 0))
            # Add ATK/DEF (for monsters)
            if 'atk' in card:
                features.append(card.get('atk', 0))
            if 'def' in card:
                features.append(card.get('def', 0))
            # Add pendulum scales if they exist
            if 'scale' in card:
                features.append(card.get('scale', 0))
            # Add card type as one-hot encoding (simplified)
            if 'Monster' in card.get('type', ''):
                features.append(1)
            else:
                features.append(0)
            if 'Spell' in card.get('type', ''):
                features.append(1)
            else:
                features.append(0)
            if 'Trap' in card.get('type', ''):
                features.append(1)
            else:
                features.append(0)
            
            # Only add this card if we have at least some features
            if features:
                card_features.append(features)
        
        # Skip if no features found for any card in this cluster
        if not card_features:
            continue
        
        # Calculate average feature vector for this cluster
        cluster_vectors[cluster_id] = np.mean(card_features, axis=0)
    
    # Calculate pairwise distances between cluster vectors
    total_distance = 0.0
    pair_count = 0
    
    for i, cluster_id1 in enumerate(cluster_vectors.keys()):
        for j, cluster_id2 in enumerate(cluster_vectors.keys()):
            if i < j:  # Only calculate each pair once
                # Euclidean distance between cluster vectors
                vec1 = cluster_vectors[cluster_id1]
                vec2 = cluster_vectors[cluster_id2]
                
                # Only calculate distance if vectors have the same shape
                if len(vec1) == len(vec2):
                    distance = np.linalg.norm(vec1 - vec2)
                    total_distance += distance
                    pair_count += 1
    
    # Return average distance
    if pair_count > 0:
        return total_distance / pair_count
    else:
        return 0.0

def calculate_cluster_co_occurrence_rarity(deck: List[Dict], clustered_cards: Dict[int, List[Dict]]) -> float:
    """
    Calculate the rarity of cluster co-occurrences in the deck.
    Higher values indicate more surprising/novel combinations.

    Args:
        deck: List of card dictionaries representing a deck
        clustered_cards: Dictionary mapping cluster IDs to lists of card dictionaries

    Returns:
        Rarity score for cluster co-occurrences
    """
    # Get unique clusters in the deck
    deck_clusters = list(set([card.get('cluster_id', -1) for card in deck]))
    
    # Need at least 2 clusters to measure co-occurrence
    if len(deck_clusters) < 2:
        return 0.0
    
    # Calculate total number of cards in the dataset
    total_cards_in_dataset = sum(len(cards) for cards in clustered_cards.values())
    
    # Calculate rarity scores for each cluster pair
    pair_rarities = []
    
    for i, cluster_id1 in enumerate(deck_clusters):
        for j, cluster_id2 in enumerate(deck_clusters):
            if i < j:  # Only calculate each pair once
                # Calculate the expected co-occurrence by chance
                cluster1_size = len(clustered_cards.get(cluster_id1, []))
                cluster2_size = len(clustered_cards.get(cluster_id2, []))
                
                # Skip if either cluster doesn't exist in the data
                if cluster1_size == 0 or cluster2_size == 0:
                    continue
                
                # Probability of these clusters co-occurring by random chance
                # P(A and B) = P(A) * P(B) if independent
                p_cluster1 = cluster1_size / total_cards_in_dataset
                p_cluster2 = cluster2_size / total_cards_in_dataset
                expected_co_occurrence = p_cluster1 * p_cluster2
                
                # Rarity is inverse of probability (lower probability = higher rarity)
                # Using log scale to prevent extremely small values
                rarity = -math.log10(expected_co_occurrence) if expected_co_occurrence > 0 else 10.0  # Cap at 10
                pair_rarities.append(rarity)
    
    # Return average rarity across all pairs
    if pair_rarities:
        return sum(pair_rarities) / len(pair_rarities)
    else:
        return 0.0

def calculate_noise_card_percentage(deck: List[Dict]) -> float:
    """
    Calculate the percentage of cards in the deck that were labeled as noise
    by the clustering algorithm (cluster_id = -1).

    Args:
        deck: List of card dictionaries representing a deck

    Returns:
        Percentage of noise cards in the deck (0.0 to 1.0)
    """
    if not deck:
        return 0.0
        
    noise_cards = [card for card in deck if card.get('cluster_id', -1) == -1]
    return len(noise_cards) / len(deck)
