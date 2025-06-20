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
        self.monster_count = sum(1 for card in main_deck if card.get('type') in ['Monster', 'Ritual Monster'])
        self.spell_count = sum(1 for card in main_deck if card.get('type') == 'Spell')
        self.trap_count = sum(1 for card in main_deck if card.get('type') == 'Trap')
        
        # Calculate archetype distribution
        self.archetype_distribution = self._calculate_archetype_distribution()
        self.dominant_archetype = self._get_dominant_archetype()
        
        # Calculate cluster distribution
        self.cluster_distribution = self._calculate_cluster_distribution()
        self.cluster_entropy = self._calculate_cluster_entropy()
        
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
    
    def __init__(self, clustered_cards: Dict[int, List[Dict]]):
        """
        Initialize the deck generator.
        
        Args:
            clustered_cards: Dictionary mapping cluster IDs to lists of card dictionaries
        """
        self.clustered_cards = clustered_cards
        
        # Separate cards by deck placement
        self.main_deck_clusters = {}
        self.extra_deck_clusters = {}
        
        # Separate cards by type within main deck clusters
        self.monster_clusters = {}
        self.spell_clusters = {}
        self.trap_clusters = {}
        
        # Define card types for main and extra deck
        main_deck_types = ['Monster', 'Spell', 'Trap', 'Ritual Monster']
        extra_deck_types = ['Fusion Monster', 'Synchro Monster', 'Xyz Monster', 'Link Monster', 'Pendulum Monster']
        
        for cluster_id, cards in clustered_cards.items():
            main_cards = [c for c in cards if c.get('type') in main_deck_types]
            extra_cards = [c for c in cards if c.get('type') in extra_deck_types]
            
            if main_cards:
                self.main_deck_clusters[cluster_id] = main_cards
                
                # Separate by card type for ratio enforcement
                monsters = [c for c in main_cards if c.get('type') in ['Monster', 'Ritual Monster']]
                spells = [c for c in main_cards if c.get('type') == 'Spell']
                traps = [c for c in main_cards if c.get('type') == 'Trap']
                
                if monsters:
                    self.monster_clusters[cluster_id] = monsters
                if spells:
                    self.spell_clusters[cluster_id] = spells
                if traps:
                    self.trap_clusters[cluster_id] = traps
                    
            if extra_cards:
                self.extra_deck_clusters[cluster_id] = extra_cards
        
        # Define main deck ratios (60% monsters, 20% spells, 20% traps, ±15% tolerance)
        self.target_monster_ratio = 0.6
        self.target_spell_ratio = 0.2
        self.target_trap_ratio = 0.2
        self.ratio_tolerance = 0.15
    
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
            
        monster_count = sum(1 for card in deck if card.get('type') in ['Monster', 'Ritual Monster'])
        spell_count = sum(1 for card in deck if card.get('type') == 'Spell')
        trap_count = sum(1 for card in deck if card.get('type') == 'Trap')
        
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
        target_monsters, target_spells, target_traps = self._calculate_target_counts(40)
        
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
        target_monsters, target_spells, target_traps = self._calculate_target_counts(40)
        
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


# Convenience function for quick deck generation
def generate_deck(clustered_cards: Optional[Dict[int, List[Dict]]] = None,
                 mode: str = "novel", 
                 target_archetype: Optional[str] = None,
                 use_mlflow: bool = True,
                 load_from_registry: bool = True) -> Tuple[List[Dict], List[Dict], DeckMetadata]:
    """
    Quick deck generation function with MLflow integration.
    
    Args:
        clustered_cards: Dictionary mapping cluster IDs to lists of card dictionaries.
                        If None and load_from_registry=True, will load from MLflow model registry.
        mode: Generation mode ("meta_aware" or "novel")
        target_archetype: Target archetype for meta_aware mode
        use_mlflow: Whether to use MLflow experiment tracking
        load_from_registry: Whether to load clustered cards from MLflow model registry
        
    Returns:
        Tuple of (main_deck, extra_deck, metadata)
    """
    # Load clustered cards from MLflow model registry if needed
    if clustered_cards is None and load_from_registry:
        try:
            logger.info("Loading clustered cards from MLflow model registry")
            
            # Get card data with cluster assignments from the model
            clustered_cards = get_card_data_with_clusters()
            logger.info(f"Loaded clustered cards: {len(clustered_cards)} clusters")
            
        except Exception as e:
            logger.error(f"Failed to load clustering model or generate clustered cards: {e}")
            logger.error("This could mean the clustering model is not registered or card data is not accessible")
            raise ValueError("No clustered cards provided and failed to load clustered cards from MLflow model. "
                           "Please provide clustered_cards parameter or ensure MLflow model registry is accessible.")
    
    if clustered_cards is None:
        raise ValueError("clustered_cards parameter is required when load_from_registry=False")
    
    generator = SimpleDeckGenerator(clustered_cards)
    return generator.generate_deck(mode, target_archetype, use_mlflow)


def generate_deck_from_registry(mode: str = "novel", 
                               target_archetype: Optional[str] = None) -> Tuple[List[Dict], List[Dict], DeckMetadata]:
    """
    Generate a deck using clustered cards loaded from MLflow model registry.
    
    This is the recommended function for production use as it ensures
    the deck generation uses the latest trained clustering model.
    
    Args:
        mode: Generation mode ("meta_aware" or "novel")
        target_archetype: Target archetype for meta_aware mode
        
    Returns:
        Tuple of (main_deck, extra_deck, metadata)
    """
    return generate_deck(
        clustered_cards=None,
        mode=mode,
        target_archetype=target_archetype,
        use_mlflow=True,
        load_from_registry=True
    )


if __name__ == "__main__":
    # Test data: simulate clustered card data with proper variety
    test_cards = {
        0: [  # Meta cluster (Blue-Eyes archetype) - Monsters
            {'name': 'Blue-Eyes White Dragon', 'type': 'Monster', 'archetype': 'Blue-Eyes', 'archetypes': ['Blue-Eyes'], 'cluster_id': 0},
            {'name': 'Blue-Eyes Alternative White Dragon', 'type': 'Monster', 'archetype': 'Blue-Eyes', 'archetypes': ['Blue-Eyes'], 'cluster_id': 0},
            {'name': 'Deep-Eyes White Dragon', 'type': 'Monster', 'archetype': 'Blue-Eyes', 'archetypes': ['Blue-Eyes'], 'cluster_id': 0},
            {'name': 'Blue-Eyes Chaos Dragon', 'type': 'Monster', 'archetype': 'Blue-Eyes', 'archetypes': ['Blue-Eyes'], 'cluster_id': 0},
            {'name': 'Blue-Eyes Spirit Dragon', 'type': 'Monster', 'archetype': 'Blue-Eyes', 'archetypes': ['Blue-Eyes'], 'cluster_id': 0},
        ],
        1: [  # Support spells
            {'name': 'Dragon Shrine', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 1},
            {'name': 'Return of the Dragon Lords', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 1},
            {'name': 'Trade-In', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 1},
            {'name': 'Cards of Consonance', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 1},
            {'name': 'Melody of Awakening Dragon', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 1},
        ],
        2: [  # Generic monsters and mixed cards
            {'name': 'Elemental HERO Sparkman', 'type': 'Monster', 'archetype': 'HERO', 'archetypes': ['HERO'], 'cluster_id': 2},
            {'name': 'Aleister the Invoker', 'type': 'Monster', 'archetype': 'Invoked', 'archetypes': ['Invoked'], 'cluster_id': 2},
            {'name': 'Ash Blossom & Joyous Spring', 'type': 'Monster', 'archetype': None, 'archetypes': [], 'cluster_id': 2},
            {'name': 'Effect Veiler', 'type': 'Monster', 'archetype': None, 'archetypes': [], 'cluster_id': 2},
            {'name': 'Mystical Space Typhoon', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 2},
        ],
        3: [  # Trap cards
            {'name': 'Mirror Force', 'type': 'Trap', 'archetype': None, 'archetypes': [], 'cluster_id': 3},
            {'name': 'Torrential Tribute', 'type': 'Trap', 'archetype': None, 'archetypes': [], 'cluster_id': 3},
            {'name': 'Solemn Judgment', 'type': 'Trap', 'archetype': None, 'archetypes': [], 'cluster_id': 3},
            {'name': 'Bottomless Trap Hole', 'type': 'Trap', 'archetype': None, 'archetypes': [], 'cluster_id': 3},
            {'name': 'Compulsory Evacuation Device', 'type': 'Trap', 'archetype': None, 'archetypes': [], 'cluster_id': 3},
        ],
        4: [  # More monsters for diversity
            {'name': 'Dark Magician', 'type': 'Monster', 'archetype': 'Dark Magician', 'archetypes': ['Dark Magician'], 'cluster_id': 4},
            {'name': 'Red-Eyes Black Dragon', 'type': 'Monster', 'archetype': 'Red-Eyes', 'archetypes': ['Red-Eyes'], 'cluster_id': 4},
            {'name': 'Summoned Skull', 'type': 'Monster', 'archetype': None, 'archetypes': [], 'cluster_id': 4},
            {'name': 'Celtic Guardian', 'type': 'Monster', 'archetype': None, 'archetypes': [], 'cluster_id': 4},
            {'name': 'Fissure', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 4},
        ],
        5: [  # Extra deck cards
            {'name': 'Blue-Eyes Ultimate Dragon', 'type': 'Fusion Monster', 'archetype': 'Blue-Eyes', 'archetypes': ['Blue-Eyes'], 'cluster_id': 5},
            {'name': 'Neo Blue-Eyes Ultimate Dragon', 'type': 'Fusion Monster', 'archetype': 'Blue-Eyes', 'archetypes': ['Blue-Eyes'], 'cluster_id': 5},
            {'name': 'Blue-Eyes Twin Burst Dragon', 'type': 'Fusion Monster', 'archetype': 'Blue-Eyes', 'archetypes': ['Blue-Eyes'], 'cluster_id': 5},
        ],
        6: [  # More spells
            {'name': 'Pot of Desires', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 6},
            {'name': 'Raigeki', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 6},
            {'name': 'Dark Hole', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 6},
            {'name': 'Monster Reborn', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 6},
            {'name': 'Heavy Storm', 'type': 'Spell', 'archetype': None, 'archetypes': [], 'cluster_id': 6},
        ]
    }
    
    print("=== Yu-Gi-Oh! Deck Generator Test with MLflow Integration ===\n")
    
    # Test 1: Traditional deck generation without MLflow (for backwards compatibility)
    print("1. Testing Traditional Deck Generation (No MLflow):")
    generator = SimpleDeckGenerator(test_cards)
    main_deck, extra_deck, metadata = generator.generate_deck("novel", use_mlflow=False)
    
    print(f"Generated deck: {len(main_deck)} main + {len(extra_deck)} extra cards")
    print(f"Dominant archetype: {metadata.dominant_archetype}")
    print(f"Cluster entropy: {metadata.cluster_entropy:.2f}")
    
    # Test 2: Deck generation with MLflow tracking (using provided test data)
    print("\n" + "="*60)
    print("2. Testing Deck Generation with MLflow (Test Data):")
    try:
        main_deck, extra_deck, metadata = generate_deck(
            clustered_cards=test_cards,
            mode="novel",
            use_mlflow=True,
            load_from_registry=False
        )
        print(f"✓ Generated deck with MLflow tracking: {len(main_deck)} main + {len(extra_deck)} extra cards")
        print(f"✓ Dominant archetype: {metadata.dominant_archetype}")
        print(f"✓ Cluster entropy: {metadata.cluster_entropy:.2f}")
    except Exception as e:
        print(f"✗ MLflow test failed: {e}")
    
    # Test 3: Meta-aware deck generation
    print("\n" + "="*60)
    print("3. Testing Meta-Aware Deck Generation:")
    try:
        main_deck, extra_deck, metadata = generate_deck(
            clustered_cards=test_cards,
            mode="meta_aware",
            target_archetype="Blue-Eyes",
            use_mlflow=True,
            load_from_registry=False
        )
        print(f"✓ Generated Blue-Eyes deck: {len(main_deck)} main + {len(extra_deck)} extra cards")
        print(f"✓ Dominant archetype: {metadata.dominant_archetype}")
        blue_eyes_cards = sum(1 for card in main_deck + extra_deck 
                             if 'Blue-Eyes' in card.get('archetypes', []))
        print(f"✓ Blue-Eyes cards in deck: {blue_eyes_cards}")
    except Exception as e:
        print(f"✗ Meta-aware test failed: {e}")
    
    # Test 4: Deck generation from MLflow model registry (this will likely fail without real model)
    print("\n" + "="*60)
    print("4. Testing Deck Generation from MLflow Model Registry:")
    try:
        main_deck, extra_deck, metadata = generate_deck_from_registry(mode="novel")
        print(f"✓ Generated deck from registry: {len(main_deck)} main + {len(extra_deck)} extra cards")
        print(f"✓ Dominant archetype: {metadata.dominant_archetype}")
    except Exception as e:
        print(f"✗ Registry test failed (expected if no model in registry): {e}")
    
    print("\n=== Test Complete ===")
    
    # Show final deck details for the last successful generation
    if 'main_deck' in locals():
        print(f"\nFinal Deck Details:")
        ratios = generator._check_ratios(main_deck)
        print(f"  Ratios - Monsters: {ratios['monster_ratio']:.1%}, "
              f"Spells: {ratios['spell_ratio']:.1%}, Traps: {ratios['trap_ratio']:.1%}")
        print(f"  Ratios acceptable: {generator._is_ratio_acceptable(main_deck)}")
        print(f"  Archetype distribution: {dict(list(metadata.archetype_distribution.items())[:3])}")
