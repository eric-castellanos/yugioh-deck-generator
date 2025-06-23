"""
Simple Yu-Gi-Oh! Deck Generator

A streamlined deck generation system that creates decks from clustered card data.
Supports both meta-aware and novel deck generation modes.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from collections import Counter, defaultdict
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
    PENDULUM = "Pendulum"  # Added Pendulum type

    @classmethod
    def main_deck_types(cls):
        """Return card types that go in the main deck."""
        return [cls.MONSTER, cls.SPELL, cls.TRAP, cls.RITUAL]
    
    @classmethod
    def extra_deck_types(cls):
        """Return card types that go in the extra deck."""
        return [cls.FUSION, cls.SYNCHRO, cls.XYZ, cls.LINK]


# Card constraint utility functions
def is_tuner(card: Dict) -> bool:
    """Check if a card is a Tuner monster."""
    return card.get('type') and 'Monster' in card.get('type') and card.get('desc') and 'Tuner' in card.get('desc')

def is_pendulum(card: Dict) -> bool:
    """Check if a card is a Pendulum monster."""
    return card.get('type') and 'Pendulum' in card.get('type')

def get_monster_level(card: Dict) -> int:
    """Get monster level/rank/link value."""
    if card.get('type') and 'Monster' in card.get('type'):
        if 'Link' in card.get('type'):
            return card.get('linkval', 0)
        else:
            return card.get('level', 0)
    return 0

def get_pendulum_scales(card: Dict) -> Tuple[int, int]:
    """Get pendulum scales for a card if it's a pendulum monster."""
    if is_pendulum(card):
        # Return left and right scales (typically found in lscale and rscale)
        return card.get('lscale', 0), card.get('rscale', 0)
    return 0, 0

def get_link_markers(card: Dict) -> List[str]:
    """Get link markers for a Link monster."""
    if card.get('type') and 'Link' in card.get('type'):
        return card.get('linkmarkers', [])
    return []

class DeckMetadata:
    """Metadata and analysis info for a generated deck."""
    
    def __init__(self, main_deck: List[Dict], extra_deck: List[Dict], clustered_cards: Dict = None):
        """
        Initialize deck metadata with generated deck cards.
        
        Args:
            main_deck: List of main deck card dictionaries
            extra_deck: List of extra deck card dictionaries
            clustered_cards: Dictionary of clustered cards for reference
        """
        self.main_deck = main_deck
        self.extra_deck = extra_deck
        self.clustered_cards = clustered_cards
        
        # Cluster-related metrics (initialized with default values)
        self.cluster_entropy = 0.0
        self.intra_deck_cluster_distance = 0.0
        self.cluster_co_occurrence_rarity = 0.0
        self.noise_card_percentage = 0.0
        self.cluster_distribution = {}
        self.archetype_distribution = {}
        self.dominant_archetype = "Unknown"
        
        # Analyze the deck composition
        self._analyze_deck()
    
    def _analyze_deck(self):
        """Analyze deck composition and calculate metrics."""
        # Count card types
        self.monster_count = sum(1 for card in self.main_deck if card.get('type') and 'Monster' in card.get('type'))
        self.spell_count = sum(1 for card in self.main_deck if card.get('type') and 'Spell' in card.get('type'))
        self.trap_count = sum(1 for card in self.main_deck if card.get('type') and 'Trap' in card.get('type'))
        self.total_main = len(self.main_deck)
        self.total_extra = len(self.extra_deck)
        
        # Count extra deck types
        self.fusion_count = sum(1 for card in self.extra_deck if card.get('type') and 'Fusion' in card.get('type'))
        self.synchro_count = sum(1 for card in self.extra_deck if card.get('type') and 'Synchro' in card.get('type'))
        self.xyz_count = sum(1 for card in self.extra_deck if card.get('type') and 'XYZ' in card.get('type'))
        self.link_count = sum(1 for card in self.extra_deck if card.get('type') and 'Link' in card.get('type'))
        
        # Track copy distribution
        self.copy_distribution = self._calculate_copy_distribution()
        
        # Track archetypes
        self.archetypes = self._identify_archetypes()
        
        # Check summoning mechanics supported
        self.has_tuners = any(is_tuner(card) for card in self.main_deck)
        self.has_pendulums = any(is_pendulum(card) for card in self.main_deck)
        
        # Calculate some derived metrics
        self.monster_ratio = self.monster_count / self.total_main if self.total_main > 0 else 0
        self.spell_ratio = self.spell_count / self.total_main if self.total_main > 0 else 0
        self.trap_ratio = self.trap_count / self.total_main if self.total_main > 0 else 0
    
    def _calculate_copy_distribution(self):
        """Calculate the distribution of card copies in the deck."""
        # Count occurrences of each card name
        name_counts = {}
        for card in self.main_deck:
            name = card.get('name', '')
            if name:
                name_counts[name] = name_counts.get(name, 0) + 1
        
        # Count how many cards appear 1, 2, or 3 times
        counts = {'1_count': 0, '2_count': 0, '3_count': 0}
        for name, count in name_counts.items():
            if count == 1:
                counts['1_count'] += 1
            elif count == 2:
                counts['2_count'] += 1
            elif count >= 3:
                counts['3_count'] += 1
        
        return counts
    
    def _identify_archetypes(self):
        """Identify archetypes present in the deck."""
        archetypes = {}
        
        for card in self.main_deck + self.extra_deck:
            # Check for archetypes list
            if 'archetypes' in card and isinstance(card['archetypes'], list):
                for archetype in card['archetypes']:
                    archetypes[archetype] = archetypes.get(archetype, 0) + 1
            
            # Check for single archetype field
            elif 'archetype' in card and card['archetype']:
                archetype = card['archetype']
                archetypes[archetype] = archetypes.get(archetype, 0) + 1
        
        # Sort by count and take top 3
        top_archetypes = sorted(archetypes.items(), key=lambda x: x[1], reverse=True)[:3]
        return dict(top_archetypes)
    
    def get_metrics(self):
        """Get metrics for MLflow logging."""
        metrics = {
            "main_deck_size": self.total_main,
            "extra_deck_size": self.total_extra,
            "monster_count": self.monster_count,
            "spell_count": self.spell_count,
            "trap_count": self.trap_count,
            "monster_ratio": self.monster_ratio,
            "spell_ratio": self.spell_ratio,
            "trap_ratio": self.trap_ratio,
            "fusion_count": self.fusion_count,
            "synchro_count": self.synchro_count,
            "xyz_count": self.xyz_count,
            "link_count": self.link_count,
            "unique_cards_count": (self.copy_distribution['1_count'] + 
                                       self.copy_distribution['2_count'] + 
                                       self.copy_distribution['3_count']),
            "cards_as_1_ofs": self.copy_distribution['1_count'],
            "cards_as_2_ofs": self.copy_distribution['2_count'],
            "cards_as_3_ofs": self.copy_distribution['3_count'],
            "has_tuners": 1 if self.has_tuners else 0,
            "has_pendulums": 1 if self.has_pendulums else 0,
            
            # Add cluster-related metrics
            "cluster_entropy": self.cluster_entropy,
            "intra_deck_cluster_distance": self.intra_deck_cluster_distance,
            "cluster_co_occurrence_rarity": self.cluster_co_occurrence_rarity,
            "noise_card_percentage": self.noise_card_percentage,
        }
        
        return metrics
    
    def to_dict(self):
        """Convert metadata to dictionary for serialization."""
        return {
            "main_deck_size": self.total_main,
            "extra_deck_size": self.total_extra,
            "monster_count": self.monster_count,
            "spell_count": self.spell_count,
            "trap_count": self.trap_count,
            "monster_ratio": self.monster_ratio,
            "spell_ratio": self.spell_ratio,
            "trap_ratio": self.trap_ratio,
            "fusion_count": self.fusion_count,
            "synchro_count": self.synchro_count,
            "xyz_count": self.xyz_count,
            "link_count": self.link_count,
            "copy_distribution": self.copy_distribution,
            "archetypes": self.archetypes,
            "has_tuners": self.has_tuners,
            "has_pendulums": self.has_pendulums,
            
            # Add cluster-related metrics
            "cluster_entropy": self.cluster_entropy,
            "intra_deck_cluster_distance": self.intra_deck_cluster_distance,
            "cluster_co_occurrence_rarity": self.cluster_co_occurrence_rarity,
            "noise_card_percentage": self.noise_card_percentage,
            "cluster_distribution": self.cluster_distribution,
            "archetype_distribution": self.archetype_distribution,
            "dominant_archetype": self.dominant_archetype,
        }

class DeckGenerator:
    """Yu-Gi-Oh! Deck Generator with game-based constraints."""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of DeckGenerator."""
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self, clustering_model=None, card_data=None, num_cards=40):
        """Initialize the deck generator."""
        # Ensure singleton pattern
        if self.__class__._instance is not None and self.__class__._instance.initialized:
            return
        
        self.num_cards = num_cards
        self.extra_deck_size = 15
        
        # Target ratios with reasonable defaults
        self.target_monster_ratio = 0.6
        self.target_spell_ratio = 0.2
        self.target_trap_ratio = 0.2
        self.ratio_tolerance = 0.25
        
        # Load card data with clustering if provided
        if clustering_model is not None and card_data is not None:
            self.load_card_data(clustering_model, card_data)
        else:
            self.clustered_cards = {}
            self.main_deck_clusters = {}
            self.extra_deck_clusters = {}
            self.monster_clusters = {}
            self.spell_clusters = {}
            self.trap_clusters = {}
        
        self.initialized = True
        self.__class__._instance = self
    
    def load_card_data(self, clustering_model, card_data):
        """Load and organize clustered card data."""
        self.clustered_cards = get_card_data_with_clusters(card_data, clustering_model)
        
        # Organize cards by type and cluster
        self.main_deck_clusters = {}
        self.extra_deck_clusters = {}
        self.monster_clusters = {}
        self.spell_clusters = {}
        self.trap_clusters = {}
        
        for cluster_id, cards in self.clustered_cards.items():
            main_cards = []
            extra_cards = []
            monsters = []
            spells = []
            traps = []
            
            for card in cards:
                card_type = card.get('type', '')
                
                # Skip if no type or name
                if not card_type or not card.get('name', ''):
                    continue
                
                # Categorize by main/extra deck
                if any(et in card_type for et in ['Fusion', 'Synchro', 'Xyz', 'Link']):
                    extra_cards.append(card)
                else:
                    main_cards.append(card)
                    # Further categorize by type for main deck
                    if 'Monster' in card_type:
                        monsters.append(card)
                    elif 'Spell' in card_type:
                        spells.append(card)
                    elif 'Trap' in card_type:
                        traps.append(card)
            
            # Store in appropriate clusters
            if main_cards:
                self.main_deck_clusters[cluster_id] = main_cards
                
                if monsters:
                    self.monster_clusters[cluster_id] = monsters
                if spells:
                    self.spell_clusters[cluster_id] = spells
                if traps:
                    self.trap_clusters[cluster_id] = traps
                    
            if extra_cards:
                self.extra_deck_clusters[cluster_id] = extra_cards

    def apply_game_constraints(self, main_deck: List[Dict], extra_deck: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply Yu-Gi-Oh! game-based constraints to ensure realistic, playable decks.
        
        Args:
            main_deck: The generated main deck
            extra_deck: The generated extra deck
            
        Returns:
            Tuple of (updated_main_deck, updated_extra_deck)
        """
        logger.info("Applying Yu-Gi-Oh! game-based constraints to deck")
        
        # 1. First, ensure no card exceeds 3 copies (core YGO rule)
        main_deck = apply_max_copies_constraint(main_deck)
        
        # 2. Apply realistic monster copy distribution (most monsters as 2-3 copies) 
        # Extract monsters, spells, and traps from main deck
        monsters = [card for card in main_deck if card.get('type') and 'Monster' in card.get('type')]
        spells = [card for card in main_deck if card.get('type') and 'Spell' in card.get('type')]
        traps = [card for card in main_deck if card.get('type') and 'Trap' in card.get('type')]
        
        # Apply copy distribution to each card type
        optimized_deck = []
        
        # Optimize monsters (higher chance of 3-ofs)
        if len(monsters) >= 3:
            optimized_monsters = apply_copy_distribution_to_monsters(monsters, deck_size=len(monsters))
            optimized_deck.extend(optimized_monsters)
        else:
            optimized_deck.extend(monsters)
            
        # Optimize spells and traps together (more variety in copy counts)
        spells_traps = spells + traps
        if len(spells_traps) >= 3:
            optimized_spells_traps = apply_copy_distribution_to_spells_traps(spells_traps)
            optimized_deck.extend(optimized_spells_traps)
        else:
            optimized_deck.extend(spells_traps)
            
        # Update main_deck with our optimized version
        main_deck = optimized_deck
        
        # 3. Check and fix summoning mechanics requirements based on extra deck
        
        # Synchro constraint
        synchro_ok, main_deck = validate_synchro_requirements(main_deck, extra_deck)
        
        # Xyz constraint 
        xyz_ok, main_deck = validate_xyz_requirements(main_deck, extra_deck)  
        
        # Link constraint
        link_ok, main_deck = validate_link_requirements(main_deck, extra_deck)
        
        # Pendulum constraint
        pendulum_ok, main_deck = validate_pendulum_requirements(main_deck)
        
        # Verify 3-copy limit again after all the adjustments
        main_deck = apply_max_copies_constraint(main_deck)
        
        # Shuffle the main deck for randomness
        random.shuffle(main_deck)
        # Trim the deck to exactly 40 cards if needed
        if len(main_deck) > 40:
            main_deck = main_deck[:40]
        
        
        return main_deck, extra_deck
    
    def _generate_extra_deck(self, target_archetype: Optional[str] = None) -> List[Dict]:
        """Generate extra deck cards. Always returns exactly 15 cards."""
        extra_deck = []
        target_size = 15
        
        # Select clusters containing target archetype for meta-aware decks
        if target_archetype:
            archetype_extra_clusters = []
            for cluster_id in self.extra_deck_clusters.keys():
                for card in self.extra_deck_clusters[cluster_id]:
                    if target_archetype in card.get('archetypes', []) or card.get('archetype') == target_archetype:
                        archetype_extra_clusters.append(cluster_id)
                        break
            
            # Prefer clusters with the target archetype
            if archetype_extra_clusters:
                selected_clusters = random.sample(
                    archetype_extra_clusters, 
                    min(3, len(archetype_extra_clusters))
                )
                
                # Sample evenly across selected clusters
                cards_per_cluster = max(1, target_size // len(selected_clusters))
                cards_added = 0
                
                for cluster_id in selected_clusters:
                    if cards_added >= target_size:
                        break
                    cluster_cards = self.extra_deck_clusters[cluster_id]
                    if cluster_cards:
                        # Add cards from this cluster
                        cards_to_add = min(cards_per_cluster, len(cluster_cards), target_size - cards_added)
                        selected = random.sample(cluster_cards, cards_to_add)
                        extra_deck.extend(selected)
                        cards_added += cards_to_add
                
                # Fill remaining slots with generic extra deck cards
                remaining_slots = target_size - len(extra_deck)
                if remaining_slots > 0:
                    all_extra_cards = []
                    for cards in self.extra_deck_clusters.values():
                        all_extra_cards.extend(cards)
                    
                    # Avoid duplicates
                    extra_names = {card.get('name', '') for card in extra_deck}
                    remaining_cards = [card for card in all_extra_cards if card.get('name', '') not in extra_names]
                    
                    if remaining_cards:
                        additional = random.sample(remaining_cards, min(remaining_slots, len(remaining_cards)))
                        extra_deck.extend(additional)
        
        # For novel decks or if meta-aware selection didn't fill the extra deck
        if len(extra_deck) < target_size:
            # Select 2-3 diverse clusters
            selected_clusters = random.sample(
                list(self.extra_deck_clusters.keys()),
                min(3, len(self.extra_deck_clusters))
            )
            
            # Sample evenly across selected clusters
            cards_per_cluster = max(1, (target_size - len(extra_deck)) // len(selected_clusters))
            
            # Currently added card names for deduplication
            extra_names = {card.get('name', '') for card in extra_deck}
            
            for cluster_id in selected_clusters:
                if len(extra_deck) >= target_size:
                    break
                
                cluster_cards = self.extra_deck_clusters[cluster_id]
                # Filter out duplicates
                unique_cards = [card for card in cluster_cards if card.get('name', '') not in extra_names]
                
                if unique_cards:
                    # Add cards from this cluster
                    cards_to_add = min(cards_per_cluster, len(unique_cards), target_size - len(extra_deck))
                    selected = random.sample(unique_cards, cards_to_add)
                    extra_deck.extend(selected)
                    # Update our set of added names
                    extra_names.update(card.get('name', '') for card in selected)
        
        # If still not at target size, add random unique extra deck cards
        if len(extra_deck) < target_size:
            all_extra_cards = []
            for cards in self.extra_deck_clusters.values():
                all_extra_cards.extend(cards)
            
            # Filter out cards we've already added
            extra_names = {card.get('name', '') for card in extra_deck}
            remaining_cards = [card for card in all_extra_cards if card.get('name', '') not in extra_names]
            
            if remaining_cards:
                cards_to_add = min(target_size - len(extra_deck), len(remaining_cards))
                selected = random.sample(remaining_cards, cards_to_add)
                extra_deck.extend(selected)
            else:
                # If we've exhausted unique cards, allow duplicates as a last resort
                while len(extra_deck) < target_size and all_extra_cards:
                    extra_deck.append(random.choice(all_extra_cards))
        
        # Ensure exactly target size (typically 15)
        extra_deck = extra_deck[:target_size]
        
        # Ensure we have a good distribution of summon types in the extra deck
        # Aim for some variety among Fusion, Synchro, Xyz, and Link
        type_counts = Counter()
        for card in extra_deck:
            card_type = card.get('type', '')
            if 'Fusion' in card_type:
                type_counts['Fusion'] += 1
            elif 'Synchro' in card_type:
                type_counts['Synchro'] += 1
            elif 'Xyz' in card_type:
                type_counts['Xyz'] += 1
            elif 'Link' in card_type:
                type_counts['Link'] += 1
        
        logger.info(f"Generated Extra Deck with {len(extra_deck)} cards: "
                  f"{type_counts.get('Fusion', 0)} Fusion, "
                  f"{type_counts.get('Synchro', 0)} Synchro, "
                  f"{type_counts.get('Xyz', 0)} Xyz, "
                  f"{type_counts.get('Link', 0)} Link")
        
        return extra_deck

    def generate_deck(self, mode: str = "novel", target_archetype: Optional[str] = None, 
                    use_mlflow: bool = True) -> Tuple[List[Dict], List[Dict], 'DeckMetadata']:
        """
        Generate a complete Yu-Gi-Oh! deck with game constraints applied.
        
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
                    ratio_tolerance=self.ratio_tolerance,
                    game_constraint_enabled=True  # Log that game constraints are enabled
                )
                
                # Generate the deck
                main_deck, extra_deck = self._generate_deck_internal(mode, target_archetype)
                
                # Apply game constraints (new!)
                main_deck, extra_deck = self.apply_game_constraints(main_deck, extra_deck)
                
                # Create metadata
                metadata = DeckMetadata(main_deck, extra_deck, self.clustered_cards)
                
                # Log metrics
                log_deck_metrics(main_deck, extra_deck, metadata)
                
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
            main_deck, extra_deck = self._generate_deck_internal(mode, target_archetype)
            main_deck, extra_deck = self.apply_game_constraints(main_deck, extra_deck)
            metadata = DeckMetadata(main_deck, extra_deck, self.clustered_cards)
            return main_deck, extra_deck, metadata
    
    def _generate_deck_internal(self, mode: str, target_archetype: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
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
    
    def _randomize_ratios(self):
        """Randomize the deck ratios for each deck generation to ensure variety."""
        # Occasionally generate a deck with very low monster count (15-20 monsters)
        low_monster_ratio = random.random() < 0.15  # 15% chance for low monster count deck
        
        if low_monster_ratio:
            # Generate a deck with 15-20 monsters (37.5-50% of 40 card deck)
            base_monster = random.uniform(0.375, 0.5)
            logger.info("Generating low monster count deck variant")
        else:
            # Normal deck with more standard ratios
            # Apply a stronger random fluctuation for more variation between decks
            fluctuation = random.uniform(-0.15, 0.2)  # Wider range of fluctuation
            base_monster = 0.6
            base_monster = base_monster + fluctuation
        
        # Apply wider bounds for monster ratio
        # Min 37.5% monsters (15/40), max 80% monsters for greater variety
        self.target_monster_ratio = max(0.375, min(0.8, base_monster))
        
        # Distribute remaining percentage between spells and traps
        # Apply additional variation to their distribution
        remaining = 1.0 - self.target_monster_ratio
        spell_trap_variation = random.uniform(-0.25, 0.25)  # Wider variation between spells and traps
        
        # Baseline weights
        spell_weight = 0.5 + spell_trap_variation  # Can shift between 0.25 and 0.75 of remaining
        spell_weight = max(0.25, min(0.75, spell_weight))  # Cap between 25% and 75% of remaining
        trap_weight = 1.0 - spell_weight
        
        self.target_spell_ratio = remaining * spell_weight
        self.target_trap_ratio = remaining * trap_weight
        
        # Wider tolerance for more variety
        self.ratio_tolerance = 0.2
        
        logger.info(f"Deck ratios randomized to: {self.target_monster_ratio:.1%}M/{self.target_spell_ratio:.1%}S/{self.target_trap_ratio:.1%}T (±{self.ratio_tolerance:.0%} tolerance)")
    
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
        
    def _generate_meta_aware_deck(self, target_archetype: str) -> Tuple[List[Dict], List[Dict]]:
        """Generate a deck anchored on a target archetype with proper card type ratios."""
        logger.info(f"Generating meta-aware deck for archetype: {target_archetype}")
        
        # Calculate target counts for each card type
        target_monsters, target_spells, target_traps = self._calculate_target_counts(self.num_cards)
        
        # Find clusters containing the target archetype
        archetype_monster_clusters = []
        archetype_spell_clusters = []
        archetype_trap_clusters = []
        
        # Find clusters with the target archetype
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
        
        # Add monsters from the archetype (60% of deck, ~24 cards)
        main_deck = []
        monsters_added = 0
        
        # First, prioritize clusters with the target archetype
        for cluster_id in archetype_monster_clusters:
            if monsters_added >= target_monsters:
                break
            
            archetype_cards = [
                card for card in self.monster_clusters[cluster_id]
                if target_archetype in card.get('archetypes', []) or card.get('archetype') == target_archetype
            ]
            
            # Take all archetype cards, but don't exceed target
            cards_to_add = min(len(archetype_cards), target_monsters - monsters_added)
            if cards_to_add > 0:
                selected = random.sample(archetype_cards, cards_to_add)
                main_deck.extend(selected)
                monsters_added += cards_to_add
        
        # If we still need more monsters, add from archetype clusters
        remaining_slots = target_monsters - monsters_added
        if remaining_slots > 0 and archetype_monster_clusters:
            # Collect all remaining monster cards from archetype clusters
            remaining_cards = []
            for cluster_id in archetype_monster_clusters:
                cluster_cards = self.monster_clusters[cluster_id]
                already_added = [card for card in main_deck if card in cluster_cards]
                remaining_cards.extend([card for card in cluster_cards if card not in already_added])
            
            # Add more unique monsters from these clusters
            if remaining_cards:
                cards_to_add = min(len(remaining_cards), remaining_slots)
                selected = random.sample(remaining_cards, cards_to_add)
                main_deck.extend(selected)
                monsters_added += cards_to_add
                remaining_slots = target_monsters - monsters_added
        
        # If we still need more monsters, add from any cluster
        if remaining_slots > 0:
            all_monster_cards = []
            for cluster_id in self.monster_clusters.keys():
                cluster_cards = self.monster_clusters[cluster_id]
                already_added = [card for card in main_deck if card in cluster_cards]
                all_monster_cards.extend([card for card in cluster_cards if card not in already_added])
            
            if all_monster_cards:
                cards_to_add = min(len(all_monster_cards), remaining_slots)
                selected = random.sample(all_monster_cards, cards_to_add)
                main_deck.extend(selected)
                monsters_added += cards_to_add
        
        # Add spells (20% of deck, ~8 cards)
        spells_added = 0
        
        # First, prioritize archetype-specific spells
        for cluster_id in archetype_spell_clusters:
            if spells_added >= target_spells:
                break
            
            archetype_cards = [
                card for card in self.spell_clusters[cluster_id]
                if target_archetype in card.get('archetypes', []) or card.get('archetype') == target_archetype
            ]
            
            # Take all archetype cards, but don't exceed target
            cards_to_add = min(len(archetype_cards), target_spells - spells_added)
            if cards_to_add > 0:
                selected = random.sample(archetype_cards, cards_to_add)
                main_deck.extend(selected)
                spells_added += cards_to_add
        
        # If we still need more spells, add from archetype clusters
        remaining_slots = target_spells - spells_added
        if remaining_slots > 0 and archetype_spell_clusters:
            remaining_cards = []
            for cluster_id in archetype_spell_clusters:
                cluster_cards = self.spell_clusters[cluster_id]
                already_added = [card for card in main_deck if card in cluster_cards]
                remaining_cards.extend([card for card in cluster_cards if card not in already_added])
            
            if remaining_cards:
                cards_to_add = min(len(remaining_cards), remaining_slots)
                selected = random.sample(remaining_cards, cards_to_add)
                main_deck.extend(selected)
                spells_added += cards_to_add
                remaining_slots = target_spells - spells_added
        
        # If we still need more spells, add from any cluster
        if remaining_slots > 0:
            all_spell_cards = []
            for cluster_id in self.spell_clusters.keys():
                cluster_cards = self.spell_clusters[cluster_id]
                already_added = [card for card in main_deck if card in cluster_cards]
                all_spell_cards.extend([card for card in cluster_cards if card not in already_added])
            
            if all_spell_cards:
                cards_to_add = min(len(all_spell_cards), remaining_slots)
                selected = random.sample(all_spell_cards, cards_to_add)
                main_deck.extend(selected)
                spells_added += cards_to_add
        
        # Add traps (20% of deck, ~8 cards)
        traps_added = 0
        
        # First, prioritize archetype-specific traps
        for cluster_id in archetype_trap_clusters:
            if traps_added >= target_traps:
                break
            
            archetype_cards = [
                card for card in self.trap_clusters[cluster_id]
                if target_archetype in card.get('archetypes', []) or card.get('archetype') == target_archetype
            ]
            
            # Take all archetype cards, but don't exceed target
            cards_to_add = min(len(archetype_cards), target_traps - traps_added)
            if cards_to_add > 0:
                selected = random.sample(archetype_cards, cards_to_add)
                main_deck.extend(selected)
                traps_added += cards_to_add
        
        # If we still need more traps, add from archetype clusters
        remaining_slots = target_traps - traps_added
        if remaining_slots > 0 and archetype_trap_clusters:
            remaining_cards = []
            for cluster_id in archetype_trap_clusters:
                cluster_cards = self.trap_clusters[cluster_id]
                already_added = [card for card in main_deck if card in cluster_cards]
                remaining_cards.extend([card for card in cluster_cards if card not in already_added])
            
            if remaining_cards:
                cards_to_add = min(len(remaining_cards), remaining_slots)
                selected = random.sample(remaining_cards, cards_to_add)
                main_deck.extend(selected)
                traps_added += cards_to_add
                remaining_slots = target_traps - traps_added
        
        # If we still need more traps, add from any cluster
        if remaining_slots > 0:
            all_trap_cards = []
            for cluster_id in self.trap_clusters.keys():
                cluster_cards = self.trap_clusters[cluster_id]
                already_added = [card for card in main_deck if card in cluster_cards]
                all_trap_cards.extend([card for card in cluster_cards if card not in already_added])
            
            if all_trap_cards:
                cards_to_add = min(len(all_trap_cards), remaining_slots)
                selected = random.sample(all_trap_cards, cards_to_add)
                main_deck.extend(selected)
                traps_added += cards_to_add
        
        # Ensure exactly 40 cards and fill any remaining slots
        if len(main_deck) < self.num_cards:
            # Determine which type we need more of based on current ratios
            ratios = self._check_ratios(main_deck)
            
            while len(main_deck) < self.num_cards:
                if ratios['monster_ratio'] < self.target_monster_ratio - 0.05 and self.monster_clusters:
                    # Need more monsters
                    all_monster_cards = [card for cards in self.monster_clusters.values() for card in cards 
                                       if card not in main_deck]
                    if all_monster_cards:
                        main_deck.append(random.choice(all_monster_cards))
                elif ratios['spell_ratio'] < self.target_spell_ratio - 0.05 and self.spell_clusters:
                    # Need more spells
                    all_spell_cards = [card for cards in self.spell_clusters.values() for card in cards
                                     if card not in main_deck]
                    if all_spell_cards:
                        main_deck.append(random.choice(all_spell_cards))
                elif ratios['trap_ratio'] < self.target_trap_ratio - 0.05 and self.trap_clusters:
                    # Need more traps
                    all_trap_cards = [card for cards in self.trap_clusters.values() for card in cards
                                    if card not in main_deck]
                    if all_trap_cards:
                        main_deck.append(random.choice(all_trap_cards))
                else:
                    # Fill with any available card
                    all_main_cards = [card for cards in self.main_deck_clusters.values() for card in cards
                                    if card not in main_deck]
                    if all_main_cards:
                        main_deck.append(random.choice(all_main_cards))
                    else:
                        break  # Can't find more unique cards
                
                # Recalculate ratios
                ratios = self._check_ratios(main_deck)
        
        # Trim to exactly 40 cards if needed
        main_deck = main_deck[:self.num_cards]
        
        # Generate extra deck
        extra_deck = self._generate_extra_deck(target_archetype)
        
        # Log ratio results
        final_ratios = self._check_ratios(main_deck)
        logger.info(f"Generated meta-aware deck with ratios: "
                   f"Monsters: {final_ratios['monster_ratio']:.1%}, "
                   f"Spells: {final_ratios['spell_ratio']:.1%}, "
                   f"Traps: {final_ratios['trap_ratio']:.1%}")
        
        return main_deck, extra_deck
    
    def _generate_novel_deck(self) -> Tuple[List[Dict], List[Dict]]:
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
                all_monster_cards = [card for cards in self.monster_clusters.values() for card in cards 
                                   if card not in main_deck]
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
                all_spell_cards = [card for cards in self.spell_clusters.values() for card in cards
                                 if card not in main_deck]
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
                all_trap_cards = [card for cards in self.trap_clusters.values() for card in cards
                                if card not in main_deck]
                if all_trap_cards:
                    additional = random.sample(all_trap_cards, min(remaining_slots, len(all_trap_cards)))
                    main_deck.extend(additional)
                    traps_added += len(additional)
                else:
                    break
        
        # Ensure exactly 40 cards and fill any remaining slots
        while len(main_deck) < self.num_cards:
            # Determine which type we need more of based on current ratios
            ratios = self._check_ratios(main_deck)
            
            if ratios['monster_ratio'] < self.target_monster_ratio - 0.05 and self.monster_clusters:
                # Need more monsters
                all_monster_cards = [card for cards in self.monster_clusters.values() for card in cards
                                   if card not in main_deck]
                if all_monster_cards:
                    main_deck.append(random.choice(all_monster_cards))
            elif ratios['spell_ratio'] < self.target_spell_ratio - 0.05 and self.spell_clusters:
                # Need more spells
                all_spell_cards = [card for cards in self.spell_clusters.values() for card in cards
                                 if card not in main_deck]
                if all_spell_cards:
                    main_deck.append(random.choice(all_spell_cards))
            elif ratios['trap_ratio'] < self.target_trap_ratio - 0.05 and self.trap_clusters:
                # Need more traps
                all_trap_cards = [card for cards in self.trap_clusters.values() for card in cards
                                if card not in main_deck]
                if all_trap_cards:
                    main_deck.append(random.choice(all_trap_cards))
            else:
                # Fill with any available card
                all_main_cards = [card for cards in self.main_deck_clusters.values() for card in cards
                                if card not in main_deck]
                if all_main_cards:
                    main_deck.append(random.choice(all_main_cards))
                else:
                    break
        
        # Trim to exactly 40 cards if needed
        main_deck = main_deck[:self.num_cards]
        
        # Generate extra deck
        extra_deck = self._generate_extra_deck()
        
        # Log ratio results
        final_ratios = self._check_ratios(main_deck)
        logger.info(f"Generated novel deck with ratios: "
                  f"Monsters: {final_ratios['monster_ratio']:.1%}, "
                  f"Spells: {final_ratios['spell_ratio']:.1%}, "
                  f"Traps: {final_ratios['trap_ratio']:.1%}")
        
        return main_deck, extra_deck

def apply_copy_distribution_to_monsters(cards: List[Dict], deck_size: int = 40) -> List[Dict]:
    """
    Apply a realistic distribution of card copies for monster cards.
    
    Most monsters should be 2-3 copies (60-70%)
    Some monsters should be tech cards at 1 copy (20-30%)
    No card should have more than 3 copies
    
    Args:
        cards: List of monster card dictionaries (already filtered to just monsters)
        deck_size: Target size for result (optional, defaults to 40)
    
    Returns:
        Updated list of cards with realistic copy distribution
    """
    # Group cards by name for easier processing
    card_groups = defaultdict(list)
    for card in cards:
        card_name = card.get('name', '')
        if card_name:  # Skip cards with no name
            card_groups[card_name].append(card)
    
    # Calculate target distribution
    unique_monsters = len(card_groups)
    if unique_monsters == 0:
        return []
    
    # Target distribution percentages with more 2-ofs and 3-ofs
    pct_3copies = 0.45  # ~45% as 3-ofs
    pct_2copies = 0.45  # ~45% as 2-ofs
    pct_1copies = 0.10  # ~10% as 1-ofs
    
    # Calculate how many of each copy count we need
    target_3copies = int(unique_monsters * pct_3copies)
    target_2copies = int(unique_monsters * pct_2copies)
    target_1copies = unique_monsters - target_3copies - target_2copies
    
    # Adjust if we have very few unique monsters
    if unique_monsters <= 3:
        # If 3 or fewer unique monsters, make them all 3-ofs
        target_3copies = unique_monsters
        target_2copies = 0
        target_1copies = 0
    elif unique_monsters <= 6:
        # If 6 or fewer unique monsters, make half 3-ofs and half 2-ofs
        target_3copies = unique_monsters // 2
        target_2copies = unique_monsters - target_3copies
        target_1copies = 0
    
    # Assign copy counts to each unique monster
    monster_names = list(card_groups.keys())
    random.shuffle(monster_names)  # Shuffle to randomize which cards get which copy count
    
    copy_assignments = {}
    
    # Assign 3-ofs
    for i in range(target_3copies):
        if i < len(monster_names):
            copy_assignments[monster_names[i]] = 3
    
    # Assign 2-ofs
    for i in range(target_3copies, target_3copies + target_2copies):
        if i < len(monster_names):
            copy_assignments[monster_names[i]] = 2
    
    # Assign 1-ofs
    for i in range(target_3copies + target_2copies, unique_monsters):
        if i < len(monster_names):
            copy_assignments[monster_names[i]] = 1
    
    # Build the new monster list with correct copy counts
    result = []
    for card_name, copy_count in copy_assignments.items():
        cards = card_groups[card_name]
        if cards:
            # Add the assigned number of copies (or all if fewer exist)
            copies_to_add = min(copy_count, len(cards))
            result.extend(cards[:copies_to_add])
    
    # Log the distribution we achieved
    final_counts = Counter()
    for card_name in copy_assignments.keys():
        copies = sum(1 for card in result if card.get('name', '') == card_name)
        final_counts[copies] += 1
    
    total_monsters = sum(count * copies for copies, count in final_counts.items())
    logger.info(f"Monster copy distribution: " +
              f"{final_counts.get(3, 0)} cards as 3-ofs, " +
              f"{final_counts.get(2, 0)} cards as 2-ofs, " +
              f"{final_counts.get(1, 0)} cards as 1-ofs " +
              f"(total {total_monsters} monsters)")
    
    return result

def apply_copy_distribution_to_spells_traps(cards: List[Dict]) -> List[Dict]:
    """
    Apply a realistic distribution of card copies for spell and trap cards.
    
    Key spells and traps should be 2-3 copies (60-70%)
    Tech/situational cards should be 1 copy (30-40%)
    No card should have more than 3 copies
    
    Args:
        cards: List of spell/trap card dictionaries
    
    Returns:
        Updated list of cards with realistic copy distribution
    """
    # Group cards by name for easier processing
    card_groups = defaultdict(list)
    for card in cards:
        card_name = card.get('name', '')
        if card_name:  # Skip cards with no name
            card_groups[card_name].append(card)
    
    # Calculate target distribution
    unique_cards = len(card_groups)
    if unique_cards == 0:
        return []
    
    # Target distribution percentages with emphasis on 3-ofs for key spells/traps
    pct_3copies = 0.4  # ~40% as 3-ofs
    pct_2copies = 0.3  # ~30% as 2-ofs
    pct_1copies = 0.3  # ~30% as 1-ofs
    
    # Calculate how many of each copy count we need
    target_3copies = int(unique_cards * pct_3copies)
    target_2copies = int(unique_cards * pct_2copies)
    target_1copies = unique_cards - target_3copies - target_2copies
    
    # Adjust if we have very few unique cards
    if unique_cards <= 3:
        # If 3 or fewer unique cards, make them all 3-ofs
        target_3copies = unique_cards
        target_2copies = 0
        target_1copies = 0
    elif unique_cards <= 6:
        # If 6 or fewer unique cards, make half 3-ofs and half 2-ofs
        target_3copies = unique_cards // 2
        target_2copies = unique_cards - target_3copies
        target_1copies = 0
    
    # Assign copy counts to each unique card
    card_names = list(card_groups.keys())
    random.shuffle(card_names)  # Shuffle to randomize which cards get which copy count
    
    copy_assignments = {}
    
    # Assign 3-ofs
    for i in range(target_3copies):
        if i < len(card_names):
            copy_assignments[card_names[i]] = 3
    
    # Assign 2-ofs
    for i in range(target_3copies, target_3copies + target_2copies):
        if i < len(card_names):
            copy_assignments[card_names[i]] = 2
    
    # Assign 1-ofs (all remaining cards)
    for i in range(target_3copies + target_2copies, len(card_names)):
        copy_assignments[card_names[i]] = 1
    
    # Build the result
    result = []
    for card_name, copy_count in copy_assignments.items():
        original_cards = card_groups[card_name]
        result.extend(original_cards[:copy_count])  # Add the required number of copies
    
    return result

def validate_synchro_requirements(main_deck: List[Dict], extra_deck: List[Dict]) -> Tuple[bool, List[Dict]]:
    """
    Validate and fix Synchro summoning requirements.
    
    If Extra Deck has Synchro monsters:
    1. Ensure Main Deck has at least one Tuner monster
    2. Try to ensure levels are compatible with Synchro monsters
    
    Args:
        main_deck: The main deck cards
        extra_deck: The extra deck cards
    
    Returns:
        Tuple of (requirements_met, updated_main_deck)
    """
    # Check if we have any Synchro monsters in the Extra Deck
    synchro_monsters = [card for card in extra_deck if card.get('type') and 'Synchro' in card.get('type')]
    if not synchro_monsters:
        return True, main_deck  # No Synchro monsters, no need to validate
    
    # Check if we have any Tuners in the Main Deck
    tuners = [card for card in main_deck if is_tuner(card)]
    
    if tuners:
        logger.info(f"Synchro requirement met: Main Deck has {len(tuners)} Tuner monsters")
        # We have tuners, check if levels can match synchro requirements
        
        # Get levels of tuners and non-tuners
        tuner_levels = [get_monster_level(card) for card in tuners]
        non_tuners = [card for card in main_deck if card.get('type') and 'Monster' in card.get('type') and not is_tuner(card)]
        non_tuner_levels = [get_monster_level(card) for card in non_tuners]
        
        # Get levels of synchro monsters
        synchro_levels = [get_monster_level(card) for card in synchro_monsters]
        
        # Check if we can make valid synchro summons
        can_synchro = False
        for tuner_level in tuner_levels:
            for non_tuner_level in non_tuner_levels:
                sum_level = tuner_level + non_tuner_level
                if any(sum_level == synchro_level for synchro_level in synchro_levels):
                    can_synchro = True
                    break
            if can_synchro:
                break
                
        if can_synchro:
            logger.info("Synchro level requirements met: Main Deck contains compatible levels for Synchro Summons")
            return True, main_deck
        else:
            logger.info("Synchro levels not optimal - but this is a soft requirement, continuing with existing monsters")
            return True, main_deck  # Still return True as this is a soft constraint
    else:
        # We don't have tuners but need them - get the DeckGenerator instance to add a tuner
        logger.info("Missing Tuner monsters for Synchro Summons - adding at least one Tuner")
        
        # Get DeckGenerator instance to access card clusters
        deck_gen = DeckGenerator.get_instance()
        
        # Find tuners in our card database
        potential_tuners = []
        for cluster_cards in deck_gen.monster_clusters.values():
            for card in cluster_cards:
                if is_tuner(card):
                    potential_tuners.append(card)
        
        if potential_tuners:
            # Choose a random tuner and add it to the main deck
            tuner_to_add = random.choice(potential_tuners)
            main_deck.append(tuner_to_add)
            logger.info(f"Added Tuner monster: {tuner_to_add.get('name', 'Unknown')} (Level {get_monster_level(tuner_to_add)})")
            return True, main_deck
        else:
            logger.warning("No Tuner monsters found in card database - cannot fix Synchro requirements")
            return False, main_deck

def validate_xyz_requirements(main_deck: List[Dict], extra_deck: List[Dict]) -> Tuple[bool, List[Dict]]:
    """
    Validate and fix Xyz summoning requirements.
    
    If Extra Deck has Xyz monsters:
    1. Ensure Main Deck has at least 2 monsters of the same level
    2. Try to ensure those levels match the ranks of Xyz monsters
    
    Args:
        main_deck: The main deck cards
        extra_deck: The extra deck cards
    
    Returns:
        Tuple of (requirements_met, updated_main_deck)
    """
    # Check if we have any Xyz monsters in the Extra Deck
    xyz_monsters = [card for card in extra_deck if card.get('type') and 'Xyz' in card.get('type')]
    if not xyz_monsters:
        return True, main_deck  # No Xyz monsters, no need to validate
    
    # Get all monsters in the main deck
    monsters = [card for card in main_deck if card.get('type') and 'Monster' in card.get('type')]
    
    # Count monsters by level
    level_counts = Counter(get_monster_level(card) for card in monsters)
    
    # Check if we have at least 2 monsters of the same level
    has_same_level = any(count >= 2 for level, count in level_counts.items() if level > 0)
    
    if has_same_level:
        logger.info("Xyz requirement met: Main Deck has monsters of the same level")
        
        # Check if levels match ranks of Xyz monsters (bonus requirement)
        xyz_ranks = set(get_monster_level(card) for card in xyz_monsters)
        matching_levels = any(level in xyz_ranks for level, count in level_counts.items() if count >= 2)
        
        if matching_levels:
            logger.info("Xyz rank requirements met: Main Deck contains matching levels for Xyz Summons")
        else:
            logger.info("Xyz ranks not matched exactly, but this is a soft requirement")
            
        return True, main_deck
    else:
        # We don't have enough monsters of the same level - fix by adding some
        logger.info("Missing monsters of the same level for Xyz Summons - adding compatible monsters")
        
        # Get DeckGenerator instance to access card clusters
        deck_gen = DeckGenerator.get_instance()
        
        # Get common Xyz ranks (typically 3, 4, 7, 8)
        xyz_ranks = [get_monster_level(card) for card in xyz_monsters]
        most_common_rank = Counter(xyz_ranks).most_common(1)[0][0] if xyz_ranks else 4  # Default to 4 if can't determine
        
        # Find monsters with matching level in our card database
        matching_monsters = []
        for cluster_cards in deck_gen.monster_clusters.values():
            for card in cluster_cards:
                if get_monster_level(card) == most_common_rank:
                    matching_monsters.append(card)
        
        if matching_monsters:
            # Choose at least 2 monsters with the desired level
            monsters_to_add = random.sample(matching_monsters, min(2, len(matching_monsters)))
            main_deck.extend(monsters_to_add)
            logger.info(f"Added {len(monsters_to_add)} Level {most_common_rank} monsters to support Xyz Summons")
            return True, main_deck
        else:
            logger.warning(f"No Level {most_common_rank} monsters found in card database - cannot fix Xyz requirements")
            return False, main_deck
def validate_link_requirements(main_deck: List[Dict], extra_deck: List[Dict]) -> Tuple[bool, List[Dict]]:
    """
    Validate and fix Link summoning requirements.
    
    If Extra Deck has Link monsters:
    1. Ensure Main Deck has enough monsters (at least 2) to serve as Link Materials
    2. Optionally check Link arrows
    
    Args:
        main_deck: The main deck cards
        extra_deck: The extra deck cards
    
    Returns:
        Tuple of (requirements_met, updated_main_deck)
    """
    # Check if we have any Link monsters in the Extra Deck
    link_monsters = [card for card in extra_deck if card.get('type') and 'Link' in card.get('type')]
    if not link_monsters:
        return True, main_deck  # No Link monsters, no need to validate
    
    # Get all monsters in the main deck
    monsters = [card for card in main_deck if card.get('type') and 'Monster' in card.get('type')]
    
    # For Link Summons, we need at least 2 monsters (for Link-1) and more for higher Link ratings
    highest_link = max((card.get('linkval', 1) for card in link_monsters), default=1)
    min_monsters_needed = min(highest_link + 1, 3)  # Need at least this many to make basic Link plays
    
    if len(monsters) >= min_monsters_needed:
        logger.info(f"Link requirement met: Main Deck has {len(monsters)} monsters for Link Materials (needed {min_monsters_needed})")
        return True, main_deck
    else:
        # We don't have enough monsters - add some generic ones
        logger.info(f"Not enough monsters for Link Summons - adding {min_monsters_needed - len(monsters)} more monsters")
        
        # Get DeckGenerator instance to access card clusters
        deck_gen = DeckGenerator.get_instance()
        
        # Find any monster cards in our database
        additional_monsters = []
        for cluster_cards in deck_gen.monster_clusters.values():
            additional_monsters.extend(cluster_cards)
        
        if additional_monsters:
            # Shuffle to get random monsters and select how many we need
            random.shuffle(additional_monsters)
            monsters_to_add = min_monsters_needed - len(monsters)
            main_deck.extend(additional_monsters[:monsters_to_add])
            
            logger.info(f"Added {monsters_to_add} monsters to support Link Summons")
            return True, main_deck
        else:
            logger.warning("No monster cards found in database - cannot fix Link requirements")
            return False, main_deck
def validate_pendulum_requirements(main_deck: List[Dict]) -> Tuple[bool, List[Dict]]:
    """
    Validate and fix Pendulum summoning requirements.
    
    If Main Deck has any Pendulum monsters:
    1. Ensure there are at least 2 Pendulum cards (for setting scales)
    2. Try to ensure the scales have different values
    
    Args:
        main_deck: The main deck cards
    
    Returns:
        Tuple of (requirements_met, updated_main_deck)
    """
    # Check if we have any Pendulum monsters in the Main Deck
    pendulum_monsters = [card for card in main_deck if is_pendulum(card)]
    if not pendulum_monsters:
        return True, main_deck  # No Pendulum monsters, no need to validate
    
    # For Pendulum Summons, ideally we want at least 2 Pendulum monsters with different scales
    if len(pendulum_monsters) >= 2:
        # Check if we have different scales
        scales = set()
        for card in pendulum_monsters:
            left, right = get_pendulum_scales(card)
            scales.add(left)
            scales.add(right)
        
        if len(scales) >= 2:
            logger.info(f"Pendulum requirement met: Main Deck has {len(pendulum_monsters)} Pendulum monsters with different scales")
            return True, main_deck
        else:
            logger.info(f"Pendulum monsters have identical scales, but we already have {len(pendulum_monsters)} Pendulum monsters (soft requirement)")
            return True, main_deck  # Consider it a soft requirement
    else:
        # We don't have enough pendulum monsters - add at least one more
        logger.info("Not enough Pendulum monsters for Pendulum Summons - adding another Pendulum monster")
        
        # Get DeckGenerator instance to access card clusters
        deck_gen = DeckGenerator.get_instance()
        
        # Find pendulum monsters in our database
        additional_pendulums = []
        for cluster_cards in deck_gen.monster_clusters.values():
            for card in cluster_cards:
                if is_pendulum(card) and card not in pendulum_monsters:
                    # Ideally find one with a different scale
                    existing_scales = set(scale for pend in pendulum_monsters for scale in get_pendulum_scales(pend))
                    new_scales = get_pendulum_scales(card)
                    
                    if new_scales[0] not in existing_scales or new_scales[1] not in existing_scales:
                        additional_pendulums.append(card)
        
        # If we couldn't find pendulum cards with different scales, accept any pendulum card
        if not additional_pendulums:
            for cluster_cards in deck_gen.monster_clusters.values():
                for card in cluster_cards:
                    if is_pendulum(card) and card not in pendulum_monsters:
                        additional_pendulums.append(card)
        
        if additional_pendulums:
            # Add a pendulum monster to the deck
            card_to_add = random.choice(additional_pendulums)
            main_deck.append(card_to_add)
            
            logger.info(f"Added Pendulum monster: {card_to_add.get('name', 'Unknown')} with scales {get_pendulum_scales(card_to_add)}")
            return True, main_deck
        else:
            logger.warning("No additional Pendulum monsters found in database - cannot fix Pendulum requirements")
            return True, main_deck  # Return True anyway since it's more of a soft requirement

def apply_max_copies_constraint(deck: List[Dict]) -> List[Dict]:
    """
    Ensure no card appears more than 3 times in the deck (Yu-Gi-Oh! core rule).
    
    Args:
        deck: List of card dictionaries
        
    Returns:
        Updated deck with no more than 3 copies of any card
    """
    # Count occurrences of each card by name
    card_counts = Counter(card.get('name', '') for card in deck)
    
    # Identify cards exceeding 3 copies
    excess_cards = {name: count for name, count in card_counts.items() if count > 3}
    
    if not excess_cards:
        return deck  # No changes needed
    
    # Remove excess copies
    new_deck = []
    current_counts = defaultdict(int)
    
    for card in deck:
        card_name = card.get('name', '')
        # Add card if we haven't hit the limit of 3 yet
        if current_counts[card_name] < 3:
            new_deck.append(card)
            current_counts[card_name] += 1
    
    logger.info(f"Removed {len(deck) - len(new_deck)} excess cards to enforce 3-copy limit")
    return new_deck

# Define SimpleDeckGenerator as an alias for DeckGenerator for backwards compatibility
SimpleDeckGenerator = DeckGenerator
