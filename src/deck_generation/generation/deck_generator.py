"""
Simple Yu-Gi-Oh! Deck Generator

A streamlined deck generation system that creates decks from clustered card data.
Supports both meta-aware and novel deck generation modes.

Enhancements:
- Optional HARD reachability for Synchro/XYZ/Link/Pendulum with graceful fallback.
- Smarter trimming back to 40 after auto-fixes (keeps enablers).
- Extra reachability metrics for MLflow (synchro coverage, xyz pairs, link material depth, pendulum span).
- Cleaner extra-deck diversity and dedupe.
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
    setup_experiment,
    log_deck_generation_tags,
    log_deck_generation_params,
    log_deck_metrics,
    log_deck_artifacts
)

from src.utils.mlflow.get_clustering_model_from_registry import (
    get_clustering_model_from_registry,
    get_card_data_with_clusters
)

# Import deck scoring utilities
from src.utils.deck_scoring.deck_scoring_utils import (
    calculate_cluster_entropy,
    calculate_intra_deck_cluster_distance, 
    calculate_cluster_cooccurrence_rarity,
    calculate_noise_card_percentage,
    get_cluster_distribution,
    get_archetype_distribution,
    get_dominant_archetype,
    create_card_to_cluster_mapping
)

logger = logging.getLogger(__name__)


# =========================
# Types & small helpers
# =========================

class CardType(Enum):
    """Yu-Gi-Oh! card types."""
    MONSTER = "Monster"
    SPELL = "Spell"
    TRAP = "Trap"
    FUSION = "Fusion"
    SYNCHRO = "Synchro"
    XYZ = "XYZ"
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


def is_monster(card: Dict) -> bool:
    return card.get('type') and 'Monster' in card.get('type')

def is_spell(card: Dict) -> bool:
    return card.get('type') and 'Spell' in card.get('type')

def is_trap(card: Dict) -> bool:
    return card.get('type') and 'Trap' in card.get('type')

# Card constraint utility functions
def is_tuner(card: Dict) -> bool:
    """Check if a card is a Tuner monster."""
    return card.get('type') and 'Monster' in card.get('type') and 'Tuner' in card.get('type')

def is_pendulum(card: Dict) -> bool:
    """Check if a card is a Pendulum monster."""
    return card.get('type') and 'Pendulum' in card.get('type')

def get_monster_level(card: Dict) -> int:
    """Get monster level/rank/link value."""
    if card.get('type') and 'Monster' in card.get('type'):
        if 'Link' in card.get('type'):
            return int(card.get('linkval', 0) or 0)
        else:
            return int(card.get('level', 0) or 0)
    return 0

def get_pendulum_scales(card: Dict) -> Tuple[int, int]:
    """Get pendulum scales for a card if it's a pendulum monster."""
    if is_pendulum(card):
        # Return left and right scales (typically found in lscale and rscale)
        return int(card.get('lscale', 0) or 0), int(card.get('rscale', 0) or 0)
    return 0, 0

def get_link_markers(card: Dict) -> List[str]:
    """Get link markers for a Link monster."""
    if card.get('type') and 'Link' in card.get('type'):
        return card.get('linkmarkers', []) or []
    return []

def pick_random(seq: List[Any]) -> Optional[Any]:
    return random.choice(seq) if seq else None


# =========================
# Deck metadata (with reachability metrics)
# =========================

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
        
        # Summon reachability metrics
        self.synchro_levels_reachable = 0
        self.xyz_level_pairs = 0
        self.link_material_depth = 0
        self.pendulum_scale_span = 0
        
        # Analyze the deck composition
        self._analyze_deck()
    
    def _analyze_deck(self):
        """Analyze deck composition and calculate metrics."""
        # Count card types
        self.monster_count = sum(1 for card in self.main_deck if is_monster(card))
        self.spell_count = sum(1 for card in self.main_deck if is_spell(card))
        self.trap_count = sum(1 for card in self.main_deck if is_trap(card))
        self.total_main = len(self.main_deck)
        self.total_extra = len(self.extra_deck)
        
        # Count extra deck types
        self.fusion_count = sum(1 for card in self.extra_deck if card.get('type') and 'Fusion' in card.get('type'))
        self.synchro_count = sum(1 for card in self.extra_deck if card.get('type') and 'Synchro' in card.get('type'))
        self.xyz_count = sum(1 for card in self.extra_deck if card.get('type') and 'XYZ' in card.get('type'))
        self.link_count = sum(1 for card in self.extra_deck if card.get('type') and 'Link' in card.get('type'))
        
        # Copy distribution
        self.copy_distribution = self._calculate_copy_distribution()
        
        # Archetypes
        all_cards = self.main_deck + self.extra_deck
        self.archetype_distribution = get_archetype_distribution(all_cards)
        self.archetypes = dict(sorted(self.archetype_distribution.items(), key=lambda x: x[1], reverse=True)[:3])
        self.dominant_archetype = get_dominant_archetype(self.archetype_distribution)
        
        # Flags & ratios
        self.has_tuners = any(is_tuner(card) for card in self.main_deck)
        self.has_pendulums = any(is_pendulum(card) for card in self.main_deck)
        
        self.monster_ratio = self.monster_count / self.total_main if self.total_main > 0 else 0
        self.spell_ratio = self.spell_count / self.total_main if self.total_main > 0 else 0
        self.trap_ratio = self.trap_count / self.total_main if self.total_main > 0 else 0
        
        # Reachability metrics (cheap, static approximations)
        self._compute_reachability_metrics()
        
        # Cluster-related metrics if available
        if self.clustered_cards:
            card_to_cluster = create_card_to_cluster_mapping(self.clustered_cards)
            self.cluster_distribution = get_cluster_distribution(all_cards, card_to_cluster)
            self.cluster_entropy = calculate_cluster_entropy(self.cluster_distribution)
            self.noise_card_percentage = calculate_noise_card_percentage(self.main_deck, self.clustered_cards, card_to_cluster)
            
            # crude but stable cluster embeddings from ids
            cluster_embeddings = {}
            for i, cluster_id in enumerate(self.cluster_distribution.keys()):
                cluster_num = int(cluster_id) if str(cluster_id).isdigit() else hash(cluster_id) % 10000
                cluster_embeddings[cluster_id] = [
                    math.sin(cluster_num * 0.1),
                    math.cos(cluster_num * 0.1),
                    math.sin(cluster_num * 0.2),
                    math.cos(cluster_num * 0.2)
                ]
            
            self.intra_deck_cluster_distance = calculate_intra_deck_cluster_distance(
                self.cluster_distribution, cluster_embeddings
            )
            
            # pseudo co-occurrence rarity (stable noise)
            global_cooccurrence = {}
            clusters = list(self.cluster_distribution.keys())
            for i, c1 in enumerate(clusters):
                for j, c2 in enumerate(clusters):
                    if i != j:
                        c1_val = int(c1) if str(c1).isdigit() else hash(c1) % 10000
                        c2_val = int(c2) if str(c2).isdigit() else hash(c2) % 10000
                        global_cooccurrence[(c1, c2)] = 0.1 + 0.8 * ((c1_val * c2_val) % 100) / 100
            
            self.cluster_co_occurrence_rarity = calculate_cluster_cooccurrence_rarity(
                self.cluster_distribution, global_cooccurrence
            )
    
    def _calculate_copy_distribution(self):
        """Calculate the distribution of card copies in the deck."""
        name_counts = {}
        for card in self.main_deck:
            name = card.get('name', '')
            if name:
                name_counts[name] = name_counts.get(name, 0) + 1
        counts = {'1_count': 0, '2_count': 0, '3_count': 0}
        for name, count in name_counts.items():
            if count == 1:
                counts['1_count'] += 1
            elif count == 2:
                counts['2_count'] += 1
            elif count >= 3:
                counts['3_count'] += 1
        return counts
    
    def _compute_reachability_metrics(self):
        """Static features to help your NN understand 'how live' each mechanic is."""
        monsters = [c for c in self.main_deck if is_monster(c)]
        tuners = [c for c in monsters if is_tuner(c)]
        non_tuners = [c for c in monsters if not is_tuner(c)]
        tuner_lvls = [get_monster_level(c) for c in tuners if get_monster_level(c) > 0]
        nt_lvls = [get_monster_level(c) for c in non_tuners if get_monster_level(c) > 0]
        # Synchro coverage: count distinct achievable sums that appear in Extra
        extra_syn_lvls = {get_monster_level(c) for c in self.extra_deck if c.get('type') and 'Synchro' in c.get('type')}
        reachable = set()
        for t in tuner_lvls:
            for n in nt_lvls:
                s = t + n
                if s in extra_syn_lvls:
                    reachable.add(s)
        self.synchro_levels_reachable = len(reachable)
        # XYZ: count levels with at least 2 copies that match some XYZ rank in Extra
        level_counts = Counter([get_monster_level(c) for c in monsters if get_monster_level(c) > 0])
        xyz_ranks = {get_monster_level(c) for c in self.extra_deck if c.get('type') and 'XYZ' in c.get('type')}
        self.xyz_level_pairs = sum(1 for lvl, cnt in level_counts.items() if cnt >= 2 and lvl in xyz_ranks)
        # Link: naive material depth proxy = min(total monsters // 2, highest link + 1)
        highest_link = 0
        for c in self.extra_deck:
            if c.get('type') and 'Link' in c.get('type'):
                highest_link = max(highest_link, get_monster_level(c))
        self.link_material_depth = min(len(monsters) // 2, highest_link + 1) if highest_link > 0 else 0
        # Pendulum scale span
        scales = [get_pendulum_scales(c) for c in self.main_deck if is_pendulum(c)]
        flat = [s for pair in scales for s in pair if s is not None]
        self.pendulum_scale_span = (max(flat) - min(flat)) if flat else 0
    
    # _identify_archetypes removed; using get_archetype_distribution
    
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
            
            # Cluster-related
            "cluster_entropy": self.cluster_entropy,
            "intra_deck_cluster_distance": self.intra_deck_cluster_distance,
            "cluster_co_occurrence_rarity": self.cluster_co_occurrence_rarity,
            "noise_card_percentage": self.noise_card_percentage,
            "dominant_archetype": self.dominant_archetype,
            "cluster_count": len(self.cluster_distribution) if self.cluster_distribution else 0,
            "archetype_count": len(self.archetype_distribution) if self.archetype_distribution else 0,

            # NEW: reachability
            "synchro_levels_reachable": self.synchro_levels_reachable,
            "xyz_level_pairs": self.xyz_level_pairs,
            "link_material_depth": self.link_material_depth,
            "pendulum_scale_span": self.pendulum_scale_span,
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
            
            # Cluster-related
            "cluster_entropy": self.cluster_entropy,
            "intra_deck_cluster_distance": self.intra_deck_cluster_distance,
            "cluster_co_occurrence_rarity": self.cluster_co_occurrence_rarity,
            "noise_card_percentage": self.noise_card_percentage,
            "cluster_distribution": self.cluster_distribution,
            "archetype_distribution": self.archetype_distribution,
            "dominant_archetype": self.dominant_archetype,

            # Reachability
            "synchro_levels_reachable": self.synchro_levels_reachable,
            "xyz_level_pairs": self.xyz_level_pairs,
            "link_material_depth": self.link_material_depth,
            "pendulum_scale_span": self.pendulum_scale_span,
        }


# =========================
# Deck generator
# =========================

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
    
    def __init__(self, clustering_model=None, card_data=None, num_cards=40,
                 # NEW toggles: choose how strict to be
                 hard_synchro: bool = True,
                 hard_xyz: bool = True,
                 hard_link: bool = True,
                 hard_pendulum: bool = False,
                 random_seed: Optional[int] = None):
        """Initialize the deck generator."""
        # Always set essential attributes (even for singleton pattern)
        self.num_cards = num_cards
        self.extra_deck_size = 15

        # Strictness toggles
        self.hard_synchro = hard_synchro
        self.hard_xyz = hard_xyz
        self.hard_link = hard_link
        self.hard_pendulum = hard_pendulum

        # Random seed support (logged to MLflow)
        self.random_seed = random_seed if random_seed is not None else random.randrange(10**9)
        random.seed(self.random_seed)

        # Ensure singleton pattern
        if self.__class__._instance is not None and self.__class__._instance.initialized:
            # Update essentials in existing instance
            inst = self.__class__._instance
            inst.num_cards = num_cards
            inst.hard_synchro = hard_synchro
            inst.hard_xyz = hard_xyz
            inst.hard_link = hard_link
            inst.hard_pendulum = hard_pendulum
            inst.random_seed = self.random_seed
            return
        
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
                if any(et in card_type for et in ['Fusion', 'Synchro', 'XYZ', 'Link']):
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

    # -----------------------------
    # Core constraint application
    # -----------------------------
    def apply_game_constraints(self, main_deck: List[Dict], extra_deck: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply Yu-Gi-Oh! game-based constraints to ensure realistic, playable decks.
        Enforces copy caps, adjusts copies, and (optionally) hard-enforces reachability
        by adding enablers or pruning unreachable Extra cards.
        """
        logger.info("Applying Yu-Gi-Oh! game-based constraints to deck")
        
        # 1) Global 3-copy cap (before we juggle things)
        main_deck = apply_max_copies_constraint(main_deck)
        
        # 2) Realistic copy distributions by type
        monsters = [card for card in main_deck if is_monster(card)]
        spells   = [card for card in main_deck if is_spell(card)]
        traps    = [card for card in main_deck if is_trap(card)]
        
        optimized_deck = []
        if len(monsters) >= 3:
            optimized_deck.extend(apply_copy_distribution_to_monsters(monsters, deck_size=len(monsters)))
        else:
            optimized_deck.extend(monsters)
        
        spells_traps = spells + traps
        if len(spells_traps) >= 3:
            optimized_deck.extend(apply_copy_distribution_to_spells_traps(spells_traps))
        else:
            optimized_deck.extend(spells_traps)
        main_deck = optimized_deck
        
        # 3) Summon-mechanic reachability (attempt to fix main; else prune extra if HARD)
        # Synchro
        sync_ok, main_deck = validate_synchro_requirements(main_deck, extra_deck)
        if not sync_ok and self.hard_synchro:
            extra_deck = prune_unreachable_extra(extra_deck, keep_types={'Fusion','XYZ','Link'})  # drop Synchros
            logger.info("HARD Synchro: pruned unreachable Synchros from Extra Deck.")
        # XYZ
        xyz_ok, main_deck = validate_xyz_requirements(main_deck, extra_deck)
        if not xyz_ok and self.hard_xyz:
            extra_deck = prune_unreachable_extra(extra_deck, keep_types={'Fusion','Synchro','Link'})  # drop XYZ
            logger.info("HARD XYZ: pruned unreachable XYZs from Extra Deck.")
        # Link
        link_ok, main_deck = validate_link_requirements(main_deck, extra_deck)
        if not link_ok and self.hard_link:
            extra_deck = prune_unreachable_extra(extra_deck, keep_types={'Fusion','Synchro','XYZ'})  # drop Links
            logger.info("HARD Link: pruned unreachable Links from Extra Deck.")
        # Pendulum (usually soft)
        pend_ok, main_deck = validate_pendulum_requirements(main_deck)
        if not pend_ok and self.hard_pendulum:
            # nothing to drop in Extra: pendulum is main mechanic; we already tried to fix main
            logger.info("HARD Pendulum: could not ensure scales; leaving as-is.")
        
        # 4) Enforce copy cap again after fixes
        main_deck = apply_max_copies_constraint(main_deck)

        # 5) Trim back to exactly 40, preferring to keep enablers
        main_deck = trim_back_to_40(main_deck)
        
        # Shuffle main to avoid positional bias
        random.shuffle(main_deck)

        # Keep Extra <= 15
        if len(extra_deck) > self.extra_deck_size:
            extra_deck = extra_deck[:self.extra_deck_size]
        
        return main_deck, extra_deck

    # -----------------------------
    # Extra deck generation
    # -----------------------------
    def _generate_extra_deck(self, target_archetype: Optional[str] = None) -> List[Dict]:
        """Generate extra deck cards. Always returns exactly 15 cards."""
        extra_deck = []
        target_size = self.extra_deck_size
        
        # Prefer archetype-aligned extra where possible
        if target_archetype:
            archetype_extra_clusters = []
            for cluster_id in self.extra_deck_clusters.keys():
                for card in self.extra_deck_clusters[cluster_id]:
                    if target_archetype in card.get('archetypes', []) or card.get('archetype') == target_archetype:
                        archetype_extra_clusters.append(cluster_id)
                        break
            if archetype_extra_clusters:
                selected_clusters = random.sample(
                    archetype_extra_clusters, 
                    min(3, len(archetype_extra_clusters))
                )
                cards_per_cluster = max(1, target_size // len(selected_clusters))
                extra_names = set()
                for cluster_id in selected_clusters:
                    if len(extra_deck) >= target_size:
                        break
                    cluster_cards = [c for c in self.extra_deck_clusters[cluster_id] if c.get('name','') not in extra_names]
                    to_add = min(cards_per_cluster, len(cluster_cards), target_size - len(extra_deck))
                    chosen = random.sample(cluster_cards, to_add) if to_add > 0 else []
                    extra_deck.extend(chosen)
                    extra_names.update(c.get('name','') for c in chosen)
        
        # Fill remaining with diverse clusters
        if len(extra_deck) < target_size and self.extra_deck_clusters:
            extra_names = {c.get('name','') for c in extra_deck}
            pool_ids = list(self.extra_deck_clusters.keys())
            selected_clusters = random.sample(pool_ids, min(3, len(pool_ids)))
            cards_per_cluster = max(1, (target_size - len(extra_deck)) // len(selected_clusters))
            for cluster_id in selected_clusters:
                if len(extra_deck) >= target_size:
                    break
                cluster_cards = [c for c in self.extra_deck_clusters[cluster_id] if c.get('name','') not in extra_names]
                to_add = min(cards_per_cluster, len(cluster_cards), target_size - len(extra_deck))
                chosen = random.sample(cluster_cards, to_add) if to_add > 0 else []
                extra_deck.extend(chosen)
                extra_names.update(c.get('name','') for c in chosen)
        
        # Final fill from all extra (dedupe names)
        if len(extra_deck) < target_size and self.extra_deck_clusters:
            all_extra = []
            for cards in self.extra_deck_clusters.values():
                all_extra.extend(cards)
            extra_names = {c.get('name','') for c in extra_deck}
            remaining = [c for c in all_extra if c.get('name','') not in extra_names]
            to_add = min(target_size - len(extra_deck), len(remaining))
            if to_add > 0:
                chosen = random.sample(remaining, to_add)
                extra_deck.extend(chosen)

        # Ensure exactly size and show type diversity stats
        extra_deck = extra_deck[:target_size]
        
        type_counts = Counter()
        for card in extra_deck:
            ct = card.get('type','')
            if 'Fusion' in ct: type_counts['Fusion'] += 1
            elif 'Synchro' in ct: type_counts['Synchro'] += 1
            elif 'XYZ' in ct: type_counts['XYZ'] += 1
            elif 'Link' in ct: type_counts['Link'] += 1
        
        logger.info(f"Generated Extra Deck with {len(extra_deck)} cards: "
                    f"{type_counts.get('Fusion', 0)} Fusion, "
                    f"{type_counts.get('Synchro', 0)} Synchro, "
                    f"{type_counts.get('XYZ', 0)} XYZ, "
                    f"{type_counts.get('Link', 0)} Link")
        
        return extra_deck

    # -----------------------------
    # Public generation API
    # -----------------------------
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
            setup_experiment()
            
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
                    game_constraint_enabled=True,  # constraints enabled
                    hard_synchro=self.hard_synchro,
                    hard_xyz=self.hard_xyz,
                    hard_link=self.hard_link,
                    hard_pendulum=self.hard_pendulum,
                    random_seed=self.random_seed
                )
                
                # Generate the deck
                main_deck, extra_deck = self._generate_deck_internal(mode, target_archetype)
                
                # Apply constraints
                main_deck, extra_deck = self.apply_game_constraints(main_deck, extra_deck)
                
                # Create metadata
                metadata = DeckMetadata(main_deck, extra_deck, self.clustered_cards)
                
                # Log metrics (now includes reachability)
                log_deck_metrics(main_deck, extra_deck, metadata)
                
                # Artifacts
                try:
                    log_deck_artifacts(main_deck, extra_deck, metadata)
                    logger.info("Successfully logged deck artifacts")
                except Exception as e:
                    logger.warning(f"Failed to log deck artifacts (likely S3 credentials issue): {e}")
                
                return main_deck, extra_deck, metadata
        else:
            main_deck, extra_deck = self._generate_deck_internal(mode, target_archetype)
            main_deck, extra_deck = self.apply_game_constraints(main_deck, extra_deck)
            metadata = DeckMetadata(main_deck, extra_deck, self.clustered_cards)
            return main_deck, extra_deck, metadata
    
    def _generate_deck_internal(self, mode: str, target_archetype: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
        """Internal deck generation logic without MLflow tracking."""
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
        low_monster_ratio = random.random() < 0.15  # 15% chance for low monster count deck
        
        if low_monster_ratio:
            base_monster = random.uniform(0.375, 0.5)
            logger.info("Generating low monster count deck variant")
        else:
            fluctuation = random.uniform(-0.15, 0.2)
            base_monster = 0.6 + fluctuation
        
        self.target_monster_ratio = max(0.375, min(0.8, base_monster))
        
        remaining = 1.0 - self.target_monster_ratio
        spell_trap_variation = random.uniform(-0.25, 0.25)
        spell_weight = max(0.25, min(0.75, 0.5 + spell_trap_variation))
        trap_weight = 1.0 - spell_weight
        
        self.target_spell_ratio = remaining * spell_weight
        self.target_trap_ratio = remaining * trap_weight
        
        self.ratio_tolerance = 0.25
        
        logger.info(f"Deck ratios randomized to: {self.target_monster_ratio:.1%}M/{self.target_spell_ratio:.1%}S/{self.target_trap_ratio:.1%}T (±{self.ratio_tolerance:.0%} tol)")
    
    def _calculate_target_counts(self, deck_size: int = 40) -> Tuple[int, int, int]:
        """Calculate target counts for each card type based on ratios."""
        target_monsters = int(deck_size * self.target_monster_ratio)
        target_spells = int(deck_size * self.target_spell_ratio)
        target_traps = int(deck_size * self.target_trap_ratio)
        
        total = target_monsters + target_spells + target_traps
        if total < deck_size:
            target_monsters += deck_size - total
        elif total > deck_size:
            target_monsters -= total - deck_size
        return target_monsters, target_spells, target_traps
    
    def _check_ratios(self, deck: List[Dict]) -> Dict[str, float]:
        """Check current ratios in a deck."""
        deck_size = len(deck)
        if deck_size == 0:
            return {'monster_ratio': 0.0, 'spell_ratio': 0.0, 'trap_ratio': 0.0}
        monster_count = sum(1 for card in deck if is_monster(card))
        spell_count = sum(1 for card in deck if is_spell(card))
        trap_count = sum(1 for card in deck if is_trap(card))
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
        
        main_deck = []
        # Monsters
        monsters_added = 0
        for cluster_id in archetype_monster_clusters:
            if monsters_added >= target_monsters:
                break
            archetype_cards = [
                card for card in self.monster_clusters[cluster_id]
                if target_archetype in card.get('archetypes', []) or card.get('archetype') == target_archetype
            ]
            to_add = min(len(archetype_cards), target_monsters - monsters_added)
            if to_add > 0:
                chosen = random.sample(archetype_cards, to_add)
                main_deck.extend(chosen)
                monsters_added += to_add
        remaining = target_monsters - monsters_added
        if remaining > 0 and archetype_monster_clusters:
            pool = []
            for cluster_id in archetype_monster_clusters:
                cluster_cards = self.monster_clusters[cluster_id]
                pool.extend([c for c in cluster_cards if c not in main_deck])
            if pool:
                chosen = random.sample(pool, min(remaining, len(pool)))
                main_deck.extend(chosen)
                monsters_added += len(chosen)
        if target_monsters - monsters_added > 0:
            pool = []
            for cluster_id in self.monster_clusters.keys():
                pool.extend([c for c in self.monster_clusters[cluster_id] if c not in main_deck])
            if pool:
                chosen = random.sample(pool, min(target_monsters - monsters_added, len(pool)))
                main_deck.extend(chosen)
                monsters_added += len(chosen)
        
        # Spells
        spells_added = 0
        for cluster_id in archetype_spell_clusters:
            if spells_added >= target_spells:
                break
            archetype_cards = [
                card for card in self.spell_clusters[cluster_id]
                if target_archetype in card.get('archetypes', []) or card.get('archetype') == target_archetype
            ]
            to_add = min(len(archetype_cards), target_spells - spells_added)
            if to_add > 0:
                chosen = random.sample(archetype_cards, to_add)
                main_deck.extend(chosen)
                spells_added += to_add
        remaining = target_spells - spells_added
        if remaining > 0 and archetype_spell_clusters:
            pool = []
            for cluster_id in archetype_spell_clusters:
                pool.extend([c for c in self.spell_clusters[cluster_id] if c not in main_deck])
            if pool:
                chosen = random.sample(pool, min(remaining, len(pool)))
                main_deck.extend(chosen)
                spells_added += len(chosen)
        if target_spells - spells_added > 0:
            pool = []
            for cluster_id in self.spell_clusters.keys():
                pool.extend([c for c in self.spell_clusters[cluster_id] if c not in main_deck])
            if pool:
                chosen = random.sample(pool, min(target_spells - spells_added, len(pool)))
                main_deck.extend(chosen)
                spells_added += len(chosen)
        
        # Traps
        traps_added = 0
        for cluster_id in archetype_trap_clusters:
            if traps_added >= target_traps:
                break
            archetype_cards = [
                card for card in self.trap_clusters[cluster_id]
                if target_archetype in card.get('archetypes', []) or card.get('archetype') == target_archetype
            ]
            to_add = min(len(archetype_cards), target_traps - traps_added)
            if to_add > 0:
                chosen = random.sample(archetype_cards, to_add)
                main_deck.extend(chosen)
                traps_added += to_add
        remaining = target_traps - traps_added
        if remaining > 0 and archetype_trap_clusters:
            pool = []
            for cluster_id in archetype_trap_clusters:
                pool.extend([c for c in self.trap_clusters[cluster_id] if c not in main_deck])
            if pool:
                chosen = random.sample(pool, min(remaining, len(pool)))
                main_deck.extend(chosen)
                traps_added += len(chosen)
        if target_traps - traps_added > 0:
            pool = []
            for cluster_id in self.trap_clusters.keys():
                pool.extend([c for c in self.trap_clusters[cluster_id] if c not in main_deck])
            if pool:
                chosen = random.sample(pool, min(target_traps - traps_added, len(pool)))
                main_deck.extend(chosen)
                traps_added += len(chosen)
        
        # Fill to 40 respecting ratios
        if len(main_deck) < self.num_cards:
            ratios = self._check_ratios(main_deck)
            while len(main_deck) < self.num_cards:
                if ratios['monster_ratio'] < self.target_monster_ratio - 0.05 and self.monster_clusters:
                    pool = [c for cards in self.monster_clusters.values() for c in cards if c not in main_deck]
                    if pool: main_deck.append(random.choice(pool))
                elif ratios['spell_ratio'] < self.target_spell_ratio - 0.05 and self.spell_clusters:
                    pool = [c for cards in self.spell_clusters.values() for c in cards if c not in main_deck]
                    if pool: main_deck.append(random.choice(pool))
                elif ratios['trap_ratio'] < self.target_trap_ratio - 0.05 and self.trap_clusters:
                    pool = [c for cards in self.trap_clusters.values() for c in cards if c not in main_deck]
                    if pool: main_deck.append(random.choice(pool))
                else:
                    pool = [c for cards in self.main_deck_clusters.values() for c in cards if c not in main_deck]
                    if pool: main_deck.append(random.choice(pool))
                    else: break
                ratios = self._check_ratios(main_deck)
        
        main_deck = main_deck[:self.num_cards]
        extra_deck = self._generate_extra_deck(target_archetype)
        
        final_ratios = self._check_ratios(main_deck)
        logger.info(f"Generated meta-aware deck ratios: "
                    f"M {final_ratios['monster_ratio']:.1%} / "
                    f"S {final_ratios['spell_ratio']:.1%} / "
                    f"T {final_ratios['trap_ratio']:.1%}")
        return main_deck, extra_deck
    
    def _generate_novel_deck(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate a novel deck by combining diverse clusters with proper card type ratios."""
        logger.info("Generating novel deck with diverse cluster combination and proper ratios")
        
        target_monsters, target_spells, target_traps = self._calculate_target_counts(self.num_cards)
        main_deck = []
        
        monster_cluster_ids = list(self.monster_clusters.keys())
        spell_cluster_ids = list(self.spell_clusters.keys())
        trap_cluster_ids = list(self.trap_clusters.keys())
        
        # Monsters
        if monster_cluster_ids:
            selected = random.sample(monster_cluster_ids, min(3, len(monster_cluster_ids)))
            per = max(1, target_monsters // len(selected))
            added = 0
            for cid in selected:
                if added >= target_monsters: break
                cards = self.monster_clusters[cid]
                to_add = min(per, len(cards), target_monsters - added)
                if to_add > 0:
                    chosen = random.sample(cards, to_add)
                    main_deck.extend(chosen)
                    added += to_add
            while added < target_monsters:
                pool = [c for cards in self.monster_clusters.values() for c in cards if c not in main_deck]
                if not pool: break
                extra = random.sample(pool, min(target_monsters - added, len(pool)))
                main_deck.extend(extra)
                added += len(extra)
        
        # Spells
        if spell_cluster_ids:
            selected = random.sample(spell_cluster_ids, min(2, len(spell_cluster_ids)))
            per = max(1, target_spells // len(selected))
            added = 0
            for cid in selected:
                if added >= target_spells: break
                cards = self.spell_clusters[cid]
                to_add = min(per, len(cards), target_spells - added)
                if to_add > 0:
                    chosen = random.sample(cards, to_add)
                    main_deck.extend(chosen)
                    added += to_add
            while added < target_spells:
                pool = [c for cards in self.spell_clusters.values() for c in cards if c not in main_deck]
                if not pool: break
                extra = random.sample(pool, min(target_spells - added, len(pool)))
                main_deck.extend(extra)
                added += len(extra)
        
        # Traps
        if trap_cluster_ids:
            selected = random.sample(trap_cluster_ids, min(2, len(trap_cluster_ids)))
            per = max(1, target_traps // len(selected))
            added = 0
            for cid in selected:
                if added >= target_traps: break
                cards = self.trap_clusters[cid]
                to_add = min(per, len(cards), target_traps - added)
                if to_add > 0:
                    chosen = random.sample(cards, to_add)
                    main_deck.extend(chosen)
                    added += to_add
            while added < target_traps:
                pool = [c for cards in self.trap_clusters.values() for c in cards if c not in main_deck]
                if not pool: break
                extra = random.sample(pool, min(target_traps - added, len(pool)))
                main_deck.extend(extra)
                added += len(extra)
        
        # Fill to 40
        while len(main_deck) < self.num_cards:
            ratios = self._check_ratios(main_deck)
            if ratios['monster_ratio'] < self.target_monster_ratio - 0.05 and self.monster_clusters:
                pool = [c for cards in self.monster_clusters.values() for c in cards if c not in main_deck]
                if pool: main_deck.append(random.choice(pool))
            elif ratios['spell_ratio'] < self.target_spell_ratio - 0.05 and self.spell_clusters:
                pool = [c for cards in self.spell_clusters.values() for c in cards if c not in main_deck]
                if pool: main_deck.append(random.choice(pool))
            elif ratios['trap_ratio'] < self.target_trap_ratio - 0.05 and self.trap_clusters:
                pool = [c for cards in self.trap_clusters.values() for c in cards if c not in main_deck]
                if pool: main_deck.append(random.choice(pool))
            else:
                pool = [c for cards in self.main_deck_clusters.values() for c in cards if c not in main_deck]
                if pool: main_deck.append(random.choice(pool))
                else: break
        
        main_deck = main_deck[:self.num_cards]
        extra_deck = self._generate_extra_deck()
        
        final_ratios = self._check_ratios(main_deck)
        logger.info(f"Generated novel deck ratios: "
                    f"M {final_ratios['monster_ratio']:.1%} / "
                    f"S {final_ratios['spell_ratio']:.1%} / "
                    f"T {final_ratios['trap_ratio']:.1%}")
        return main_deck, extra_deck


# =========================
# Copy distribution helpers
# =========================

def apply_copy_distribution_to_monsters(cards: List[Dict], deck_size: int = 40) -> List[Dict]:
    """
    Apply a realistic distribution of card copies for monster cards using a refined greedy strategy.
    """
    card_groups = defaultdict(list)
    for card in cards:
        name = card.get('name', '')
        if name:
            card_groups[name].append(card)
    num_monsters = len(cards)
    if num_monsters == 0:
        return []
    monster_names = list(card_groups.keys())
    random.shuffle(monster_names)
    result = []
    copy_assignments = {}
    total_added = 0
    copy_weights = {3: 50, 2: 40, 1: 10}
    while total_added < num_monsters and monster_names:
        remaining = num_monsters - total_added
        if remaining <= 3:
            possible = [c for c in [1,2,3] if c <= remaining]
            if not possible: break
            copy_count = max(possible)
        else:
            weights = [copy_weights[c] for c in [1,2,3]]
            copy_count = random.choices([1,2,3], weights=weights, k=1)[0]
        if not monster_names:
            break
        name = monster_names.pop(0)
        if remaining >= 3 and random.random() < 0.6:
            actual = 3
        else:
            actual = min(copy_count, remaining)
        copy_assignments[name] = actual
        total_added += actual
        if total_added >= num_monsters:
            break
    for name, cnt in copy_assignments.items():
        variants = card_groups[name]
        if len(variants) >= cnt:
            result.extend(variants[:cnt])
        else:
            result.extend(variants * cnt)
    if total_added < num_monsters and result:
        name_counts = Counter(card.get('name','') for card in result)
        candidates = [n for n, c in name_counts.items() if 1 <= c <= 2]
        while total_added < num_monsters and candidates:
            name = random.choice(candidates)
            cur = name_counts[name]
            if cur < 3:
                if name in card_groups and card_groups[name]:
                    result.append(card_groups[name][0])
                    total_added += 1
                    name_counts[name] += 1
                    if name_counts[name] >= 3:
                        candidates.remove(name)
            else:
                candidates.remove(name)
    final_counts = Counter()
    for name in set(card.get('name','') for card in result):
        copies = sum(1 for card in result if card.get('name','') == name)
        final_counts[copies] += 1
    total_monsters = sum(count * copies for copies, count in final_counts.items())
    logger.info(f"Monster copies: 3-of {final_counts.get(3,0)}, 2-of {final_counts.get(2,0)}, 1-of {final_counts.get(1,0)} (total {total_monsters})")
    return result

def apply_copy_distribution_to_spells_traps(cards: List[Dict]) -> List[Dict]:
    """
    Apply a realistic distribution of card copies for spell and trap cards.
    """
    card_groups = defaultdict(list)
    for card in cards:
        name = card.get('name','')
        if name:
            card_groups[name].append(card)
    unique = len(card_groups)
    if unique == 0:
        return []
    pct_3, pct_2, pct_1 = 0.25, 0.25, 0.50
    target_3 = max(1, int(unique * pct_3))
    target_2 = max(1, int(unique * pct_2))
    target_1 = unique - target_3 - target_2
    if unique <= 3:
        target_3, target_2, target_1 = unique, 0, 0
    elif unique <= 6:
        target_3, target_2, target_1 = unique // 2, unique - (unique // 2), 0
    names = list(card_groups.keys())
    random.shuffle(names)
    assign = {}
    for i in range(target_3):
        if i < len(names):
            assign[names[i]] = 3
    for i in range(target_3, target_3 + target_2):
        if i < len(names):
            assign[names[i]] = 2
    for i in range(target_3 + target_2, len(names)):
        assign[names[i]] = 1
    result = []
    for name, cnt in assign.items():
        orig = card_groups[name]
        if len(orig) >= cnt:
            result.extend(orig[:cnt])
        else:
            result.extend(orig * cnt)
    final_counts = Counter()
    for name in set(card.get('name','') for card in result):
        copies = sum(1 for card in result if card.get('name','') == name)
        final_counts[copies] += 1
    total_st = sum(count * copies for copies, count in final_counts.items())
    logger.info(f"Spell/Trap copies: 3-of {final_counts.get(3,0)}, 2-of {final_counts.get(2,0)}, 1-of {final_counts.get(1,0)} (total {total_st})")
    return result


# =========================
# Reachability validators
# =========================

def validate_synchro_requirements(main_deck: List[Dict], extra_deck: List[Dict]) -> Tuple[bool, List[Dict]]:
    """
    Validate and (soft-)fix Synchro summoning requirements.
    """
    synchro_monsters = [card for card in extra_deck if card.get('type') and 'Synchro' in card.get('type')]
    if not synchro_monsters:
        return True, main_deck
    tuners = [card for card in main_deck if is_tuner(card)]
    if tuners:
        tuner_levels = [get_monster_level(c) for c in tuners]
        non_tuners = [c for c in main_deck if is_monster(c) and not is_tuner(c)]
        nt_levels = [get_monster_level(c) for c in non_tuners]
        syn_levels = [get_monster_level(c) for c in synchro_monsters]
        for t in tuner_levels:
            for n in nt_levels:
                if (t + n) in syn_levels:
                    logger.info("Synchro levels reachable.")
                    return True, main_deck
        logger.info("Synchro levels not optimal - treating as soft if tuners exist.")
        return True, main_deck
    else:
        logger.info("Missing Tuner monsters - adding one.")
        deck_gen = DeckGenerator.get_instance()
        potential = []
        for cards in deck_gen.monster_clusters.values():
            for card in cards:
                if is_tuner(card):
                    potential.append(card)
        if potential:
            main_deck.append(random.choice(potential))
            return True, main_deck
        else:
            logger.warning("No Tuners found in database.")
            return False, main_deck

def validate_xyz_requirements(main_deck: List[Dict], extra_deck: List[Dict]) -> Tuple[bool, List[Dict]]:
    """
    Validate and (soft-)fix XYZ summoning requirements.
    """
    xyz_monsters = [card for card in extra_deck if card.get('type') and 'XYZ' in card.get('type')]
    if not xyz_monsters:
        return True, main_deck
    monsters = [card for card in main_deck if is_monster(card)]
    level_counts = Counter(get_monster_level(c) for c in monsters)
    has_same_level = any(cnt >= 2 for lvl, cnt in level_counts.items() if lvl > 0)
    if has_same_level:
        logger.info("XYZ requirement met: have at least one level pair.")
        xyz_ranks = set(get_monster_level(c) for c in xyz_monsters)
        if any(lvl in xyz_ranks and cnt >= 2 for lvl, cnt in level_counts.items()):
            logger.info("XYZ rank match present.")
        else:
            logger.info("XYZ ranks not matched exactly (soft).")
        return True, main_deck
    else:
        logger.info("Adding two monsters of common XYZ level to enable XYZ lines.")
        deck_gen = DeckGenerator.get_instance()
        xyz_ranks = [get_monster_level(c) for c in xyz_monsters]
        most_common_rank = Counter(xyz_ranks).most_common(1)[0][0] if xyz_ranks else 4
        pool = []
        for cards in deck_gen.monster_clusters.values():
            for card in cards:
                if get_monster_level(card) == most_common_rank:
                    pool.append(card)
        if pool:
            pick = random.sample(pool, min(2, len(pool)))
            main_deck.extend(pick)
            return True, main_deck
        else:
            logger.warning(f"No Level {most_common_rank} monsters in database.")
            return False, main_deck

def validate_link_requirements(main_deck: List[Dict], extra_deck: List[Dict]) -> Tuple[bool, List[Dict]]:
    """
    Validate and (soft-)fix Link summoning requirements.
    """
    link_monsters = [card for card in extra_deck if card.get('type') and 'Link' in card.get('type')]
    if not link_monsters:
        return True, main_deck
    monsters = [card for card in main_deck if is_monster(card)]
    highest_link = max((get_monster_level(c) for c in link_monsters), default=1)
    min_needed = min(highest_link + 1, 3)  # heuristic
    if len(monsters) >= min_needed:
        logger.info(f"Link requirement met: have {len(monsters)} monsters (need {min_needed}).")
        return True, main_deck
    else:
        logger.info(f"Adding {min_needed - len(monsters)} more monsters for Link materials.")
        deck_gen = DeckGenerator.get_instance()
        pool = []
        for cards in deck_gen.monster_clusters.values():
            pool.extend(cards)
        if pool:
            random.shuffle(pool)
            need = min_needed - len(monsters)
            main_deck.extend(pool[:need])
            return True, main_deck
        else:
            logger.warning("No monsters available to add for Link enablement.")
            return False, main_deck

def validate_pendulum_requirements(main_deck: List[Dict]) -> Tuple[bool, List[Dict]]:
    """
    Validate and (soft-)fix Pendulum summoning requirements.
    """
    pendulum_monsters = [card for card in main_deck if is_pendulum(card)]
    if not pendulum_monsters:
        return True, main_deck
    if len(pendulum_monsters) >= 2:
        scales = set()
        for c in pendulum_monsters:
            l, r = get_pendulum_scales(c)
            scales.add(l); scales.add(r)
        if len(scales) >= 2:
            logger.info("Pendulum requirement met (two scales).")
            return True, main_deck
        else:
            logger.info("Pendulum scales identical (soft).")
            return True, main_deck
    else:
        logger.info("Adding another Pendulum monster to establish scales.")
        deck_gen = DeckGenerator.get_instance()
        existing_scales = set(s for p in pendulum_monsters for s in get_pendulum_scales(p))
        pool = []
        for cards in deck_gen.monster_clusters.values():
            for card in cards:
                if is_pendulum(card) and card not in pendulum_monsters:
                    l, r = get_pendulum_scales(card)
                    if l not in existing_scales or r not in existing_scales:
                        pool.append(card)
        if not pool:
            for cards in deck_gen.monster_clusters.values():
                for card in cards:
                    if is_pendulum(card) and card not in pendulum_monsters:
                        pool.append(card)
        if pool:
            main_deck.append(random.choice(pool))
            return True, main_deck
        else:
            logger.warning("No additional Pendulum monsters found.")
            return True, main_deck  # treat as soft


# =========================
# Utility: enforce copy cap & trimming
# =========================

def apply_max_copies_constraint(deck: List[Dict]) -> List[Dict]:
    """
    Ensure no card appears more than 3 times in the deck (Yu-Gi-Oh! core rule).
    """
    card_counts = Counter(card.get('name', '') for card in deck)
    excess = {name: count for name, count in card_counts.items() if count > 3}
    if not excess:
        return deck
    new_deck = []
    current = defaultdict(int)
    for card in deck:
        name = card.get('name','')
        if current[name] < 3:
            new_deck.append(card)
            current[name] += 1
    removed = len(deck) - len(new_deck)
    if removed > 0:
        logger.info(f"Removed {removed} excess copies to enforce 3-copy limit.")
    return new_deck

def trim_back_to_40(main_deck: List[Dict]) -> List[Dict]:
    """
    Trim to exactly 40 cards, preferring to keep key enablers (tuners, pendulums, monsters).
    Heuristic: drop traps first, then non-archetype spells, then excess monsters (non-tuners).
    """
    if len(main_deck) <= 40:
        return main_deck
    # Partition by rough importance
    tuners = [c for c in main_deck if is_tuner(c)]
    pends  = [c for c in main_deck if is_pendulum(c)]
    monsters = [c for c in main_deck if is_monster(c) and not is_tuner(c) and not is_pendulum(c)]
    spells = [c for c in main_deck if is_spell(c)]
    traps  = [c for c in main_deck if is_trap(c)]
    # Removal order: traps -> spells -> monsters (non-tuner) -> pendulums -> tuners (last)
    buckets = [traps, spells, monsters, pends, tuners]
    keep = []
    for b in buckets[::-1]:  # start from most important; build keep list
        keep.extend(b)
    # Now drop from the front of the least important (traps first)
    target = 40
    if len(main_deck) <= target:
        return main_deck
    # Rebuild in preferred order (tuners/pends/monsters/spells/traps)
    preferred = tuners + pends + monsters + spells + traps
    if len(preferred) <= target:
        return preferred
    # remove from the tail (least important end)
    trimmed = preferred[:target]
    return apply_max_copies_constraint(trimmed)


def prune_unreachable_extra(extra_deck: List[Dict], keep_types: Set[str]) -> List[Dict]:
    """
    Remove Extra Deck cards whose type is NOT in keep_types.
    keep_types is a set of strings like {'Fusion','XYZ','Link'}
    """
    out = []
    for c in extra_deck:
        t = c.get('type','')
        keep = False
        for kt in keep_types:
            if kt in t:
                keep = True
                break
        if keep:
            out.append(c)
    return out[:15]


# =========================
# Define alias for backwards compatibility
# =========================

SimpleDeckGenerator = DeckGenerator
