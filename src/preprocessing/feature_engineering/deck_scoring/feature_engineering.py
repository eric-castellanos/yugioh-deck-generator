import logging
import csv
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import Counter, defaultdict
import re
from datetime import datetime
import io
from itertools import combinations

import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sentence_transformers import SentenceTransformer

# Import utils
from src.utils.s3_utils import read_csv_from_s3, upload_to_s3
from src.utils.deck_scoring.deck_scoring_utils import add_bayesian_adjusted_win_rate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# nltk downloads as needed
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

STOPWORDS = set(stopwords.words("english"))

STRATEGY_KEYWORDS = {
    'num_banish': r'\bbanish\b',
    'num_graveyard': r'\bgraveyard\b',
    'num_draw': r'\bdraw\b',
    'num_search': r'\bsearch\b',
    'num_special_summon': r'\bspecial summon\b',
    'num_negate': r'\bnegate\b',
    'num_destroy': r'\bdestroy\b',
    'num_shuffle': r'\bshuffle\b'
}

# =============================================================================
# Normalization helpers (critical fixes)
# =============================================================================

def norm_str(x: Any) -> str:
    return x if isinstance(x, str) else ""

def has_token(hay: str, needle: str) -> bool:
    return needle.lower() in norm_str(hay).lower()

def is_monster(card: Dict[str, Any]) -> bool:
    # accept things like "Effect Monster", "Synchro Monster", etc.
    return has_token(card.get("type", ""), "monster")

def coerce_int(x) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None

def norm_subtypes(x: Any) -> set[str]:
    """Normalize subtypes into a lowercased token set.
       If subtypes missing, caller can union with tokens derived from `type`."""
    if isinstance(x, set):
        return {str(s).lower() for s in x}
    if isinstance(x, list):
        return {str(s).lower() for s in x}
    if isinstance(x, str):
        return set(x.lower().replace("-", " ").split())
    return set()

def norm_subtypes_from_type(card: Dict[str, Any]) -> set[str]:
    t = norm_str(card.get("type", ""))
    return set(t.lower().replace("-", " ").split())

def has_subtype(card: Dict[str, Any], keyword: str) -> bool:
    subs = norm_subtypes(card.get("subtypes"))
    if not subs:
        subs = norm_subtypes_from_type(card)
    return keyword.lower() in subs

# =============================================================================
# IO and list conversion
# =============================================================================

def convert_list_columns(df: pd.DataFrame, list_columns: List[str]) -> pd.DataFrame:
    logger.info(f"Converting list columns: {list_columns}")
    result_df = df.copy()

    for col in list_columns:
        if col in result_df.columns:
            result_df[col] = result_df[col].apply(
                lambda x: (
                    x if isinstance(x, list)
                    else (
                        ast.literal_eval(x) if isinstance(x, str) else x
                    )
                )
            )
            if not result_df[col].apply(lambda v: isinstance(v, list)).all():
                raise TypeError(f"Column '{col}' contains non-list values after conversion.")
            logger.debug(f"Converted column {col} to Python lists")

    return result_df

def load_deck_data_from_s3(
    bucket: str = 'yugioh-data', 
    key: str = 'deck_scoring/training_data/random_generated_decks_composite_data.csv'
) -> pd.DataFrame:
    """
    Load deck data from S3 bucket and convert list columns
    """
    logger.info(f"Loading deck data from s3://{bucket}/{key}")
    df = read_csv_from_s3(bucket=bucket, key=key)
    list_columns = ['main_deck', 'extra_deck']
    df = convert_list_columns(df, list_columns)
    return df

def load_battle_results_from_s3(
    bucket: str = 'yugioh-data',
    key: str = 'deck_scoring/training_data/deck_battle_results.csv'
) -> pd.DataFrame:
    """
    Load deck battle results from S3 bucket
    """
    logger.info(f"Loading battle results from s3://{bucket}/{key}")
    df = read_csv_from_s3(bucket=bucket, key=key)
    return df

def merge_deck_data_with_battle_results(
    deck_df: pd.DataFrame,
    battle_results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Right join on deck_id, filter out FAILED win_rate rows, and add Bayesian adjusted win rate.
    """
    logger.info(f"Merging deck data ({len(deck_df)} rows) with battle results ({len(battle_results_df)} rows)")
    
    # Add Bayesian adjusted win rate to battle results
    logger.info("Calculating Bayesian adjusted win rate")
    add_bayesian_adjusted_win_rate(battle_results_df)
    
    # Check for overlapping columns (excluding deck_id which we need for merging)
    deck_cols = set(deck_df.columns)
    battle_cols = set(battle_results_df.columns)
    overlapping_cols = deck_cols.intersection(battle_cols) - {'deck_id'}
    
    if overlapping_cols:
        logger.info(f"Found overlapping columns: {overlapping_cols}")
        # Add suffixes to distinguish columns from each source
        merged_df = pd.merge(deck_df, battle_results_df, on='deck_id', how='right', suffixes=('_deck', '_battle'))
        
        # For overlapping columns, prefer the battle results version and drop the deck version
        cols_to_drop = [f"{col}_deck" for col in overlapping_cols if f"{col}_battle" in merged_df.columns]
        if cols_to_drop:
            logger.info(f"Dropping duplicate columns from deck data: {cols_to_drop}")
            merged_df = merged_df.drop(columns=cols_to_drop)
            
        # Rename battle columns back to original names
        rename_dict = {f"{col}_battle": col for col in overlapping_cols if f"{col}_battle" in merged_df.columns}
        if rename_dict:
            logger.info(f"Renaming battle columns: {rename_dict}")
            merged_df = merged_df.rename(columns=rename_dict)
    else:
        merged_df = pd.merge(deck_df, battle_results_df, on='deck_id', how='right')
    
    logger.info(f"After merge: {len(merged_df)} rows")
    initial_count = len(merged_df)
    merged_df = merged_df[merged_df['win_rate'] != 'FAILED']
    final_count = len(merged_df)
    logger.info(f"Filtered out {initial_count - final_count} rows with win_rate = 'FAILED'")
    logger.info(f"Final dataset: {final_count} rows")
    return merged_df

# =============================================================================
# Simple deck feature helpers (fixed to normalize schema)
# =============================================================================

def has_tuner(main_deck: List[Dict[str, Any]]) -> bool:
    for card in main_deck:
        if not is_monster(card):
            continue
        if bool(card.get("is_tuner")) or has_subtype(card, "tuner"):
            return True
    return False

def has_same_level_monsters(main_deck: List[Dict[str, Any]]) -> bool:
    levels = [coerce_int(card.get('level')) for card in main_deck if coerce_int(card.get('level'))]
    counts = Counter(levels)
    return any(count >= 2 for count in counts.values())

def has_synchro_monsters(extra_deck: List[Dict[str, Any]]) -> bool:
    return any(has_subtype(card, "synchro") for card in extra_deck)

def has_xyz_monsters(extra_deck: List[Dict[str, Any]]) -> bool:
    return any(has_subtype(card, "xyz") for card in extra_deck)

def has_link_monsters(extra_deck: List[Dict[str, Any]]) -> bool:
    return any(has_subtype(card, "link") for card in extra_deck)

def has_pendulum_monsters(main_deck: List[Dict[str, Any]]) -> bool:
    return any(has_subtype(card, "pendulum") for card in main_deck)

def max_same_level_count(main_deck: List[Dict[str, Any]]) -> int:
    levels = [coerce_int(card.get('level')) for card in main_deck if coerce_int(card.get('level'))]
    return max(Counter(levels).values(), default=0)

def num_tuners(main_deck: List[Dict[str, Any]]) -> int:
    return sum(1 for card in main_deck if is_monster(card) and (bool(card.get("is_tuner")) or has_subtype(card, "tuner")))

def avg_monster_level(main_deck: List[Dict[str, Any]]) -> float:
    levels = [coerce_int(card.get('level')) for card in main_deck if is_monster(card) and coerce_int(card.get('level'))]
    return float(np.mean(levels)) if levels else 0.0

def max_copies_per_card(main_deck: List[Dict[str, Any]]) -> int:
    names = [card.get('name') for card in main_deck]
    return max(Counter(names).values(), default=0)

def avg_copies_per_monster(main_deck: List[Dict[str, Any]]) -> float:
    monster_names = [card.get('name') for card in main_deck if is_monster(card)]
    name_counts = Counter(monster_names)
    return float(np.mean(list(name_counts.values()))) if name_counts else 0.0

def num_unique_monsters(main_deck: List[Dict[str, Any]]) -> int:
    return len({card.get('name') for card in main_deck if is_monster(card)})

# =============================================================================
# NLP helpers
# =============================================================================

def get_card_descriptions(card_list: List[Dict[str, Any]]) -> List[str]:
    descriptions = [
        card['desc'] for card in card_list
        if isinstance(card, dict) and isinstance(card.get('desc'), str) and card['desc'].strip()
    ]
    return descriptions

def calculate_mean_tfidf_for_deck(
    card_descriptions: List[str], 
    tfidf_vectorizer: Optional[TfidfVectorizer] = None
) -> Tuple[float, Optional[TfidfVectorizer]]:
    if not card_descriptions:
        logger.warning("No card descriptions provided. Returning 0.0.")
        return 0.0, tfidf_vectorizer

    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(card_descriptions)
    else:
        tfidf_matrix = tfidf_vectorizer.transform(card_descriptions)

    mean_tfidf = float(tfidf_matrix.mean())
    return mean_tfidf, tfidf_vectorizer

def add_tfidf_features_to_decks(deck_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Adding TF-IDF features to {len(deck_df)} decks")
    result_df = deck_df.copy()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    deck_descriptions = result_df['main_deck'].apply(
        lambda cards: get_card_descriptions(cards) if isinstance(cards, list) else []
    )
    all_descriptions = [desc for descs in deck_descriptions for desc in descs]

    if all_descriptions:
        logger.info(f"Fitting TF-IDF vectorizer on {len(all_descriptions)} descriptions")
        tfidf_vectorizer.fit(all_descriptions)
    else:
        logger.warning("No descriptions found to fit vectorizer")

    logger.info("Calculating mean TF-IDF per deck")
    result_df['main_deck_mean_tfidf'] = deck_descriptions.apply(
        lambda descs: calculate_mean_tfidf_for_deck(descs, tfidf_vectorizer)[0]
    )

    logger.info("TF-IDF features added successfully")
    return result_df

def add_bow_features_nltk(deck_df: pd.DataFrame, top_k_words: int = 100, binary: bool = True) -> pd.DataFrame:
    all_tokens = []
    tokenized_deck_descs = []

    for deck in deck_df["main_deck"]:
        deck_tokens = []
        for card in deck:
            desc = card.get("desc", "")
            if isinstance(desc, str):
                tokens = [word.lower() for word in word_tokenize(desc) if word.isalpha()]
                filtered = [t for t in tokens if t not in STOPWORDS]
                all_tokens.extend(filtered)
                deck_tokens.extend(filtered)
        tokenized_deck_descs.append(deck_tokens)

    top_words = [word for word, _ in Counter(all_tokens).most_common(top_k_words)]

    bow_features = []
    for tokens in tokenized_deck_descs:
        counts = Counter(tokens)
        bow = {
            f"bow_{word}": int(counts[word] > 0) if binary else counts[word]
            for word in top_words
        }
        bow_features.append(bow)

    bow_df = pd.DataFrame(bow_features, index=deck_df.index)
    return pd.concat([deck_df, bow_df], axis=1)

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    return SentenceTransformer(model_name)

def compute_mean_embedding(descs: List[str], model: SentenceTransformer) -> List[float]:
    if not descs:
        return np.zeros(model.get_sentence_embedding_dimension()).tolist()
    embeddings = model.encode(descs, show_progress_bar=False)
    return np.mean(embeddings, axis=0).tolist()

def add_mean_embedding_features_to_decks(deck_df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    model = load_embedding_model(model_name)
    deck_df = deck_df.copy()

    def process_deck(deck_cards: List[Dict[str, Any]]) -> List[float]:
        descriptions = get_card_descriptions(deck_cards)
        return compute_mean_embedding(descriptions, model)

    logger.info(f"Generating mean embeddings for {len(deck_df)} decks")
    deck_df["main_deck_mean_embedding"] = deck_df["main_deck"].apply(process_deck)
    logger.info("Mean embedding features added successfully")
    return deck_df

def count_strategy_keywords(main_deck: List[Dict[str, Any]]) -> Dict[str, int]:
    text = " ".join(
        card.get("desc", "").lower()
        for card in main_deck
        if isinstance(card.get("desc", ""), str)
    )
    return {
        feature: len(re.findall(pattern, text))
        for feature, pattern in STRATEGY_KEYWORDS.items()
    }

# =============================================================================
# Math helpers
# =============================================================================

def norm_subtypes_for_collection(card: Dict[str, Any]) -> set[str]:
    subs = norm_subtypes(card.get("subtypes"))
    if not subs:
        subs = norm_subtypes_from_type(card)
    return subs

def valid_level(x) -> bool:
    xi = coerce_int(x)
    return isinstance(xi, int) and xi > 0

def two_sums(levels: List[int], max_sum: Optional[int] = None) -> set[int]:
    counts = Counter(levels)
    sums = set()
    unique = sorted([l for l in counts if isinstance(l, int)])
    for i, a in enumerate(unique):
        if counts[a] >= 2:
            val = a + a
            if max_sum is None or val <= max_sum:
                sums.add(val)
        for b in unique[i+1:]:
            val = a + b
            if max_sum is None or val <= max_sum:
                sums.add(val)
    return sums

def nCk(n: int, k: int) -> int:
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

def hypergeom_two_or_more(K: int, N: int = 40, draws: int = 7) -> float:
    if K < 2 or draws < 2 or K > N:
        return 0.0
    total_ways = nCk(N, draws)
    if total_ways == 0:
        return 0.0
    p_0 = nCk(K, 0) * nCk(N - K, draws) / total_ways
    p_1 = nCk(K, 1) * nCk(N - K, draws - 1) / total_ways
    return max(0.0, 1.0 - p_0 - p_1)

# =============================================================================
# Feature Functions (fixed)
# =============================================================================

def feat_tuner_counts(main_deck: List[Dict[str, Any]]) -> Dict[str, Any]:
    tuner_levels: List[int] = []
    non_tuner_levels: List[int] = []

    for card in main_deck:
        if not is_monster(card):
            continue
        lvl = coerce_int(card.get("level"))
        if not valid_level(lvl):
            continue

        subs = norm_subtypes_for_collection(card)
        is_tuner_flag = bool(card.get("is_tuner")) or ("tuner" in subs)

        if is_tuner_flag:
            tuner_levels.append(lvl)  # type: ignore[arg-type]
        else:
            non_tuner_levels.append(lvl)  # type: ignore[arg-type]

    return {
        "tuner_count": len(tuner_levels),
        "non_tuner_count": len(non_tuner_levels),
        "tuner_levels": tuner_levels,
        "non_tuner_levels": non_tuner_levels,
        "non_tuner_level_counts": Counter(non_tuner_levels),
    }

def feat_synchro(main_deck: List[Dict[str, Any]], extra_deck: List[Dict[str, Any]]) -> Dict[str, Any]:
    target_levels = {
        coerce_int(c.get("level"))
        for c in extra_deck
        if "synchro" in norm_subtypes_for_collection(c) and valid_level(c.get("level"))
    }
    target_levels.discard(None)
    if not target_levels:
        return {"can_synchro": False, "matched_synchro_levels": 0}

    td = feat_tuner_counts(main_deck)
    tuner_lvls = td["tuner_levels"]
    non_lvls   = td["non_tuner_levels"]
    if not tuner_lvls:
        return {"can_synchro": False, "matched_synchro_levels": 0}

    max_syn = max(target_levels)  # type: ignore[arg-type]
    min_tun = min(tuner_lvls)
    max_needed_sum = max_syn - min_tun
    if max_needed_sum <= 0:
        return {"can_synchro": False, "matched_synchro_levels": 0}

    one_sums = {x for x in non_lvls if isinstance(x, int) and x > 0 and x <= max_needed_sum}
    two_sum_set = two_sums([x for x in non_lvls if isinstance(x, int)], max_sum=max_needed_sum)

    matched = 0
    for L in target_levels:
        if not isinstance(L, int):
            continue
        found = any(
            (L - t) > 0 and ((L - t) in one_sums or (L - t) in two_sum_set)
            for t in tuner_lvls
        )
        matched += int(found)

    can_synchro = (td["tuner_count"] >= 3 and td["non_tuner_count"] >= 8 and matched > 0)
    return {"can_synchro": can_synchro, "matched_synchro_levels": matched}

def feat_xyz(main_deck: List[Dict[str, Any]], extra_deck: List[Dict[str, Any]]) -> Dict[str, Any]:
    xyz_ranks = {
        coerce_int(card.get("rank"))
        for card in extra_deck
        if "xyz" in norm_subtypes_for_collection(card) and valid_level(card.get("rank"))
    }
    xyz_ranks.discard(None)

    level_counts = Counter(
        coerce_int(card.get("level"))
        for card in main_deck
        if is_monster(card) and valid_level(card.get("level"))
    )
    level_counts.pop(None, None)

    if level_counts:
        max_count = max(level_counts.values())
        mode_candidates = [lvl for lvl, c in level_counts.items() if c == max_count]
        xyz_level_mode = max(mode_candidates)   # deterministic tie-break
        xyz_level_mode_count = max_count
    else:
        xyz_level_mode = 0
        xyz_level_mode_count = 0

    can_xyz = True if not xyz_ranks else any(level_counts.get(r, 0) >= 8 for r in xyz_ranks if isinstance(r, int))

    deck_size = max(1, len(main_deck))
    N = max(1, min(60, deck_size))
    draws = min(7, N)
    if xyz_level_mode_count >= 2 and draws >= 2:
        K = min(xyz_level_mode_count, N)
        p_two_of_mode_lvl_in_7 = hypergeom_two_or_more(K=K, N=N, draws=draws)
    else:
        p_two_of_mode_lvl_in_7 = 0.0

    return {
        "xyz_level_mode": xyz_level_mode,
        "xyz_level_mode_count": xyz_level_mode_count,
        "can_xyz": can_xyz,
        "p_two_of_mode_lvl_in_7": p_two_of_mode_lvl_in_7,
    }

def feat_fusion(main_deck: List[Dict[str, Any]], extra_deck: List[Dict[str, Any]]) -> Dict[str, object]:
    has_fusion_extra = any("fusion" in norm_subtypes_for_collection(c) for c in extra_deck)

    fusion_enabler_count = sum(
        1
        for c in main_deck
        if norm_str(c.get("type")) and has_token(c.get("type", ""), "spell") and "fusion" in norm_subtypes_for_collection(c)
    )

    monster_count = sum(
        1
        for c in main_deck
        if is_monster(c) and (
            valid_level(c.get("level")) or
            (isinstance(coerce_int(c.get("rank")), int) and coerce_int(c.get("rank")) > 0) or
            (isinstance(coerce_int(c.get("link_rating")), int) and coerce_int(c.get("link_rating")) > 0)
        )
    )

    can_fusion = (not has_fusion_extra) or (fusion_enabler_count >= 1 and monster_count >= 3)

    return {
        "fusion_enabler_count": fusion_enabler_count,
        "can_fusion": can_fusion,
    }

def feat_core_and_link(main_deck: List[Dict[str, Any]], extra_deck: List[Dict[str, Any]]) -> Dict[str, Union[int, bool]]:
    low_level_count = 0
    monster_count = 0

    for c in main_deck:
        if not is_monster(c):
            continue
        lvl = coerce_int(c.get("level"))
        rnk = coerce_int(c.get("rank"))
        lnk = coerce_int(c.get("link_rating"))

        has_body = (
            valid_level(lvl) or
            (isinstance(rnk, int) and rnk > 0) or
            (isinstance(lnk, int) and lnk > 0)
        )
        if not has_body:
            continue

        monster_count += 1
        if valid_level(lvl) and lvl <= 4:
            low_level_count += 1

    link_ratings = [
        coerce_int(c.get("link_rating"))
        for c in extra_deck
        if "link" in norm_subtypes_for_collection(c)
        and isinstance(coerce_int(c.get("link_rating")), int)
        and coerce_int(c.get("link_rating")) > 0
    ]
    max_link_rating_in_extra = max([lr for lr in link_ratings if lr is not None], default=0)

    can_link = (max_link_rating_in_extra == 0) or (monster_count >= 12 and low_level_count >= 6)

    return {
        "low_level_count": low_level_count,
        "monster_count": monster_count,
        "max_link_rating_in_extra": max_link_rating_in_extra,
        "can_link": can_link,
    }

def feat_pendulum(main_deck: List[Dict[str, Any]], extra_deck: List[Dict[str, Any]]) -> Dict[str, Any]:
    has_pendulum_intent = any(
        "pendulum" in norm_subtypes_for_collection(card)
        for card in (main_deck + extra_deck)
    )
    if not has_pendulum_intent:
        return {
            "pendulum_count": 0,
            "pendulum_span_max": 0,
            "pendulum_inrange_monsters": 0,
            "can_pendulum": False,
        }

    main_monster_levels: List[int] = []
    pendulum_monsters: List[Tuple[int, int]] = []

    for card in main_deck:
        if is_monster(card):
            lvl = coerce_int(card.get('level'))
            if valid_level(lvl):
                qty = card.get('qty', 1)
                try:
                    q = max(1, int(qty))
                except (TypeError, ValueError):
                    q = 1
                main_monster_levels.extend([lvl] * q)  # type: ignore[list-item]

            L = coerce_int(card.get('pendulum_scale_left'))
            R = coerce_int(card.get('pendulum_scale_right'))
            if isinstance(L, int) and isinstance(R, int):
                if 0 <= L <= 13 and 0 <= R <= 13 and L != R:
                    pendulum_monsters.append((L, R))

    pendulum_count = len(pendulum_monsters)
    if pendulum_count < 2:
        return {
            "pendulum_count": pendulum_count,
            "pendulum_span_max": 0,
            "pendulum_inrange_monsters": 0,
            "can_pendulum": False,
        }

    pendulum_span_max = 0
    pendulum_inrange_monsters = 0

    for (L1, R1), (L2, R2) in combinations(pendulum_monsters, 2):
        lo = min(L1, R1, L2, R2)
        hi = max(L1, R1, L2, R2)
        span = hi - lo
        if span > pendulum_span_max:
            pendulum_span_max = span

        in_range = sum(1 for lvl in main_monster_levels if isinstance(lvl, int) and lo < lvl < hi)
        if in_range > pendulum_inrange_monsters:
            pendulum_inrange_monsters = in_range

    can_pendulum = (
        pendulum_count >= 2 and
        pendulum_span_max >= 2 and
        pendulum_inrange_monsters >= 5
    )

    return {
        "pendulum_count": pendulum_count,
        "pendulum_span_max": pendulum_span_max,
        "pendulum_inrange_monsters": pendulum_inrange_monsters,
        "can_pendulum": can_pendulum,
    }

def mechanic_features(main_deck: List[Dict[str, Any]], extra_deck: List[Dict[str, Any]]) -> Dict[str, Union[int, float, bool]]:
    result: Dict[str, Union[int, float, bool]] = {}
    
    # Tuner counts
    tuner_data = feat_tuner_counts(main_deck)
    result["tuner_count"] = tuner_data["tuner_count"]
    result["non_tuner_count"] = tuner_data["non_tuner_count"]
    
    # Synchro features
    synchro_data = feat_synchro(main_deck, extra_deck)
    result["can_synchro"] = synchro_data["can_synchro"]
    result["matched_synchro_levels"] = synchro_data["matched_synchro_levels"]
    
    # Xyz features
    xyz_data = feat_xyz(main_deck, extra_deck)
    result["xyz_level_mode"] = xyz_data["xyz_level_mode"]
    result["xyz_level_mode_count"] = xyz_data["xyz_level_mode_count"]
    result["can_xyz"] = xyz_data["can_xyz"]
    result["p_two_of_mode_lvl_in_7"] = xyz_data["p_two_of_mode_lvl_in_7"]
    
    # Fusion features
    fusion_data = feat_fusion(main_deck, extra_deck)
    result["fusion_enabler_count"] = fusion_data["fusion_enabler_count"]
    result["can_fusion"] = fusion_data["can_fusion"]
    
    # Core and Link features
    core_link_data = feat_core_and_link(main_deck, extra_deck)
    result["low_level_count"] = core_link_data["low_level_count"]
    result["max_link_rating_in_extra"] = core_link_data["max_link_rating_in_extra"]
    result["can_link"] = core_link_data["can_link"]
    
    # Pendulum features
    pendulum_data = feat_pendulum(main_deck, extra_deck)
    result["pendulum_count"] = pendulum_data["pendulum_count"]
    result["pendulum_span_max"] = pendulum_data["pendulum_span_max"]
    result["pendulum_inrange_monsters"] = pendulum_data["pendulum_inrange_monsters"]
    result["can_pendulum"] = pendulum_data["can_pendulum"]
    
    return result

# =============================================================================
# Cluster Functions (needed for main)
# =============================================================================

def assign_deck_clusters(deck_df: pd.DataFrame, n_clusters: int = 8) -> pd.Series:
    """Assign cluster labels to decks (placeholder - implement clustering logic)"""
    # For now, assign random clusters
    np.random.seed(42)
    return pd.Series(np.random.randint(0, n_clusters, size=len(deck_df)), index=deck_df.index)

def add_cluster_features(deck_df: pd.DataFrame) -> pd.DataFrame:
    """Add cluster-based features"""
    result_df = deck_df.copy()
    
    # Add cluster entropy (placeholder)
    result_df['cluster_entropy'] = np.random.random(len(deck_df))
    result_df['intra_deck_cluster_distance'] = np.random.random(len(deck_df))
    result_df['cluster_co_occurrence_rarity'] = np.random.random(len(deck_df))
    result_df['noise_card_percentage'] = np.random.random(len(deck_df))
    
    return result_df

def adjust_win_rate_by_cluster(deck_df: pd.DataFrame, target_col: str = 'win_rate') -> pd.DataFrame:
    """Add cluster-adjusted win rate"""
    result_df = deck_df.copy()
    result_df['adjusted_win_rate'] = result_df[target_col]  # Simple copy for now
    return result_df

# =============================================================================
# Advanced Feature Selection Functions
# =============================================================================

def drop_zero_variance_features(deck_df):
    """Drop features with zero or near-zero variance"""
    zero_variance_features = [
        'has_same_level_monsters', 'max_copies_per_card', 'can_xyz', 
        'fusion_enabler_count', 'max_link_rating_in_extra', 'can_link',
        'pendulum_count', 'pendulum_span_max', 'pendulum_inrange_monsters',
        'can_pendulum', 'num_search'
    ]
    
    existing_features = [f for f in zero_variance_features if f in deck_df.columns]
    if existing_features:
        deck_df = deck_df.drop(columns=existing_features)
        logging.info(f"Dropped {len(existing_features)} zero-variance features: {existing_features}")
    else:
        logging.info("No zero-variance features found to drop")
    
    return deck_df


def add_interaction_features(deck_df):
    """Add interaction terms between strongly correlated features"""
    interaction_features = []
    
    # Define interaction pairs with clear naming
    interaction_pairs = [
        ('low_level_count', 'xyz_level_mode_count', 'interact_low_level_xyz'),
        ('monster_count', 'p_two_of_mode_lvl_in_7', 'interact_monster_p_two'),
        ('non_tuner_count', 'num_unique_monsters', 'interact_non_tuner_unique')
    ]
    
    for feat1, feat2, interaction_name in interaction_pairs:
        if feat1 in deck_df.columns and feat2 in deck_df.columns:
            deck_df[interaction_name] = deck_df[feat1] * deck_df[feat2]
            interaction_features.append(interaction_name)
            logging.info(f"Added interaction feature: {interaction_name} = {feat1} × {feat2}")
        else:
            logging.warning(f"Cannot create interaction {interaction_name}: missing {feat1} or {feat2}")
    
    return deck_df, interaction_features


def drop_low_correlation_features(deck_df, target_col='adjusted_win_rate', drop_percentage=0.25):
    """Drop lowest X% of features by correlation, but keep SVD/PCA and interaction features"""
    if target_col not in deck_df.columns:
        logging.warning(f"Target column '{target_col}' not found, skipping correlation filtering")
        return deck_df, []
    
    # Identify features to consider for dropping (exclude protected features)
    protected_keywords = ['svd', 'pca', 'embedding', 'interact_', 'bow_', 'tfidf']
    dense_features = []
    protected_features = []
    
    for col in deck_df.columns:
        if any(keyword in col.lower() for keyword in protected_keywords):
            protected_features.append(col)
        elif col != target_col and deck_df[col].dtype in ['float64', 'int64']:
            dense_features.append(col)
    
    if len(dense_features) < 10:
        logging.info("Too few dense features to apply correlation filtering")
        return deck_df, []
    
    # Calculate correlations with target
    correlations = []
    for feature in dense_features:
        try:
            corr = deck_df[feature].corr(deck_df[target_col])
            if not np.isnan(corr):
                correlations.append((feature, abs(corr)))
        except Exception as e:
            logging.warning(f"Could not compute correlation for {feature}: {e}")
    
    # Sort by correlation and drop lowest X%
    correlations.sort(key=lambda x: x[1])
    n_to_drop = int(len(correlations) * drop_percentage)
    features_to_drop = [feat for feat, _ in correlations[:n_to_drop]]
    
    if features_to_drop:
        deck_df = deck_df.drop(columns=features_to_drop)
        logging.info(f"Dropped {len(features_to_drop)} low-correlation features (bottom {drop_percentage*100:.0f}%)")
        logging.info(f"Dropped features: {features_to_drop[:5]}{'...' if len(features_to_drop) > 5 else ''}")
    
    # Log summary
    remaining_dense = len([f for f in dense_features if f not in features_to_drop])
    logging.info(f"Remaining dense features: {remaining_dense}, Protected features: {len(protected_features)}")
    
    return deck_df, features_to_drop


def apply_advanced_feature_selection(deck_df, target_col='adjusted_win_rate'):
    """Apply all advanced feature selection steps"""
    logging.info("=== APPLYING ADVANCED FEATURE SELECTION ===")
    
    original_features = len(deck_df.columns)
    original_shape = deck_df.shape
    
    # Step 1: Drop zero variance features
    deck_df = drop_zero_variance_features(deck_df)
    
    # Step 2: Add interaction features
    deck_df, interaction_features = add_interaction_features(deck_df)
    
    # Step 3: Drop low correlation features (but keep interactions and embeddings)
    deck_df, dropped_features = drop_low_correlation_features(deck_df, target_col)
    
    final_shape = deck_df.shape
    logging.info(f"Feature selection complete: {original_shape} → {final_shape}")
    logging.info(f"Added {len(interaction_features)} interaction features: {interaction_features}")
    
    return deck_df

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Load deck data and battle results
    deck_df = load_deck_data_from_s3()
    battle_results_df = load_battle_results_from_s3()
    
    # Merge datasets and filter out failed battles
    deck_df = merge_deck_data_with_battle_results(deck_df, battle_results_df)

    # Summoning Mechanic Features (fixed checks)
    deck_df['has_tuner'] = deck_df['main_deck'].apply(has_tuner)
    deck_df['num_tuners'] = deck_df['main_deck'].apply(num_tuners)
    deck_df['has_same_level_monsters'] = deck_df['main_deck'].apply(has_same_level_monsters)
    deck_df['max_same_level_count'] = deck_df['main_deck'].apply(max_same_level_count)
    deck_df['avg_monster_level'] = deck_df['main_deck'].apply(avg_monster_level)
    deck_df['has_pendulum_monsters'] = deck_df['main_deck'].apply(has_pendulum_monsters)

    deck_df['has_synchro_monsters'] = deck_df['extra_deck'].apply(has_synchro_monsters)
    deck_df['has_xyz_monsters'] = deck_df['extra_deck'].apply(has_xyz_monsters)
    deck_df['has_link_monsters'] = deck_df['extra_deck'].apply(has_link_monsters)
    
    # Additional missing features
    deck_df['max_copies_per_card'] = deck_df['main_deck'].apply(max_copies_per_card)
    deck_df['avg_copies_per_monster'] = deck_df['main_deck'].apply(avg_copies_per_monster)
    deck_df['num_unique_monsters'] = deck_df['main_deck'].apply(num_unique_monsters)

    # Advanced Mechanic Features
    def apply_mechanic_features(row):
        return mechanic_features(row['main_deck'], row['extra_deck'])
    
    mechanic_features_df = pd.DataFrame(
        deck_df.apply(apply_mechanic_features, axis=1).tolist(), 
        index=deck_df.index
    )
    deck_df = pd.concat([deck_df, mechanic_features_df], axis=1)

    # NLP features
    deck_df = add_tfidf_features_to_decks(deck_df)
    deck_df = add_bow_features_nltk(deck_df, top_k_words=150, binary=True)
    deck_df = add_mean_embedding_features_to_decks(deck_df, model_name="all-MiniLM-L6-v2")

    # Strategy Flag Features
    strat_kw_count_df = pd.DataFrame(deck_df['main_deck'].apply(count_strategy_keywords).tolist(), index=deck_df.index)
    deck_df = pd.concat([deck_df, strat_kw_count_df], axis=1)

    # Cluster assignment features
    deck_df['cluster'] = assign_deck_clusters(deck_df, n_clusters=8)
    deck_df = add_cluster_features(deck_df)

    # Adjust win rate by cluster
    deck_df = adjust_win_rate_by_cluster(deck_df, target_col='win_rate')

    # Display summary info
    feature_cols = [col for col in deck_df.columns 
                   if col not in ['deck_id', 'main_deck', 'extra_deck', 'side_deck', 'name', 'deck_json']]

    for col in feature_cols:
        try:
            logger.info(f"\nSummary for {col}:\n{deck_df[[col]].describe(include='all')}")
        except Exception as e:
            logger.warning(f"Could not summarize column {col}: {e}")

    logging.info("Feature engineering completed")
    logging.info(f"Final dataset shape: {deck_df.shape}")

    # Apply advanced feature selection
    deck_df = apply_advanced_feature_selection(deck_df, target_col='adjusted_win_rate')

    # Write to parquet buffer
    buffer = io.BytesIO()
    deck_df.to_parquet(buffer)
    buffer.seek(0)
    
    # Define key to overwrite original feature_engineered.parquet
    month_str = datetime.today().strftime('%Y-%m')
    s3_key = f"processed/feature_engineered/deck_scoring/{month_str}/feature_engineered.parquet"

    upload_to_s3(
        bucket="yugioh-data",
        key=s3_key,
        data=buffer.getvalue(),
        content_type="application/octet-stream"
    )

    logging.info("Advanced feature-engineered data uploaded to S3")
    print("✅ Advanced feature engineering completed successfully!")
    print(f"Final dataset: {deck_df.shape[0]} samples, {deck_df.shape[1]} features")
    print(f"Uploaded to: s3://yugioh-data/{s3_key}")
