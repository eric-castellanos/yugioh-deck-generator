import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import Counter, defaultdict
import re
from datetime import datetime
import io

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#nltk.download('punkt')
#nltk.download('punkt_tab')
#nltk.download('stopwords')

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
            # Raise error if any value is not a list
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
    
    Parameters:
    -----------
    bucket : str
        S3 bucket name
    key : str
        Object key (path to file in S3)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the deck data with properly converted lists
    """
    logger.info(f"Loading deck data from s3://{bucket}/{key}")
    
    # Read data from S3
    df = read_csv_from_s3(bucket=bucket, key=key)
    
    # Convert string representations of lists to actual lists
    list_columns = ['main_deck', 'extra_deck']
    df = convert_list_columns(df, list_columns)
            
    return df

def load_battle_results_from_s3(
    bucket: str = 'yugioh-data',
    key: str = 'deck_scoring/training_data/deck_battle_results.csv'
) -> pd.DataFrame:
    """
    Load deck battle results from S3 bucket
    
    Parameters:
    -----------
    bucket : str
        S3 bucket name
    key : str
        Object key (path to file in S3)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the battle results data
    """
    logger.info(f"Loading battle results from s3://{bucket}/{key}")
    
    # Read data from S3
    df = read_csv_from_s3(bucket=bucket, key=key)
    
    return df

def merge_deck_data_with_battle_results(
    deck_df: pd.DataFrame,
    battle_results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Perform right join between deck data and battle results on deck_id,
    then filter out rows where win_rate is "FAILED"
    
    Parameters:
    -----------
    deck_df : pd.DataFrame
        DataFrame with deck composition and features
    battle_results_df : pd.DataFrame
        DataFrame with battle results including win_rate
        
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with failed battles filtered out
    """
    logger.info(f"Merging deck data ({len(deck_df)} rows) with battle results ({len(battle_results_df)} rows)")
    
    # Perform right join on deck_id
    merged_df = pd.merge(
        deck_df, 
        battle_results_df, 
        on='deck_id', 
        how='right'
    )
    
    logger.info(f"After merge: {len(merged_df)} rows")
    
    # Filter out rows where win_rate is "FAILED"
    initial_count = len(merged_df)
    merged_df = merged_df[merged_df['win_rate'] != 'FAILED']
    final_count = len(merged_df)
    
    failed_count = initial_count - final_count
    logger.info(f"Filtered out {failed_count} rows with win_rate = 'FAILED'")
    logger.info(f"Final dataset: {final_count} rows")
    
    return merged_df

def has_tuner(main_deck: List[Dict[str, Any]]) -> bool:
    return any('Tuner' in card.get('type', '') for card in main_deck)

def has_same_level_monsters(main_deck: List[Dict[str, Any]]) -> bool:
    levels = [card.get('level') for card in main_deck if card.get('level') is not None]
    counts = Counter(levels)
    return any(count >= 2 for count in counts.values())

def has_synchro_monsters(extra_deck: List[Dict[str, Any]]) -> bool:
    return any('Synchro' in card.get('type', '') for card in extra_deck)

def has_xyz_monsters(extra_deck: List[Dict[str, Any]]) -> bool:
    return any('XYZ' in card.get('type', '') for card in extra_deck)

def has_link_monsters(extra_deck: List[Dict[str, Any]]) -> bool:
    return any('Link' in card.get('type', '') for card in extra_deck)

def has_pendulum_monsters(main_deck: List[Dict[str, Any]]) -> bool:
    return any('Pendulum' in card.get('type', '') for card in main_deck)

def max_same_level_count(main_deck: List[Dict[str, Any]]) -> int:
    levels = [card.get('level') for card in main_deck if card.get('level') is not None]
    return max(Counter(levels).values(), default=0)

def num_tuners(main_deck: List[Dict[str, Any]]) -> int:
    return sum(1 for card in main_deck if 'Tuner' in card.get('type', ''))

def avg_monster_level(main_deck: List[Dict[str, Any]]) -> float:
    levels = [card.get('level') for card in main_deck if isinstance(card.get('level'), (int, float))]
    return float(np.mean(levels)) if levels else 0.0

def max_copies_per_card(main_deck: List[Dict[str, Any]]) -> int:
    names = [card.get('name') for card in main_deck]
    return max(Counter(names).values(), default=0)

def avg_copies_per_monster(main_deck: List[Dict[str, Any]]) -> float:
    monster_names = [card.get('name') for card in main_deck if 'Monster' in card.get('type', '')]
    name_counts = Counter(monster_names)
    return float(np.mean(list(name_counts.values()))) if name_counts else 0.0

def num_unique_monsters(main_deck: List[Dict[str, Any]]) -> int:
    return len({card.get('name') for card in main_deck if 'Monster' in card.get('type', '')})

def overreplicated_cards(main_deck: List[Dict[str, Any]]) -> int:
    name_counts = Counter(card.get('name') for card in main_deck)
    return sum(1 for count in name_counts.values() if count > 3)

def get_card_descriptions(card_list: List[Dict[str, Any]]) -> List[str]:
    """
    Extracts the 'desc' field from each card dictionary.
    """
    #logger.info(f"Getting descriptions from {len(card_list)} card dictionaries")

    descriptions = [
        card['desc'] for card in card_list
        if isinstance(card, dict) and isinstance(card.get('desc'), str) and card['desc'].strip()
    ]

    logger.debug(f"Extracted {len(descriptions)} valid descriptions")
    return descriptions


def calculate_mean_tfidf_for_deck(
    card_descriptions: List[str], 
    tfidf_vectorizer: Optional[TfidfVectorizer] = None
) -> Tuple[float, TfidfVectorizer]:
    """
    Calculate mean TF-IDF value for a deck based on card descriptions.
    
    Parameters:
    -----------
    card_descriptions : List[str]
        List of card descriptions.
    tfidf_vectorizer : TfidfVectorizer, optional
        Fitted TF-IDF vectorizer. If None, a new one will be created and fitted.
    
    Returns:
    --------
    Tuple[float, TfidfVectorizer]
        (mean_tfidf_value, fitted_vectorizer)
    """
    if not card_descriptions:
        logger.warning("No card descriptions provided. Returning 0.0.")
        return 0.0, tfidf_vectorizer

    if tfidf_vectorizer is None:
        logger.debug("Fitting new TF-IDF vectorizer.")
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(card_descriptions)
    else:
        logger.debug("Transforming with existing TF-IDF vectorizer.")
        tfidf_matrix = tfidf_vectorizer.transform(card_descriptions)

    # Mean of all non-zero TF-IDF values
    mean_tfidf = float(tfidf_matrix.mean())
    logger.debug(f"Mean TF-IDF value: {mean_tfidf:.6f}")

    return mean_tfidf, tfidf_vectorizer

def add_tfidf_features_to_decks(deck_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TF-IDF based features to the deck dataframe using embedded card descriptions.
    """
    logger.info(f"Adding TF-IDF features to {len(deck_df)} decks")
    result_df = deck_df.copy()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Extract all descriptions
    logger.info("Extracting descriptions from main decks")
    deck_descriptions = result_df['main_deck'].apply(
        lambda cards: get_card_descriptions(cards) if isinstance(cards, list) else []
    )
    all_descriptions = [desc for descs in deck_descriptions for desc in descs]

    if all_descriptions:
        logger.info(f"Fitting TF-IDF vectorizer on {len(all_descriptions)} descriptions")
        tfidf_vectorizer.fit(all_descriptions)
    else:
        logger.warning("No descriptions found to fit vectorizer")

    # Calculate mean TF-IDF
    logger.info("Calculating mean TF-IDF per deck")
    result_df['main_deck_mean_tfidf'] = deck_descriptions.apply(
        lambda descs: calculate_mean_tfidf_for_deck(descs, tfidf_vectorizer)[0]
    )

    logger.info("TF-IDF features added successfully")
    return result_df

def add_bow_features_nltk(deck_df: pd.DataFrame, top_k_words: int = 100, binary: bool = True) -> pd.DataFrame:
    # Step 1: Pre-tokenize all descriptions and cache tokens
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

    # Step 2: Get top K frequent words
    top_words = [word for word, _ in Counter(all_tokens).most_common(top_k_words)]

    # Step 3: Construct BoW matrix
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
    """
    Load the SentenceTransformer model.
    """
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    return SentenceTransformer(model_name)


def compute_mean_embedding(descs: List[str], model: SentenceTransformer) -> List[float]:
    """
    Compute the mean embedding for a list of descriptions using the given model.
    """
    if not descs:
        return np.zeros(model.get_sentence_embedding_dimension()).tolist()
    
    embeddings = model.encode(descs, show_progress_bar=False)
    return np.mean(embeddings, axis=0).tolist()


def add_mean_embedding_features_to_decks(deck_df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    """
    Add a new column `main_deck_mean_embedding` with mean description embeddings for each deck.
    """
    model = load_embedding_model(model_name)
    deck_df = deck_df.copy()

    def process_deck(deck_cards: List[Dict[str, Any]]) -> List[float]:
        descriptions = get_card_descriptions(deck_cards)
        return compute_mean_embedding(descriptions, model)

    logger.info(f"Generating mean embeddings for {len(deck_df)} decks")
    deck_df["main_deck_mean_embedding"] = deck_df["main_deck"].apply(process_deck)
    logger.info("Mean embedding features added successfully")

    return deck_df

def count_strategy_keywords(main_deck: list[dict]) -> dict:
    """
    Returns counts of how many times each strategy keyword appears across card descriptions.
    """
    text = " ".join(
        card.get("desc", "").lower()
        for card in main_deck
        if isinstance(card.get("desc", ""), str)
    )

    return {
        feature: len(re.findall(pattern, text))
        for feature, pattern in STRATEGY_KEYWORDS.items()
    }

if __name__ == "__main__":
    # Load deck data and battle results
    deck_df = load_deck_data_from_s3()
    battle_results_df = load_battle_results_from_s3()
    
    # Merge datasets and filter out failed battles
    deck_df = merge_deck_data_with_battle_results(deck_df, battle_results_df)

    # Summoning Mechanic Features
    deck_df['has_tuner'] = deck_df['main_deck'].apply(has_tuner)
    deck_df['num_tuners'] = deck_df['main_deck'].apply(num_tuners)
    deck_df['has_same_level_monsters'] = deck_df['main_deck'].apply(has_same_level_monsters)
    deck_df['max_same_level_count'] = deck_df['main_deck'].apply(max_same_level_count)
    deck_df['avg_monster_level'] = deck_df['main_deck'].apply(avg_monster_level)
    deck_df['has_pendulum_monsters'] = deck_df['main_deck'].apply(has_pendulum_monsters)

    deck_df['has_synchro_monsters'] = deck_df['extra_deck'].apply(has_synchro_monsters)
    deck_df['has_xyz_monsters'] = deck_df['extra_deck'].apply(has_xyz_monsters)
    deck_df['has_link_monsters'] = deck_df['extra_deck'].apply(has_link_monsters)

    # Deck Composition Features
    deck_df['max_copies_per_card'] = deck_df['main_deck'].apply(max_copies_per_card)
    deck_df['avg_copies_per_monster'] = deck_df['main_deck'].apply(avg_copies_per_monster)
    deck_df['num_unique_monsters'] = deck_df['main_deck'].apply(num_unique_monsters)

    # NLP features
    deck_df = add_tfidf_features_to_decks(deck_df)
    deck_df = add_bow_features_nltk(deck_df, top_k_words=150, binary=True)
    deck_df = add_mean_embedding_features_to_decks(deck_df, model_name="all-MiniLM-L6-v2")

    # Strategy Flag Features
    strat_kw_count_df = pd.DataFrame(deck_df['main_deck'].apply(count_strategy_keywords).tolist(), index=deck_df.index)
    deck_df = pd.concat([deck_df, strat_kw_count_df], axis=1)

    feature_cols = [
    'has_tuner',
    'num_tuners',
    'has_same_level_monsters',
    'max_same_level_count',
    'avg_monster_level',
    'has_pendulum_monsters',
    'has_synchro_monsters',
    'has_xyz_monsters',
    'has_link_monsters',
    'max_copies_per_card',
    'avg_copies_per_monster',
    'num_unique_monsters',
    'main_deck_mean_tfidf',
    'main_deck_mean_embedding',
    'num_banish',
    'num_graveyard',
    'num_draw',
    'num_search',
    'num_special_summon',
    'num_negate',
    'num_destroy',
    'num_shuffle'
    ]

    for col in feature_cols:
        logger.info(f"\nSummary for {col}:\n{deck_df[[col]].describe(include='all')}")

    deck_df = deck_df.drop(columns=['cluster_entropy', 'intra_deck_cluster_distance', 'cluster_co_occurrence_rarity',
                                    'noise_card_percentage', 'raw_entropy', 'raw_distance', 'raw_rarity',
                                    'raw_noise'], axis=1)

    # Write Polars DataFrame to in-memory Parquet buffer
    buffer = io.BytesIO()
    deck_df.to_parquet(buffer)
    buffer.seek(0)
    
    # Define key and upload
    month_str = datetime.today().strftime('%Y-%m')
    s3_key = f"processed/feature_engineered/deck_scoring/{month_str}/feature_engineered.parquet"

    upload_to_s3(
        bucket="yugioh-data",
        key=s3_key,
        data=buffer.getvalue(),
        content_type="application/octet-stream"
    )