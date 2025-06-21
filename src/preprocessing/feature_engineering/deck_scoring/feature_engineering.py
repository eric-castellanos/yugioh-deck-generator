import pandas as pd
import numpy as np
import ast
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer

# Import utils
from src.utils.s3_utils import read_csv_from_s3

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_list_columns(
    df: pd.DataFrame, 
    list_columns: List[str]
) -> pd.DataFrame:
    """
    Convert string representations of lists to actual Python lists
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing columns with string representations of lists
    list_columns : List[str]
        List of column names to convert
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with converted list columns
    """
    logger.info(f"Converting list columns: {list_columns}")
    result_df = df.copy()
    
    for col in list_columns:
        if col in result_df.columns:
            result_df[col] = result_df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
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

def get_card_descriptions(
    card_list: List[str], 
    card_data: pd.DataFrame
) -> List[str]:
    """
    Get descriptions for a list of cards
    
    Parameters:
    -----------
    card_list : List[str]
        List of card IDs or names
    card_data : pd.DataFrame
        DataFrame containing card information with 'name' and 'desc' columns
        
    Returns:
    --------
    List[str]
        List of card descriptions
    """
    logger.info(f"Getting descriptions for {len(card_list)} cards")
    descriptions = []
    
    for card in card_list:
        # Try to match by ID or name depending on what's provided
        card_info = card_data[card_data['name'] == card]
        if not card_info.empty:
            desc = card_info['desc'].iloc[0]
            if isinstance(desc, str) and len(desc) > 0:
                descriptions.append(desc)
    
    logger.debug(f"Found {len(descriptions)} valid descriptions")
    return descriptions

def calculate_mean_tfidf_for_deck(
    card_descriptions: List[str], 
    tfidf_vectorizer: Optional[TfidfVectorizer] = None
) -> Tuple[float, TfidfVectorizer]:
    """
    Calculate mean TF-IDF value for a deck based on card descriptions
    
    Parameters:
    -----------
    card_descriptions : List[str]
        List of card descriptions
    tfidf_vectorizer : TfidfVectorizer, optional
        Fitted TF-IDF vectorizer. If None, a new one will be created and fitted
        
    Returns:
    --------
    Tuple[float, TfidfVectorizer]
        (mean_tfidf_value, fitted_vectorizer)
    """
    logger.debug(f"Calculating mean TF-IDF for {len(card_descriptions)} descriptions")
    
    if not card_descriptions:
        logger.warning("No card descriptions provided, returning 0.0")
        return 0.0, tfidf_vectorizer
    
    # If no vectorizer provided, create and fit a new one
    if tfidf_vectorizer is None:
        logger.debug("Creating new TF-IDF vectorizer")
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(card_descriptions)
    else:
        # Use the provided vectorizer
        logger.debug("Using existing TF-IDF vectorizer")
        tfidf_matrix = tfidf_vectorizer.transform(card_descriptions)
    
    # Calculate mean TF-IDF value across all terms and documents
    mean_tfidf = tfidf_matrix.mean()
    logger.debug(f"Mean TF-IDF value: {mean_tfidf}")
    
    return mean_tfidf, tfidf_vectorizer

def add_tfidf_features_to_decks(
    deck_df: pd.DataFrame, 
    card_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add TF-IDF based features to the deck dataframe
    
    Parameters:
    -----------
    deck_df : pd.DataFrame
        DataFrame containing deck data with 'main_deck' and 'extra_deck' columns
    card_df : pd.DataFrame
        DataFrame containing card information with 'name' and 'desc' columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added TF-IDF features
    """
    logger.info(f"Adding TF-IDF features to {len(deck_df)} decks")
    
    # Create a copy to avoid modifying the original
    result_df = deck_df.copy()
    
    # Initialize the vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    # First pass - collect all descriptions to fit the vectorizer
    logger.info("First pass: collecting all descriptions to fit the vectorizer")
    all_descriptions = []
    for _, row in result_df.iterrows():
        main_deck = row['main_deck'] 
        if isinstance(main_deck, list):
            all_descriptions.extend(get_card_descriptions(main_deck, card_df))
    
    # Fit the vectorizer on all descriptions (if we have any)
    if all_descriptions:
        logger.info(f"Fitting TF-IDF vectorizer on {len(all_descriptions)} descriptions")
        tfidf_vectorizer.fit(all_descriptions)
    else:
        logger.warning("No card descriptions found to fit the vectorizer")
    
    # Second pass - calculate mean TF-IDF for each deck
    logger.info("Second pass: calculating mean TF-IDF for each deck")
    result_df['main_deck_mean_tfidf'] = 0.0
    
    for idx, row in result_df.iterrows():
        main_deck = row['main_deck']
        if isinstance(main_deck, list):
            descriptions = get_card_descriptions(main_deck, card_df)
            mean_tfidf, _ = calculate_mean_tfidf_for_deck(descriptions, tfidf_vectorizer)
            result_df.at[idx, 'main_deck_mean_tfidf'] = mean_tfidf
    
    logger.info("TF-IDF features added successfully")
    return result_df

# Example usage:
if __name__ == "__main__":
    # Load deck data from S3
    deck_df = load_deck_data_from_s3()
    
    # This would require loading card data with descriptions
    # For example:
    # card_df = pd.read_csv("path_to_card_data.csv")
    
    # Print a preview of the loaded data
    logger.info("Loaded deck data:")
    logger.info(f"Shape: {deck_df.shape}")
    logger.info(f"Columns: {deck_df.columns.tolist()}")
    
    # Check that main_deck column was properly parsed as list
    if 'main_deck' in deck_df.columns:
        sample = deck_df['main_deck'].iloc[0] if not deck_df.empty else []
        logger.info(f"Sample main deck (first entry): {sample[:5]}...")
        logger.info(f"Type of main_deck column values: {type(sample)}")
    
    logger.info("\nTo add TF-IDF features, you need to also load card data with descriptions")
    logger.info("Example: card_df = read_csv_from_s3('yugioh-data', 'path/to/card_data.csv')")
    logger.info("Then run: result_df = add_tfidf_features_to_decks(deck_df, card_df)")
