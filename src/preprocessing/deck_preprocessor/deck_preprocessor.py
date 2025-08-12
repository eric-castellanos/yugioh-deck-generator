# src/preprocessing/deck_preprocessor/deck_preprocessor.py
from typing import List, Optional, Tuple, Literal
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

class DeckPreprocessor:
    def __init__(
        self,
        method: Literal["pca", "svd"] = "svd",
        embedding_n_components: int = 48,
        dense_features: Optional[List[str]] = None,
        target_col: str = "adjusted_win_rate",
        standardize_embeddings: bool = False,
        random_state: int = 42,
    ):
        self.method = method
        self.embedding_n_components = embedding_n_components
        self.target_col = target_col
        self.random_state = random_state

        # Reducers/scalers created on fit
        self.reducer = None
        self.scaler_dense = StandardScaler()
        self.scaler_emb = StandardScaler(with_mean=True, with_std=True) if standardize_embeddings else None

        # Simplified dense features - assuming advanced feature selection already done
        self._all_dense_features_cfg = dense_features or [
            # Basic deck structure 
            'has_tuner', 'num_tuners', 'max_same_level_count',
            'avg_monster_level', 'has_pendulum_monsters', 'has_synchro_monsters',
            'has_xyz_monsters', 'has_link_monsters',
            'avg_copies_per_monster', 'num_unique_monsters',
            # Mechanic features
            'tuner_count', 'non_tuner_count', 'can_synchro', 'matched_synchro_levels',
            'xyz_level_mode', 'xyz_level_mode_count', 'p_two_of_mode_lvl_in_7',
            'can_fusion', 'low_level_count', 'monster_count',
            # Strategy features
            'main_deck_mean_tfidf', 'num_banish', 'num_graveyard', 'num_draw',
            'num_special_summon', 'num_negate', 'num_destroy', 'num_shuffle',
            # Cluster features
            'cluster_entropy', 'intra_deck_cluster_distance', 'cluster_co_occurrence_rarity',
            'noise_card_percentage',
            # Interaction features (if present from feature engineering)
            'interact_low_xyz', 'interact_monster_p', 'interact_non_unique',
            # Battle confidence (if present)
            'battle_confidence', 'combo_potential'
        ]

        # Fitted state
        self.embedding_cols: List[str] = []            # expanded embedding column names (embedding_*)
        self.dense_features_used_: List[str] = []      # subset present at fit time (ordered)
        self.feature_names_: List[str] = []            # final feature names (dense + reduced embeddings)
        self.n_original_embedding_dims_: Optional[int] = None

    # --------------------------- helpers ---------------------------
    def _add_simple_ranking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple ranking-focused features (idempotent)."""
        df = df.copy()
        # Battle confidence - bounded [0,1]
        if 'battle_confidence' not in df.columns:
            if 'successful_battles' in df.columns:
                denom = df['total_battles'] if 'total_battles' in df.columns else (df['successful_battles'] + 0)
                denom = denom.replace(0, np.nan)
                raw_confidence = df['successful_battles'] / (denom + 1)
                df['battle_confidence'] = np.clip(raw_confidence.fillna(0.0), 0.0, 1.0)
        # Combo potential (count of available mechanics)
        if 'combo_potential' not in df.columns:
            combo_features = [c for c in ['can_synchro', 'can_xyz', 'can_link', 'can_fusion'] if c in df.columns]
            if combo_features:
                df['combo_potential'] = df[combo_features].sum(axis=1)
        return df

    def _extract_and_expand_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        if "main_deck_mean_embedding" not in df.columns:
            raise ValueError("Column 'main_deck_mean_embedding' is missing from the DataFrame. This column is required.")
        # Convert list-like column to a 2D array
        emb_values = df["main_deck_mean_embedding"].tolist()
        if len(emb_values) == 0:
            raise ValueError("'main_deck_mean_embedding' column is empty.")
        # Ensure all rows have same length
        first_len = len(emb_values[0]) if isinstance(emb_values[0], (list, np.ndarray)) else None
        if first_len is None:
            raise ValueError("Each entry in 'main_deck_mean_embedding' must be a list/ndarray.")
        for i, v in enumerate(emb_values):
            if not isinstance(v, (list, np.ndarray)) or len(v) != first_len:
                raise ValueError(f"Embedding at row {i} has inconsistent length (expected {first_len}).")
        emb_mat = np.asarray(emb_values, dtype=np.float32)
        self.n_original_embedding_dims_ = emb_mat.shape[1]
        emb_df = pd.DataFrame(emb_mat, index=df.index).add_prefix("embedding_")
        return pd.concat([df.drop(columns=["main_deck_mean_embedding"]), emb_df], axis=1)

    def _select_dense_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            self.dense_features_used_ = [c for c in self._all_dense_features_cfg if c in df.columns]
            if not self.dense_features_used_:
                raise ValueError("No configured dense features were found in the DataFrame during fit.")
        # Ensure all required columns exist; if missing at transform-time, add zeros
        for c in self.dense_features_used_:
            if c not in df.columns:
                df[c] = 0.0
        return df[self.dense_features_used_]

    def _reduce_embeddings(self, emb: np.ndarray, fit: bool) -> Tuple[np.ndarray, List[str]]:
        X = emb
        if self.scaler_emb is not None:
            if fit:
                X = self.scaler_emb.fit_transform(X)
            else:
                X = self.scaler_emb.transform(X)
        if self.method == "pca":
            if fit:
                self.reducer = PCA(n_components=self.embedding_n_components, random_state=self.random_state)
                Z = self.reducer.fit_transform(X)
            else:
                Z = self.reducer.transform(X)
        elif self.method == "svd":
            if fit:
                self.reducer = TruncatedSVD(n_components=self.embedding_n_components, random_state=self.random_state)
                Z = self.reducer.fit_transform(X)
            else:
                Z = self.reducer.transform(X)
        else:
            Z = X
        names = [f"reduced_embedding_{i}" for i in range(Z.shape[1])]
        return Z, names

    # --------------------------- public API ---------------------------
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = self._add_simple_ranking_features(df)
        df = self._extract_and_expand_embeddings(df)
        self.embedding_cols = [c for c in df.columns if c.startswith("embedding_")]

        # Dense features (fit-time selection + scaling)
        dense_df = self._select_dense_features(df, fit=True).astype(np.float32)
        dense_X = self.scaler_dense.fit_transform(dense_df)

        # Embeddings
        emb = df[self.embedding_cols].to_numpy(dtype=np.float32)
        Z, z_names = self._reduce_embeddings(emb, fit=True)

        # Combine
        X = np.hstack([dense_X, Z]).astype(np.float32)

        # Names
        self.feature_names_ = list(self.dense_features_used_) + z_names

        # Target
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found")
        y = df[self.target_col].to_numpy(dtype=np.float32)
        return X, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Guard: ensure fitted
        if not hasattr(self.scaler_dense, 'scale_'):
            raise ValueError("Must call fit_transform before transform")
        if self.embedding_cols is None or len(self.embedding_cols) == 0:
            raise ValueError("Preprocessor not fitted: missing embedding column state.")
        if self.reducer is None and self.method in {"pca", "svd"}:
            raise ValueError("Preprocessor not fitted: missing reducer.")

        df = self._add_simple_ranking_features(df)
        df = self._extract_and_expand_embeddings(df)

        # Ensure we have the same embedding columns
        for c in self.embedding_cols:
            if c not in df.columns:
                raise ValueError(f"Expected embedding column '{c}' missing at transform time.")
        emb = df[self.embedding_cols].to_numpy(dtype=np.float32)

        # Dense
        dense_df = self._select_dense_features(df, fit=False).astype(np.float32)
        dense_X = self.scaler_dense.transform(dense_df)

        # Embeddings reduce with fitted reducer
        Z, _ = self._reduce_embeddings(emb, fit=False)

        X = np.hstack([dense_X, Z]).astype(np.float32)

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found")
        y = df[self.target_col].to_numpy(dtype=np.float32)
        return X, y
