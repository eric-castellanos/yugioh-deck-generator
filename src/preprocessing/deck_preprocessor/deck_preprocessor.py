# src/model/preprocessing/deck_preprocessor.py

from typing import List, Optional, Tuple, Literal
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

class DeckPreprocessor:
    def __init__(
        self,
        method: Literal["pca", "svd"] = "svd",
        embedding_n_components: int = 40,
        dense_features: Optional[List[str]] = None,
    ):
        self.method = method
        self.embedding_n_components = embedding_n_components
        self.reducer = None
        self.scaler = StandardScaler()
        self.dense_features = dense_features or [
            'has_tuner', 'num_tuners', 'has_same_level_monsters', 'max_same_level_count',
            'avg_monster_level', 'has_pendulum_monsters', 'has_synchro_monsters',
            'has_xyz_monsters', 'has_link_monsters', 'max_copies_per_card',
            'avg_copies_per_monster', 'num_unique_monsters', 'main_deck_mean_tfidf',
            'num_banish', 'num_graveyard', 'num_draw', 'num_search',
            'num_special_summon', 'num_negate', 'num_destroy', 'num_shuffle'
        ]
        self.embedding_cols = []
        self.feature_names_: List[str] = []

    def _reduce_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if self.method == "pca":
            self.reducer = PCA(n_components=self.embedding_n_components, random_state=42)
        else:
            self.reducer = TruncatedSVD(n_components=self.embedding_n_components, random_state=42)
        return self.reducer.fit_transform(embeddings)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        embedding_df = pd.DataFrame(
            df["main_deck_mean_embedding"].to_list(),
            index=df.index
        ).add_prefix("embedding_")

        df = pd.concat([df.drop(columns=["main_deck_mean_embedding"]), embedding_df], axis=1)

        self.embedding_cols = [col for col in df.columns if col.startswith("embedding_")]
        dense_X = df[self.dense_features]
        embeddings = df[self.embedding_cols].values

        dense_X_scaled = self.scaler.fit_transform(dense_X)
        reduced_embeddings = self._reduce_embeddings(embeddings)

        self.feature_names_ = (
            self.dense_features +
            [f"{self.method}_embed_{i}" for i in range(self.embedding_n_components)]
        )

        X = np.hstack([dense_X_scaled, reduced_embeddings])
        y = df["composite_score"].values.astype(np.float32)

        return X.astype(np.float32), y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        embedding_df = pd.DataFrame(
            df["main_deck_mean_embedding"].to_list(),
            index=df.index
        ).add_prefix("embedding_")

        df = pd.concat([df.drop(columns=["main_deck_mean_embedding"]), embedding_df], axis=1)
        dense_X = df[self.dense_features]
        embeddings = df[self.embedding_cols].values

        dense_X_scaled = self.scaler.transform(dense_X)
        reduced_embeddings = self.reducer.transform(embeddings)

        X = np.hstack([dense_X_scaled, reduced_embeddings])
        y = df["composite_score"].values.astype(np.float32)

        return X.astype(np.float32), y
