from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from sklearn.model_selection import GroupKFold
try:
    # Available in scikit-learn >= 1.3
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
    HAS_SGK = True
except Exception:
    HAS_SGK = False

from src.utils.s3_utils import read_parquet_from_s3
from src.preprocessing.deck_preprocessor.deck_preprocessor import DeckPreprocessor

S3_BUCKET = "yugioh-data"
S3_KEY = "processed/feature_engineered/deck_scoring/2025-08/feature_engineered.parquet"


class YugiohDeckDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
        # Ensure contiguous float32 tensors for speed
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.weights[idx], dtype=torch.float32),
        )


def _ensure_pandas(df_or_table) -> pd.DataFrame:
    if hasattr(df_or_table, "to_pandas"):
        return df_or_table.to_pandas()
    return df_or_table


def _clean_frame(df: pd.DataFrame,
                 target_col: str,
                 weight_col: Optional[str]) -> pd.DataFrame:
    # Coerce to numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    if weight_col is not None and weight_col in df.columns:
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    # Drop rows without target
    before = len(df)
    df = df.dropna(subset=[target_col]).copy()
    if weight_col is not None and weight_col in df.columns:
        df[weight_col] = df[weight_col].fillna(0)
    # Clip target into [0,1] just in case
    df[target_col] = df[target_col].clip(0.0, 1.0)
    return df


def _make_weights(arr: np.ndarray, cap_quantile: float = 0.99) -> np.ndarray:
    """Stabilize highly skewed battle counts with log1p + optional capping."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    # Cap extreme values to alleviate outliers
    cap = np.quantile(arr, cap_quantile) if arr.size > 0 else 0.0
    if cap > 0:
        arr = np.minimum(arr, cap)
    # Log1p compress
    w = np.log1p(arr)
    # Normalize median to 1.0 for stable loss scales
    med = np.median(w) if w.size > 0 else 1.0
    if med > 0:
        w = w / med
    # Floor to small epsilon to avoid zeroing gradients
    return np.maximum(w.astype(np.float32), 1e-3)


def create_cv_splits_no_leakage(
    use_pca: bool = False,
    n_splits: int = 5,
    target_col: str = "adjusted_win_rate",
    group_col: str = "deck_id",
    weight_col: str = "successful_battles",
    s3_bucket: str = S3_BUCKET,
    s3_key: str = S3_KEY,
) -> List[Tuple[YugiohDeckDataset, YugiohDeckDataset, DeckPreprocessor]]:
    """Create train/val splits with leakage prevention.

    - Uses GroupKFold to keep all examples of the same group (deck_id) in the same fold.
    - If available, uses StratifiedGroupKFold on binned targets for better target balance.
    - Fits DeckPreprocessor ONLY on training split, then transforms val.
    - Returns (train_dataset, val_dataset, fitted_preprocessor) per fold.
    """
    df_raw = read_parquet_from_s3(s3_bucket, s3_key)
    df = _ensure_pandas(df_raw)

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    # Fallback group key if missing
    if group_col not in df.columns:
        df[group_col] = (np.arange(len(df)) // 100).astype(str)

    df = _clean_frame(df, target_col=target_col, weight_col=weight_col)

    # Base weights
    if weight_col in df.columns:
        base_weights = _make_weights(df[weight_col].to_numpy())
    else:
        base_weights = np.ones(len(df), dtype=np.float32)

    # Prepare stratification bins if available
    y = df[target_col].to_numpy()
    groups = df[group_col].astype(str).to_numpy()

    # 10-bin quantiles for stratification
    try:
        y_bins = pd.qcut(y, q=min(10, max(2, int(np.sqrt(len(df))))), labels=False, duplicates="drop")
    except Exception:
        y_bins = None

    method = "pca" if use_pca else "svd"
    splits: List[Tuple[YugiohDeckDataset, YugiohDeckDataset, DeckPreprocessor]] = []

    if HAS_SGK and y_bins is not None and len(np.unique(groups)) >= n_splits:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(df, y_bins, groups)
    else:
        # GroupKFold fallback (no stratification, deterministic order)
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(df, y, groups)

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        print(f"Fold {fold}: Train={len(train_idx)}, Val={len(val_idx)}")

        df_train, df_val = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
        weights_train, weights_val = base_weights[train_idx], base_weights[val_idx]

        # Fit preprocessor ONLY on training data
        preprocessor = DeckPreprocessor(method=method, target_col=target_col)
        X_train, y_train = preprocessor.fit_transform(df_train)
        X_val, y_val = preprocessor.transform(df_val)

        # Final dtype normalization
        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
        y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)

        train_dataset = YugiohDeckDataset(X_train, y_train, weights_train)
        val_dataset = YugiohDeckDataset(X_val, y_val, weights_val)
        splits.append((train_dataset, val_dataset, preprocessor))

    return splits
