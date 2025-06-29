from torch.utils.data import Dataset
import torch
from src.utils.s3_utils import read_parquet_from_s3
from src.preprocessing.deck_preprocessor.deck_preprocessor import DeckPreprocessor

S3_BUCKET = "yugioh-data"
S3_KEY = "processed/feature_engineered/deck_scoring/2025-06/feature_engineered.parquet"

class YugiohDeckDataset(Dataset):
    def __init__(self, use_pca: bool = False):
        super().__init__()
        df = read_parquet_from_s3(S3_BUCKET, S3_KEY).to_pandas()
        method = "pca" if use_pca else "svd"
        self.X, self.y = DeckPreprocessor(method=method).fit_transform(df)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

