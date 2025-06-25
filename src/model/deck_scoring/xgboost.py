import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.utils.s3_utils import read_parquet_from_s3

S3_BUCKET = "yugioh-data"
S3_KEY = "processed/feature_engineered/deck_scoring/2025-06/feature_engineered.parquet"

# === Load your feature-enriched dataset ===
deck_df = read_parquet_from_s3(S3_BUCKET, S3_KEY).to_pandas()

# Split the embedding column into separate columns
embedding_df = pd.DataFrame(
    deck_df['main_deck_mean_embedding'].to_list(),
    index=deck_df.index
).add_prefix("embedding_")

# Drop the original embedding column and concatenate the expanded version
deck_df = pd.concat([deck_df.drop(columns=['main_deck_mean_embedding']), embedding_df], axis=1)

# Add all embedding dimensions to your feature list
embedding_cols = [col for col in deck_df.columns if col.startswith("embedding_")]
bow_cols = [col for col in deck_df.columns if col.startswith("bow_")]
feature_cols = [
    'has_tuner', 'num_tuners', 'has_same_level_monsters', 'max_same_level_count',
    'avg_monster_level', 'has_pendulum_monsters', 'has_synchro_monsters',
    'has_xyz_monsters', 'has_link_monsters', 'max_copies_per_card',
    'avg_copies_per_monster', 'num_unique_monsters', 'main_deck_mean_tfidf',
    'mentions_banish', 'mentions_graveyard', 'mentions_draw', 'mentions_search',
    'mentions_special_summon', 'mentions_negate', 'mentions_destroy', 'mentions_shuffle'
] + embedding_cols + bow_cols

X = deck_df[feature_cols]

# 1. Separate feature groups
dense_cols = [col for col in X.columns if col not in bow_cols and col not in embedding_cols]

X_bow = X[bow_cols]
X_dense = X[dense_cols]

# 2. Reduce BoW with SVD
svd = TruncatedSVD(n_components=50, random_state=42)
X_bow_reduced = svd.fit_transform(X_bow)

# 3. (Optional) Scale dense features
scaler = StandardScaler()
X_dense_scaled = scaler.fit_transform(X_dense)

embedding_matrix = np.vstack(deck_df[embedding_cols].values)
embedding_svd = TruncatedSVD(n_components=20, random_state=42)
embedding_reduced = embedding_svd.fit_transform(embedding_matrix)

# 4. Concatenate reduced BoW and dense features
X_final = np.hstack([X_dense_scaled, X_bow_reduced, embedding_reduced])

y = deck_df['composite_score']

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# === Train XGBoost model ===
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# === Plot feature importances ===

dense_feature_names = X_dense.columns.tolist()
bow_feature_names = [f"svd_bow_{i}" for i in range(X_bow_reduced.shape[1])]
embedding_feature_names = [f"svd_embed_{i}" for i in range(embedding_reduced.shape[1])]
final_feature_names = dense_feature_names + bow_feature_names + embedding_feature_names

X_final_df = pd.DataFrame(X_final, columns=final_feature_names)
X_final_df.index = X.index

model.get_booster().feature_names = X_final_df.columns.tolist()

fig, ax = plt.subplots()
xgb.plot_importance(model, max_num_features=20)
plt.tight_layout()
plt.title("Top Feature Importances")
plt.savefig("src/model/deck_scoring/xgb_importance.png", dpi=300, bbox_inches="tight")
