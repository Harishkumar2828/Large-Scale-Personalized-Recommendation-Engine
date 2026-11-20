import numpy as np
import joblib

from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from pathlib import Path

# Paths
model_path = Path("models/temporal")

matrix_file = model_path / "user_movie_matrix.npz"

print("Loading temporal training matrix...")

matrix = load_npz(matrix_file)

print("Matrix shape:", matrix.shape)

# Train SVD
print("\nTraining Truncated SVD...")

svd = TruncatedSVD(
    n_components=50,
    n_iter=5,
    random_state=42
)

user_factors = svd.fit_transform(matrix)
movie_factors = svd.components_.T

# Save temporal model artifacts
joblib.dump(
    svd,
    model_path / "svd_recommendation_model.pkl"
)

np.save(model_path / "user_factors.npy", user_factors)
np.save(model_path / "movie_factors.npy", movie_factors)

print("\n===== TEMPORAL SVD TRAINING COMPLETE =====")
print("User factors:", user_factors.shape)
print("Movie factors:", movie_factors.shape)
print(
    "Explained variance:",
    round(svd.explained_variance_ratio_.sum(), 4)
)
print("Model saved to:", model_path)