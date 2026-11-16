
import numpy as np
import joblib

from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from pathlib import Path

# Paths
matrix_path = Path("models/user_movie_matrix.npz")
output_path = Path("models")

print("Loading sparse user-movie matrix...")

matrix = load_npz(matrix_path)

print("Matrix shape:", matrix.shape)
print("Training ratings:", matrix.nnz)

# Initialize SVD
print("\nInitializing Truncated SVD...")

svd = TruncatedSVD(
    n_components=50,
    n_iter=5,
    random_state=42
)

# Train model
print("\nTraining SVD model...")

user_factors = svd.fit_transform(matrix)

movie_factors = svd.components_.T

# Save trained model
joblib.dump(
    svd,
    output_path / "svd_recommendation_model.pkl"
)

np.save(
    output_path / "user_factors.npy",
    user_factors
)

np.save(
    output_path / "movie_factors.npy",
    movie_factors
)

# Results
print("\n===== SVD TRAINING RESULTS =====")

print("User factors shape:", user_factors.shape)
print("Movie factors shape:", movie_factors.shape)

print(
    "Total explained variance:",
    round(svd.explained_variance_ratio_.sum(), 4)
)

print("\nSVD model trained and saved successfully!")