
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from pathlib import Path

# Project paths
model_path = Path("models")
data_path = Path("data/raw/ml-25m")

# Load trained factors and mappings
print("Loading trained recommendation model...")

user_factors = np.load(model_path / "user_factors.npy")
movie_factors = np.load(model_path / "movie_factors.npy")

user_ids = np.load(model_path / "user_ids.npy")
movie_ids = np.load(model_path / "movie_ids.npy")

matrix = load_npz(model_path / "user_movie_matrix.npz")

movies = pd.read_csv(data_path / "movies.csv")

# Select a user
target_user_id = 1

# Find user index
user_index = np.searchsorted(user_ids, target_user_id)

if user_index >= len(user_ids) or user_ids[user_index] != target_user_id:
    raise ValueError("User ID not found in training data.")

print("Generating recommendations for User:", target_user_id)

# Calculate recommendation scores
scores = user_factors[user_index] @ movie_factors.T

# Find movies already rated by this user
rated_indices = matrix.indices[
    matrix.indptr[user_index]:matrix.indptr[user_index + 1]
]

# Exclude already rated movies
scores[rated_indices] = -np.inf

# Get top 10 recommendations
top_indices = np.argsort(scores)[-10:][::-1]

# Map movie IDs to titles
recommendations = pd.DataFrame({
    "movieId": movie_ids[top_indices],
    "svd_score": scores[top_indices]
})

recommendations = recommendations.merge(
    movies[["movieId", "title", "genres"]],
    on="movieId",
    how="left"
)

print("\n===== PERSONALIZED RECOMMENDATIONS =====")

print(
    recommendations[
        ["movieId", "title", "genres", "svd_score"]
    ].to_string(index=False)
)

print("\nRecommendations generated successfully!")