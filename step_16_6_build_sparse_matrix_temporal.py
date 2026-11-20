
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, save_npz
from pathlib import Path

# Paths
data_path = Path("data/processed")
output_path = Path("models/temporal")

output_path.mkdir(parents=True, exist_ok=True)

train_file = data_path / "ratings_train_temporal.csv"

print("Loading temporal training ratings...")

ratings = pd.read_csv(
    train_file,
    usecols=["userId", "movieId", "rating"],
    dtype={
        "userId": np.int32,
        "movieId": np.int32,
        "rating": np.float32
    }
)

print("Training ratings:", f"{len(ratings):,}")

# Create user and movie indexes
user_codes, user_ids = pd.factorize(
    ratings["userId"],
    sort=True
)

movie_codes, movie_ids = pd.factorize(
    ratings["movieId"],
    sort=True
)

# Build sparse user-movie matrix
print("Building sparse matrix...")

matrix = csr_matrix(
    (
        ratings["rating"].to_numpy(),
        (user_codes, movie_codes)
    ),
    shape=(len(user_ids), len(movie_ids)),
    dtype=np.float32
)

# Save temporal artifacts separately
save_npz(
    output_path / "user_movie_matrix.npz",
    matrix
)

np.save(output_path / "user_ids.npy", user_ids)
np.save(output_path / "movie_ids.npy", movie_ids)

print("\n===== TEMPORAL MATRIX COMPLETE =====")
print("Matrix shape:", matrix.shape)
print("Non-zero ratings:", f"{matrix.nnz:,}")
print("Users:", len(user_ids))
print("Movies:", len(movie_ids))
print("Saved to:", output_path)
