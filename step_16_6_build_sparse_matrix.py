
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, save_npz
from pathlib import Path

# File paths
data_path = Path("data/processed")
train_file = data_path / "ratings_train.csv"

# Load training ratings using memory-efficient data types
print("Loading training ratings...")

ratings = pd.read_csv(
    train_file,
    usecols=["userId", "movieId", "rating"],
    dtype={
        "userId": np.int32,
        "movieId": np.int32,
        "rating": np.float32
    }
)

print("Training data loaded:", len(ratings))

# Create user and movie mappings
print("\nCreating user and movie mappings...")

user_codes, user_ids = pd.factorize(
    ratings["userId"],
    sort=True
)

movie_codes, movie_ids = pd.factorize(
    ratings["movieId"],
    sort=True
)

# Build sparse matrix
print("\nBuilding sparse user-movie matrix...")

matrix = csr_matrix(
    (
        ratings["rating"].to_numpy(),
        (user_codes, movie_codes)
    ),
    shape=(len(user_ids), len(movie_ids)),
    dtype=np.float32
)

# Prepare output directory
matrix_path = Path("models")
matrix_path.mkdir(exist_ok=True)

# Save matrix and mappings
save_npz(
    matrix_path / "user_movie_matrix.npz",
    matrix
)

np.save(
    matrix_path / "user_ids.npy",
    user_ids
)

np.save(
    matrix_path / "movie_ids.npy",
    movie_ids
)

# Display results
print("\n===== SPARSE MATRIX RESULTS =====")

print("Matrix shape:", matrix.shape)
print("Number of users:", len(user_ids))
print("Number of movies:", len(movie_ids))
print("Non-zero ratings:", matrix.nnz)

print(
    "Matrix density:",
    round(
        matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100,
        4
    ),
    "%"
)

print(
    "Matrix memory:",
    round(
        (matrix.data.nbytes +
         matrix.indices.nbytes +
         matrix.indptr.nbytes) / (1024 ** 2),
        2
    ),
    "MB"
)

print("\nSparse matrix created and saved successfully!")