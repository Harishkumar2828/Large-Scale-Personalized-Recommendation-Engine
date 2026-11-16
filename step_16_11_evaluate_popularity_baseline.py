
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from pathlib import Path

model_path = Path("models")
data_path = Path("data/processed")

print("Loading training matrix and test data...")

matrix = load_npz(
    model_path / "user_movie_matrix.npz"
)

user_ids = np.load(model_path / "user_ids.npy")
movie_ids = np.load(model_path / "movie_ids.npy")

test = pd.read_csv(
    data_path / "ratings_test.csv",
    dtype={
        "userId": np.int32,
        "movieId": np.int32,
        "rating": np.float32
    }
)

# Relevant movies
relevant = test[test["rating"] >= 4.0].copy()

# Keep only movies known to the training model
positions = np.searchsorted(
    movie_ids,
    relevant["movieId"].to_numpy()
)

valid = positions < len(movie_ids)
valid_indices = np.flatnonzero(valid)

valid[valid_indices] = (
    movie_ids[positions[valid_indices]]
    == relevant["movieId"].to_numpy()[valid_indices]
)

relevant = relevant.loc[valid].copy()

eligible_users = relevant["userId"].unique()

rng = np.random.default_rng(42)

sampled_users = rng.choice(
    eligible_users,
    size=min(500, len(eligible_users)),
    replace=False
)

relevant_by_user = (
    relevant[relevant["userId"].isin(sampled_users)]
    .groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

# Calculate movie popularity using training ratings
print("Calculating movie popularity...")

binary_matrix = matrix.copy()
binary_matrix.data = np.ones(
    binary_matrix.nnz,
    dtype=np.int32
)

popularity = np.asarray(
    binary_matrix.tocsc().sum(axis=0)
).ravel()

K = 10

precision_scores = []
recall_scores = []
ndcg_scores = []

print("Evaluating popularity recommendations...")

for user_id in sampled_users:

    user_index = np.searchsorted(user_ids, user_id)

    if user_index >= len(user_ids):
        continue

    if user_ids[user_index] != user_id:
        continue

    scores = popularity.astype(np.float64)

    # Exclude movies already rated by the user
    rated_indices = matrix.indices[
        matrix.indptr[user_index]:
        matrix.indptr[user_index + 1]
    ]

    scores[rated_indices] = -np.inf

    top_indices = np.argsort(scores)[-K:][::-1]

    recommended_movies = movie_ids[top_indices]

    relevant_movies = relevant_by_user[user_id]

    hits = [
        int(movie_id in relevant_movies)
        for movie_id in recommended_movies
    ]

    hit_count = sum(hits)

    precision_scores.append(hit_count / K)

    recall_scores.append(
        hit_count / len(relevant_movies)
    )

    dcg = sum(
        hit / np.log2(rank + 2)
        for rank, hit in enumerate(hits)
    )

    ideal_hits = min(len(relevant_movies), K)

    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    ndcg_scores.append(dcg / idcg)

print("\n===== POPULARITY BASELINE RESULTS =====")

print("Evaluated users:", len(precision_scores))
print("Precision@10:", round(np.mean(precision_scores), 4))
print("Recall@10:", round(np.mean(recall_scores), 4))
print("NDCG@10:", round(np.mean(ndcg_scores), 4))

print("\nBaseline evaluation completed successfully!")