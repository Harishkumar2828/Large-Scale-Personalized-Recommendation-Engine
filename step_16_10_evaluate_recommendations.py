
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from pathlib import Path

# Paths
model_path = Path("models")
data_path = Path("data/processed")

print("Loading model artifacts...")

user_factors = np.load(model_path / "user_factors.npy")
movie_factors = np.load(model_path / "movie_factors.npy")

user_ids = np.load(model_path / "user_ids.npy")
movie_ids = np.load(model_path / "movie_ids.npy")

train_matrix = load_npz(
    model_path / "user_movie_matrix.npz"
)

# Load test data
print("Loading test ratings...")

test = pd.read_csv(
    data_path / "ratings_test.csv",
    dtype={
        "userId": np.int32,
        "movieId": np.int32,
        "rating": np.float32
    }
)

# Relevant means rating >= 4.0
relevant = test[test["rating"] >= 4.0].copy()

# Keep only movies known to the trained model
movie_positions = np.searchsorted(
    movie_ids,
    relevant["movieId"].to_numpy()
)

valid = movie_positions < len(movie_ids)

valid_indices = np.flatnonzero(valid)

valid[valid_indices] = (
    movie_ids[movie_positions[valid_indices]]
    == relevant["movieId"].to_numpy()[valid_indices]
)

relevant = relevant.loc[valid].copy()

# Select users with relevant test movies
eligible_users = relevant["userId"].unique()

rng = np.random.default_rng(42)

sample_size = min(500, len(eligible_users))

sampled_users = rng.choice(
    eligible_users,
    size=sample_size,
    replace=False
)

sampled_relevant = relevant[
    relevant["userId"].isin(sampled_users)
]

relevant_by_user = (
    sampled_relevant.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

# Evaluation
precision_scores = []
recall_scores = []
ndcg_scores = []

K = 10

print("\nEvaluating recommendations...")

for user_id in sampled_users:

    user_index = np.searchsorted(user_ids, user_id)

    if user_index >= len(user_ids):
        continue

    if user_ids[user_index] != user_id:
        continue

    # Calculate scores
    scores = (
        user_factors[user_index] @ movie_factors.T
    )

    # Exclude movies already rated in training
    rated_indices = train_matrix.indices[
        train_matrix.indptr[user_index]:
        train_matrix.indptr[user_index + 1]
    ]

    scores[rated_indices] = -np.inf

    # Find top 10 recommendations
    top_indices = np.argpartition(
        scores, -K
    )[-K:]

    top_indices = top_indices[
        np.argsort(scores[top_indices])[::-1]
    ]

    recommended_movies = set(movie_ids[top_indices])

    # Relevant test movies
    relevant_movies = relevant_by_user[user_id]

    # Calculate hits
    hits = [
        1 if movie_id in relevant_movies else 0
        for movie_id in movie_ids[top_indices]
    ]

    hit_count = sum(hits)

    precision_scores.append(hit_count / K)

    recall_scores.append(
        hit_count / len(relevant_movies)
    )

    # Calculate NDCG
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

# Final results
print("\n===== RECOMMENDATION EVALUATION =====")

print("Evaluated users:", len(precision_scores))

print(
    "Precision@10:",
    round(np.mean(precision_scores), 4)
)

print(
    "Recall@10:",
    round(np.mean(recall_scores), 4)
)

print(
    "NDCG@10:",
    round(np.mean(ndcg_scores), 4)
)

print("\nEvaluation completed successfully!")