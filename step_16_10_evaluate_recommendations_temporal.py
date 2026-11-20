import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from pathlib import Path

# Paths
model_path = Path("models/temporal")
data_path = Path("data/processed")

print("Loading temporal model artifacts...")

user_factors = np.load(model_path / "user_factors.npy")
movie_factors = np.load(model_path / "movie_factors.npy")

user_ids = np.load(model_path / "user_ids.npy")
movie_ids = np.load(model_path / "movie_ids.npy")

train_matrix = load_npz(
    model_path / "user_movie_matrix.npz"
)

print("Loading temporal test ratings...")

test = pd.read_csv(
    data_path / "ratings_test_temporal.csv",
    usecols=["userId", "movieId", "rating"],
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

# Keep only users known to the trained model
known_users = np.isin(
    relevant["userId"].to_numpy(),
    user_ids
)

relevant = relevant.loc[known_users].copy()

# Group relevant movies by user
relevant_by_user = (
    relevant.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

eligible_users = np.array(list(relevant_by_user.keys()))

# Reproducible sample of up to 500 users
rng = np.random.default_rng(42)
sample_size = min(500, len(eligible_users))

sampled_users = rng.choice(
    eligible_users,
    size=sample_size,
    replace=False
)

# Evaluation
precision_scores = []
recall_scores = []
ndcg_scores = []

K = 10

print("\nEvaluating temporal recommendations...")

for user_id in sampled_users:

    user_index = np.searchsorted(user_ids, user_id)

    scores = user_factors[user_index] @ movie_factors.T

    # Exclude movies rated during training
    rated_indices = train_matrix.indices[
        train_matrix.indptr[user_index]:
        train_matrix.indptr[user_index + 1]
    ]

    scores[rated_indices] = -np.inf

    # Top K recommendations
    top_indices = np.argpartition(
        scores, -K
    )[-K:]

    top_indices = top_indices[
        np.argsort(scores[top_indices])[::-1]
    ]

    recommended_ids = movie_ids[top_indices]
    relevant_movies = relevant_by_user[user_id]

    hits = [
        1 if movie_id in relevant_movies else 0
        for movie_id in recommended_ids
    ]

    hit_count = sum(hits)

    precision_scores.append(hit_count / K)
    recall_scores.append(hit_count / len(relevant_movies))

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

print("\n===== TEMPORAL EVALUATION RESULTS =====")

print("Test ratings:", f"{len(test):,}")
print("Relevant test ratings:", f"{len(test[test.rating >= 4.0]):,}")
print("Relevant ratings for known movies and users:", f"{len(relevant):,}")
print("Eligible users:", len(eligible_users))
print("Evaluated users:", len(precision_scores))

print("Precision@10:", round(np.mean(precision_scores), 4))
print("Recall@10:", round(np.mean(recall_scores), 4))
print("NDCG@10:", round(np.mean(ndcg_scores), 4))

print("\nTemporal evaluation completed!")