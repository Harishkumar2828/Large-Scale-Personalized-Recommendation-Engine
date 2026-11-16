
import numpy as np
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from scipy.sparse import load_npz

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "raw" / "ml-25m"

# Initialize FastAPI
app = FastAPI(
    title="Movie Recommendation Engine",
    description="Personalized movie recommendations using SVD",
    version="1.0.0"
)

# Load model artifacts
print("Loading recommendation model...")

user_factors = np.load(MODEL_DIR / "user_factors.npy")
movie_factors = np.load(MODEL_DIR / "movie_factors.npy")

user_ids = np.load(MODEL_DIR / "user_ids.npy")
movie_ids = np.load(MODEL_DIR / "movie_ids.npy")

train_matrix = load_npz(
    MODEL_DIR / "user_movie_matrix.npz"
)

movies = pd.read_csv(
    DATA_DIR / "movies.csv"
)

# Create movie lookup
movie_lookup = movies.set_index("movieId").to_dict("index")

print("Recommendation model loaded successfully!")


@app.get("/")
def home():
    return {
        "message": "Movie Recommendation Engine API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Truncated SVD",
        "users": len(user_ids),
        "movies": len(movie_ids)
    }


@app.get("/recommend/{user_id}")
def recommend(
    user_id: int,
    top_n: int = Query(default=10, ge=1, le=50)
):

    # Find user index
    user_index = np.searchsorted(user_ids, user_id)

    if user_index >= len(user_ids) or user_ids[user_index] != user_id:
        raise HTTPException(
            status_code=404,
            detail=f"User ID {user_id} not found in training data."
        )

    # Calculate recommendation scores
    scores = user_factors[user_index] @ movie_factors.T

    # Exclude previously rated movies
    rated_indices = train_matrix.indices[
        train_matrix.indptr[user_index]:
        train_matrix.indptr[user_index + 1]
    ]

    scores[rated_indices] = -np.inf

    # Get top recommendations
    top_indices = np.argpartition(
        scores, -top_n
    )[-top_n:]

    top_indices = top_indices[
        np.argsort(scores[top_indices])[::-1]
    ]

    recommendations = []

    for index in top_indices:

        movie_id = int(movie_ids[index])
        movie_info = movie_lookup.get(movie_id)

        if movie_info is None:
            continue

        recommendations.append({
            "movie_id": movie_id,
            "title": movie_info["title"],
            "genres": movie_info["genres"],
            "score": round(float(scores[index]), 4)
        })

    return {
        "user_id": user_id,
        "recommendations_count": len(recommendations),
        "recommendations": recommendations
    }