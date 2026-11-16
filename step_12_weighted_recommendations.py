import pandas as pd
from pathlib import Path

data_path = Path("data/raw/ml-25m")

movies = pd.read_csv(data_path / "movies.csv")

all_stats = []

for chunk in pd.read_csv(
    data_path / "ratings.csv",
    chunksize=1_000_000
):
    stats = chunk.groupby("movieId")["rating"].agg(
        ["count", "sum"]
    )

    all_stats.append(stats)

movie_stats = pd.concat(all_stats)

movie_stats = movie_stats.groupby(level=0).sum()

movie_stats["avg_rating"] = (
    movie_stats["sum"] / movie_stats["count"]
)

# Global average rating
C = 3.53

# Minimum number of ratings
m = 1000

# Calculate weighted rating
movie_stats["weighted_rating"] = (
    (movie_stats["count"] /
     (movie_stats["count"] + m))
    * movie_stats["avg_rating"]
    +
    (m /
     (movie_stats["count"] + m))
    * C
)

movie_stats = movie_stats.reset_index()

movie_stats = movie_stats.merge(
    movies[["movieId", "title"]],
    on="movieId",
    how="left"
)

# Filter movies with at least 1000 ratings
popular_movies = movie_stats[
    movie_stats["count"] >= m
]

# Rank by weighted rating
top_movies = popular_movies.sort_values(
    "weighted_rating",
    ascending=False
)

print("\n===== TOP 20 WEIGHTED RECOMMENDATIONS =====")

print(
    top_movies[
        ["title", "count", "avg_rating", "weighted_rating"]
    ].head(20).to_string(index=False)
)

# Save results
output_path = Path("data/processed")
output_path.mkdir(parents=True, exist_ok=True)

top_movies.to_csv(
    output_path / "weighted_movie_recommendations.csv",
    index=False
)

print("\nRecommendations saved successfully!")