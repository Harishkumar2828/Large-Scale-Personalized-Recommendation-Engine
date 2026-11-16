import pandas as pd
from pathlib import Path

data_path = Path("data/raw/ml-25m")

# Load movie details
movies = pd.read_csv(data_path / "movies.csv")

# Store statistics from each chunk
all_stats = []

# Process ratings in chunks
for chunk in pd.read_csv(
    data_path / "ratings.csv",
    chunksize=1_000_000
):
    stats = chunk.groupby("movieId")["rating"].agg(
        ["count", "sum"]
    )

    all_stats.append(stats)

# Combine chunk statistics
movie_stats = pd.concat(all_stats)

movie_stats = movie_stats.groupby(level=0).sum()

# Calculate average rating
movie_stats["avg_rating"] = (
    movie_stats["sum"] / movie_stats["count"]
)

# Reset index
movie_stats = movie_stats.reset_index()

# Attach movie titles
movie_stats = movie_stats.merge(
    movies[["movieId", "title"]],
    on="movieId",
    how="left"
)

# Consider movies with at least 1000 ratings
popular_movies = movie_stats[
    movie_stats["count"] >= 1000
]

# Sort by number of ratings
popular_movies = popular_movies.sort_values(
    by="count",
    ascending=False
)

print("\n===== TOP 20 MOST POPULAR MOVIES =====")

print(
    popular_movies[
        ["title", "count", "avg_rating"]
    ].head(20).to_string(index=False)
)

print("\nMovies with at least 1000 ratings:",
      len(popular_movies))