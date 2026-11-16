import pandas as pd

from pathlib import Path

data_path = Path("data/raw/ml-25m")

# Load movie information
movies = pd.read_csv(data_path / "movies.csv")

# Initialize statistics
total_ratings = 0
unique_users = set()
unique_movies = set()
rating_sum = 0

# Process ratings in chunks
for chunk in pd.read_csv(
    data_path / "ratings.csv",
    chunksize=1_000_000
):
    total_ratings += len(chunk)

    unique_users.update(chunk["userId"].unique())
    unique_movies.update(chunk["movieId"].unique())

    rating_sum += chunk["rating"].sum()

# Calculate average rating
average_rating = rating_sum / total_ratings

# Display results
print("\n===== DATASET STATISTICS =====")

print("Total movies:", len(movies))
print("Total ratings:", total_ratings)
print("Unique users:", len(unique_users))
print("Unique rated movies:", len(unique_movies))
print("Average rating:", round(average_rating, 2))