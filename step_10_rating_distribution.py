import pandas as pd
from pathlib import Path

data_path = Path("data/raw/ml-25m")

# Read ratings in chunks
rating_counts = {}

for chunk in pd.read_csv(
    data_path / "ratings.csv",
    chunksize=1_000_000
):
    counts = chunk["rating"].value_counts()

    for rating, count in counts.items():
        rating_counts[rating] = (
            rating_counts.get(rating, 0) + count
        )

# Display rating distribution
print("\n===== RATING DISTRIBUTION =====")

for rating in sorted(rating_counts):
    print(f"Rating {rating}: {rating_counts[rating]:,}")

# Calculate sparsity
total_users = 162541
total_movies = 59047
total_ratings = sum(rating_counts.values())

total_possible = total_users * total_movies

density = (total_ratings / total_possible) * 100
sparsity = 100 - density

print("\n===== MATRIX SPARSITY =====")
print("Matrix density:", round(density, 4), "%")
print("Matrix sparsity:", round(sparsity, 4), "%")