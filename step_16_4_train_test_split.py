import pandas as pd
from pathlib import Path

# Input and output paths
data_path = Path("data/raw/ml-25m/ratings.csv")
output_path = Path("data/processed")
output_path.mkdir(parents=True, exist_ok=True)

# Unix timestamp for 2016-06-25 06:49:57 UTC
cutoff_timestamp = 1466837397

# Separate output files: preserve the original random split
train_file = output_path / "ratings_train_temporal.csv"
test_file = output_path / "ratings_test_temporal.csv"

# Remove previous temporal outputs if they exist
train_file.unlink(missing_ok=True)
test_file.unlink(missing_ok=True)

train_count = 0
test_count = 0

print("Creating chronological train/test split...")

for chunk in pd.read_csv(
    data_path,
    chunksize=1_000_000,
    dtype={
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
        "timestamp": "int64"
    }
):
    # Earlier interactions go to training
    train_data = chunk[
        chunk["timestamp"] <= cutoff_timestamp
    ]

    # Later interactions go to testing
    test_data = chunk[
        chunk["timestamp"] > cutoff_timestamp
    ]

    train_data.to_csv(
        train_file,
        mode="a",
        header=(train_count == 0),
        index=False
    )

    test_data.to_csv(
        test_file,
        mode="a",
        header=(test_count == 0),
        index=False
    )

    train_count += len(train_data)
    test_count += len(test_data)

    print(
        f"Processed: {train_count + test_count:,} | "
        f"Train: {train_count:,} | Test: {test_count:,}"
    )

print("\n===== TEMPORAL SPLIT COMPLETE =====")
print("Training ratings:", f"{train_count:,}")
print("Testing ratings:", f"{test_count:,}")
print("Total ratings:", f"{train_count + test_count:,}")
print("Cutoff: 2016-06-25 06:49:57 UTC")