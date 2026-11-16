
import pandas as pd
import numpy as np
from pathlib import Path

# Dataset paths
data_path = Path("data/raw/ml-25m/ratings.csv")
output_path = Path("data/processed")

output_path.mkdir(parents=True, exist_ok=True)

train_file = output_path / "ratings_train.csv"
test_file = output_path / "ratings_test.csv"

# Reproducible random splitting
rng = np.random.default_rng(42)

train_count = 0
test_count = 0

# Remove previous outputs if they exist
train_file.unlink(missing_ok=True)
test_file.unlink(missing_ok=True)

# Process ratings in chunks
for chunk in pd.read_csv(data_path, chunksize=1_000_000):

    # Randomly assign 80% to training
    mask = rng.random(len(chunk)) < 0.8

    train_data = chunk[mask]
    test_data = chunk[~mask]

    # Save training data
    train_data.to_csv(
        train_file,
        mode="a",
        header=(train_count == 0),
        index=False
    )

    # Save testing data
    test_data.to_csv(
        test_file,
        mode="a",
        header=(test_count == 0),
        index=False
    )

    train_count += len(train_data)
    test_count += len(test_data)

    print(
        f"Processed: {train_count + test_count:,} ratings"
    )

print("\n===== TRAIN TEST SPLIT RESULTS =====")
print("Training ratings:", f"{train_count:,}")
print("Testing ratings:", f"{test_count:,}")
print("Total ratings:", f"{train_count + test_count:,}")

print("\nDataset splitting completed successfully!")