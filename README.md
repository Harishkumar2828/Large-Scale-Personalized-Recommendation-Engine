# Large-Scale Personalized Recommendation Engine

## Project Overview

A personalized movie recommendation system built using the MovieLens 25M dataset. The project uses collaborative filtering with Truncated Singular Value Decomposition (SVD) to generate personalized movie recommendations for users.

A popularity-based recommendation model is also implemented as a baseline for performance comparison.

## Dataset

**MovieLens 25M Dataset**

- Total ratings: 25,000,095
- Unique users: 162,541
- Total movies: 62,423
- Average rating: 3.53

Dataset source: https://grouplens.org/datasets/movielens/25m/

## Technology Stack

- Python 3.11
- NumPy and Pandas
- SciPy
- Scikit-learn
- Truncated SVD
- FastAPI
- Uvicorn

## Model Development

1. Dataset exploration and statistical analysis.
2. Rating distribution and movie popularity analysis.
3. Weighted movie recommendation baseline.
4. Random 80:20 rating-level split for the original model.
5. Chronological 80:20 train/test split for temporal evaluation.
6. Sparse user-item matrix construction using SciPy.
7. Truncated SVD model training with 50 latent components.
8. Personalized top-10 recommendation generation.
9. Evaluation against a popularity-based baseline.
10. FastAPI integration for serving recommendations.

## Model Evaluation

### Temporal Evaluation

A chronological split was introduced to evaluate future-interaction recommendation performance.

- Training ratings: 20,000,076 (80%)
- Testing ratings: 5,000,019 (20%)
- Timestamp cutoff: 2016-06-25 06:49:57 UTC
- Relevant rating threshold: 4.0
- Evaluation sample: 500 users
- Eligible warm-start users: 6,181

**Temporal evaluation results:**

| Metric | SVD Model | Popularity Baseline |
|---|---:|---:|
| Precision@10 | 0.1654 | 0.0842 |
| Recall@10 | 0.0631 | 0.0303 |
| NDCG@10 | 0.1935 | 0.1022 |

The temporal SVD model outperformed the popularity baseline across all three metrics.

### Historical Random-Split Evaluation

The original random rating-level evaluation produced these results:

| Metric | SVD Model | Popularity Baseline |
|---|---:|---:|
| Precision@10 | 0.2250 | 0.1200 |
| Recall@10 | 0.2085 | 0.1061 |
| NDCG@10 | 0.3065 | 0.1635 |

These results are retained for historical reference and are not directly comparable with the temporal evaluation.

## API Features

- API health monitoring.
- Personalized movie recommendations by user ID.
- Configurable recommendation count (1–50).
- Movie titles, genres, and recommendation scores.
- HTTP 404 handling for unknown users.
- Interactive Swagger API documentation.

The currently configured FastAPI service uses the original model artifacts stored in `models/`. The temporal SVD artifacts are stored separately in `models/temporal/` and have been evaluated offline.

## Running the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Project_2_Recommendation_Engine
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install numpy pandas scipy scikit-learn joblib fastapi uvicorn
```

### 4. Prepare the dataset

Download and extract MovieLens 25M into:

`data/raw/ml-25m/`

### 5. Reproduce the original model and API artifacts

Run the following scripts from the project root directory:

**1. Create the original random training and testing split**

```cmd
python step_16_4_train_test_split_random_backup.py
```

**2. Build the sparse user-item matrix**

```cmd
python step_16_6_build_sparse_matrix.py
```

**3. Train the Truncated SVD model**

```cmd
python step_16_8_train_svd_model.py
```

**4. Generate recommendations**

```cmd
python step_16_9_generate_recommendations.py
```

**5. Evaluate the SVD model**

```cmd
python step_16_10_evaluate_recommendations.py
```

**6. Evaluate the popularity baseline**

```cmd
python step_16_11_evaluate_popularity_baseline.py
```

These scripts generate the original model artifacts used by the current API.

### 6. Reproduce the temporal evaluation

The temporal pipeline uses separate files and artifacts so that the original model remains unchanged.

Run the scripts in this order:

```cmd
python step_16_4_train_test_split.py
python step_16_6_build_sparse_matrix_temporal.py
python step_16_8_train_svd_model_temporal.py
python step_16_10_evaluate_recommendations_temporal.py
python step_16_11_evaluate_popularity_temporal.py
```

The temporal model artifacts are saved in:

`models/temporal/`

### 7. Start the API

```bash
python -m uvicorn api.app:app --reload
```

### 8. Open Swagger documentation

`http://127.0.0.1:8000/docs`

## Documentation

- `docs/model_comparison_report.md` – Temporal and historical model evaluation and baseline comparison.
- `docs/api_documentation.md` – API endpoints and usage.

## Important Notes and Limitations

- Evaluation uses a reproducible sample of 500 users.
- Temporal SVD evaluation includes warm-start users and movies known during training.
- Unseen users and movies are excluded from the temporal SVD evaluation.
- The relevance threshold is set to 4.0.
- Recommendation scores are ranking scores, not probabilities or predicted user ratings.
- The project uses local model artifacts and is not currently deployed to a public server.
- Results represent offline evaluation and do not establish real-world production performance.

## Future Improvements

- Apache Spark-based distributed data processing.
- TensorFlow Recommenders two-tower retrieval architecture.
- FAISS-based approximate nearest-neighbor retrieval.
- Deep-learning ranking model.
- Cold-start recommendation strategies.
- Redis caching.
- Triton model serving.
- Airflow-based model retraining orchestration.
- Online A/B testing and monitoring dashboard.

## Project Status

The core recommendation model, popularity baseline, temporal evaluation pipeline, model comparison documentation, and FastAPI recommendation endpoints have been implemented and tested.

The advanced distributed retrieval, deep ranking, caching, orchestration, and online experimentation components remain future improvements.