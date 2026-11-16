\# Large-Scale Personalized Recommendation Engine



\## Project Overview



A personalized movie recommendation system built using the MovieLens 25M dataset. The project uses collaborative filtering with Truncated Singular Value Decomposition (SVD) to generate personalized movie recommendations for users.



A popularity-based recommendation model is also implemented as a baseline for performance comparison.



\## Dataset



\*\*MovieLens 25M Dataset\*\*



\- Total ratings: 25,000,095

\- Unique users: 162,541

\- Total movies: 62,423

\- Average rating: 3.53



Dataset source: https://grouplens.org/datasets/movielens/25m/



\## Technology Stack



\- Python 3.11

\- NumPy and Pandas

\- SciPy

\- Scikit-learn

\- Truncated SVD

\- FastAPI

\- Uvicorn



\## Model Development



1\. Dataset exploration and statistical analysis.

2\. Rating distribution and movie popularity analysis.

3\. Weighted movie recommendation baseline.

4\. 80:20 training and testing split.

5\. Sparse user-item matrix construction using SciPy.

6\. Truncated SVD model training with 50 latent components.

7\. Personalized top-10 recommendation generation.

8\. Evaluation against a popularity-based baseline.

9\. FastAPI integration for serving recommendations.



\## Model Evaluation



Offline evaluation on a sample of 500 users:



| Metric | SVD Model | Popularity Baseline |

|---|---:|---:|

| Precision@10 | 0.2250 | 0.1200 |

| Recall@10 | 0.2085 | 0.1061 |

| NDCG@10 | 0.3065 | 0.1635 |



The SVD model outperformed the popularity baseline on all three evaluation metrics in this experiment.



\## API Features



\- API health monitoring.

\- Personalized movie recommendations by user ID.

\- Configurable recommendation count (1–50).

\- Movie titles, genres, and recommendation scores.

\- HTTP 404 handling for unknown users.

\- Interactive Swagger API documentation.



\## Running the Project



\### 1. Clone the repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd Project\_2\_Recommendation\_Engine

```



\### 2. Create and activate a virtual environment



```bash

python -m venv .venv

.venv\\Scripts\\activate

```



\### 3. Install dependencies



```bash

pip install numpy pandas scipy scikit-learn joblib fastapi uvicorn

```



\### 4. Prepare the dataset and model artifacts



Download and extract MovieLens 25M into:



`data/raw/ml-25m/`



Run the dataset preparation, sparse matrix construction, and model training scripts in the appropriate order.



\### 5. Start the API



```bash

python -m uvicorn api.app:app --reload

```



\### 6. Open Swagger documentation



`http://127.0.0.1:8000/docs`



\## Documentation



\- `docs/model\_comparison\_report.md` – Model evaluation and baseline comparison.

\- `docs/api\_documentation.md` – API endpoints and usage.



\## Important Notes



\- The model was evaluated offline using a sample of 500 users.

\- The train/test split was performed at the rating-row level.

\- Recommendation scores are ranking scores, not probabilities or predicted user ratings.

\- The project uses local model artifacts and is not currently deployed to a public server.



\## Project Status



Core recommendation model, offline evaluation, and FastAPI recommendation endpoints have been implemented and tested.

