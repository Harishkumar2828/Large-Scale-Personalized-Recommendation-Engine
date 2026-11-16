\# Movie Recommendation Engine – API Documentation



\## 1. Project Overview



This API provides personalized movie recommendations using a trained Truncated SVD recommendation model.



\*\*Technology Stack:\*\*

\- Python 3.11

\- FastAPI

\- Uvicorn

\- NumPy

\- Pandas

\- SciPy

\- Scikit-learn



\## 2. Model Information



\- Algorithm: Truncated Singular Value Decomposition (SVD)

\- Latent factors: 50

\- Dataset: MovieLens 25M

\- Training ratings: 20,001,483

\- Users: 162,541

\- Movies in training matrix: 56,722



\## 3. API Endpoints



\### GET /



Checks whether the API is running.



\### GET /health



Returns API health and loaded model information.



\### GET /recommend/{user\_id}



Generates personalized movie recommendations for a given user.



\*\*Parameters:\*\*

\- `user\_id`: User ID from the training dataset.

\- `top\_n`: Number of recommendations (1–50, default 10).



\*\*Example:\*\*



`/recommend/1?top\_n=10`



Returns movie IDs, titles, genres, and recommendation scores.



\## 4. Error Handling



If a user ID is not present in the training data, the API returns HTTP 404.



Example:



`{"detail": "User ID 999999 not found in training data."}`



\## 5. Running the API



Activate the virtual environment and execute:



`python -m uvicorn api.app:app --reload`



API URL: `http://127.0.0.1:8000`



Swagger documentation: `http://127.0.0.1:8000/docs`



\## 6. Testing Results



\- API home endpoint: Passed

\- Health endpoint: Passed

\- Personalized recommendations: Passed

\- Invalid user handling: Passed



\## 7. Recommendation Evaluation



| Metric | SVD Model | Popularity Baseline |

|---|---:|---:|

| Precision@10 | 0.2250 | 0.1200 |

| Recall@10 | 0.2085 | 0.1061 |

| NDCG@10 | 0.3065 | 0.1635 |



Evaluation was performed on a sample of 500 users.



These are offline evaluation results, not a guarantee of production performance.

