\# Recommendation Model Comparison Report



\## 1. Project Dataset



\*\*Dataset:\*\* MovieLens 25M



\- Total ratings: 25,000,095

\- Unique users in original dataset: 162,541

\- Total movies in original dataset: 62,423



\## 2. Models Evaluated



1\. Truncated Singular Value Decomposition (SVD)

2\. Popularity-Based Recommendation Baseline



\## 3. Evaluation Methodology



\### Temporal Train/Test Split



A chronological split was used to simulate recommendations for future user interactions.



\- Training split: 80%

\- Testing split: 20%

\- Training ratings: 20,000,076

\- Testing ratings: 5,000,019

\- Timestamp cutoff: 2016-06-25 06:49:57 UTC

\- Training data: Ratings at or before the cutoff

\- Testing data: Ratings after the cutoff



\### Evaluation Configuration



\- Evaluation users: 500 sampled users

\- Eligible warm-start users: 6,181

\- Relevant movie threshold: Rating >= 4.0

\- Recommendation cutoff: Top 10

\- Evaluation metrics: Precision@10, Recall@10 and NDCG@10



Only users and movies present in the temporal training data were included in the SVD evaluation. Therefore, the results represent warm-start recommendation performance.



\## 4. Temporal Evaluation Results



| Metric | Truncated SVD | Popularity |

|---|---:|---:|

| Precision@10 | 0.1654 | 0.0842 |

| Recall@10 | 0.0631 | 0.0303 |

| NDCG@10 | 0.1935 | 0.1022 |



Truncated SVD outperformed the popularity-based baseline across all three evaluation metrics.



\## 5. Historical Random-Split Results



The earlier random rating-level evaluation produced the following results:



| Metric | SVD | Popularity |

|---|---:|---:|

| Precision@10 | 0.2250 | 0.1200 |

| Recall@10 | 0.2085 | 0.1061 |

| NDCG@10 | 0.3065 | 0.1635 |



These results are retained for historical reference. They should not be directly compared with the temporal results because the training and testing methodologies differ.



\## 6. Conclusion



The Truncated SVD recommendation model outperformed the popularity-based baseline across Precision@10, Recall@10 and NDCG@10 in the chronological evaluation.



The results demonstrate that the SVD model captures useful user preference patterns and provides better personalized recommendations than a popularity-only approach for users and movies known during training.



\## 7. Limitations



\- Evaluation was performed on a reproducible sample of 500 users rather than all eligible users.

\- Only warm-start users and movies were evaluated.

\- Unseen users and movies were excluded from the SVD evaluation.

\- The relevance threshold was set to 4.0.

\- The evaluation was performed offline and does not establish real-world production performance.

\- The current implementation uses Truncated SVD rather than a deep-learning two-tower retrieval and ranking architecture.

\- Distributed Spark processing, FAISS retrieval, Redis caching, Triton serving, Airflow orchestration and online A/B testing have not been implemented.



Further development and testing are required before production deployment.

