

\# Recommendation Model Comparison Report



\## 1. Project Dataset



Dataset: MovieLens 25M



Total ratings: 25,000,095

Unique users: 162,541

Total movies: 62,423



\## 2. Models Evaluated



1\. Truncated Singular Value Decomposition (SVD)

2\. Popularity-Based Recommendation Baseline



\## 3. Evaluation Methodology



\- Training split: 80%

\- Testing split: 20%

\- Evaluation users: 500

\- Relevant movie threshold: Rating >= 4.0

\- Recommendation cutoff: Top 10



\## 4. Evaluation Results



| Metric | SVD | Popularity |

|---|---:|---:|

| Precision@10 | 0.2250 | 0.1200 |

| Recall@10 | 0.2085 | 0.1061 |

| NDCG@10 | 0.3065 | 0.1635 |



\## 5. Performance Improvement



\- Precision@10: 87.5%

\- Recall@10: 96.5%

\- NDCG@10: 87.5%



\## 6. Conclusion



The SVD-based personalized recommendation model

outperformed the popularity-based baseline across

all three evaluation metrics.



This demonstrates the benefit of incorporating

individual user preferences into movie recommendations.



\## 7. Limitations



\- Evaluation was performed on a sample of 500 users.

\- The relevance threshold was set to 4.0.

\- The dataset used a random rating-level split.

\- Results represent offline evaluation and do not

&#x20; establish real-world production performance.



Further testing is required before deployment.

