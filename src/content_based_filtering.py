import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def user_content_scores(user_id: str, courses_df: pd.DataFrame, interactions_df: pd.DataFrame, tfidf: csr_matrix) -> np.ndarray:
	user_hist = interactions_df[(interactions_df["user_id"] == user_id) & (interactions_df["rating"] >= 4)]
	if user_hist.empty:
		return courses_df["rating"].fillna(0).to_numpy(dtype=float)
	liked_idx = courses_df.merge(user_hist[["course_id"]], on="course_id", how="inner").index.to_numpy()
	if liked_idx.size == 0:
		return courses_df["rating"].fillna(0).to_numpy(dtype=float)
	user_vec = tfidf[liked_idx].mean(axis=0)
	# Ensure ndarray (not np.matrix) for sklearn
	user_vec = np.asarray(user_vec)
	sims = cosine_similarity(user_vec, tfidf).ravel()
	# Blend with course rating to stabilize
	return 0.9 * sims + 0.1 * courses_df["rating"].fillna(0).to_numpy(dtype=float)


def top_n_from_scores(scores: np.ndarray, courses_df: pd.DataFrame, exclude_ids: set, n: int = 10):
	idx = np.argsort(-scores)
	recs = []
	for i in idx:
		cid = courses_df.iloc[i]["course_id"]
		if cid in exclude_ids:
			continue
		recs.append((cid, float(scores[i])))
		if len(recs) >= n:
			break
	return recs


