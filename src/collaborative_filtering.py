from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF


class CollabModel:
	def __init__(self, algo: str, model, user_index: Dict[str, int], item_index: Dict[str, int], users: list, items: list):
		self.algo = algo
		self.model = model
		self.user_index = user_index
		self.item_index = item_index
		self.users = users
		self.items = items

	def scores_for_user(self, user_id: str) -> np.ndarray:
		if self.algo == "nmf":
			u_idx = self.user_index.get(user_id, None)
			if u_idx is None:
				return np.zeros(len(self.items))
			W = self.model["W"]
			H = self.model["H"]
			return W[u_idx] @ H
		elif self.algo == "surprise":
			algo = self.model["algo"]
			trainset = self.model["trainset"]
			iid_inner = {i: trainset.to_inner_iid(i) for i in self.items if i in trainset._raw2inner_id_items}
			scores = np.zeros(len(self.items))
			for i, item in enumerate(self.items):
				if item in iid_inner:
					est = algo.predict(user_id, item, clip=False).est
				else:
					est = 0.0
				scores[i] = est
			return scores
		return np.zeros(len(self.items))


def _build_indices(interactions: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], list, list]:
	users = sorted(interactions["user_id"].unique().tolist())
	items = sorted(interactions["course_id"].unique().tolist())
	user_index = {u: i for i, u in enumerate(users)}
	item_index = {it: i for i, it in enumerate(items)}
	return user_index, item_index, users, items


def _build_urm(interactions: pd.DataFrame, user_index: Dict[str, int], item_index: Dict[str, int]) -> csr_matrix:
	rows = interactions["user_id"].map(user_index).to_numpy()
	cols = interactions["course_id"].map(item_index).to_numpy()
	data = interactions["rating"].astype(float).to_numpy()
	n_users = len(user_index)
	n_items = len(item_index)
	return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def train_collab(interactions: pd.DataFrame) -> Optional[CollabModel]:
	try:
		from surprise import Dataset, Reader, SVD
		reader = Reader(rating_scale=(1, 5))
		data = Dataset.load_from_df(interactions[["user_id", "course_id", "rating"]], reader)
		trainset = data.build_full_trainset()
		algo = SVD(n_factors=50, n_epochs=20, biased=True, random_state=42)
		algo.fit(trainset)
		# Build item list from interactions
		user_index, item_index, users, items = _build_indices(interactions)
		return CollabModel("surprise", {"algo": algo, "trainset": trainset}, user_index, item_index, users, items)
	except Exception:
		# Fallback to NMF (no-surprise)
		user_index, item_index, users, items = _build_indices(interactions)
		URM = _build_urm(interactions, user_index, item_index)
		# Normalize ratings to 0-1 for NMF
		X = (URM / 5.0).astype(float)
		k = min(20, max(2, min(X.shape) - 1))
		nmf = NMF(n_components=k, init="nndsvda", random_state=42, max_iter=300)
		W = nmf.fit_transform(X)
		H = nmf.components_
		return CollabModel("nmf", {"W": W, "H": H}, user_index, item_index, users, items)


