from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_courses(path: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	df["tags"] = df["tags"].fillna("").astype(str)
	df["text"] = (
		df["title"].fillna("").astype(str) + " "
		+ df["subject"].fillna("").astype(str) + " "
		+ df["level"].fillna("").astype(str) + " "
		+ df["tags"].str.replace("|", " ", regex=False)
	)
	return df


def load_interactions(path: str) -> pd.DataFrame:
	return pd.read_csv(path)


def build_content_matrix(courses_df: pd.DataFrame) -> Tuple[TfidfVectorizer, 'scipy.sparse.csr_matrix']:
	vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
	tfidf = vectorizer.fit_transform(courses_df["text"])
	return vectorizer, tfidf


