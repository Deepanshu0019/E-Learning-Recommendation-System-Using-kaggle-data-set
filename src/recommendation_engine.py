from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class Recommender:
    def __init__(self, courses_path: str, interactions_path: str):
        self.courses = pd.read_csv(courses_path)
        self.interactions = pd.read_csv(interactions_path)

        for col in ["description", "title", "tags"]:
            if col not in self.courses.columns:
                self.courses[col] = ""

        text_data = (
            self.courses["title"].fillna("") + " " +
            self.courses["description"].fillna("") + " " +
            self.courses["tags"].fillna("")
        )
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = self.tfidf.fit_transform(text_data)

    def recommend_content(self, query: str, n: int = 10):
        if not query or not isinstance(query, str):
            return []

        q_vec = self.tfidf.transform([query])
        cosine_sim = linear_kernel(q_vec, self.tfidf_matrix).flatten()
        top_idx = cosine_sim.argsort()[::-1][:n]

        results = []
        for idx in top_idx:
            row = self.courses.iloc[idx].to_dict()
            row["score"] = float(cosine_sim[idx])
            results.append(row)
        return results

    def recommend(self, user_id: str, n: int = 10, alpha: float = 0.5) -> List[Tuple[str, float]]:
        if user_id not in self.interactions["user_id"].unique():
            top = self.courses.nlargest(n, "rating")[["course_id"]].to_dict(orient="records")
            return [(r["course_id"], float(r.get("rating", 0))) for r in top]

        user_rows = self.interactions[self.interactions["user_id"] == user_id]
        user_courses = user_rows.sort_values("rating", ascending=False)["course_id"].tolist()

        if not user_courses:
            top = self.courses.nlargest(n, "rating")[["course_id"]].to_dict(orient="records")
            return [(r["course_id"], float(r.get("rating", 0))) for r in top]

        candidates = []
        for ref_course_id in user_courses[:3]:
            title = self.courses[self.courses["course_id"] == ref_course_id]["title"].values
            if len(title) == 0:
                continue
            recs = self.recommend_content(title[0], n=n+len(user_courses))
            for r in recs:
                if r["course_id"] not in user_courses:
                    candidates.append(r)

        cand_map = {}
        for c in candidates:
            cid = c["course_id"]
            cand_map[cid] = max(cand_map.get(cid, 0), c.get("score", 0.0))

        sorted_cands = sorted(cand_map.items(), key=lambda x: x[1], reverse=True)
        sorted_cids = [t[0] for t in sorted_cands][:n]

        if not sorted_cids:
            top = self.courses[~self.courses["course_id"].isin(user_courses)].nlargest(n, "rating")[["course_id"]].to_dict(orient="records")
            return [(r["course_id"], float(r.get("rating", 0))) for r in top]

        results = []
        for cid in sorted_cids:
            score = cand_map.get(cid, 0.0)
            results.append((cid, float(score)))

        if len(results) < n:
            needed = n - len(results)
            seen = set(user_courses)
            filler = self.courses[~self.courses["course_id"].isin(list(seen) + list(cand_map.keys()))].nlargest(needed, "rating")[["course_id","rating"]].to_dict(orient="records")
            for f in filler:
                results.append((f["course_id"], float(f.get("rating", 0))))

        if not results:
            top = self.courses.nlargest(1, "rating")[["course_id","rating"]].to_dict(orient="records")
            return [(top[0]["course_id"], float(top[0].get("rating", 0)))]

        return results

    def course_metadata(self, course_id: str) -> dict:
        row = self.courses[self.courses["course_id"] == course_id].iloc[0].to_dict()
        return row
