import os
import sys
import streamlit as st
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.recommendation_engine import Recommender  # noqa


@st.cache_resource
def get_rec():
    data_dir = os.path.join(ROOT, "data")
    return Recommender(
        courses_path=os.path.join(data_dir, "courses.csv"),
        interactions_path=os.path.join(data_dir, "interactions.csv"),
    )


def render_card(course: dict):
    title = course.get("title", "N/A")
    rating = course.get("rating", "N/A")
    with st.expander(f"📘 **{title}** — 📊 Rating: {rating}"):
        st.markdown(f"**Subject:** {course.get('subject','N/A')}  ")
        st.markdown(f"**Level:** {course.get('level','N/A')}  ")
        st.markdown(f"**Tags:** {course.get('tags','N/A')}  ")
        st.markdown(f"**Relevance Score:** {round(course.get('score',0.0),4)}")
        st.markdown('---')
        desc = str(course.get('description',''))
        st.markdown(desc[:400] + ("..." if len(desc) > 400 else ""))


def apply_filters(courses, subject_filter, level_filter, tag_query=None):
    """Filter course dicts by subject, level and optional tag/keyword query.
       Guarantee at least one course is returned by falling back to top-rated courses.
    """
    filtered = courses
    if subject_filter != "All":
        filtered = [c for c in filtered if c.get("subject") == subject_filter]
    if level_filter != "All":
        filtered = [c for c in filtered if c.get("level") == level_filter]

    if tag_query:
        tq = tag_query.lower()
        filtered = [c for c in filtered if tq in str(c.get("tags","")).lower() or tq in str(c.get("title","")).lower()]

    if not filtered:
        all_courses = sorted(courses, key=lambda x: float(x.get("rating", 0) or 0), reverse=True)
        filtered = all_courses[:max(1, min(3, len(all_courses)))]
    return filtered


def main():
    st.title("🎓 E-Learning Recommendation System")
    st.markdown("""
    Welcome to the **E-Learning Recommendation System** 👋

    This project demonstrates how recommendation engines can personalize online learning experiences. 
    It combines **content-based filtering** and **collaborative filtering** to suggest relevant courses, 
    making it easier for learners to discover new skills.

    ✅ Search by keywords  
    ✅ Get personalized recommendations  
    ✅ Explore trending courses  
    ✅ Filter by subject and level

    ---
    """)

    rec = get_rec()

    st.sidebar.header("🔧 Controls")
    users = sorted(rec.interactions["user_id"].unique().tolist())
    user_id = st.sidebar.selectbox("Select user", users)
    alpha = st.sidebar.slider("Blend (Content vs Collaborative)", 0.0, 1.0, 0.5, 0.05)
    n = st.sidebar.slider("How many recommendations?", 1, 20, 5, 1)

    all_subjects = ["All"] + sorted(rec.courses["subject"].dropna().unique().tolist())
    all_levels = ["All"] + sorted(rec.courses["level"].dropna().unique().tolist())

    subject_filter = st.sidebar.selectbox("Filter by subject", all_subjects)
    level_filter = st.sidebar.selectbox("Filter by level", all_levels)

    st.subheader("🔍 Search / Keyword (matches titles, tags)")
    query = st.text_input("Enter a course title, tag, or keyword")

    if query:
        if st.button("Search Recommendations"):
            recs = rec.recommend_content(query, n=n)
            recs = apply_filters(recs, subject_filter, level_filter, tag_query=query)

            if recs:
                st.success(f"Content-based results for '{query}'")
                for c in recs:
                    render_card(c)
            else:
                st.info("No similar courses found. Showing top-rated suggestions instead.")
                top = rec.courses.sort_values("rating", ascending=False).head(n).to_dict(orient="records")
                for c in top:
                    render_card(c)

    st.subheader("👤 Personalized Hybrid Recommendations")
    if st.button("Recommend for User"):
        recs = rec.recommend(user_id, n=n, alpha=alpha)
        rows = []
        if recs and isinstance(recs[0], tuple):
            for cid, score in recs:
                meta = rec.course_metadata(cid)
                meta["score"] = round(float(score), 4)
                rows.append(meta)
        else:
            for r in recs:
                r["score"] = float(r.get("score", 0.0))
                rows.append(r)

        rows = apply_filters(rows, subject_filter, level_filter)

        if not rows:
            top = rec.courses.sort_values("rating", ascending=False).head(max(1, n)).to_dict(orient="records")
            rows = top

        for c in rows:
            render_card(c)

    st.subheader("📈 Trending Courses")
    top_courses = rec.courses.sort_values("rating", ascending=False).head(8).to_dict(orient="records")
    top_courses = apply_filters(top_courses, subject_filter, level_filter)

    for c in top_courses:
        render_card(c)

    with st.expander("📂 Show raw dataset"):
        st.dataframe(rec.courses)

    st.caption("ℹ️ Tip: If collaborative model isn't available, it falls back to content-based only.")


if __name__ == "__main__":
    main()
