"""Simple Streamlit interface for the movie recommendation system."""

from __future__ import annotations

import logging
from typing import Any, Dict

import streamlit as st

from recommender import MovieRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("movie_recommender_app")

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Minimal styling only; recommendation and data-processing code is untouched.
st.markdown(
    """
    <style>
    .stApp { background: #0e0e0e; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #1a1a1a;
        border-color: #303030;
        border-radius: 8px;
    }
    div.stButton > button[kind="primary"] {
        background: #ffffff;
        color: #111111;
        border: 1px solid #ffffff;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #dddddd;
        color: #111111;
        border-color: #dddddd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading movie data...")
def get_recommender() -> MovieRecommender:
    """Load the existing recommender and its persisted model."""
    return MovieRecommender(auto_init=True)


def render_movie_card(movie: Dict[str, Any]) -> None:
    """Display one recommendation in a plain card."""
    with st.container(border=True):
        poster_url = movie.get("poster_url")
        if poster_url:
            try:
                st.image(poster_url, use_container_width=True)
            except Exception:
                st.markdown(
                    "<div style='height:260px;background:#242424;border-radius:6px'></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='height:260px;background:#242424;border-radius:6px'></div>",
                unsafe_allow_html=True,
            )

        st.markdown(f"**{movie.get('title', 'Untitled')}**")
        year = movie.get("release_year")
        genres = movie.get("genres") or []
        genre_text = ", ".join(genres[:2]) if genres else "Genre unavailable"
        st.caption(f"{year or 'Year unavailable'} · {genre_text}")

        rating = movie.get("vote_average")
        if rating is not None and rating > 0:
            st.caption(f"Rating: {rating:.1f}/10")
        else:
            st.caption("Rating unavailable")


def main() -> None:
    st.title("Movie Recommendation System")
    recommender = get_recommender()

    query = st.text_input("Search for a movie", placeholder="Enter a movie title")
    candidates = recommender.search_movies(query.strip(), limit=10) if query.strip() else []

    selected_id = None
    if query.strip():
        if candidates:
            options = {item["display_label"]: item["movie_id"] for item in candidates}
            selected_label = st.selectbox("Select a movie", list(options.keys()))
            selected_id = options[selected_label]
        else:
            st.info("No matching movie found. Try another title.")

    if st.button(
        "Recommend Movies",
        type="primary",
        disabled=selected_id is None,
    ):
        st.session_state["recommended_movie_id"] = selected_id

    movie_id = st.session_state.get("recommended_movie_id")
    if movie_id is not None:
        with st.spinner("Finding similar movies..."):
            source_movie, recommendations = recommender.get_recommendations(
                movie_id, top_n=10
            )

        if not source_movie:
            st.warning("The selected movie could not be loaded. Please try another title.")
            st.session_state.pop("recommended_movie_id", None)
            return

        st.subheader(f"Recommendations for {source_movie.get('title', 'Selected movie')}")
        if not recommendations:
            st.info("No recommendations were found for this movie.")
            return

        for start in range(0, min(10, len(recommendations)), 3):
            columns = st.columns(3)
            for column, movie in zip(columns, recommendations[start:start + 3]):
                with column:
                    render_movie_card(movie)


if __name__ == "__main__":
    main()
