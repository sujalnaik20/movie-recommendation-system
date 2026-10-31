"""Unit tests for Content-Based Recommender and benchmark validation."""

import pytest
from recommender import MovieRecommender


@pytest.fixture(scope="module")
def recommender_instance():
    """Shared recommender instance initialized with trained models."""
    return MovieRecommender(auto_init=True)


REQUIRED_BENCHMARK_MOVIES = [
    "Interstellar",
    "Inception",
    "The Dark Knight",
    "Avatar",
    "Titanic",
    "The Avengers",
    "The Conjuring",
    "Finding Nemo",
    "Jurassic Park",
    "The Matrix",
]


@pytest.mark.parametrize("movie_title", REQUIRED_BENCHMARK_MOVIES)
def test_benchmark_movie_recommendations(recommender_instance, movie_title):
    """Validate each required benchmark movie produces 10 high-quality recommendations."""
    src, recs = recommender_instance.get_recommendations(movie_title, top_n=10)

    # 1. Source movie exists
    assert src is not None, f"Source movie '{movie_title}' was not found."
    assert src["title"].lower() == movie_title.lower()

    # 2. Exactly 10 recommendations
    assert len(recs) == 10, f"Expected 10 recommendations for '{movie_title}', got {len(recs)}."

    # 3. Selected movie is excluded from recommendations
    rec_ids = [r["movie_id"] for r in recs]
    assert src["movie_id"] not in rec_ids, f"Selected movie '{movie_title}' was recommended to itself."

    # 4. Duplicate recommendations are prevented
    assert len(rec_ids) == len(set(rec_ids)), f"Duplicate recommendations found for '{movie_title}'."

    # 5. Recommendations are sorted by descending similarity
    scores = [r["similarity_score"] for r in recs]
    assert scores == sorted(scores, reverse=True), (
        f"Recommendations for '{movie_title}' are not strictly sorted by similarity: {scores}"
    )

    # 6. Metadata integrity for all returned recommendations
    for r in recs:
        assert r["title"], "Recommendation missing title."
        assert "similarity_score" in r and r["similarity_score"] >= 0.0
        assert "explanation" in r and len(r["explanation"]) > 0
        assert "genres" in r and isinstance(r["genres"], list)
        assert "poster_url" in r and r["poster_url"] is not None


def test_recommendation_by_movie_id(recommender_instance):
    """Verify recommendation works via TMDB movie_id (e.g. Inception ID: 27205)."""
    # Inception TMDB ID is 27205
    src, recs = recommender_instance.get_recommendations(27205, top_n=10)
    assert src is not None
    assert src["title"] == "Inception"
    assert len(recs) == 10
    assert src["movie_id"] not in [r["movie_id"] for r in recs]


def test_recommendation_unknown_movie(recommender_instance):
    """Unknown movie identifier should return (None, []) without crashing."""
    src, recs = recommender_instance.get_recommendations("NonExistentMovie12345XYZ", top_n=10)
    assert src is None
    assert recs == []


def test_popular_and_top_rated(recommender_instance):
    """Verify discovery feeds return valid movies."""
    popular = recommender_instance.get_popular_movies(limit=8)
    top_rated = recommender_instance.get_top_rated_movies(limit=8)

    assert len(popular) == 8
    assert len(top_rated) == 8
    for m in popular + top_rated:
        assert m["title"]
        assert m["poster_url"]
