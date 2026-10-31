"""Unit tests for the movie search system and title disambiguation."""

import pytest
from recommender import MovieRecommender


@pytest.fixture(scope="module")
def recommender_instance():
    """Shared recommender instance for search tests."""
    return MovieRecommender(auto_init=True)


def test_search_empty_and_whitespace(recommender_instance):
    """Empty or whitespace search queries should return an empty list."""
    assert recommender_instance.search_movies("") == []
    assert recommender_instance.search_movies("   ") == []


def test_search_exact_match(recommender_instance):
    """Exact title search should find the movie accurately."""
    results = recommender_instance.search_movies("Inception", limit=5)
    assert len(results) >= 1
    assert results[0]["title"] == "Inception"
    assert results[0]["release_year"] == 2010


def test_search_case_insensitivity(recommender_instance):
    """Search must be case-insensitive."""
    res_lower = recommender_instance.search_movies("avatar", limit=3)
    res_upper = recommender_instance.search_movies("AVATAR", limit=3)
    res_mixed = recommender_instance.search_movies("AvAtAr", limit=3)

    assert len(res_lower) >= 1
    assert res_lower[0]["title"] == "Avatar"
    assert res_upper[0]["title"] == "Avatar"
    assert res_mixed[0]["title"] == "Avatar"


def test_search_substring_and_partial(recommender_instance):
    """Partial title matching should return relevant movies."""
    results = recommender_instance.search_movies("Interstell", limit=5)
    assert len(results) >= 1
    matched_titles = [r["title"] for r in results]
    assert "Interstellar" in matched_titles


def test_search_title_disambiguation_with_year(recommender_instance):
    """Distinguish movies with identical or similar titles using release years."""
    results = recommender_instance.search_movies("Batman", limit=10)
    assert len(results) >= 2
    # Verify display labels contain year
    for r in results:
        assert "(" in r["display_label"] and ")" in r["display_label"]

    # Verify Batman 1989 vs Batman 1966 vs Batman Begins 2005
    batman_titles = [r["display_label"] for r in results]
    assert any("1989" in label for label in batman_titles)


def test_search_typo_tolerance(recommender_instance):
    """Basic spelling tolerance should resolve common typos."""
    results = recommender_instance.search_movies("Interstelar", limit=5)
    assert len(results) >= 1
    assert results[0]["title"] == "Interstellar"


def test_search_unknown_movie(recommender_instance):
    """Nonsense query should return empty list gracefully."""
    results = recommender_instance.search_movies("xyznonexistentmovie99999", limit=5)
    assert results == []
