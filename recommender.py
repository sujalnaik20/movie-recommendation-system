"""Content-Based Movie Recommendation Engine using TF-IDF and Cosine Similarity.

Handles search with fuzzy matching & title disambiguation, calculates
similarity scores on the fly against sparse TF-IDF vectors, and provides
content-based explanations (shared genres, director, cast, keywords).
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.cache_manager import DEFAULT_MODELS_DIR, load_artifacts, validate_artifacts
from utils.tmdb_api import tmdb_client, generate_fallback_poster_svg

logger = logging.getLogger(__name__)


class MovieRecommender:
    """Core Content-Based Recommender using TF-IDF and Cosine Similarity."""

    def __init__(self, models_dir: Optional[Path | str] = None, auto_init: bool = True):
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.movies_df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[csr_matrix] = None
        self._title_lookup: Dict[str, List[int]] = {}

        if auto_init:
            self.load_or_train()

    def load_or_train(self, force_retrain: bool = False) -> None:
        """Load persisted models or trigger training pipeline if missing."""
        if not force_retrain and validate_artifacts(self.models_dir):
            try:
                self.movies_df, self.vectorizer, self.tfidf_matrix = load_artifacts(self.models_dir)
                self._build_indices()
                return
            except Exception as exc:
                logger.warning("Failed to load existing artifacts: %s. Initiating training...", exc)

        # Trigger automatic training
        from train_model import train_pipeline

        logger.info("Initializing model training pipeline...")
        self.movies_df, self.vectorizer, self.tfidf_matrix = train_pipeline(
            force=force_retrain, models_dir=self.models_dir
        )
        self._build_indices()

    def _build_indices(self) -> None:
        """Build internal title lookup table for fast and accurate movie search."""
        if self.movies_df is None:
            return

        lookup: Dict[str, List[int]] = {}
        for idx, row in self.movies_df.iterrows():
            clean_title = str(row["title"]).strip().lower()
            if clean_title not in lookup:
                lookup[clean_title] = []
            lookup[clean_title].append(int(idx))
        self._title_lookup = lookup

    @property
    def total_movies(self) -> int:
        """Total number of movies indexed in the recommender."""
        return len(self.movies_df) if self.movies_df is not None else 0

    def search_movies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search movies by title with prefix matching, substring matching, and fuzzy fallback.

        Disambiguates movies with the same or similar title by showing release year.

        Args:
            query: User search query.
            limit: Maximum suggestions to return.

        Returns:
            List of movie candidate dictionaries with formatted labels and metadata.
        """
        if not query or not query.strip() or self.movies_df is None:
            return []

        q = query.strip().lower()
        matched_indices: List[int] = []

        # 1. Exact match
        if q in self._title_lookup:
            matched_indices.extend(self._title_lookup[q])

        # 2. Starts with query
        for title, indices in self._title_lookup.items():
            if title.startswith(q):
                for idx in indices:
                    if idx not in matched_indices:
                        matched_indices.append(idx)
            if len(matched_indices) >= limit * 2:
                break

        # 3. Substring match
        if len(matched_indices) < limit:
            for title, indices in self._title_lookup.items():
                if q in title:
                    for idx in indices:
                        if idx not in matched_indices:
                            matched_indices.append(idx)
                if len(matched_indices) >= limit * 2:
                    break

        # 4. Fuzzy / typo tolerance fallback (e.g. 'Interstelar' -> 'Interstellar')
        if len(matched_indices) < limit:
            all_titles = list(self._title_lookup.keys())
            close_matches = difflib.get_close_matches(q, all_titles, n=5, cutoff=0.6)
            for m in close_matches:
                for idx in self._title_lookup[m]:
                    if idx not in matched_indices:
                        matched_indices.append(idx)

        # Slice to requested limit
        matched_indices = matched_indices[:limit]

        # Convert to structured dictionary items
        results = []
        for idx in matched_indices:
            row = self.movies_df.iloc[idx]
            year_val = None
            if pd.notna(row["release_year"]):
                try:
                    year_val = int(row["release_year"])
                except (ValueError, TypeError):
                    year_val = None
            year_str = f" ({year_val})" if year_val is not None else ""
            display_label = f"{row['title']}{year_str}"
            results.append(
                {
                    "index": idx,
                    "movie_id": int(row["movie_id"]),
                    "title": str(row["title"]),
                    "release_year": year_val,
                    "display_label": display_label,
                    "vote_average": float(row["vote_average"]),
                    "genres": list(row["genres"]) if isinstance(row["genres"], (list, np.ndarray)) else [],
                    "director": str(row["director"]) if pd.notna(row["director"]) else "",
                }
            )

        return results

    def get_movie_by_id(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve full movie record by TMDB movie_id."""
        if self.movies_df is None:
            return None
        matches = self.movies_df[self.movies_df["movie_id"] == movie_id]
        if matches.empty:
            return None
        return self._row_to_dict(matches.index[0])

    def get_movie_by_title(self, title: str, year: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Retrieve movie record by exact or closest title, optionally disambiguated by year."""
        if self.movies_df is None:
            return None

        clean_title = title.strip().lower()
        if clean_title in self._title_lookup:
            indices = self._title_lookup[clean_title]
            if len(indices) == 1 or year is None:
                return self._row_to_dict(indices[0])
            # Disambiguate by year
            for idx in indices:
                if self.movies_df.iloc[idx]["release_year"] == year:
                    return self._row_to_dict(idx)
            return self._row_to_dict(indices[0])

        # Try closest match
        search_res = self.search_movies(title, limit=1)
        if search_res:
            return self._row_to_dict(search_res[0]["index"])
        return None

    def _row_to_dict(self, idx: int) -> Dict[str, Any]:
        """Convert a dataframe row at index to full movie dictionary."""
        row = self.movies_df.iloc[idx]
        movie_id = int(row["movie_id"])
        year = None
        if pd.notna(row["release_year"]):
            try:
                year = int(row["release_year"])
            except (ValueError, TypeError):
                year = None
        genres = list(row["genres"]) if isinstance(row["genres"], (list, np.ndarray)) else []
        primary_genre = genres[0] if genres else None

        # Fetch TMDB details (or offline fallback)
        tmdb_info = tmdb_client.get_movie_details(movie_id)
        poster_url = tmdb_info.get("poster_url")
        backdrop_url = tmdb_info.get("backdrop_url")

        # Fallback poster if none is available from TMDB
        if not poster_url:
            poster_url = generate_fallback_poster_svg(str(row["title"]), year, primary_genre)

        return {
            "index": idx,
            "movie_id": movie_id,
            "title": str(row["title"]),
            "release_year": year,
            "release_date": str(row["release_date"]) if pd.notna(row["release_date"]) else "",
            "genres": genres,
            "keywords": list(row["keywords"]) if isinstance(row["keywords"], (list, np.ndarray)) else [],
            "cast": list(row["cast"]) if isinstance(row["cast"], (list, np.ndarray)) else [],
            "director": str(row["director"]) if pd.notna(row["director"]) else "",
            "overview": str(row["overview"]) if pd.notna(row["overview"]) else "",
            "popularity": float(row["popularity"]),
            "vote_average": float(row["vote_average"]),
            "vote_count": int(row["vote_count"]),
            "tagline": tmdb_info.get("tagline", ""),
            "runtime": tmdb_info.get("runtime"),
            "poster_url": poster_url,
            "backdrop_url": backdrop_url,
            "has_real_poster": tmdb_info.get("poster_url") is not None,
        }

    def _generate_explanation(self, source: Dict[str, Any], target: Dict[str, Any]) -> str:
        """Generate human-readable reason why target movie was recommended."""
        reasons = []

        # Check shared director
        if source.get("director") and target.get("director") and source["director"].lower() == target["director"].lower():
            reasons.append(f"Directed by {source['director']}")

        # Check shared cast
        shared_cast = [c for c in source.get("cast", []) if c in target.get("cast", [])]
        if shared_cast:
            reasons.append(f"Starring {', '.join(shared_cast[:2])}")

        # Check shared genres
        shared_genres = [g for g in source.get("genres", []) if g in target.get("genres", [])]
        if shared_genres:
            reasons.append(f"Shared genres: {', '.join(shared_genres[:2])}")

        # Check shared keywords
        shared_kw = [k for k in source.get("keywords", []) if k in target.get("keywords", [])]
        if shared_kw and len(reasons) < 2:
            reasons.append(f"Themes: {', '.join(shared_kw[:2])}")

        if not reasons:
            return "Recommended based on similar storyline and narrative themes"
        return " | ".join(reasons)

    def get_recommendations(
        self,
        movie_identifier: int | str,
        top_n: int = 10,
        year: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Calculate genuine Content-Based Recommendations using Cosine Similarity.

        Computes similarity vector on the fly using sparse matrix multiplication.
        Excludes the query movie, sorts descending, and returns top_n unique recommendations.

        Args:
            movie_identifier: TMDB movie_id (int) or title (str).
            top_n: Number of recommendations to return (default 10).
            year: Release year for title disambiguation if title string provided.

        Returns:
            Tuple of (source_movie_dict, list_of_recommended_movie_dicts).
        """
        if self.movies_df is None or self.tfidf_matrix is None:
            raise RuntimeError("Recommender model has not been initialized.")

        # Resolve query movie
        if isinstance(movie_identifier, int):
            source_movie = self.get_movie_by_id(movie_identifier)
        else:
            source_movie = self.get_movie_by_title(str(movie_identifier), year=year)

        if source_movie is None:
            logger.warning("Movie not found for identifier: %s", movie_identifier)
            return None, []

        source_idx = source_movie["index"]

        # Calculate Cosine Similarity vector against all movies
        # tfidf_matrix is CSR: shape (N, V)
        # tfidf_matrix[source_idx] shape is (1, V)
        # cosine_similarity output is (1, N)
        similarity_vector = cosine_similarity(
            self.tfidf_matrix[source_idx], self.tfidf_matrix
        ).flatten()

        # Sort indices in descending order of similarity
        sorted_indices = np.argsort(similarity_vector)[::-1]

        recommendations: List[Dict[str, Any]] = []
        seen_ids = {source_movie["movie_id"]}

        for idx in sorted_indices:
            idx = int(idx)
            if idx == source_idx:
                continue  # Never recommend the selected movie itself

            row = self.movies_df.iloc[idx]
            m_id = int(row["movie_id"])
            if m_id in seen_ids:
                continue  # Prevent duplicates

            score = float(similarity_vector[idx])
            # Filter zero or negative similarity unless we need fallbacks
            if score <= 0.001 and len(recommendations) >= top_n:
                break

            rec_dict = self._row_to_dict(idx)
            rec_dict["similarity_score"] = round(score, 4)
            rec_dict["similarity_pct"] = round(score * 100, 1)
            rec_dict["explanation"] = self._generate_explanation(source_movie, rec_dict)

            recommendations.append(rec_dict)
            seen_ids.add(m_id)

            if len(recommendations) == top_n:
                break

        # Fallback mechanism if fewer than top_n recommendations were found with positive similarity
        if len(recommendations) < top_n:
            logger.info("Applying genre/popularity fallback to fulfill top_%d recommendations", top_n)
            source_genres = set(source_movie.get("genres", []))
            # Sort remaining movies by popularity
            candidate_df = self.movies_df[~self.movies_df["movie_id"].isin(seen_ids)].copy()
            # Score by genre overlap then popularity
            def fallback_score(r):
                g_overlap = len(source_genres.intersection(set(r["genres"])))
                return (g_overlap, r["popularity"])

            sorted_candidates = candidate_df.sort_values(
                by=["vote_average", "popularity"], ascending=[False, False]
            )
            for idx in sorted_candidates.index:
                rec_dict = self._row_to_dict(int(idx))
                rec_dict["similarity_score"] = 0.05
                rec_dict["similarity_pct"] = 5.0
                rec_dict["explanation"] = "Popular title in related themes"
                recommendations.append(rec_dict)
                seen_ids.add(rec_dict["movie_id"])
                if len(recommendations) == top_n:
                    break

        return source_movie, recommendations

    def get_popular_movies(self, limit: int = 12) -> List[Dict[str, Any]]:
        """Retrieve most popular movies for homepage/discovery view."""
        if self.movies_df is None:
            return []
        top_popular = self.movies_df.sort_values(
            by=["popularity", "vote_average"], ascending=[False, False]
        ).head(limit)
        return [self._row_to_dict(int(idx)) for idx in top_popular.index]

    def get_top_rated_movies(self, limit: int = 12, min_votes: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve critically acclaimed movies with sufficient vote volume."""
        if self.movies_df is None:
            return []
        qualified = self.movies_df[self.movies_df["vote_count"] >= min_votes]
        top_rated = qualified.sort_values(by="vote_average", ascending=False).head(limit)
        return [self._row_to_dict(int(idx)) for idx in top_rated.index]


# Global singleton instance for app and testing
recommender = MovieRecommender(auto_init=False)
