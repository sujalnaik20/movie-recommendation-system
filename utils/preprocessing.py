"""Data preprocessing and feature engineering for Movie Recommendation System.

Cleans metadata, extracts JSON structures (genres, keywords, cast, crew),
creates weighted tags, and prepares data for TF-IDF vectorization.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def safe_literal_eval(val: Any) -> Any:
    """Safely evaluate Python literals or JSON-like strings.

    Args:
        val: Input string or object.

    Returns:
        Parsed object or empty list on failure.
    """
    if not isinstance(val, str) or not val.strip():
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        return []


def parse_names(val: Any, limit: Optional[int] = None) -> List[str]:
    """Extract 'name' attribute from a list of dictionaries.

    Args:
        val: Raw string containing serialized list of dicts.
        limit: Maximum number of names to extract.

    Returns:
        List of extracted names.
    """
    items = safe_literal_eval(val)
    if not isinstance(items, list):
        return []
    if limit is not None:
        items = items[:limit]
    names = []
    for item in items:
        if isinstance(item, dict) and "name" in item and item["name"]:
            names.append(str(item["name"]).strip())
    return names


def extract_director(val: Any) -> str:
    """Extract director name from crew list of dictionaries.

    Args:
        val: Raw string containing crew list.

    Returns:
        Director's name or empty string if not found.
    """
    items = safe_literal_eval(val)
    if not isinstance(items, list):
        return ""
    for item in items:
        if isinstance(item, dict) and item.get("job") == "Director":
            name = item.get("name")
            if name:
                return str(name).strip()
    return ""


def collapse_token(text: str) -> str:
    """Remove spaces and special characters from entity names (e.g. 'Tom Cruise' -> 'tomcruise').

    Ensures multi-word entities act as single distinctive tokens in TF-IDF.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[^\w]", "", text.lower())
    return cleaned


def extract_year(date_val: Any) -> Optional[int]:
    """Extract four-digit release year from date string (e.g. '2014-11-05' -> 2014)."""
    if not date_val or pd.isna(date_val):
        return None
    match = re.search(r"\b(19\d\d|20\d\d)\b", str(date_val))
    if match:
        return int(match.group(1))
    return None


def clean_overview(text: Any) -> str:
    """Clean and normalize overview string."""
    if not text or pd.isna(text):
        return ""
    cleaned = str(text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def build_weighted_tags(
    overview: str,
    genres: List[str],
    keywords: List[str],
    cast: List[str],
    director: str,
    genre_weight: int = 3,
    keyword_weight: int = 2,
    director_weight: int = 3,
    cast_weight: int = 2,
) -> str:
    """Construct a weighted tags string for TF-IDF vectorization.

    Feature weighting is achieved through frequency repetition of key content signals
    (genres, keywords, director, top cast). Collapsed tokens prevent accidental
    sub-word collisions (e.g. 'Christopher Nolan' becomes 'christophernolan').
    """
    tag_parts: List[str] = []

    # 1. Overview text (base narrative context)
    if overview:
        tag_parts.append(overview.lower())

    # 2. Genres (repeated to boost thematic relevance)
    for g in genres:
        collapsed = collapse_token(g)
        raw_clean = g.lower()
        if collapsed:
            # Add both collapsed and raw forms for comprehensive matching
            tag_parts.extend([collapsed] * genre_weight)
            tag_parts.extend([raw_clean] * (genre_weight - 1))

    # 3. Keywords (repeated for specific topic similarity)
    for kw in keywords:
        collapsed = collapse_token(kw)
        raw_clean = kw.lower()
        if collapsed:
            tag_parts.extend([collapsed] * keyword_weight)
            tag_parts.extend([raw_clean] * (keyword_weight - 1))

    # 4. Director (high directorial style correlation)
    if director:
        collapsed = collapse_token(director)
        if collapsed:
            tag_parts.extend([collapsed] * director_weight)

    # 5. Top cast members
    for actor in cast:
        collapsed = collapse_token(actor)
        if collapsed:
            tag_parts.extend([collapsed] * cast_weight)

    return " ".join(tag_parts)


def preprocess_data(
    movies_df: pd.DataFrame,
    credits_df: pd.DataFrame,
    cast_limit: int = 4,
    genre_weight: int = 3,
    keyword_weight: int = 2,
    director_weight: int = 3,
    cast_weight: int = 2,
) -> pd.DataFrame:
    """Merge movies and credits, clean features, and produce processed movie DataFrame.

    Args:
        movies_df: Raw movies dataframe from tmdb_5000_movies.csv.
        credits_df: Raw credits dataframe from tmdb_5000_credits.csv.
        cast_limit: Number of top cast members to consider.
        genre_weight: Repetition weight for genres.
        keyword_weight: Repetition weight for keywords.
        director_weight: Repetition weight for director.
        cast_weight: Repetition weight for cast.

    Returns:
        DataFrame containing cleaned metadata, structured lists, and 'tags' column.
    """
    logger.info("Merging movies and credits datasets...")

    # Identify merge key
    movie_id_col = "id" if "id" in movies_df.columns else "movie_id"
    credit_id_col = "movie_id" if "movie_id" in credits_df.columns else "id"

    # Merge while avoiding duplicated 'title' columns
    credits_subset = credits_df[[credit_id_col, "cast", "crew"]].copy()
    credits_subset = credits_subset.rename(columns={credit_id_col: "movie_id"})

    movies_copy = movies_df.copy()
    movies_copy = movies_copy.rename(columns={movie_id_col: "movie_id"})

    merged = movies_copy.merge(credits_subset, on="movie_id", how="inner")
    logger.info("Merged shape: %s", merged.shape)

    # Remove duplicates based on movie_id
    initial_count = len(merged)
    merged = merged.drop_duplicates(subset=["movie_id"]).copy()
    if len(merged) < initial_count:
        logger.info("Removed %d duplicate movie_id records.", initial_count - len(merged))

    # Ensure title is present and string
    merged["title"] = merged["title"].fillna("").astype(str).str.strip()
    merged = merged[merged["title"] != ""].copy()

    # Extract clean release year
    merged["release_year"] = merged["release_date"].apply(extract_year)

    # Clean overview
    merged["overview"] = merged["overview"].apply(clean_overview)

    # Extract structured metadata lists
    logger.info("Parsing structured metadata (genres, keywords, cast, crew)...")
    merged["genres"] = merged["genres"].apply(parse_names)
    merged["keywords"] = merged["keywords"].apply(parse_names)
    merged["cast"] = merged["cast"].apply(lambda x: parse_names(x, limit=cast_limit))
    merged["director"] = merged["crew"].apply(extract_director)

    # Ensure numeric columns are clean
    merged["popularity"] = pd.to_numeric(merged["popularity"], errors="coerce").fillna(0.0)
    merged["vote_average"] = pd.to_numeric(merged["vote_average"], errors="coerce").fillna(0.0)
    merged["vote_count"] = pd.to_numeric(merged["vote_count"], errors="coerce").fillna(0).astype(int)

    # Construct weighted tags
    logger.info("Constructing weighted recommendation tags...")
    tags = []
    for _, row in merged.iterrows():
        tag = build_weighted_tags(
            overview=row["overview"],
            genres=row["genres"],
            keywords=row["keywords"],
            cast=row["cast"],
            director=row["director"],
            genre_weight=genre_weight,
            keyword_weight=keyword_weight,
            director_weight=director_weight,
            cast_weight=cast_weight,
        )
        tags.append(tag)
    merged["tags"] = tags

    # Filter out records where tags are completely empty
    merged = merged[merged["tags"].str.strip() != ""].copy()
    merged = merged.reset_index(drop=True)

    # Select final standardized schema
    final_cols = [
        "movie_id",
        "title",
        "release_year",
        "release_date",
        "genres",
        "keywords",
        "cast",
        "director",
        "overview",
        "popularity",
        "vote_average",
        "vote_count",
        "tags",
    ]
    result_df = merged[final_cols].copy()
    logger.info("Preprocessing complete. Final dataset size: %d movies.", len(result_df))
    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from utils.data_loader import load_raw_data

    m_df, c_df = load_raw_data()
    clean_df = preprocess_data(m_df, c_df)
    print("Clean DF sample:")
    print(clean_df[["movie_id", "title", "release_year", "director"]].head())
    print("\nSample tags for first movie:")
    print(clean_df["tags"].iloc[0][:300])
