"""Cache manager and model artifact validation utilities.

Handles safe loading, serialization, artifact verification, and metadata tracking.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MOVIE_LIST_FILENAME = "movie_list.pkl"
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
MATRIX_FILENAME = "tfidf_matrix.pkl"
METADATA_FILENAME = "model_metadata.json"


def get_model_paths(models_dir: Optional[Path | str] = None) -> Dict[str, Path]:
    """Get absolute paths for all model artifacts."""
    base = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    base.mkdir(parents=True, exist_ok=True)
    return {
        "dir": base,
        "movie_list": base / MOVIE_LIST_FILENAME,
        "vectorizer": base / VECTORIZER_FILENAME,
        "matrix": base / MATRIX_FILENAME,
        "metadata": base / METADATA_FILENAME,
    }


def save_artifacts(
    movies_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: csr_matrix,
    models_dir: Optional[Path | str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Persist processed movie list, fitted TF-IDF vectorizer, and sparse matrix.

    Args:
        movies_df: Processed movies DataFrame.
        vectorizer: Fitted TfidfVectorizer.
        tfidf_matrix: Sparse TF-IDF feature matrix.
        models_dir: Directory to save model files in.
        extra_metadata: Additional key-value metadata to record.

    Returns:
        Dict mapping artifact keys to their Paths.
    """
    paths = get_model_paths(models_dir)
    logger.info("Persisting model artifacts to %s...", paths["dir"])

    # 1. Save movie_list
    joblib.dump(movies_df, paths["movie_list"], compress=3)

    # 2. Save vectorizer
    joblib.dump(vectorizer, paths["vectorizer"], compress=3)

    # 3. Save tfidf_matrix (sparse CSR matrix)
    joblib.dump(tfidf_matrix, paths["matrix"], compress=3)

    # 4. Save metadata
    vocab_size = len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") else None
    metadata: Dict[str, Any] = {
        "num_movies": len(movies_df),
        "vocab_size": vocab_size,
        "matrix_shape": list(tfidf_matrix.shape),
        "columns": list(movies_df.columns),
        "version": "1.0.0",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Successfully persisted all artifacts to %s.", paths["dir"])
    return paths


def validate_artifacts(models_dir: Optional[Path | str] = None) -> bool:
    """Verify that all model artifacts exist, are non-empty, and are structurally intact.

    Returns:
        True if all files exist and are valid, False otherwise.
    """
    paths = get_model_paths(models_dir)
    required_keys = ["movie_list", "vectorizer", "matrix"]

    for key in required_keys:
        p = paths[key]
        if not p.is_file():
            logger.debug("Artifact missing: %s", p)
            return False
        if p.stat().st_size == 0:
            logger.warning("Artifact is zero bytes: %s", p)
            return False

    return True


def load_artifacts(
    models_dir: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, TfidfVectorizer, csr_matrix]:
    """Load model artifacts with structural sanity checks.

    Returns:
        Tuple of (movies_df, vectorizer, tfidf_matrix).

    Raises:
        FileNotFoundError: If any artifact is missing.
        ValueError: If loaded artifacts are incompatible or corrupted.
    """
    paths = get_model_paths(models_dir)

    if not validate_artifacts(models_dir):
        raise FileNotFoundError(
            f"One or more model artifacts are missing or invalid in '{paths['dir']}'."
        )

    logger.info("Loading model artifacts from %s...", paths["dir"])
    try:
        movies_df = joblib.load(paths["movie_list"])
        vectorizer = joblib.load(paths["vectorizer"])
        tfidf_matrix = joblib.load(paths["matrix"])
    except Exception as exc:
        raise ValueError(f"Corrupted model file detected in '{paths['dir']}': {exc}") from exc

    # Sanity checks
    if not isinstance(movies_df, pd.DataFrame):
        raise ValueError("movie_list.pkl does not contain a pandas DataFrame.")

    if not hasattr(vectorizer, "transform"):
        raise ValueError("tfidf_vectorizer.pkl does not contain a valid vectorizer.")

    if not issparse(tfidf_matrix):
        logger.warning("Loaded tfidf_matrix is dense; expected sparse matrix.")

    if tfidf_matrix.shape[0] != len(movies_df):
        raise ValueError(
            f"Shape mismatch: tfidf_matrix has {tfidf_matrix.shape[0]} rows, "
            f"but movie_list has {len(movies_df)} rows."
        )

    logger.info(
        "Model artifacts loaded successfully. (%d movies, %d features)",
        len(movies_df),
        tfidf_matrix.shape[1],
    )
    return movies_df, vectorizer, tfidf_matrix
