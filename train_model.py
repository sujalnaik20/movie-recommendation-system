"""Model training pipeline for Content-Based Movie Recommendation System.

Executes KaggleHub dataset acquisition, preprocessing, TF-IDF vectorization,
and sparse matrix persistence.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.cache_manager import (
    DEFAULT_MODELS_DIR,
    load_artifacts,
    save_artifacts,
    validate_artifacts,
)
from utils.data_loader import DataLoaderError, load_raw_data
from utils.preprocessing import preprocess_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_pipeline(
    force: bool = False,
    models_dir: Optional[Path | str] = None,
    max_features: int = 7500,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[pd.DataFrame, TfidfVectorizer, csr_matrix]:
    """Execute complete end-to-end training pipeline.

    If valid models already exist and `force` is False, loads and returns them.
    Otherwise, downloads dataset, preprocesses features, fits TF-IDF, and saves artifacts.

    Args:
        force: If True, forces retraining even if model artifacts exist.
        models_dir: Target directory for model files (default: project's models/ folder).
        max_features: Maximum vocabulary size for TF-IDF.
        ngram_range: N-gram range for TF-IDF (default: unigrams and bigrams).

    Returns:
        Tuple of (movies_df, vectorizer, tfidf_matrix).
    """
    target_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR

    # 1. Check if existing models can be reused
    if not force and validate_artifacts(target_dir):
        logger.info("Found existing validated model artifacts in %s. Loading...", target_dir)
        try:
            return load_artifacts(target_dir)
        except Exception as exc:
            logger.warning(
                "Existing model artifacts in %s failed validation: %s. Retraining...",
                target_dir,
                exc,
            )

    start_time = time.time()
    logger.info("Starting training pipeline...")

    # 2. Automatically load raw datasets via KaggleHub
    try:
        raw_movies, raw_credits = load_raw_data()
    except DataLoaderError as exc:
        logger.error("Dataset loading failed: %s", exc)
        raise

    # 3. Preprocess and build weighted tags
    clean_movies = preprocess_data(raw_movies, raw_credits)

    # 4. TF-IDF Vectorization
    logger.info(
        "Fitting TfidfVectorizer (max_features=%d, ngram_range=%s, stop_words='english')...",
        max_features,
        ngram_range,
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(clean_movies["tags"])
    # Convert to CSR format if not already
    tfidf_matrix = csr_matrix(tfidf_matrix)

    logger.info(
        "TF-IDF matrix shape: %s (non-zero entries: %d, memory: ~%.2f MB)",
        tfidf_matrix.shape,
        tfidf_matrix.nnz,
        (tfidf_matrix.data.nbytes + tfidf_matrix.indices.nbytes + tfidf_matrix.indptr.nbytes)
        / (1024 * 1024),
    )

    # 5. Persist artifacts
    save_artifacts(
        movies_df=clean_movies,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        models_dir=target_dir,
        extra_metadata={
            "max_features": max_features,
            "ngram_range": list(ngram_range),
            "training_time_sec": round(time.time() - start_time, 2),
        },
    )

    elapsed = time.time() - start_time
    logger.info("Training pipeline completed successfully in %.2f seconds.", elapsed)
    return clean_movies, vectorizer, tfidf_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Content-Based Movie Recommendation System")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain and overwrite existing model artifacts.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Custom directory to store models.",
    )
    args = parser.parse_args()

    try:
        train_pipeline(force=args.force, models_dir=args.models_dir)
        print("\nModel training & verification complete!")
    except Exception as exc:
        logger.critical("Fatal error during training pipeline: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
