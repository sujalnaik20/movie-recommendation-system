"""Data loader module for TMDB 5000 Movie Metadata dataset.

Downloads dataset automatically using kagglehub and loads into pandas DataFrames.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DATASET_IDENTIFIER = "tmdb/tmdb-movie-metadata"
MOVIES_FILENAME = "tmdb_5000_movies.csv"
CREDITS_FILENAME = "tmdb_5000_credits.csv"


class DataLoaderError(Exception):
    """Custom exception raised when dataset loading fails."""


def find_csv_file(search_dir: Path | str, target_filename: str) -> Path:
    """Recursively search for target_filename within search_dir.

    Args:
        search_dir: The root directory to search.
        target_filename: Name of the file to find.

    Returns:
        Path to the target file.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    search_path = Path(search_dir)
    # Check directly first for speed
    direct = search_path / target_filename
    if direct.is_file():
        return direct

    # Recursively check subdirectories
    for root, _, files in os.walk(search_path):
        if target_filename in files:
            return Path(root) / target_filename

    raise FileNotFoundError(
        f"Could not locate '{target_filename}' inside '{search_dir}'."
    )


def load_raw_data(dataset_id: str = DATASET_IDENTIFIER) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download and load the TMDB 5000 movies and credits datasets via KaggleHub.

    Uses KaggleHub's built-in local caching so subsequent calls do not re-download.

    Args:
        dataset_id: Kaggle dataset identifier (default: "tmdb/tmdb-movie-metadata").

    Returns:
        Tuple of (movies_df, credits_df).

    Raises:
        DataLoaderError: If download or loading encounters an unrecoverable error.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise DataLoaderError(
            "kagglehub package is not installed. Please run `pip install kagglehub`."
        ) from exc

    logger.info("Accessing dataset '%s' via kagglehub...", dataset_id)
    try:
        download_path = kagglehub.dataset_download(dataset_id)
    except Exception as exc:
        error_msg = str(exc)
        auth_hint = (
            "\nTip: If Kaggle requires credentials, ensure `KAGGLE_USERNAME` and "
            "`KAGGLE_KEY` are set in your environment or ~/.kaggle/kaggle.json."
            if "credentials" in error_msg.lower() or "401" in error_msg or "unauthorized" in error_msg.lower()
            else ""
        )
        raise DataLoaderError(
            f"Failed to acquire dataset '{dataset_id}' via kagglehub: {error_msg}{auth_hint}"
        ) from exc

    if not download_path or not os.path.exists(download_path):
        raise DataLoaderError(
            f"kagglehub returned an invalid or nonexistent path: {download_path}"
        )

    logger.info("Dataset available at: %s", download_path)

    # Locate both CSV files
    try:
        movies_csv = find_csv_file(download_path, MOVIES_FILENAME)
        credits_csv = find_csv_file(download_path, CREDITS_FILENAME)
    except FileNotFoundError as exc:
        raise DataLoaderError(f"Dataset verification failed: {exc}") from exc

    logger.info("Loading '%s' and '%s'...", movies_csv.name, credits_csv.name)
    try:
        movies_df = pd.read_csv(movies_csv)
        credits_df = pd.read_csv(credits_csv)
    except Exception as exc:
        raise DataLoaderError(f"Error reading CSV files: {exc}") from exc

    # Sanity check on required columns
    if "id" not in movies_df.columns:
        raise DataLoaderError(f"Expected 'id' column in {MOVIES_FILENAME}, got {movies_df.columns.tolist()}")
    if "movie_id" not in credits_df.columns and "id" not in credits_df.columns:
        raise DataLoaderError(f"Expected 'movie_id' or 'id' column in {CREDITS_FILENAME}, got {credits_df.columns.tolist()}")

    logger.info(
        "Successfully loaded %d movies and %d credits records.",
        len(movies_df),
        len(credits_df),
    )
    return movies_df, credits_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m, c = load_raw_data()
    print(f"Movies: {m.shape}, Credits: {c.shape}")
