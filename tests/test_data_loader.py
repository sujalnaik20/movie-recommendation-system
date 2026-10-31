"""Unit tests for dataset loading and KaggleHub integration."""

import os
from pathlib import Path
import pytest
import pandas as pd

from utils.data_loader import (
    DATASET_IDENTIFIER,
    DataLoaderError,
    find_csv_file,
    load_raw_data,
)


def test_find_csv_file_success(tmp_path: Path):
    """Test locating a CSV file in the root or nested directory."""
    # Direct file
    f1 = tmp_path / "test1.csv"
    f1.write_text("a,b\n1,2", encoding="utf-8")
    assert find_csv_file(tmp_path, "test1.csv") == f1

    # Nested file
    sub = tmp_path / "subdir" / "nested"
    sub.mkdir(parents=True)
    f2 = sub / "test2.csv"
    f2.write_text("x,y\n3,4", encoding="utf-8")
    assert find_csv_file(tmp_path, "test2.csv") == f2


def test_find_csv_file_not_found(tmp_path: Path):
    """Test exception when target file does not exist."""
    with pytest.raises(FileNotFoundError):
        find_csv_file(tmp_path, "nonexistent.csv")


def test_load_raw_data_kagglehub():
    """Verify automatic KaggleHub integration loads both TMDB 5000 datasets."""
    movies_df, credits_df = load_raw_data()

    assert isinstance(movies_df, pd.DataFrame)
    assert isinstance(credits_df, pd.DataFrame)
    assert not movies_df.empty
    assert not credits_df.empty

    # Verify key columns exist
    assert "id" in movies_df.columns
    assert "title" in movies_df.columns
    assert "overview" in movies_df.columns
    assert "genres" in movies_df.columns
    assert "keywords" in movies_df.columns

    assert "movie_id" in credits_df.columns or "id" in credits_df.columns
    assert "cast" in credits_df.columns
    assert "crew" in credits_df.columns

    # Verify expected row count (~4800 movies)
    assert len(movies_df) >= 4500
    assert len(credits_df) >= 4500
