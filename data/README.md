# Dataset Information

This recommendation system uses the **TMDB 5000 Movie Metadata** dataset from Kaggle (`tmdb/tmdb-movie-metadata`).

The dataset contains two key files:
1. `tmdb_5000_movies.csv` - Contains movie attributes such as budget, genres, homepage, id, keywords, original_language, original_title, overview, popularity, production_companies, production_countries, release_date, revenue, runtime, spoken_languages, status, tagline, title, vote_average, vote_count.
2. `tmdb_5000_credits.csv` - Contains cast and crew information including movie_id, title, cast, and crew.

## Automatic KaggleHub Integration
Manual download or file copying is **not required**.
When `utils/data_loader.py` is invoked, `kagglehub.dataset_download("tmdb/tmdb-movie-metadata")` automatically downloads and caches the dataset, locating both CSV files seamlessly.
