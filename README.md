# Movie Recommendation System

A production-grade, end-to-end Machine Learning Movie Recommendation Platform developed in Python with **TF-IDF Vectorization**, **Cosine Similarity**, **KaggleHub** automatic dataset acquisition, and a simple dark **Streamlit** interface.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Machine Learning Methodology](#machine-learning-methodology)
4. [Technology Stack](#technology-stack)
5. [Project Architecture](#project-architecture)
6. [Dataset & Automatic KaggleHub Integration](#dataset--automatic-kagglehub-integration)
7. [Installation & Setup](#installation--setup)
8. [TMDB API Configuration (Optional)](#tmdb-api-configuration-optional)
9. [Running the Application](#running-the-application)
10. [Automated Testing & Validation](#automated-testing--validation)
11. [Recommendation Quality & Edge Case Handling](#recommendation-quality--edge-case-handling)
12. [Known Limitations & Future Roadmap](#known-limitations--future-roadmap)

---

## 1. Project Overview

This project is a movie recommendation system that analyzes actual film metadata—including plot narratives, thematic keywords, genres, directors, and top cast members—to generate genuine content-based movie recommendations.

Unlike hardcoded demonstration systems or black-box collaborative filtering that suffers from cold-start problems, CineVerse AI calculates high-dimensional textual similarity using **Scikit-learn** and **Scipy Sparse Matrices**, serving recommendations in milliseconds with minimal RAM consumption.

---

## 2. Key Features

- **Live Movie Search & Suggestion Engine**: Fast case-insensitive search with prefix matching, substring scanning, and typo/spelling tolerance (e.g. typing `"Interstelar"` resolves to `"Interstellar (2014)"`).
- **Title Disambiguation**: Distinguishes between films with identical or similar titles using release years (e.g., *Batman (1989)* vs *Batman Begins (2005)*).
- **Interactive Continuous Discovery Loop**: Selecting any recommended movie instantly generates a new set of 10 similar movies, enabling endless exploration.
- **Explainable Recommendations**: Transparent AI explanations highlight why each movie was recommended (e.g., *Directed by Christopher Nolan*, *Themes: space travel, nasa*, *Shared genres: Sci-Fi, Adventure*).
- **Curated Feeds**: Explore trending popular films and critically acclaimed masterpieces directly from the dashboard.
- **Offline-Resilient TMDB Integration**: Live official posters and backdrops when TMDB API credentials are provided, with clean SVG poster generators and dataset fallbacks when offline.
- **Zero-Manual-Download Workflow**: Automatically downloads and caches Kaggle's TMDB 5000 Movie Metadata dataset via `kagglehub`.

---

## 3. Machine Learning Methodology

### Content-Based Filtering Architecture

The recommendation engine models each film as a vector in a high-dimensional feature space derived from its composite attributes:

1. **Metadata Feature Extraction & Normalization**:
   - **Director**: Extracted from crew metadata, repeated ($3\times$ weight), and collapsed into a single token (e.g. `Christopher Nolan` $\rightarrow$ `christophernolan`) to prevent token collision with unrelated persons.
   - **Genres**: Multi-label lists extracted and weighted ($3\times$).
   - **Keywords**: Thematic plot tags extracted and weighted ($2\times$).
   - **Top Cast**: Top 4 actors extracted, collapsed, and weighted ($2\times$).
   - **Overview**: Full narrative synopsis included as base context.

2. **TF-IDF Vectorization**:
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
   - Vocabulary size: 7,500 features.
   - N-gram range: $(1, 2)$ unigrams and bigrams (capturing terms like `"science fiction"`, `"time travel"`, `"secret agent"`).
   - Stop words: Standard English stop word elimination.
   - Sublinear term-frequency scaling applied to dampen the impact of repetitive mentions.

3. **Sparse Cosine Similarity Computation**:
   $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
   - The TF-IDF matrix is stored on disk and loaded as a **Compressed Sparse Row matrix (`scipy.sparse.csr_matrix`)**, consuming only **~2.5 MB** of RAM instead of a dense matrix that would require over 180 MB.
   - Similarities are computed on the fly for the selected movie vector against all 4,800+ films in **$< 2\text{ ms}$**.

---

## 4. Technology Stack

- **Language**: Python 3.10+ (tested on Python 3.14)
- **Machine Learning & NLP**: `scikit-learn`, `scipy`, `numpy`, `pandas`
- **Frontend Framework**: `streamlit`
- **Data Acquisition**: `kagglehub`
- **API & Networking**: `requests`, `python-dotenv`
- **Model Serialization**: `joblib`
- **Testing**: `pytest`

---

## 5. Project Architecture

```text
movie-recommendation-system/
│
├── app.py                     # Streamlit frontend & interactive discovery UI
├── train_model.py             # End-to-end training & artifact persistence pipeline
├── recommender.py             # Search, Cosine Similarity, and recommendation logic
│
├── data/
│   └── README.md              # Dataset documentation
│
├── models/
│   ├── .gitkeep
│   ├── movie_list.pkl         # Cleaned DataFrame with structured metadata
│   ├── tfidf_vectorizer.pkl   # Fitted TfidfVectorizer
│   ├── tfidf_matrix.pkl       # Sparse TF-IDF CSR matrix
│   └── model_metadata.json    # Training parameters & timestamp
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # KaggleHub dataset acquisition & CSV locator
│   ├── preprocessing.py       # Data cleaning, JSON parsing & weighted tags
│   ├── tmdb_api.py            # TMDB API client with disk caching & offline SVG fallback
│   └── cache_manager.py       # Artifact validation, serialization, and safe loading
│
├── assets/
│   └── config.toml            # Streamlit dark theme
│   └── logo.png               # Custom brand icon
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py    # KaggleHub & CSV ingestion tests
│   ├── test_recommender.py    # Recommendation accuracy & benchmark tests
│   └── test_search.py         # Search, disambiguation & typo tolerance tests
│
├── .env                       # Local environment variables (git-ignored)
├── .env.example               # Template for environment configuration
├── .gitignore                 # Secrets, cache, and artifact ignore rules
├── requirements.txt           # Project dependencies
└── README.md                  # Comprehensive project documentation
```

---

## 6. Dataset & Automatic KaggleHub Integration

The application integrates directly with Kaggle via **KaggleHub** using dataset identifier:
`tmdb/tmdb-movie-metadata`

Files acquired automatically:
- `tmdb_5000_movies.csv` (4,803 records, 20 columns)
- `tmdb_5000_credits.csv` (4,803 records, 4 columns)

### No Manual Downloads Required
When running `app.py` or `train_model.py`, the system calls `kagglehub.dataset_download("tmdb/tmdb-movie-metadata")`. KaggleHub caches files locally in `~/.cache/kagglehub/datasets/`—subsequent runs reuse the cached dataset immediately without re-downloading.

### Kaggle Authentication (If Prompted)
Public datasets typically download anonymously. If Kaggle requires credentials on your machine:
1. Log into your Kaggle account and navigate to **Settings -> Create New API Token** (`kaggle.json`).
2. Set environment variables or place `kaggle.json` in `~/.kaggle/`:
   ```bash
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_api_key"
   ```

---

## 7. Installation & Setup

### Step 1: Clone or navigate to the directory
```bash
cd "c:\Antigravity\Movie Recommendation"
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

---

## 8. TMDB API Configuration (Optional)

The recommendation engine is **fully functional offline** without an API key. To view official live movie posters and backdrops from The Movie Database (TMDB):

1. Obtain a free API key at [themoviedb.org](https://www.themoviedb.org/settings/api).
2. Create or edit your `.env` file in the project root:
   ```env
   TMDB_API_KEY=your_tmdb_api_key_here
   ```
3. When no key is provided, the application automatically uses styled SVG poster cards and dataset overviews.

---

## 9. Running the Application

Launch the Streamlit app with:

```bash
streamlit run app.py
```
*(Or `python -m streamlit run app.py`)*

Open your browser at `http://localhost:8501`.

### Pre-training (Optional)
The application automatically checks for trained models on startup and executes training if files are missing. You can also explicitly trigger model training from the CLI:
```bash
python train_model.py --force
```

---

## 10. Automated Testing & Validation

Run the comprehensive pytest suite:

```bash
python -m pytest -v
```

### Benchmark Validation
The test suite validates recommendation generation against 10 diverse benchmark films:
1. **Interstellar** (2014) — matches *Apollo 13*, *The Martian*, *2001: A Space Odyssey*
2. **Inception** (2010) — matches *Cypher*, *The Matrix*, *Source Code*
3. **The Dark Knight** (2008) — matches *The Dark Knight Rises*, *Batman Begins*, *Batman Returns*
4. **Avatar** (2009) — matches *Aliens*, *Star Trek Into Darkness*, *The Abyss*
5. **Titanic** (1997) — matches *Poseidon*, *The Abyss*, *Revolutionary Road*
6. **The Avengers** (2012) — matches *Avengers: Age of Ultron*, *Iron Man 2*, *Captain America*
7. **The Conjuring** (2013) — matches *The Conjuring 2*, *Annabelle*, *Insidious*
8. **Finding Nemo** (2003) — matches *Shark Tale*, *The Reef*, *Finding Dory*
9. **Jurassic Park** (1993) — matches *The Lost World: Jurassic Park*, *Jurassic World*
10. **The Matrix** (1999) — matches *The Matrix Reloaded*, *The Matrix Revolutions*, *Terminator*

---

## 11. Recommendation Quality & Edge Case Handling

- **Exclusion of Self**: The selected movie is guaranteed never to appear in its own recommendations.
- **Duplicate Prevention**: Movie IDs are tracked in a hash set to eliminate repeated titles.
- **Sparse Feature Memory**: Keeps feature matrices in sparse CSR format, operating comfortably on standard consumer laptops.
- **Cold-Start Fallback**: If a title shares minimal vocabulary, the engine gracefully falls back to genre-overlapping critically acclaimed films to ensure 10 recommendations are always delivered.

---

## 12. Known Limitations & Future Roadmap

- **Dataset Cutoff**: The TMDB 5000 dataset includes feature films released up to ~2016-2017. Later releases can be incorporated via TMDB API batch ingestion.
- **Future Enhancements**:
  - Hybrid Collaborative-Filtering: Incorporating user rating matrices with Matrix Factorization (SVD).
  - Neural Semantic Embeddings: Integrating sentence-transformers (`all-MiniLM-L6-v2`) for latent semantic understanding.
  - User Watchlist & Personalized Accounts: Persistent watchlists via SQLite / Supabase.
