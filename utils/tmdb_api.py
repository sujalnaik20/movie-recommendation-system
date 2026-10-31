"""TMDB (The Movie Database) API Client with robust caching and offline fallback.

Provides movie posters, backdrops, taglines, and runtime metadata while
gracefully degrading to offline mode when credentials or connectivity are absent.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file automatically
load_dotenv()

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_FILE = CACHE_DIR / "tmdb_cache.json"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p"
POSTER_SIZE = "w500"
BACKDROP_SIZE = "w1280"
DEFAULT_TIMEOUT = 4.0  # seconds


class TMDBClient:
    """Client for fetching movie metadata and image assets from TMDB API."""

    def __init__(self, api_key: Optional[str] = None, enable_cache: bool = True):
        self.api_key = api_key or os.getenv("TMDB_API_KEY", "").strip()
        self.enable_cache = enable_cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        if self.enable_cache:
            self._load_cache()

    def _load_cache(self) -> None:
        """Load disk cache if available."""
        if CACHE_FILE.is_file():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception as exc:
                logger.warning("Could not read TMDB cache: %s. Initializing empty cache.", exc)
                self._cache = {}

    def _save_cache(self) -> None:
        """Persist memory cache to disk safely."""
        if not self.enable_cache:
            return
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as exc:
            logger.warning("Could not write TMDB cache: %s", exc)

    @property
    def is_configured(self) -> bool:
        """Check if an API key is configured and non-empty."""
        return bool(self.api_key and self.api_key != "your_tmdb_api_key_here")

    def get_movie_details(self, tmdb_id: int) -> Dict[str, Any]:
        """Fetch details for a given TMDB movie ID.

        Returns a dictionary containing:
            - poster_url (str or None)
            - backdrop_url (str or None)
            - tagline (str)
            - runtime (int or None)
            - status (str)
            - tmdb_rating (float or None)

        If the API key is not configured or an error occurs, returns safe fallbacks.
        """
        cache_key = str(tmdb_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        fallback_result: Dict[str, Any] = {
            "poster_url": None,
            "backdrop_url": None,
            "tagline": "",
            "runtime": None,
            "status": "offline_or_uncached",
            "tmdb_rating": None,
        }

        if not self.is_configured:
            # Offline mode: gracefully return empty image URLs without crashing
            return fallback_result

        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        params = {"api_key": self.api_key, "language": "en-US"}

        try:
            response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                poster_path = data.get("poster_path")
                backdrop_path = data.get("backdrop_path")

                details: Dict[str, Any] = {
                    "poster_url": f"{TMDB_IMAGE_BASE_URL}/{POSTER_SIZE}{poster_path}"
                    if poster_path
                    else None,
                    "backdrop_url": f"{TMDB_IMAGE_BASE_URL}/{BACKDROP_SIZE}{backdrop_path}"
                    if backdrop_path
                    else None,
                    "tagline": data.get("tagline", "") or "",
                    "runtime": data.get("runtime"),
                    "status": "online",
                    "tmdb_rating": data.get("vote_average"),
                }
                self._cache[cache_key] = details
                self._save_cache()
                return details

            if response.status_code == 401:
                logger.warning("TMDB API key invalid or unauthorized (HTTP 401).")
            elif response.status_code == 404:
                logger.info("Movie ID %s not found on TMDB (HTTP 404).", tmdb_id)
            elif response.status_code == 429:
                logger.warning("TMDB API rate limit exceeded (HTTP 429).")
            else:
                logger.warning("TMDB API returned HTTP %s for movie %s.", response.status_code, tmdb_id)

        except requests.exceptions.Timeout:
            logger.warning("TMDB API request timed out for movie %s.", tmdb_id)
        except requests.exceptions.RequestException as exc:
            logger.warning("TMDB network request failed: %s", exc)

        return fallback_result


def generate_fallback_poster_svg(title: str, year: Optional[int] = None, genre: Optional[str] = None) -> str:
    """Generate a clean SVG data URI placeholder for movies without TMDB posters.

    Produces an elegant, minimalist cinematic card so no random or broken images are shown.
    """
    safe_title = (title[:22] + "...") if len(title) > 25 else title
    safe_year = str(year) if year else ""
    safe_genre = genre.upper() if genre else "FEATURE FILM"

    # Minimalist dark gradient card with movie typography
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 450" width="300" height="450">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#141923" />
      <stop offset="100%" stop-color="#090b10" />
    </linearGradient>
    <linearGradient id="accent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#3b82f6" />
      <stop offset="100%" stop-color="#8b5cf6" />
    </linearGradient>
  </defs>
  <rect width="300" height="450" rx="12" fill="url(#bg)" />
  <rect x="20" y="20" width="260" height="410" rx="8" fill="none" stroke="#ffffff" stroke-opacity="0.08" stroke-width="1.5" />
  <circle cx="150" cy="170" r="48" fill="#1e293b" fill-opacity="0.6" stroke="url(#accent)" stroke-width="2" />
  <path d="M 142 152 L 166 170 L 142 188 Z" fill="#93c5fd" />
  <text x="150" y="260" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="700" letter-spacing="2.5" fill="#38bdf8" text-anchor="middle">{safe_genre}</text>
  <text x="150" y="295" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="17" font-weight="700" fill="#f8fafc" text-anchor="middle">{safe_title}</text>
  <text x="150" y="325" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="500" fill="#94a3b8" text-anchor="middle">{safe_year}</text>
</svg>"""

    encoded = urllib.parse.quote(svg)
    return f"data:image/svg+xml;utf8,{encoded}"


# Global client instance
tmdb_client = TMDBClient()
