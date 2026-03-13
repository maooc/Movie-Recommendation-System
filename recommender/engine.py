"""
Movie Recommendation Engine - Single Source of Truth
Consolidates recommendation logic from views.py and training/infer.py
"""
import logging
import hashlib
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from difflib import get_close_matches

import pandas as pd
import numpy as np
from scipy.sparse import load_npz, issparse
import json

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
_RECOMMENDATION_CACHE: Dict[str, Tuple[Dict, float]] = {}
_CACHE_TTL = 300
_CACHE_MAX_SIZE = 1000


class ModelLoadError(Exception):
    """Exception raised when model loading fails"""
    pass


class MovieRecommender:
    """
    Unified Movie Recommender Engine
    
    This is the single source of truth for all recommendation logic,
    used by both web views and API endpoints.
    """
    
    def __init__(self, model_dir: str = 'models', progress_callback=None):
        """
        Initialize with trained model directory
        
        Args:
            model_dir: Directory containing trained model artifacts
            progress_callback: Optional callback function for progress updates
        """
        self.model_dir = Path(model_dir)
        self.metadata: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self._sparse_similarity = False
        self._sparse_matrix = None
        self.title_to_idx: Optional[Dict[str, int]] = None
        self.config: Optional[Dict] = None
        self._loaded = False
        self._load_models(progress_callback)
    
    def _load_models(self, progress_callback=None):
        """Load all model artifacts with progress tracking"""
        logger.info(f"Loading models from {self.model_dir}...")
        
        if progress_callback:
            progress_callback(10)
        
        self._load_metadata()
        if progress_callback:
            progress_callback(25)
        
        self._load_similarity_matrix()
        if progress_callback:
            progress_callback(65)
        
        self._load_title_mapping()
        if progress_callback:
            progress_callback(80)
        
        self._load_config()
        if progress_callback:
            progress_callback(100)
        
        self._loaded = True
        logger.info(f"Loaded {self.config['n_movies']:,} movies successfully")
    
    def _load_metadata(self):
        """Load movie metadata from parquet file"""
        metadata_path = self.model_dir / 'movie_metadata.parquet'
        if not metadata_path.exists():
            raise ModelLoadError(f"Metadata file not found: {metadata_path}")
        self.metadata = pd.read_parquet(metadata_path)
    
    def _load_similarity_matrix(self):
        """Load similarity matrix, keeping sparse format for memory efficiency"""
        npz_path = self.model_dir / 'similarity_matrix.npz'
        npy_path = self.model_dir / 'similarity_matrix.npy'
        
        if npz_path.exists():
            self._sparse_matrix = load_npz(npz_path)
            self._sparse_similarity = True
            logger.info("Loaded sparse similarity matrix (memory efficient mode)")
        elif npy_path.exists():
            self.similarity_matrix = np.load(npy_path)
            self._sparse_similarity = False
            logger.info("Loaded dense similarity matrix")
        else:
            raise ModelLoadError(
                f"Similarity matrix not found. Expected {npz_path} or {npy_path}"
            )
    
    def _load_title_mapping(self):
        """Load title to index mapping"""
        mapping_path = self.model_dir / 'title_to_idx.json'
        if not mapping_path.exists():
            raise ModelLoadError(f"Title mapping file not found: {mapping_path}")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.title_to_idx = json.load(f)
    
    def _load_config(self):
        """Load model configuration"""
        config_path = self.model_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {'n_movies': len(self.title_to_idx)}
    
    def _get_similarity_row(self, idx: int) -> np.ndarray:
        """
        Get similarity scores for a single movie index.
        Avoids converting entire sparse matrix to dense.
        """
        if self._sparse_similarity:
            row = self._sparse_matrix[idx].toarray().flatten()
            return row
        return self.similarity_matrix[idx]
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded
    
    @property
    def n_movies(self) -> int:
        """Get number of movies in the model"""
        return len(self.title_to_idx) if self.title_to_idx else 0
    
    def find_movie(self, title: str, threshold: float = 0.6) -> Optional[str]:
        """
        Fuzzy search for movie title
        
        Args:
            title: Movie title to search
            threshold: Similarity threshold (0-1)
        
        Returns:
            Best matching title or None
        """
        matches = get_close_matches(title, self.title_to_idx.keys(), n=1, cutoff=threshold)
        return matches[0] if matches else None
    
    def search_movies(self, query: str, n: int = 20, min_rating: float = None) -> List[str]:
        """
        Search movies by partial title match
        
        Args:
            query: Search query
            n: Maximum number of results
            min_rating: Optional minimum rating filter
        
        Returns:
            List of matching movie titles
        """
        query_lower = query.lower()
        matches = []
        
        for title in self.title_to_idx.keys():
            if query_lower in title.lower():
                if min_rating is not None:
                    idx = self.title_to_idx[title]
                    rating = self.metadata.iloc[idx]['vote_average']
                    if pd.isna(rating) or rating < min_rating:
                        continue
                matches.append(title)
                if len(matches) >= n:
                    break
        
        return matches
    
    def _extract_year(self, release_date) -> Optional[int]:
        """
        Safely extract year from release_date field.
        Handles various formats and missing values.
        """
        if pd.isna(release_date):
            return None
        
        try:
            release_str = str(release_date).strip()
            if not release_str or release_str.lower() in ('nan', 'none', 'unknown', ''):
                return None
            
            if '-' in release_str:
                year_str = release_str.split('-')[0]
            elif '/' in release_str:
                year_str = release_str.split('/')[-1]
            else:
                year_str = release_str[:4]
            
            year = int(year_str)
            if 1900 <= year <= 2100:
                return year
            return None
        except (ValueError, TypeError, IndexError):
            return None
    
    def _parse_genres(self, genres_input) -> List[str]:
        """
        Parse genres input to list of genre strings.
        Accepts comma-separated string or list.
        """
        if genres_input is None:
            return []
        
        if isinstance(genres_input, list):
            return [g.strip().lower().replace(' ', '') for g in genres_input if g]
        
        if isinstance(genres_input, str):
            genres = [g.strip().lower().replace(' ', '') for g in genres_input.split(',') if g.strip()]
            return genres
        
        return []
    
    def _movie_matches_genres(self, movie_genres: Any, filter_genres: List[str]) -> bool:
        """
        Check if movie genres match any of the filter genres.
        """
        if not filter_genres:
            return True
        
        if not isinstance(movie_genres, list):
            return False
        
        movie_genres_normalized = [g.lower().replace(' ', '') for g in movie_genres if g]
        
        return any(fg in movie_genres_normalized for fg in filter_genres)
    
    def get_recommendations(
        self,
        movie_title: str,
        n: int = 15,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        genres: Optional[str] = None,
        min_rating: Optional[float] = None,
        exclude_same_company: bool = False
    ) -> Dict:
        """
        Get movie recommendations with advanced filtering
        
        Args:
            movie_title: Title of the movie to base recommendations on
            n: Number of recommendations to return (default: 15)
            min_year: Minimum release year filter
            max_year: Maximum release year filter
            genres: Comma-separated list of genres to filter by
            min_rating: Minimum vote_average (0-10)
            exclude_same_company: Exclude movies by same production company
        
        Returns:
            Dictionary with recommendations and metadata
        """
        matched_title = self.find_movie(movie_title)
        
        if not matched_title:
            suggestions = self.search_movies(movie_title, n=5)
            return {
                'error': f"Movie '{movie_title}' not found",
                'suggestions': suggestions,
                'query_movie': None,
                'source_movie': None,
                'recommendations': [],
                'filters_applied': self._build_filters_summary(
                    n, min_year, max_year, genres, min_rating, exclude_same_company
                )
            }
        
        movie_idx = self.title_to_idx[matched_title]
        source_movie = self.metadata.iloc[movie_idx]
        
        sim_scores = self._get_similarity_row(movie_idx)
        indexed_scores = list(enumerate(sim_scores))
        indexed_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        indexed_scores = indexed_scores[1:]
        
        filter_genres = self._parse_genres(genres)
        source_company = source_movie.get('primary_company')
        
        recommendations = []
        filtered_count = {'year': 0, 'rating': 0, 'genre': 0, 'company': 0}
        
        for idx, score in indexed_scores:
            if len(recommendations) >= n:
                break
            
            movie = self.metadata.iloc[idx]
            
            if min_year is not None or max_year is not None:
                year = self._extract_year(movie.get('release_date'))
                if year is None:
                    filtered_count['year'] += 1
                    continue
                if min_year is not None and year < min_year:
                    filtered_count['year'] += 1
                    continue
                if max_year is not None and year > max_year:
                    filtered_count['year'] += 1
                    continue
            
            if min_rating is not None:
                rating = movie.get('vote_average')
                if pd.isna(rating) or rating < min_rating:
                    filtered_count['rating'] += 1
                    continue
            
            if filter_genres:
                movie_genres = movie.get('genres', [])
                if not self._movie_matches_genres(movie_genres, filter_genres):
                    filtered_count['genre'] += 1
                    continue
            
            if exclude_same_company and source_company:
                movie_company = movie.get('primary_company')
                if pd.notna(movie_company) and movie_company == source_company:
                    filtered_count['company'] += 1
                    continue
            
            recommendations.append(self._build_recommendation_entry(movie, score, len(recommendations) + 1))
        
        filters_summary = self._build_filters_summary(
            n, min_year, max_year, genres, min_rating, exclude_same_company
        )
        
        return {
            'query_movie': matched_title,
            'source_movie': {
                'production': self._safe_str(source_movie.get('primary_company'), 'Unknown'),
                'rating': self._format_rating(source_movie.get('vote_average')),
                'genres': self._format_genres(source_movie.get('genres', [])),
                'release_date': self._safe_str(source_movie.get('release_date'), 'Unknown')
            },
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'filters_applied': filters_summary,
            'filtered_stats': filtered_count if any(filtered_count.values()) else None
        }
    
    def _build_recommendation_entry(self, movie: pd.Series, score: float, rank: int) -> Dict:
        """Build a single recommendation entry"""
        return {
            'rank': rank,
            'title': movie.get('title', 'Unknown'),
            'release_date': self._safe_str(movie.get('release_date'), 'Unknown'),
            'production': self._safe_str(movie.get('primary_company'), 'Unknown'),
            'genres': self._format_genres(movie.get('genres', [])),
            'rating': self._format_rating(movie.get('vote_average')),
            'votes': self._format_votes(movie.get('vote_count')),
            'similarity_score': round(float(score), 3),
            'imdb_id': self._safe_str(movie.get('imdb_id')),
            'poster_url': self._build_poster_url(movie.get('poster_path')),
            'google_link': self._build_google_link(movie.get('title')),
            'imdb_link': self._build_imdb_link(movie.get('imdb_id'))
        }
    
    def _build_filters_summary(
        self,
        n: int,
        min_year: Optional[int],
        max_year: Optional[int],
        genres: Optional[str],
        min_rating: Optional[float],
        exclude_same_company: bool
    ) -> Dict:
        """Build a summary of applied filters"""
        filters = {'n': n}
        
        if min_year is not None:
            filters['min_year'] = min_year
        if max_year is not None:
            filters['max_year'] = max_year
        if genres:
            filters['genres'] = genres
        if min_rating is not None:
            filters['min_rating'] = min_rating
        if exclude_same_company:
            filters['exclude_same_company'] = True
        
        return filters
    
    @staticmethod
    def _safe_str(value, default: str = None) -> Optional[str]:
        """Safely convert value to string"""
        if pd.isna(value):
            return default
        return str(value)
    
    @staticmethod
    def _format_rating(rating) -> str:
        """Format rating for display"""
        if pd.isna(rating):
            return 'N/A'
        try:
            return f"{float(rating):.1f}/10"
        except (ValueError, TypeError):
            return 'N/A'
    
    @staticmethod
    def _format_votes(votes) -> str:
        """Format vote count for display"""
        if pd.isna(votes):
            return 'N/A'
        try:
            return f"{int(votes):,}"
        except (ValueError, TypeError):
            return 'N/A'
    
    @staticmethod
    def _format_genres(genres) -> str:
        """Format genres list for display"""
        if not isinstance(genres, list) or not genres:
            return 'N/A'
        return ', '.join(str(g) for g in genres[:3])
    
    @staticmethod
    def _build_poster_url(poster_path) -> Optional[str]:
        """Build TMDB poster URL"""
        if pd.isna(poster_path):
            return None
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    
    @staticmethod
    def _build_google_link(title) -> Optional[str]:
        """Build Google search link"""
        if not title:
            return None
        return f"https://www.google.com/search?q={'+'.join(str(title).split())}+movie"
    
    @staticmethod
    def _build_imdb_link(imdb_id) -> Optional[str]:
        """Build IMDb link"""
        if pd.isna(imdb_id) or not imdb_id:
            return None
        return f"https://www.imdb.com/title/{imdb_id}"


class RecommenderManager:
    """
    Singleton manager for the MovieRecommender instance.
    Handles background loading, progress tracking, and caching.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._recommender: Optional[MovieRecommender] = None
        self._loading = False
        self._progress = 0
        self._error: Optional[str] = None
        self._loading_thread: Optional[threading.Thread] = None
        self._model_dir: Optional[str] = None
        self._initialized = True
    
    def start_loading(self, model_dir: str = None):
        """Start loading model in background"""
        with self._lock:
            if self._recommender is not None:
                return
            
            if self._loading:
                return
            
            self._loading = True
            self._progress = 0
            self._error = None
            self._model_dir = model_dir
            
            if self._loading_thread is None or not self._loading_thread.is_alive():
                self._loading_thread = threading.Thread(
                    target=self._load_model_background,
                    daemon=True
                )
                self._loading_thread.start()
    
    def _load_model_background(self):
        """Load model in background thread"""
        try:
            def progress_callback(progress):
                with self._lock:
                    self._progress = progress
            
            self._recommender = MovieRecommender(
                model_dir=self._model_dir or 'models',
                progress_callback=progress_callback
            )
            
            with self._lock:
                self._loading = False
                self._progress = 100
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            with self._lock:
                self._loading = False
                self._error = str(e)
            logger.error(f"Failed to load recommender: {e}")
    
    def get_status(self) -> Dict:
        """Get current loading status"""
        with self._lock:
            if self._error:
                return {
                    'loaded': False,
                    'progress': 0,
                    'status': 'error',
                    'error': self._error
                }
            elif self._recommender is not None:
                return {
                    'loaded': True,
                    'progress': 100,
                    'status': 'ready',
                    'n_movies': self._recommender.n_movies
                }
            elif self._loading:
                return {
                    'loaded': False,
                    'progress': self._progress,
                    'status': 'loading'
                }
            else:
                return {
                    'loaded': False,
                    'progress': 0,
                    'status': 'initializing'
                }
    
    def get_recommender(self) -> Optional[MovieRecommender]:
        """Get the recommender instance if loaded"""
        with self._lock:
            return self._recommender
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        with self._lock:
            return self._recommender is not None
    
    def get_error(self) -> Optional[str]:
        """Get any loading error"""
        with self._lock:
            return self._error


def get_cache_key(movie_title: str, **filters) -> str:
    """Generate cache key from movie title and filters"""
    filter_str = json.dumps(filters, sort_keys=True, default=str)
    combined = f"{movie_title.lower()}:{filter_str}"
    return hashlib.md5(combined.encode()).hexdigest()


def get_cached_recommendations(cache_key: str) -> Optional[Dict]:
    """Get cached recommendations if still valid"""
    with _CACHE_LOCK:
        if cache_key in _RECOMMENDATION_CACHE:
            result, timestamp = _RECOMMENDATION_CACHE[cache_key]
            if time.time() - timestamp < _CACHE_TTL:
                return result
            else:
                del _RECOMMENDATION_CACHE[cache_key]
    return None


def set_cached_recommendations(cache_key: str, result: Dict):
    """Cache recommendations with TTL"""
    with _CACHE_LOCK:
        if len(_RECOMMENDATION_CACHE) >= _CACHE_MAX_SIZE:
            oldest_key = min(
                _RECOMMENDATION_CACHE.keys(),
                key=lambda k: _RECOMMENDATION_CACHE[k][1]
            )
            del _RECOMMENDATION_CACHE[oldest_key]
        
        _RECOMMENDATION_CACHE[cache_key] = (result, time.time())


def get_recommendations_with_cache(
    recommender: MovieRecommender,
    movie_title: str,
    use_cache: bool = True,
    **filters
) -> Dict:
    """
    Get recommendations with optional caching.
    
    Args:
        recommender: MovieRecommender instance
        movie_title: Movie title to search
        use_cache: Whether to use cache
        **filters: Filter parameters (n, min_year, max_year, genres, min_rating, exclude_same_company)
    
    Returns:
        Recommendation results dictionary
    """
    cache_key = get_cache_key(movie_title, **filters)
    
    if use_cache:
        cached = get_cached_recommendations(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for: {movie_title}")
            return cached
    
    result = recommender.get_recommendations(movie_title, **filters)
    
    if use_cache and 'error' not in result:
        set_cached_recommendations(cache_key, result)
    
    return result


def clear_cache():
    """Clear all cached recommendations"""
    global _RECOMMENDATION_CACHE
    with _CACHE_LOCK:
        _RECOMMENDATION_CACHE = {}


def get_cache_stats() -> Dict:
    """Get cache statistics"""
    with _CACHE_LOCK:
        return {
            'size': len(_RECOMMENDATION_CACHE),
            'max_size': _CACHE_MAX_SIZE,
            'ttl_seconds': _CACHE_TTL
        }
