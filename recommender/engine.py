"""
Movie Recommendation Engine
Core implementation for movie recommendations with advanced filtering
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
from difflib import get_close_matches

import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import json

logger = logging.getLogger(__name__)


class MovieRecommender:
    """Integrated recommender system with advanced filtering capabilities"""
    
    def __init__(self, model_dir='models', progress_callback=None):
        """Initialize with trained model directory"""
        self.model_dir = Path(model_dir)
        self.metadata = None
        self.similarity_matrix = None
        self.title_to_idx = None
        self.config = None
        self._load_models(progress_callback)
    
    def _load_models(self, progress_callback=None):
        """Load all model artifacts with progress tracking"""
        logger.info(f"Loading models from {self.model_dir}...")
        
        # Load metadata (25%)
        if progress_callback:
            progress_callback(10)
        self.metadata = pd.read_parquet(self.model_dir / 'movie_metadata.parquet')
        if progress_callback:
            progress_callback(25)
        
        # Load similarity matrix (sparse or dense) (50%)
        if progress_callback:
            progress_callback(40)
        if (self.model_dir / 'similarity_matrix.npz').exists():
            # Don't convert to array immediately to save memory
            self.similarity_matrix = load_npz(self.model_dir / 'similarity_matrix.npz')
        else:
            self.similarity_matrix = np.load(self.model_dir / 'similarity_matrix.npy')
        if progress_callback:
            progress_callback(65)
        
        # Load title mapping (75%)
        with open(self.model_dir / 'title_to_idx.json', 'r') as f:
            self.title_to_idx = json.load(f)
        if progress_callback:
            progress_callback(80)
        
        # Load config (100%)
        with open(self.model_dir / 'config.json', 'r') as f:
            self.config = json.load(f)
        if progress_callback:
            progress_callback(100)
        
        logger.info(f"Loaded {self.config['n_movies']:,} movies successfully")
    
    def find_movie(self, title: str, threshold: float = 0.6) -> Optional[str]:
        """Find closest matching movie title"""
        matches = get_close_matches(title, self.title_to_idx.keys(), n=1, cutoff=threshold)
        return matches[0] if matches else None
    
    def search_movies(self, query: str, n: int = 20, min_rating: float = None) -> List[str]:
        """Search movies by partial title"""
        query_lower = query.lower()
        matches = []
        
        for title in self.title_to_idx.keys():
            if query_lower in title.lower():
                if min_rating:
                    idx = self.title_to_idx[title]
                    rating = self.metadata.iloc[idx]['vote_average']
                    if rating < min_rating:
                        continue
                matches.append(title)
        
        return matches[:n]
    
    def get_recommendations(
        self,
        movie_title: str,
        n: int = 15,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        genres: Optional[List[str]] = None,
        min_rating: Optional[float] = None,
        exclude_same_company: bool = False
    ) -> Dict:
        """Get movie recommendations with advanced filtering"""
        matched_title = self.find_movie(movie_title)
        if not matched_title:
            return {'error': f"Movie '{movie_title}' not found", 'suggestions': self.search_movies(movie_title, 5)}
        
        movie_idx = self.title_to_idx[matched_title]
        source_movie = self.metadata.iloc[movie_idx]
        
        # Get similarity scores
        if hasattr(self.similarity_matrix, 'toarray'):
            # For sparse matrix, only convert the specific row to array
            sim_scores = list(enumerate(self.similarity_matrix[movie_idx].toarray()[0]))
        else:
            # For dense matrix
            sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]  # Exclude self
        
        recommendations = []
        source_company = source_movie['primary_company']
        
        for idx, score in sim_scores:
            if len(recommendations) >= n:
                break
            
            movie = self.metadata.iloc[idx]
            
            # Year filter
            if min_year or max_year:
                try:
                    release_str = str(movie['release_date'])
                    if pd.notna(release_str) and len(release_str) >= 4:
                        year = int(release_str.split('-')[0]) if '-' in release_str else int(release_str[:4])
                        if min_year and year < min_year:
                            continue
                        if max_year and year > max_year:
                            continue
                except:
                    # Skip if release_date is invalid or empty
                    continue
            
            # Rating filter
            if min_rating and movie['vote_average'] < min_rating:
                continue
            
            # Genre filter
            if genres:
                movie_genres = movie['genres'] if isinstance(movie['genres'], list) else []
                movie_genres_lower = [g.lower().replace(' ', '') for g in movie_genres]
                genres_lower = [g.lower().replace(' ', '') for g in genres]
                if not any(g in movie_genres_lower for g in genres_lower):
                    continue
            
            # Company filter
            if exclude_same_company and movie['primary_company'] == source_company:
                continue
            
            recommendations.append({
                'title': movie['title'],
                'release_date': movie['release_date'] if pd.notna(movie['release_date']) else 'Unknown',
                'production': movie['primary_company'] if pd.notna(movie['primary_company']) else 'Unknown',
                'genres': ', '.join(movie['genres'][:3]) if isinstance(movie['genres'], list) else 'N/A',
                'rating': f"{movie['vote_average']:.1f}/10" if pd.notna(movie['vote_average']) else 'N/A',
                'votes': f"{movie['vote_count']:,}" if pd.notna(movie['vote_count']) else 'N/A',
                'similarity_score': f"{score:.3f}",
                'imdb_id': movie['imdb_id'] if pd.notna(movie['imdb_id']) else None,
                'poster_url': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if pd.notna(movie['poster_path']) else None,
                'google_link': f"https://www.google.com/search?q={'+'.join(movie['title'].split())}+movie",
                'imdb_link': f"https://www.imdb.com/title/{movie['imdb_id']}" if pd.notna(movie['imdb_id']) else None
            })
        
        return {
            'query_movie': matched_title,
            'source_movie': {
                'production': source_movie['primary_company'] if pd.notna(source_movie['primary_company']) else 'Unknown',
                'rating': f"{source_movie['vote_average']:.1f}/10" if pd.notna(source_movie['vote_average']) else 'N/A',
                'genres': ', '.join(source_movie['genres'][:3]) if isinstance(source_movie['genres'], list) else 'N/A'
            },
            'recommendations': recommendations
        }
