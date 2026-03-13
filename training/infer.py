"""
Advanced Movie Recommendation System - Inference Engine
Optimized for TMDB Movies Dataset 2023 (930K+ movies)

This script can be run standalone or imported as a module.
It uses the unified recommendation engine from recommender/engine.py
for consistent behavior with the web/API interfaces.
"""

import sys
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import json
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from recommender.engine import MovieRecommender as UnifiedRecommender
    USING_UNIFIED_ENGINE = True
except ImportError:
    USING_UNIFIED_ENGINE = False
    print("Note: Running in standalone mode (unified engine not available)")


class StandaloneMovieRecommender:
    """
    Standalone recommender for when the unified engine is not available.
    Provides the same interface as the unified engine.
    """
    
    def __init__(self, model_dir='./models'):
        """
        Initialize recommender with trained models
        
        Args:
            model_dir: Directory containing trained model artifacts
        """
        self.model_dir = Path(model_dir)
        self.metadata = None
        self.similarity_matrix = None
        self._sparse_similarity = False
        self._sparse_matrix = None
        self.title_to_idx = None
        self.config = None
        self._loaded = False
        self.load_models()
    
    def load_models(self):
        """Load all model artifacts"""
        print("Loading TMDB Movie Recommendation Engine...")
        
        self.metadata = pd.read_parquet(self.model_dir / 'movie_metadata.parquet')
        
        if (self.model_dir / 'similarity_matrix.npz').exists():
            print("Loading sparse similarity matrix...")
            self._sparse_matrix = load_npz(self.model_dir / 'similarity_matrix.npz')
            self._sparse_similarity = True
        else:
            print("Loading dense similarity matrix...")
            self.similarity_matrix = np.load(self.model_dir / 'similarity_matrix.npy')
            self._sparse_similarity = False
        
        with open(self.model_dir / 'title_to_idx.json', 'r') as f:
            self.title_to_idx = json.load(f)
        
        with open(self.model_dir / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        self._loaded = True
        print(f"Loaded {self.config['n_movies']:,} movies from {self.config.get('dataset', 'dataset')}")
        print(f"   Model ready for inference!")
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def n_movies(self) -> int:
        return len(self.title_to_idx) if self.title_to_idx else 0
    
    def _get_similarity_row(self, idx: int) -> np.ndarray:
        """Get similarity scores for a single movie index"""
        if self._sparse_similarity:
            return self._sparse_matrix[idx].toarray().flatten()
        return self.similarity_matrix[idx]
    
    def find_movie(self, title: str, threshold: float = 0.6) -> Optional[str]:
        """
        Fuzzy search for movie title
        
        Args:
            title: Movie title to search
            threshold: Similarity threshold (0-1)
        
        Returns:
            Best matching title or None
        """
        from difflib import get_close_matches
        matches = get_close_matches(title, self.title_to_idx.keys(), n=1, cutoff=threshold)
        return matches[0] if matches else None
    
    def search_movies(self, query: str, n: int = 10, min_rating: float = None) -> List[str]:
        """
        Search for movies by partial title match
        
        Args:
            query: Search query
            n: Number of results
            min_rating: Minimum rating filter
        
        Returns:
            List of matching movie titles
        """
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
    
    def _extract_year(self, release_date) -> Optional[int]:
        """Safely extract year from release_date field"""
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
        """Parse genres input to list of genre strings"""
        if genres_input is None:
            return []
        
        if isinstance(genres_input, list):
            return [g.strip().lower().replace(' ', '') for g in genres_input if g]
        
        if isinstance(genres_input, str):
            return [g.strip().lower().replace(' ', '') for g in genres_input.split(',') if g.strip()]
        
        return []
    
    def _movie_matches_genres(self, movie_genres, filter_genres: List[str]) -> bool:
        """Check if movie genres match any of the filter genres"""
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
        n_recommendations: int = None,
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
            n_recommendations: Alias for n (for backward compatibility)
            min_year: Minimum release year filter
            max_year: Maximum release year filter
            genres: Comma-separated string or list of genres to filter by
            min_rating: Minimum vote_average (0-10)
            exclude_same_company: Exclude movies by same production company
        
        Returns:
            Dictionary with recommendations and metadata
        """
        if n_recommendations is not None:
            n = n_recommendations
        
        matched_title = self.find_movie(movie_title)
        if not matched_title:
            suggestions = self.search_movies(movie_title, n=5)
            return {
                'error': f"Movie '{movie_title}' not found",
                'suggestions': suggestions if suggestions else "Try different spelling or search by partial title",
                'query_movie': None,
                'recommendations': []
            }
        
        if matched_title != movie_title:
            print(f"Found closest match: '{matched_title}'")
        
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
            
            recommendations.append({
                'rank': len(recommendations) + 1,
                'title': movie.get('title', 'Unknown'),
                'production': movie.get('primary_company') if pd.notna(movie.get('primary_company')) else 'N/A',
                'release_date': str(movie.get('release_date', 'Unknown')),
                'genres': movie.get('genres') if isinstance(movie.get('genres'), list) else [],
                'rating': f"{movie.get('vote_average', 0):.1f}/10" if pd.notna(movie.get('vote_average')) else 'N/A',
                'votes': f"{int(movie.get('vote_count', 0)):,}" if pd.notna(movie.get('vote_count')) else 'N/A',
                'similarity_score': float(score),
                'tmdb_id': int(movie.get('id')) if pd.notna(movie.get('id')) else None,
                'imdb_id': movie.get('imdb_id') if pd.notna(movie.get('imdb_id')) else None,
                'poster_url': f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if pd.notna(movie.get('poster_path')) else None,
                'google_link': f"https://www.google.com/search?q={'+'.join(str(movie.get('title', '')).split())}+movie",
                'imdb_link': f"https://www.imdb.com/title/{movie.get('imdb_id')}" if pd.notna(movie.get('imdb_id')) else None
            })
        
        return {
            'query_movie': matched_title,
            'source_movie': {
                'production': source_movie.get('primary_company') if pd.notna(source_movie.get('primary_company')) else 'N/A',
                'genres': source_movie.get('genres') if isinstance(source_movie.get('genres'), list) else [],
                'rating': f"{source_movie.get('vote_average', 0):.1f}/10",
                'release_date': str(source_movie.get('release_date', 'Unknown'))
            },
            'query_details': {
                'production': source_movie.get('primary_company') if pd.notna(source_movie.get('primary_company')) else 'N/A',
                'genres': source_movie.get('genres') if isinstance(source_movie.get('genres'), list) else [],
                'rating': f"{source_movie.get('vote_average', 0):.1f}/10",
                'release_date': str(source_movie.get('release_date', 'Unknown'))
            },
            'total_recommendations': len(recommendations),
            'recommendations': recommendations,
            'filtered_stats': filtered_count if any(filtered_count.values()) else None
        }
    
    def get_movie_details(self, movie_title: str) -> Dict:
        """Get detailed information about a movie"""
        matched_title = self.find_movie(movie_title)
        if not matched_title:
            return {'error': f"Movie '{movie_title}' not found"}
        
        idx = self.title_to_idx[matched_title]
        movie = self.metadata.iloc[idx]
        
        return {
            'title': movie['title'],
            'release_date': movie['release_date'],
            'production': movie['primary_company'] if pd.notna(movie['primary_company']) else 'N/A',
            'genres': movie['genres'] if isinstance(movie['genres'], list) else [],
            'rating': f"{movie['vote_average']:.1f}/10",
            'votes': f"{movie['vote_count']:,}",
            'popularity': f"{movie['popularity']:.1f}" if 'popularity' in movie else 'N/A',
            'overview': movie['overview'][:200] + '...' if len(str(movie.get('overview', ''))) > 200 else movie.get('overview', 'N/A'),
            'imdb_id': movie['imdb_id'] if pd.notna(movie['imdb_id']) else 'N/A',
            'poster_url': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if pd.notna(movie['poster_path']) else None
        }
    
    def get_top_rated(self, n: int = 10, min_votes: int = 1000, genres: List[str] = None) -> List[Dict]:
        """
        Get top-rated movies
        
        Args:
            n: Number of movies to return
            min_votes: Minimum vote count
            genres: Filter by genres
        
        Returns:
            List of top-rated movies
        """
        df = self.metadata[self.metadata['vote_count'] >= min_votes].copy()
        
        if genres:
            genres_lower = [g.lower().replace(' ', '') for g in genres]
            df = df[
                df['genres'].apply(
                    lambda x: any(
                        g in [genre.lower().replace(' ', '') for genre in (x if isinstance(x, list) else [])]
                        for g in genres_lower
                    ) if isinstance(x, list) else False
                )
            ]
        
        df = df.nlargest(n, 'vote_average')
        
        results = []
        for idx, row in df.iterrows():
            results.append({
                'title': row['title'],
                'rating': f"{row['vote_average']:.1f}/10",
                'votes': f"{row['vote_count']:,}",
                'release_date': row['release_date'],
                'genres': row['genres'] if isinstance(row['genres'], list) else [],
                'production': row['primary_company'] if pd.notna(row['primary_company']) else 'N/A'
            })
        
        return results
    
    def get_diverse_recommendations(
        self, 
        movie_title: str, 
        n_recommendations: int = 10,
        diversity_weight: float = 0.3
    ) -> Dict:
        """
        Get diverse recommendations using MMR (Maximal Marginal Relevance)
        
        Args:
            movie_title: Input movie title
            n_recommendations: Number of recommendations
            diversity_weight: Weight for diversity (0-1, higher = more diverse)
        
        Returns:
            Diverse list of recommendations
        """
        matched_title = self.find_movie(movie_title)
        if not matched_title:
            return {'error': f"Movie '{movie_title}' not found", 'recommendations': []}
        
        movie_idx = self.title_to_idx[matched_title]
        sim_to_query = self._get_similarity_row(movie_idx)
        
        selected = []
        candidates = list(range(len(self.metadata)))
        candidates.remove(movie_idx)
        
        for _ in range(min(n_recommendations, len(candidates))):
            mmr_scores = []
            
            for candidate in candidates:
                if candidate in selected:
                    continue
                
                relevance = sim_to_query[candidate]
                
                if selected:
                    candidate_row = self._get_similarity_row(candidate)
                    max_sim = max(candidate_row[s] for s in selected)
                else:
                    max_sim = 0
                
                mmr = (1 - diversity_weight) * relevance - diversity_weight * max_sim
                mmr_scores.append((candidate, mmr))
            
            if not mmr_scores:
                break
            
            best = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best)
            candidates.remove(best)
        
        recommendations = []
        for rank, idx in enumerate(selected, 1):
            movie = self.metadata.iloc[idx]
            recommendations.append({
                'rank': rank,
                'title': movie['title'],
                'production': movie['primary_company'] if pd.notna(movie['primary_company']) else 'N/A',
                'rating': f"{movie['vote_average']:.1f}/10",
                'genres': movie['genres'] if isinstance(movie['genres'], list) else [],
                'similarity_score': float(sim_to_query[idx])
            })
        
        return {
            'query_movie': matched_title,
            'recommendations': recommendations
        }
    
    def print_recommendations(self, results: Dict, show_scores: bool = False):
        """Pretty print recommendations"""
        if 'error' in results:
            print(f"\nError: {results['error']}")
            if 'suggestions' in results:
                sugg = results['suggestions']
                if isinstance(sugg, list) and sugg:
                    print("\nDid you mean:")
                    for s in sugg[:5]:
                        print(f"   - {s}")
                elif isinstance(sugg, str):
                    print(f"\n{sugg}")
            return
        
        print(f"\n{'='*100}")
        print(f"Recommendations for: {results['query_movie']}")
        if 'query_details' in results:
            details = results['query_details']
            genres_str = ", ".join(details['genres'][:3]) if details['genres'] else 'N/A'
            print(f"   Production: {details['production']} | Rating: {details['rating']} | Genres: {genres_str}")
        print(f"{'='*100}\n")
        
        for rec in results['recommendations']:
            score_str = f" [Similarity: {rec['similarity_score']:.3f}]" if show_scores else ""
            genres_str = ", ".join(rec['genres'][:3]) if rec['genres'] else 'N/A'
            
            print(f"{rec['rank']:2d}. {rec['title']}")
            print(f"    Rating: {rec['rating']} ({rec['votes']} votes) | Date: {rec['release_date']}")
            print(f"    Genres: {genres_str} | Studio: {rec['production']}{score_str}")
            
            if rec.get('imdb_link'):
                print(f"    Link: {rec['imdb_link']}")
            print()


def get_recommender(model_dir='./models'):
    """
    Get the appropriate recommender instance.
    Uses the unified engine if available, otherwise uses standalone.
    
    Args:
        model_dir: Directory containing trained model artifacts
    
    Returns:
        MovieRecommender instance
    """
    if USING_UNIFIED_ENGINE:
        print("Using unified recommendation engine")
        return UnifiedRecommender(model_dir=model_dir)
    else:
        return StandaloneMovieRecommender(model_dir=model_dir)


MovieRecommender = get_recommender


if __name__ == "__main__":
    recommender = get_recommender(model_dir='./models')
    
    print("\n" + "="*100)
    print("TMDB Movie Recommendation System - Examples")
    print("="*100)
    
    print("\nExample 1: Recommendations for 'Inception'")
    print("-" * 100)
    results = recommender.get_recommendations("Inception", n=5)
    recommender.print_recommendations(results, show_scores=True)
    
    print("\nExample 2: Filtered recommendations - Recent Action movies like 'The Dark Knight'")
    print("-" * 100)
    results = recommender.get_recommendations(
        "The Dark Knight",
        n=5,
        min_year=2015,
        genres=['Action'],
        min_rating=7.0
    )
    recommender.print_recommendations(results)
    
    print("\nExample 3: Top Rated Sci-Fi Movies")
    print("-" * 100)
    top_scifi = recommender.get_top_rated(n=5, min_votes=5000, genres=['Science Fiction'])
    for i, movie in enumerate(top_scifi, 1):
        print(f"{i}. {movie['title']} - {movie['rating']} ({movie['votes']} votes)")
    
    print("\nExample 4: Movie Details")
    print("-" * 100)
    details = recommender.get_movie_details("Interstellar")
    if 'error' not in details:
        print(f"Title: {details['title']}")
        print(f"Rating: {details['rating']} ({details['votes']} votes)")
        print(f"Genres: {', '.join(details['genres'])}")
        print(f"Production: {details['production']}")
        print(f"Overview: {details['overview']}")
    
    print("\n" + "="*100)
    print("Interactive Mode")
    print("="*100)
    
    movie_name = input("\nEnter a movie title (or press Enter for 'The Matrix'): ").strip()
    if not movie_name:
        movie_name = "The Matrix"
    
    results = recommender.get_recommendations(movie_name, n=10, min_rating=6.5)
    recommender.print_recommendations(results, show_scores=True)
