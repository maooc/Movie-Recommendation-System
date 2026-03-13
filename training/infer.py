"""
Advanced Movie Recommendation System - Inference Engine
Optimized for TMDB Movies Dataset 2023 (930K+ movies)
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import from recommender module
sys.path.append(str(Path(__file__).parent.parent))

from recommender.engine import MovieRecommender as CoreMovieRecommender

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class MovieRecommender(CoreMovieRecommender):
    def __init__(self, model_dir='./models'):
        """
        Initialize recommender with trained models
        
        Args:
            model_dir: Directory containing trained model artifacts
        """
        super().__init__(model_dir=model_dir)
    
    def load_models(self):
        """Load all model artifacts with custom logging"""
        print("🎬 Loading TMDB Movie Recommendation Engine...")
        super()._load_models()
        print(f"✅ Loaded {self.config['n_movies']:,} movies from {self.config.get('dataset', 'dataset')}")
        print(f"   Model ready for inference!")
    
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
            'production': movie['primary_company'],
            'genres': movie['genres'] if isinstance(movie['genres'], list) else [],
            'rating': f"{movie['vote_average']:.1f}/10",
            'votes': f"{movie['vote_count']:,}",
            'popularity': f"{movie['popularity']:.1f}",
            'overview': movie['overview'][:200] + '...' if len(str(movie['overview'])) > 200 else movie['overview'],
            'imdb_id': movie['imdb_id'] if pd.notna(movie['imdb_id']) else 'N/A',
            'poster_url': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if pd.notna(movie['poster_path']) else None
        }
    
    def get_recommendations(
        self, 
        movie_title: str, 
        n_recommendations: int = 10,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        genres: Optional[List[str]] = None,
        min_rating: Optional[float] = None,
        exclude_same_company: bool = False
    ) -> Dict:
        """
        Get movie recommendations with advanced filtering
        
        Args:
            movie_title: Title of the movie to base recommendations on
            n_recommendations: Number of recommendations to return
            min_year: Minimum release year filter
            max_year: Maximum release year filter
            genres: List of genres to filter by
            min_rating: Minimum vote_average (0-10)
            exclude_same_company: Exclude movies by same production company
        
        Returns:
            Dictionary with recommendations and metadata
        """
        # Use core implementation
        result = super().get_recommendations(
            movie_title,
            n=n_recommendations,
            min_year=min_year,
            max_year=max_year,
            genres=genres,
            min_rating=min_rating,
            exclude_same_company=exclude_same_company
        )
        
        # Add rank to recommendations
        if 'recommendations' in result:
            for i, rec in enumerate(result['recommendations'], 1):
                rec['rank'] = i
                # Add tmdb_id if available
                idx = self.title_to_idx.get(rec['title'])
                if idx is not None:
                    movie = self.metadata.iloc[idx]
                    if hasattr(movie, 'id'):
                        rec['tmdb_id'] = int(movie['id'])
                # Rename google_link to google_search for consistency
                if 'google_link' in rec:
                    rec['google_search'] = rec.pop('google_link')
        
        # Add query_details for compatibility
        if 'source_movie' in result:
            result['query_details'] = {
                'production': result['source_movie']['production'],
                'genres': result['source_movie']['genres'].split(', ') if isinstance(result['source_movie']['genres'], str) else result['source_movie']['genres'],
                'rating': result['source_movie']['rating'],
                'release_date': None  # Not available in core result
            }
        
        if 'recommendations' in result:
            result['total_recommendations'] = len(result['recommendations'])
        
        return result
    
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
            return {'error': f"Movie '{movie_title}' not found"}
        
        movie_idx = self.title_to_idx[matched_title]
        
        # Get similarity scores
        if hasattr(self.similarity_matrix, 'toarray'):
            sim_to_query = self.similarity_matrix[movie_idx].toarray()[0]
        else:
            sim_to_query = self.similarity_matrix[movie_idx]
        
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
                    if hasattr(self.similarity_matrix, 'toarray'):
                        max_sim = max(self.similarity_matrix[candidate, s] for s in selected)
                    else:
                        max_sim = max(self.similarity_matrix[candidate][s] for s in selected)
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
            print(f"\n❌ {results['error']}")
            if 'suggestions' in results:
                sugg = results['suggestions']
                if isinstance(sugg, list) and sugg:
                    print("\n💡 Did you mean:")
                    for s in sugg[:5]:
                        print(f"   • {s}")
                elif isinstance(sugg, str):
                    print(f"\n💡 {sugg}")
            return
        
        print(f"\n{'='*100}")
        print(f"🎬 Recommendations for: {results['query_movie']}")
        if 'query_details' in results:
            details = results['query_details']
            genres_str = ", ".join(details['genres'][:3]) if details['genres'] else 'N/A'
            print(f"   Production: {details['production']} | Rating: {details['rating']} | Genres: {genres_str}")
        print(f"{'='*100}\n")
        
        for rec in results['recommendations']:
            score_str = f" [Similarity: {rec.get('similarity_score', 0):.3f}]" if show_scores else ""
            genres_str = ", ".join(rec.get('genres', [])[:3]) if isinstance(rec.get('genres'), list) else rec.get('genres', 'N/A')
            
            print(f"{rec.get('rank', 0):2d}. {rec['title']}")
            print(f"    ⭐ {rec['rating']} ({rec.get('votes', 'N/A')} votes) | 📅 {rec.get('release_date', 'Unknown')}")
            print(f"    🎭 {genres_str} | 🏢 {rec.get('production', 'Unknown')}{score_str}")
            
            if rec.get('imdb_link'):
                print(f"    🔗 {rec['imdb_link']}")
            print()


# Example usage
if __name__ == "__main__":
    # Initialize recommender
    recommender = MovieRecommender(model_dir='./models')
    
    print("\n" + "="*100)
    print("🎬 TMDB Movie Recommendation System - Examples")
    print("="*100)
    
    # Example 1: Basic recommendations
    print("\n📌 Example 1: Recommendations for 'Inception'")
    print("-" * 100)
    results = recommender.get_recommendations("Inception", n_recommendations=5)
    recommender.print_recommendations(results, show_scores=True)
    
    # Example 2: Filtered recommendations
    print("\n📌 Example 2: Recent Action movies like 'The Dark Knight'")
    print("-" * 100)
    results = recommender.get_recommendations(
        "The Dark Knight",
        n_recommendations=5,
        min_year=2015,
        genres=['Action'],
        min_rating=7.0
    )
    recommender.print_recommendations(results)
    
    # Example 3: Top rated movies
    print("\n📌 Example 3: Top Rated Sci-Fi Movies")
    print("-" * 100)
    top_scifi = recommender.get_top_rated(n=5, min_votes=5000, genres=['Science Fiction'])
    for i, movie in enumerate(top_scifi, 1):
        print(f"{i}. {movie['title']} - {movie['rating']} ({movie['votes']} votes)")
    
    # Example 4: Movie details
    print("\n📌 Example 4: Movie Details")
    print("-" * 100)
    details = recommender.get_movie_details("Interstellar")
    if 'error' not in details:
        print(f"Title: {details['title']}")
        print(f"Rating: {details['rating']} ({details['votes']} votes)")
        print(f"Genres: {', '.join(details['genres'])}")
        print(f"Production: {details['production']}")
        print(f"Overview: {details['overview']}")
    
    # Example 5: Interactive mode
    print("\n" + "="*100)
    print("🎮 Interactive Mode")
    print("="*100)
    
    movie_name = input("\n🎬 Enter a movie title (or press Enter for random): ").strip()
    if not movie_name:
        movie_name = "The Matrix"
    
    results = recommender.get_recommendations(movie_name, n_recommendations=10, min_rating=6.5)
    recommender.print_recommendations(results, show_scores=True)
