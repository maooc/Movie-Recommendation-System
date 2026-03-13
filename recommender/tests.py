"""
Unit tests for Movie Recommendation System
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from django.test import TestCase, Client
from django.urls import reverse
from recommender.engine import MovieRecommender


class MockMovieRecommender(MovieRecommender):
    """Mock version of MovieRecommender for testing"""
    
    def __init__(self):
        # Skip the actual model loading
        self.metadata = None
        self.similarity_matrix = None
        self.title_to_idx = None
        self.config = None
        self._setup_mock_data()
    
    def _setup_mock_data(self):
        """Set up mock data for testing"""
        # Create mock metadata
        data = {
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
            'primary_company': ['Company 1', 'Company 1', 'Company 2', 'Company 3', 'Company 1'],
            'vote_average': [8.5, 7.2, 6.8, 9.0, 7.5],
            'vote_count': [1000, 500, 300, 2000, 800],
            'release_date': ['2020-01-01', '2019-05-15', '2021-03-10', '2018-11-20', '2022-07-05'],
            'genres': [['Action', 'Adventure'], ['Action', 'Comedy'], ['Drama', 'Romance'], ['Action', 'Sci-Fi'], ['Comedy', 'Drama']],
            'imdb_id': ['tt123456', 'tt234567', 'tt345678', 'tt456789', 'tt567890'],
            'poster_path': ['/poster1.jpg', '/poster2.jpg', '/poster3.jpg', '/poster4.jpg', '/poster5.jpg']
        }
        self.metadata = pd.DataFrame(data)
        
        # Create mock similarity matrix
        # Movie A is most similar to Movie D, then B, then E, then C
        self.similarity_matrix = np.array([
            [1.0, 0.8, 0.3, 0.9, 0.7],  # Movie A
            [0.8, 1.0, 0.4, 0.7, 0.6],  # Movie B
            [0.3, 0.4, 1.0, 0.2, 0.5],  # Movie C
            [0.9, 0.7, 0.2, 1.0, 0.6],  # Movie D
            [0.7, 0.6, 0.5, 0.6, 1.0]   # Movie E
        ])
        
        # Create title to index mapping
        self.title_to_idx = {title: i for i, title in enumerate(data['title'])}
        
        # Create mock config
        self.config = {'n_movies': 5, 'dataset': 'test'}
    
    def _load_models(self, progress_callback=None):
        """Override to skip actual loading"""
        pass


class EngineTests(TestCase):
    """Tests for the recommendation engine"""
    
    def setUp(self):
        self.recommender = MockMovieRecommender()
    
    def test_find_movie(self):
        """Test movie title fuzzy matching"""
        # Exact match
        self.assertEqual(self.recommender.find_movie('Movie A'), 'Movie A')
        # Fuzzy match
        self.assertEqual(self.recommender.find_movie('movie a'), 'Movie A')
        # No match
        self.assertIsNone(self.recommender.find_movie('Nonexistent Movie'))
    
    def test_search_movies(self):
        """Test movie search functionality"""
        results = self.recommender.search_movies('Movie')
        self.assertEqual(len(results), 5)
        self.assertIn('Movie A', results)
        
        results = self.recommender.search_movies('Movie A')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], 'Movie A')
    
    def test_basic_recommendations(self):
        """Test basic recommendation functionality"""
        result = self.recommender.get_recommendations('Movie A', n=3)
        self.assertEqual(result['query_movie'], 'Movie A')
        self.assertEqual(len(result['recommendations']), 3)
        # Check order (should be D, B, E based on similarity)
        self.assertEqual(result['recommendations'][0]['title'], 'Movie D')
        self.assertEqual(result['recommendations'][1]['title'], 'Movie B')
        self.assertEqual(result['recommendations'][2]['title'], 'Movie E')
    
    def test_rating_filter(self):
        """Test rating filter functionality"""
        result = self.recommender.get_recommendations('Movie A', n=3, min_rating=8.0)
        # Should only include movies with rating >= 8.0 (A, D)
        self.assertEqual(len(result['recommendations']), 1)  # A is excluded (self), so only D
        self.assertEqual(result['recommendations'][0]['title'], 'Movie D')
    
    def test_year_filter(self):
        """Test year filter functionality"""
        result = self.recommender.get_recommendations('Movie A', n=3, min_year=2020)
        # Movie A's similarity order: D (2018), B (2019), E (2022), C (2021)
        # With min_year=2020, only E and C qualify
        self.assertEqual(len(result['recommendations']), 2)
        titles = [rec['title'] for rec in result['recommendations']]
        self.assertIn('Movie E', titles)
        self.assertIn('Movie C', titles)
        # Verify order based on similarity
        self.assertEqual(result['recommendations'][0]['title'], 'Movie E')  # E is more similar than C
        self.assertEqual(result['recommendations'][1]['title'], 'Movie C')
    
    def test_genre_filter(self):
        """Test genre filter functionality"""
        result = self.recommender.get_recommendations('Movie A', n=3, genres=['Action'])
        # Should only include movies with Action genre (B, D)
        self.assertEqual(len(result['recommendations']), 2)
        titles = [rec['title'] for rec in result['recommendations']]
        self.assertIn('Movie D', titles)
        self.assertIn('Movie B', titles)
    
    def test_company_filter(self):
        """Test company filter functionality"""
        result = self.recommender.get_recommendations('Movie A', n=3, exclude_same_company=True)
        # Should exclude movies from Company 1 (B, E)
        self.assertEqual(len(result['recommendations']), 2)
        titles = [rec['title'] for rec in result['recommendations']]
        self.assertIn('Movie D', titles)
        self.assertIn('Movie C', titles)
    
    def test_combined_filters(self):
        """Test combined filtering functionality"""
        result = self.recommender.get_recommendations(
            'Movie A', 
            n=3, 
            min_rating=7.0, 
            min_year=2019, 
            genres=['Action'],
            exclude_same_company=True
        )
        # Should only include movies that meet all filters
        # Let's see: Action genre, min_rating 7.0, min_year 2019, not Company 1
        # Candidates: Movie D (2018) - no, Movie B (2019, Company 1) - no
        # So no results
        self.assertEqual(len(result['recommendations']), 0)


class APITests(TestCase):
    """Tests for the API endpoints"""
    
    def setUp(self):
        self.client = Client()
    
    @patch('recommender.views._RECOMMENDER', None)
    @patch('recommender.views._MODEL_LOADING', True)
    @patch('recommender.views._MODEL_LOAD_PROGRESS', 50)
    def test_api_recommend_loading(self):
        """Test /api/recommend/ when model is loading"""
        response = self.client.get(reverse('recommender:api_recommend'), {'movie_title': 'Movie A'})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['loading'])
        self.assertEqual(data['progress'], 50)
        self.assertEqual(data['status'], 'loading')
    
    @patch('recommender.views._RECOMMENDER')
    def test_api_recommend_movie_not_found(self, mock_recommender):
        """Test /api/recommend/ when movie is not found"""
        mock_recommender.get_recommendations.return_value = {
            'error': 'Movie not found',
            'suggestions': ['Movie A', 'Movie B']
        }
        response = self.client.get(reverse('recommender:api_recommend'), {'movie_title': 'Nonexistent Movie'})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data['loading'])
        self.assertEqual(data['error'], 'Movie not found')
        self.assertEqual(data['suggestions'], ['Movie A', 'Movie B'])
    
    @patch('recommender.views._RECOMMENDER')
    def test_api_recommend_success(self, mock_recommender):
        """Test /api/recommend/ with successful recommendation"""
        mock_recommender.get_recommendations.return_value = {
            'query_movie': 'Movie A',
            'source_movie': {
                'production': 'Company 1',
                'rating': '8.5/10',
                'genres': 'Action, Adventure'
            },
            'recommendations': [
                {
                    'title': 'Movie D',
                    'rating': '9.0/10'
                }
            ]
        }
        response = self.client.get(reverse('recommender:api_recommend'), {'movie_title': 'Movie A'})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data['loading'])
        self.assertEqual(data['query_movie'], 'Movie A')
        self.assertEqual(len(data['recommendations']), 1)
        self.assertEqual(data['recommendations'][0]['title'], 'Movie D')
    
    def test_api_recommend_missing_param(self):
        """Test /api/recommend/ with missing movie_title parameter"""
        response = self.client.get(reverse('recommender:api_recommend'))
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data['error'], 'movie_title parameter is required')


if __name__ == '__main__':
    unittest.main()
