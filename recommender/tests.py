"""
Unit Tests for Movie Recommendation System
Tests the unified recommendation engine and API endpoints
"""
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

from django.test import TestCase, Client
from django.urls import reverse


def create_test_model_files(model_dir: Path) -> dict:
    """
    Create minimal test model files for unit testing.
    Returns info about the created test data.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    
    test_movies = [
        {
            'title': 'Test Movie 1',
            'release_date': '2020-01-15',
            'vote_average': 8.5,
            'vote_count': 1000,
            'genres': ['Action', 'Sci-Fi'],
            'primary_company': 'Test Studio A',
            'imdb_id': 'tt1234567',
            'poster_path': '/poster1.jpg'
        },
        {
            'title': 'Test Movie 2',
            'release_date': '2019-06-20',
            'vote_average': 7.2,
            'vote_count': 500,
            'genres': ['Comedy', 'Romance'],
            'primary_company': 'Test Studio B',
            'imdb_id': 'tt2345678',
            'poster_path': '/poster2.jpg'
        },
        {
            'title': 'Test Movie 3',
            'release_date': '2021-03-10',
            'vote_average': 6.8,
            'vote_count': 300,
            'genres': ['Action', 'Thriller'],
            'primary_company': 'Test Studio A',
            'imdb_id': 'tt3456789',
            'poster_path': '/poster3.jpg'
        },
        {
            'title': 'Test Movie 4',
            'release_date': '2018-11-05',
            'vote_average': 9.0,
            'vote_count': 2000,
            'genres': ['Drama'],
            'primary_company': 'Test Studio C',
            'imdb_id': 'tt4567890',
            'poster_path': '/poster4.jpg'
        },
        {
            'title': 'Test Movie 5',
            'release_date': None,
            'vote_average': 5.5,
            'vote_count': 100,
            'genres': ['Horror'],
            'primary_company': 'Test Studio D',
            'imdb_id': 'tt5678901',
            'poster_path': None
        },
    ]
    
    df = pd.DataFrame(test_movies)
    metadata_path = model_dir / 'movie_metadata.parquet'
    df.to_parquet(metadata_path, index=False)
    
    n_movies = len(test_movies)
    similarity = np.random.rand(n_movies, n_movies)
    similarity = (similarity + similarity.T) / 2
    np.fill_diagonal(similarity, 1.0)
    
    similarity_path = model_dir / 'similarity_matrix.npy'
    np.save(similarity_path, similarity)
    
    title_to_idx = {movie['title']: i for i, movie in enumerate(test_movies)}
    mapping_path = model_dir / 'title_to_idx.json'
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(title_to_idx, f)
    
    config = {
        'n_movies': n_movies,
        'model_version': 'test_v1.0',
        'training_date': '2024-01-01'
    }
    config_path = model_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f)
    
    return {
        'n_movies': n_movies,
        'titles': [m['title'] for m in test_movies],
        'model_dir': str(model_dir)
    }


class EngineUnitTests(TestCase):
    """Unit tests for the recommendation engine core logic"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_info = create_test_model_files(Path(self.temp_dir))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_engine_initialization(self):
        """Test that engine initializes correctly with model files"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        self.assertTrue(recommender.is_loaded)
        self.assertEqual(recommender.n_movies, self.model_info['n_movies'])
        self.assertIsNotNone(recommender.metadata)
        self.assertIsNotNone(recommender.similarity_matrix)
        self.assertIsNotNone(recommender.title_to_idx)
    
    def test_find_movie_exact_match(self):
        """Test finding a movie with exact title match"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        result = recommender.find_movie('Test Movie 1')
        self.assertEqual(result, 'Test Movie 1')
    
    def test_find_movie_fuzzy_match(self):
        """Test finding a movie with fuzzy matching"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        result = recommender.find_movie('Test Movie')
        self.assertIsNotNone(result)
        self.assertIn('Test Movie', result)
    
    def test_find_movie_not_found(self):
        """Test finding a movie that doesn't exist"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        result = recommender.find_movie('Nonexistent Movie XYZ')
        self.assertIsNone(result)
    
    def test_search_movies(self):
        """Test searching movies by partial title"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        results = recommender.search_movies('Test', n=10)
        self.assertEqual(len(results), self.model_info['n_movies'])
    
    def test_search_movies_with_limit(self):
        """Test search results are limited correctly"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        results = recommender.search_movies('Test', n=2)
        self.assertEqual(len(results), 2)
    
    def test_get_recommendations_basic(self):
        """Test basic recommendation retrieval"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        result = recommender.get_recommendations('Test Movie 1', n=3)
        
        self.assertIn('query_movie', result)
        self.assertIn('recommendations', result)
        self.assertIn('source_movie', result)
        self.assertEqual(result['query_movie'], 'Test Movie 1')
        self.assertLessEqual(len(result['recommendations']), 3)
    
    def test_get_recommendations_movie_not_found(self):
        """Test recommendations for non-existent movie"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        result = recommender.get_recommendations('Nonexistent Movie', n=5)
        
        self.assertIn('error', result)
        self.assertIn('suggestions', result)
        self.assertEqual(result['recommendations'], [])
    
    def test_get_recommendations_with_min_rating_filter(self):
        """Test recommendations with minimum rating filter"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        result = recommender.get_recommendations('Test Movie 1', n=5, min_rating=7.0)
        
        for rec in result['recommendations']:
            rating_str = rec['rating'].replace('/10', '')
            if rating_str != 'N/A':
                rating = float(rating_str)
                self.assertGreaterEqual(rating, 7.0)
    
    def test_get_recommendations_with_year_filter(self):
        """Test recommendations with year range filter"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        result = recommender.get_recommendations(
            'Test Movie 1', 
            n=5, 
            min_year=2019, 
            max_year=2021
        )
        
        self.assertIn('filters_applied', result)
        self.assertEqual(result['filters_applied']['min_year'], 2019)
        self.assertEqual(result['filters_applied']['max_year'], 2021)
    
    def test_get_recommendations_with_genre_filter(self):
        """Test recommendations with genre filter"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        result = recommender.get_recommendations(
            'Test Movie 1',
            n=5,
            genres='Action'
        )
        
        self.assertIn('filters_applied', result)
        self.assertEqual(result['filters_applied']['genres'], 'Action')
    
    def test_get_recommendations_exclude_same_company(self):
        """Test recommendations excluding same production company"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        result = recommender.get_recommendations(
            'Test Movie 1',
            n=5,
            exclude_same_company=True
        )
        
        self.assertTrue(result['filters_applied']['exclude_same_company'])
    
    def test_extract_year_valid_date(self):
        """Test year extraction from valid date"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        year = recommender._extract_year('2020-01-15')
        self.assertEqual(year, 2020)
    
    def test_extract_year_none_value(self):
        """Test year extraction from None value"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        year = recommender._extract_year(None)
        self.assertIsNone(year)
    
    def test_extract_year_invalid_format(self):
        """Test year extraction from invalid format"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        year = recommender._extract_year('invalid-date')
        self.assertIsNone(year)
    
    def test_parse_genres_string(self):
        """Test parsing genres from comma-separated string"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        genres = recommender._parse_genres('Action, Comedy, Drama')
        self.assertEqual(genres, ['action', 'comedy', 'drama'])
    
    def test_parse_genres_list(self):
        """Test parsing genres from list"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        genres = recommender._parse_genres(['Action', 'Comedy'])
        self.assertEqual(genres, ['action', 'comedy'])
    
    def test_parse_genres_none(self):
        """Test parsing genres from None"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        genres = recommender._parse_genres(None)
        self.assertEqual(genres, [])


class RecommenderManagerTests(TestCase):
    """Tests for the RecommenderManager singleton"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_info = create_test_model_files(Path(self.temp_dir))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('recommender.engine.RecommenderManager._instance', None)
    def test_manager_singleton(self):
        """Test that manager is a singleton"""
        from recommender.engine import RecommenderManager
        
        manager1 = RecommenderManager()
        manager2 = RecommenderManager()
        
        self.assertIs(manager1, manager2)
    
    @patch('recommender.engine.RecommenderManager._instance', None)
    def test_manager_start_loading(self):
        """Test starting model loading"""
        from recommender.engine import RecommenderManager
        
        manager = RecommenderManager()
        manager.start_loading(self.temp_dir)
        
        import time
        max_wait = 10
        start = time.time()
        while not manager.is_loaded() and (time.time() - start) < max_wait:
            time.sleep(0.1)
        
        self.assertTrue(manager.is_loaded())
    
    @patch('recommender.engine.RecommenderManager._instance', None)
    def test_manager_get_status(self):
        """Test getting manager status"""
        from recommender.engine import RecommenderManager
        
        manager = RecommenderManager()
        status = manager.get_status()
        
        self.assertIn('loaded', status)
        self.assertIn('progress', status)
        self.assertIn('status', status)


class CacheTests(TestCase):
    """Tests for the recommendation caching system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_info = create_test_model_files(Path(self.temp_dir))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        from recommender.engine import get_cache_key
        
        key1 = get_cache_key('Test Movie', n=10, min_rating=7.0)
        key2 = get_cache_key('Test Movie', n=10, min_rating=7.0)
        key3 = get_cache_key('Test Movie', n=15, min_rating=7.0)
        
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)
    
    def test_cache_set_and_get(self):
        """Test setting and getting cached recommendations"""
        from recommender.engine import (
            get_cached_recommendations,
            set_cached_recommendations,
            clear_cache
        )
        
        clear_cache()
        
        cache_key = 'test_cache_key'
        test_result = {'test': 'data', 'recommendations': []}
        
        set_cached_recommendations(cache_key, test_result)
        cached = get_cached_recommendations(cache_key)
        
        self.assertEqual(cached, test_result)
    
    def test_cache_clear(self):
        """Test clearing the cache"""
        from recommender.engine import (
            get_cached_recommendations,
            set_cached_recommendations,
            clear_cache,
            get_cache_stats
        )
        
        clear_cache()
        
        set_cached_recommendations('key1', {'test': 1})
        set_cached_recommendations('key2', {'test': 2})
        
        stats = get_cache_stats()
        self.assertEqual(stats['size'], 2)
        
        clear_cache()
        stats = get_cache_stats()
        self.assertEqual(stats['size'], 0)


class APIEndpointTests(TestCase):
    """Tests for the API endpoints"""
    
    def setUp(self):
        self.client = Client()
        self.temp_dir = tempfile.mkdtemp()
        self.model_info = create_test_model_files(Path(self.temp_dir))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('recommender.views._get_model_dir')
    def test_api_recommend_success(self, mock_get_dir):
        """Test successful recommendation API call"""
        mock_get_dir.return_value = self.temp_dir
        
        from recommender.engine import RecommenderManager
        manager = RecommenderManager()
        manager.start_loading(self.temp_dir)
        
        import time
        max_wait = 10
        start = time.time()
        while not manager.is_loaded() and (time.time() - start) < max_wait:
            time.sleep(0.1)
        
        response = self.client.get('/api/recommend/', {
            'movie_title': 'Test Movie 1'
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)
    
    @patch('recommender.views._get_model_dir')
    def test_api_recommend_missing_title(self, mock_get_dir):
        """Test API call without movie_title parameter"""
        mock_get_dir.return_value = self.temp_dir
        
        response = self.client.get('/api/recommend/', {})
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)
    
    @patch('recommender.views._get_model_dir')
    def test_api_recommend_with_filters(self, mock_get_dir):
        """Test API call with filter parameters"""
        mock_get_dir.return_value = self.temp_dir
        
        from recommender.engine import RecommenderManager
        manager = RecommenderManager()
        manager.start_loading(self.temp_dir)
        
        import time
        max_wait = 10
        start = time.time()
        while not manager.is_loaded() and (time.time() - start) < max_wait:
            time.sleep(0.1)
        
        response = self.client.get('/api/recommend/', {
            'movie_title': 'Test Movie 1',
            'n': 3,
            'min_rating': 7.0,
            'min_year': 2019,
            'max_year': 2021,
            'genres': 'Action'
        })
        
        self.assertEqual(response.status_code, 200)
    
    def test_api_search_too_short(self):
        """Test search API with query too short"""
        response = self.client.get('/api/search/', {'q': 'T'})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['movies'], [])
    
    def test_api_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get('/api/health/')
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)


class SparseMatrixTests(TestCase):
    """Tests for sparse matrix handling"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_sparse_matrix_loading(self):
        """Test loading sparse similarity matrix"""
        model_dir = Path(self.temp_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        test_movies = [
            {'title': 'Sparse Test 1', 'vote_average': 7.0, 'vote_count': 100,
             'genres': ['Action'], 'primary_company': 'Studio A'},
            {'title': 'Sparse Test 2', 'vote_average': 8.0, 'vote_count': 200,
             'genres': ['Drama'], 'primary_company': 'Studio B'},
        ]
        
        df = pd.DataFrame(test_movies)
        df.to_parquet(model_dir / 'movie_metadata.parquet', index=False)
        
        n_movies = len(test_movies)
        sparse_matrix = csr_matrix(np.eye(n_movies))
        save_npz(model_dir / 'similarity_matrix.npz', sparse_matrix)
        
        title_to_idx = {m['title']: i for i, m in enumerate(test_movies)}
        with open(model_dir / 'title_to_idx.json', 'w') as f:
            json.dump(title_to_idx, f)
        
        with open(model_dir / 'config.json', 'w') as f:
            json.dump({'n_movies': n_movies}, f)
        
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=str(model_dir))
        
        self.assertTrue(recommender._sparse_similarity)
        self.assertIsNotNone(recommender._sparse_matrix)
        
        row = recommender._get_similarity_row(0)
        self.assertEqual(len(row), n_movies)


class EdgeCaseTests(TestCase):
    """Tests for edge cases and error handling"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_info = create_test_model_files(Path(self.temp_dir))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_missing_release_date(self):
        """Test handling of movies with missing release dates"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        result = recommender.get_recommendations(
            'Test Movie 1',
            n=5,
            min_year=2020
        )
        
        self.assertIn('recommendations', result)
    
    def test_empty_genres_filter(self):
        """Test with empty genres filter"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        result = recommender.get_recommendations(
            'Test Movie 1',
            n=5,
            genres=''
        )
        
        self.assertIn('recommendations', result)
    
    def test_very_high_rating_filter(self):
        """Test with very high minimum rating"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        result = recommender.get_recommendations(
            'Test Movie 1',
            n=5,
            min_rating=9.5
        )
        
        self.assertIn('recommendations', result)
    
    def test_invalid_year_range(self):
        """Test with invalid year range"""
        from recommender.engine import MovieRecommender
        
        recommender = MovieRecommender(model_dir=self.temp_dir)
        
        result = recommender.get_recommendations(
            'Test Movie 1',
            n=5,
            min_year=2025,
            max_year=2020
        )
        
        self.assertIn('recommendations', result)
        self.assertLessEqual(len(result['recommendations']), 5)
