"""
Movie Recommendation System Views
Integrates with advanced TMDB model training system
"""
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.core.cache import cache

from recommender.engine import MovieRecommender

logger = logging.getLogger(__name__)

# Global cache for recommender system
_RECOMMENDER = None
_MODEL_LOADING = False
_MODEL_LOAD_PROGRESS = 0
_LOADING_THREAD = None
_LOAD_ERROR = None


def _load_model_in_background():
    """Load model in background thread"""
    global _RECOMMENDER, _MODEL_LOADING, _MODEL_LOAD_PROGRESS, _LOAD_ERROR
    
    _MODEL_LOADING = True
    _MODEL_LOAD_PROGRESS = 0
    _LOAD_ERROR = None
    
    # Check for model directory (configurable via settings or environment)
    model_dir = getattr(settings, 'MODEL_DIR', os.environ.get('MODEL_DIR', 'models'))
    
    # Fallback to static directory if models directory doesn't exist
    if not Path(model_dir).exists():
        model_dir = 'static'
        logger.warning(f"Model directory not found, using static directory")
    
    try:
        def progress_callback(progress):
            global _MODEL_LOAD_PROGRESS
            _MODEL_LOAD_PROGRESS = progress
            logger.info(f"Model loading progress: {progress}%")
        
        _RECOMMENDER = MovieRecommender(model_dir, progress_callback)
        _MODEL_LOADING = False
        _MODEL_LOAD_PROGRESS = 100
        logger.info("Model loaded successfully")
    except Exception as e:
        _MODEL_LOADING = False
        _LOAD_ERROR = str(e)
        logger.error(f"Failed to load recommender: {e}")


def _start_model_loading():
    """Start model loading in background if not already started"""
    global _LOADING_THREAD, _RECOMMENDER, _MODEL_LOADING
    
    if _RECOMMENDER is None and not _MODEL_LOADING:
        if _LOADING_THREAD is None or not _LOADING_THREAD.is_alive():
            logger.info("Starting model loading in background...")
            _LOADING_THREAD = threading.Thread(target=_load_model_in_background, daemon=True)
            _LOADING_THREAD.start()


def _get_recommender():
    """Get or initialize the recommender singleton"""
    global _RECOMMENDER, _LOAD_ERROR
    
    if _RECOMMENDER is None:
        _start_model_loading()
        if _LOAD_ERROR:
            raise Exception(_LOAD_ERROR)
        return None
    
    return _RECOMMENDER


@require_http_methods(["GET", "POST"])
def main(request):
    """
    Main view for movie recommendation system.
    GET: Display search interface
    POST: Process search and display recommendations
    """
    # Start loading model if not already loading/loaded
    _start_model_loading()
    
    recommender = _get_recommender()
    
    # If model is still loading, show the page with loading state
    if recommender is None:
        if request.method == 'GET':
            return render(request, 'recommender/index.html', {
                'all_movie_names': [],
                'total_movies': 0,
            })
        else:
            # For POST requests, return error if model not ready
            return render(request, 'recommender/index.html', {
                'all_movie_names': [],
                'total_movies': 0,
                'error_message': 'Model is still loading. Please wait a moment and try again.',
            })
    
    # Model is loaded, proceed normally
    titles_list = list(recommender.title_to_idx.keys())
    
    if request.method == 'GET':
        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': titles_list,
                'total_movies': len(titles_list),
            }
        )
    
    # POST request - process search
    movie_name = request.POST.get('movie_name', '').strip()
    
    if not movie_name:
        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': titles_list,
                'total_movies': len(titles_list),
                'error_message': 'Please enter a movie name.',
            }
        )
    
    # Get filtering parameters
    try:
        n = int(request.POST.get('n', 15))
    except ValueError:
        n = 15
    
    try:
        min_year = int(request.POST.get('min_year', '')) if request.POST.get('min_year', '').strip() else None
    except ValueError:
        min_year = None
    
    try:
        max_year = int(request.POST.get('max_year', '')) if request.POST.get('max_year', '').strip() else None
    except ValueError:
        max_year = None
    
    genres = request.POST.get('genres', '').strip()
    genres_list = [g.strip() for g in genres.split(',')] if genres else None
    
    try:
        min_rating = float(request.POST.get('min_rating', '')) if request.POST.get('min_rating', '').strip() else None
    except ValueError:
        min_rating = None
    
    exclude_same_company = request.POST.get('exclude_same_company', '').lower() == 'true'
    
    # Get recommendations
    result = recommender.get_recommendations(
        movie_name,
        n=n,
        min_year=min_year,
        max_year=max_year,
        genres=genres_list,
        min_rating=min_rating,
        exclude_same_company=exclude_same_company
    )
    
    if 'error' in result:
        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': titles_list,
                'total_movies': len(titles_list),
                'input_movie_name': movie_name,
                'error_message': result['error'],
                'suggestions': result.get('suggestions', [])
            }
        )
    
    return render(
        request,
        'recommender/result.html',
        {
            'all_movie_names': titles_list,
            'input_movie_name': result['query_movie'],
            'source_movie': result['source_movie'],
            'recommended_movies': result['recommendations'],
            'total_recommendations': len(result['recommendations']),
            'requested_n': n,
            'filters': {
                'n': n,
                'min_year': min_year,
                'max_year': max_year,
                'genres': genres,
                'min_rating': min_rating,
                'exclude_same_company': exclude_same_company
            }
        }
    )


@require_http_methods(["GET"])
def search_movies(request):
    """API endpoint for searching movies (autocomplete)"""
    query = request.GET.get('q', '').strip()
    
    if len(query) < 2:
        return JsonResponse({'movies': [], 'count': 0})
    
    try:
        recommender = _get_recommender()
        
        if recommender is None:
            return JsonResponse({'movies': [], 'count': 0, 'loading': True})
        
        matching_movies = recommender.search_movies(query, n=20)
        
        return JsonResponse({
            'movies': matching_movies,
            'count': len(matching_movies)
        })
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return JsonResponse({'error': 'Search failed'}, status=500)


@require_http_methods(["GET"])
def model_status(request):
    """API endpoint to check model loading status"""
    global _RECOMMENDER, _MODEL_LOADING, _MODEL_LOAD_PROGRESS, _LOAD_ERROR
    
    # Start loading if not already started
    _start_model_loading()
    
    if _LOAD_ERROR:
        return JsonResponse({
            'loaded': False,
            'progress': 0,
            'status': 'error',
            'error': _LOAD_ERROR
        })
    elif _RECOMMENDER is not None:
        return JsonResponse({
            'loaded': True,
            'progress': 100,
            'status': 'ready'
        })
    elif _MODEL_LOADING:
        return JsonResponse({
            'loaded': False,
            'progress': _MODEL_LOAD_PROGRESS,
            'status': 'loading'
        })
    else:
        return JsonResponse({
            'loaded': False,
            'progress': 0,
            'status': 'initializing'
        })


@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint for monitoring"""
    try:
        recommender = _get_recommender()
        return JsonResponse({
            'status': 'healthy',
            'movies_loaded': recommender.config['n_movies'],
            'model_dir': str(recommender.model_dir),
            'model_loaded': True
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=503)


@require_http_methods(["GET"])
def api_recommend(request):
    """API endpoint for movie recommendations with filtering"""
    global _RECOMMENDER, _MODEL_LOADING, _MODEL_LOAD_PROGRESS, _LOAD_ERROR
    
    # Start loading if not already started
    _start_model_loading()
    
    # Check model status
    if _LOAD_ERROR:
        return JsonResponse({
            'error': _LOAD_ERROR,
            'loading': False
        }, status=500)
    elif _RECOMMENDER is None:
        return JsonResponse({
            'loading': True,
            'progress': _MODEL_LOAD_PROGRESS,
            'status': 'loading'
        })
    
    # Get parameters
    movie_title = request.GET.get('movie_title', '').strip()
    if not movie_title:
        return JsonResponse({
            'error': 'movie_title parameter is required'
        }, status=400)
    
    # Get filtering parameters
    try:
        n = int(request.GET.get('n', 15))
    except ValueError:
        n = 15
    
    try:
        min_year = int(request.GET.get('min_year', '')) if request.GET.get('min_year', '').strip() else None
    except ValueError:
        min_year = None
    
    try:
        max_year = int(request.GET.get('max_year', '')) if request.GET.get('max_year', '').strip() else None
    except ValueError:
        max_year = None
    
    genres = request.GET.get('genres', '').strip()
    genres_list = [g.strip() for g in genres.split(',')] if genres else None
    
    try:
        min_rating = float(request.GET.get('min_rating', '')) if request.GET.get('min_rating', '').strip() else None
    except ValueError:
        min_rating = None
    
    exclude_same_company = request.GET.get('exclude_same_company', '').lower() == 'true'
    
    # Generate cache key based on all parameters
    cache_key = f"recommend:{movie_title}:{n}:{min_year}:{max_year}:{genres}:{min_rating}:{exclude_same_company}"
    
    # Try to get from cache
    cached_result = cache.get(cache_key)
    if cached_result:
        return JsonResponse(cached_result)
    
    # Get recommendations
    try:
        result = _RECOMMENDER.get_recommendations(
            movie_title,
            n=n,
            min_year=min_year,
            max_year=max_year,
            genres=genres_list,
            min_rating=min_rating,
            exclude_same_company=exclude_same_company
        )
        
        # Add loading status to response
        result['loading'] = False
        
        # Cache the result for 30 minutes
        cache.set(cache_key, result, 1800)
        
        return JsonResponse(result)
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return JsonResponse({
            'error': str(e),
            'loading': False
        }, status=500)
