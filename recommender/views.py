"""
Movie Recommendation System Views
Delegates to the unified recommendation engine
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .engine import (
    RecommenderManager,
    get_recommendations_with_cache,
    clear_cache,
    get_cache_stats
)

logger = logging.getLogger(__name__)

_manager = RecommenderManager()


def _get_model_dir() -> str:
    """Get model directory from settings or environment"""
    model_dir = getattr(settings, 'MODEL_DIR', os.environ.get('MODEL_DIR', 'models'))
    
    if not Path(model_dir).exists():
        static_dir = 'static'
        if Path(static_dir).exists():
            logger.warning(f"Model directory not found, using static directory")
            return static_dir
    
    return model_dir


def _parse_filter_params(request) -> Dict:
    """Parse filter parameters from request (GET or POST)"""
    params = {}
    
    n = request.GET.get('n') or request.POST.get('n')
    if n:
        try:
            params['n'] = int(n)
        except ValueError:
            pass
    
    min_year = request.GET.get('min_year') or request.POST.get('min_year')
    if min_year:
        try:
            params['min_year'] = int(min_year)
        except ValueError:
            pass
    
    max_year = request.GET.get('max_year') or request.POST.get('max_year')
    if max_year:
        try:
            params['max_year'] = int(max_year)
        except ValueError:
            pass
    
    genres = request.GET.get('genres') or request.POST.get('genres')
    if genres and genres.strip():
        params['genres'] = genres.strip()
    
    min_rating = request.GET.get('min_rating') or request.POST.get('min_rating')
    if min_rating:
        try:
            params['min_rating'] = float(min_rating)
        except ValueError:
            pass
    
    exclude_same_company = request.GET.get('exclude_same_company') or request.POST.get('exclude_same_company')
    if exclude_same_company in ('true', 'True', '1', 'on', 'yes'):
        params['exclude_same_company'] = True
    
    return params


def _ensure_model_loading():
    """Ensure model loading has started"""
    model_dir = _get_model_dir()
    _manager.start_loading(model_dir)


@require_http_methods(["GET", "POST"])
def main(request):
    """
    Main view for movie recommendation system.
    GET: Display search interface
    POST: Process search and display recommendations
    """
    _ensure_model_loading()
    
    status = _manager.get_status()
    
    if not status['loaded']:
        if request.method == 'GET':
            return render(request, 'recommender/index.html', {
                'all_movie_names': [],
                'total_movies': 0,
                'model_status': status,
            })
        else:
            return render(request, 'recommender/index.html', {
                'all_movie_names': [],
                'total_movies': 0,
                'error_message': 'Model is still loading. Please wait a moment and try again.',
                'model_status': status,
            })
    
    recommender = _manager.get_recommender()
    titles_list = list(recommender.title_to_idx.keys())
    
    if request.method == 'GET':
        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': titles_list,
                'total_movies': len(titles_list),
                'model_status': status,
            }
        )
    
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
    
    filter_params = _parse_filter_params(request)
    
    result = get_recommendations_with_cache(
        recommender,
        movie_name,
        use_cache=True,
        **filter_params
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
                'suggestions': result.get('suggestions', []),
                'model_status': status,
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
            'filters_applied': result.get('filters_applied', {}),
            'filtered_stats': result.get('filtered_stats'),
            'model_status': status,
        }
    )


@require_http_methods(["GET"])
def search_movies(request):
    """API endpoint for searching movies (autocomplete)"""
    query = request.GET.get('q', '').strip()
    
    if len(query) < 2:
        return JsonResponse({'movies': [], 'count': 0})
    
    _ensure_model_loading()
    
    status = _manager.get_status()
    if not status['loaded']:
        return JsonResponse({'movies': [], 'count': 0, 'loading': True, 'progress': status.get('progress', 0)})
    
    recommender = _manager.get_recommender()
    
    try:
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
    _ensure_model_loading()
    
    status = _manager.get_status()
    return JsonResponse(status)


@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint for monitoring"""
    _ensure_model_loading()
    
    status = _manager.get_status()
    
    if status['loaded']:
        recommender = _manager.get_recommender()
        return JsonResponse({
            'status': 'healthy',
            'movies_loaded': recommender.n_movies,
            'model_dir': str(recommender.model_dir),
            'model_loaded': True,
            'cache_stats': get_cache_stats()
        })
    elif status['status'] == 'error':
        return JsonResponse({
            'status': 'unhealthy',
            'error': status.get('error', 'Unknown error')
        }, status=503)
    else:
        return JsonResponse({
            'status': 'loading',
            'progress': status.get('progress', 0)
        })


@require_http_methods(["GET"])
def recommend_api(request):
    """
    JSON API endpoint for movie recommendations.
    
    GET Parameters:
        - movie_title (required): Movie title to base recommendations on
        - n: Number of recommendations (default: 15)
        - min_year: Minimum release year
        - max_year: Maximum release year
        - genres: Comma-separated list of genres
        - min_rating: Minimum rating (0-10)
        - exclude_same_company: 'true' to exclude same production company
    
    Returns:
        JSON response with recommendations or error
    """
    movie_title = request.GET.get('movie_title', '').strip()
    
    if not movie_title:
        return JsonResponse({
            'error': 'movie_title parameter is required',
            'status': 'error'
        }, status=400)
    
    _ensure_model_loading()
    
    status = _manager.get_status()
    
    if not status['loaded']:
        return JsonResponse({
            'error': 'Model is still loading',
            'status': 'loading',
            'progress': status.get('progress', 0)
        }, status=503)
    
    recommender = _manager.get_recommender()
    
    filter_params = _parse_filter_params(request)
    
    try:
        result = get_recommendations_with_cache(
            recommender,
            movie_title,
            use_cache=True,
            **filter_params
        )
        
        result['status'] = 'success' if 'error' not in result else 'error'
        result['cached'] = True
        
        return JsonResponse(result)
        
    except Exception as e:
        logger.error(f"Error in recommend API: {e}")
        return JsonResponse({
            'error': 'Internal server error',
            'status': 'error'
        }, status=500)


@require_http_methods(["POST"])
def clear_cache_api(request):
    """API endpoint to clear recommendation cache (admin use)"""
    clear_cache()
    return JsonResponse({
        'status': 'success',
        'message': 'Cache cleared'
    })
