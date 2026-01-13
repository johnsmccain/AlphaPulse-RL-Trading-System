"""
Performance optimization utilities for AlphaPulse-RL trading system.

This module implements caching, memory optimization, and performance monitoring
to improve system efficiency and reduce latency.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, Callable, Tuple, List
from functools import wraps, lru_cache
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import torch
import psutil
import gc
from collections import defaultdict, deque
import pickle
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance monitoring metrics."""
    inference_time_ms: float
    memory_usage_mb: float
    cache_hit_rate: float
    feature_calc_time_ms: float
    risk_calc_time_ms: float
    total_latency_ms: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None


class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.start_times = {}
        self.lock = threading.Lock()
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        with self.lock:
            self.start_times[operation] = time.perf_counter()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration in milliseconds."""
        with self.lock:
            if operation not in self.start_times:
                logger.warning(f"Timer not started for operation: {operation}")
                return 0.0
            
            duration_ms = (time.perf_counter() - self.start_times[operation]) * 1000
            self.metrics_history[f"{operation}_time_ms"].append(duration_ms)
            del self.start_times[operation]
            return duration_ms
    
    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a performance metric."""
        with self.lock:
            self.metrics_history[metric_name].append(value)
    
    def get_average_metric(self, metric_name: str) -> float:
        """Get average value for a metric over the window."""
        with self.lock:
            values = self.metrics_history.get(metric_name, [])
            return sum(values) / len(values) if values else 0.0
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource usage."""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            metrics = {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_mb': memory_mb,
                'memory_available_mb': memory.available / (1024 * 1024),
                'memory_percent': memory.percent
            }
            
            # GPU usage if available
            try:
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_memory_cached = torch.cuda.memory_reserved() / (1024 * 1024)
                    metrics.update({
                        'gpu_memory_mb': gpu_memory,
                        'gpu_memory_cached_mb': gpu_memory_cached,
                        'gpu_available': True
                    })
                else:
                    metrics['gpu_available'] = False
            except Exception as e:
                logger.debug(f"GPU metrics unavailable: {e}")
                metrics['gpu_available'] = False
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self.lock:
            summary = {}
            
            # Average metrics over window
            for metric_name, values in self.metrics_history.items():
                if values:
                    summary[f"avg_{metric_name}"] = sum(values) / len(values)
                    summary[f"max_{metric_name}"] = max(values)
                    summary[f"min_{metric_name}"] = min(values)
            
            # Current system metrics
            summary.update(self.get_system_metrics())
            
            return summary


class IntelligentCache:
    """Intelligent caching system with TTL and memory management."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.expiry_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, expiry in self.expiry_times.items()
            if expiry < current_time
        ]
        
        for key in expired_keys:
            self._remove_key(key)
    
    def _remove_key(self, key: str) -> None:
        """Remove a key from all cache structures."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.expiry_times.pop(key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used items if cache is full."""
        if len(self.cache) >= self.max_size:
            # Find least recently used key
            lru_key = min(self.access_times.keys(), key=self.access_times.get)
            self._remove_key(lru_key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            self._cleanup_expired()
            
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache with optional TTL."""
        with self.lock:
            self._cleanup_expired()
            self._evict_lru()
            
            current_time = time.time()
            ttl = ttl or self.default_ttl
            
            self.cache[key] = value
            self.access_times[key] = current_time
            self.expiry_times[key] = current_time + ttl
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries matching pattern."""
        with self.lock:
            if pattern is None:
                # Clear all
                count = len(self.cache)
                self.cache.clear()
                self.access_times.clear()
                self.expiry_times.clear()
                return count
            else:
                # Clear matching pattern
                matching_keys = [key for key in self.cache.keys() if pattern in key]
                for key in matching_keys:
                    self._remove_key(key)
                return len(matching_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'utilization': len(self.cache) / self.max_size
            }


class ModelOptimizer:
    """Optimize PyTorch model inference performance."""
    
    def __init__(self):
        self.optimized_models = {}
        self.device_cache = {}
    
    def optimize_model(self, model: torch.nn.Module, model_name: str) -> torch.nn.Module:
        """Optimize model for inference performance."""
        if model_name in self.optimized_models:
            return self.optimized_models[model_name]
        
        try:
            # Set to evaluation mode
            model.eval()
            
            # Disable gradient computation
            for param in model.parameters():
                param.requires_grad = False
            
            # Try to compile model if PyTorch 2.0+
            try:
                if hasattr(torch, 'compile'):
                    model = torch.compile(model, mode='reduce-overhead')
                    logger.info(f"Model {model_name} compiled with torch.compile")
            except Exception as e:
                logger.debug(f"torch.compile not available or failed: {e}")
            
            # Cache optimized model
            self.optimized_models[model_name] = model
            
            logger.info(f"Model {model_name} optimized for inference")
            return model
            
        except Exception as e:
            logger.error(f"Failed to optimize model {model_name}: {e}")
            return model
    
    def get_optimal_device(self, model_name: str) -> torch.device:
        """Get optimal device for model inference."""
        if model_name in self.device_cache:
            return self.device_cache[model_name]
        
        # Determine best device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA for {model_name}")
        else:
            device = torch.device('cpu')
            logger.info(f"Using CPU for {model_name}")
        
        self.device_cache[model_name] = device
        return device
    
    def batch_inference(self, model: torch.nn.Module, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Perform batched inference for better GPU utilization."""
        if not inputs:
            return []
        
        try:
            # Stack inputs into batch
            batch_input = torch.stack(inputs)
            
            # Perform batched inference
            with torch.no_grad():
                batch_output = model(batch_input)
            
            # Split batch output back to individual results
            return [batch_output[i] for i in range(len(inputs))]
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            # Fallback to individual inference
            results = []
            with torch.no_grad():
                for input_tensor in inputs:
                    results.append(model(input_tensor.unsqueeze(0)).squeeze(0))
            return results


class FeatureCache:
    """Specialized cache for feature engineering results."""
    
    def __init__(self, max_pairs: int = 10, history_length: int = 200):
        self.max_pairs = max_pairs
        self.history_length = history_length
        self.feature_cache = {}
        self.indicator_cache = {}
        self.lock = threading.RLock()
    
    def cache_features(self, pair: str, timestamp: datetime, features: Dict[str, float]) -> None:
        """Cache computed features for a trading pair."""
        with self.lock:
            if pair not in self.feature_cache:
                self.feature_cache[pair] = deque(maxlen=self.history_length)
            
            self.feature_cache[pair].append({
                'timestamp': timestamp,
                'features': features.copy()
            })
    
    def get_cached_features(self, pair: str, timestamp: datetime, 
                          tolerance_seconds: int = 60) -> Optional[Dict[str, float]]:
        """Get cached features if available within tolerance."""
        with self.lock:
            if pair not in self.feature_cache:
                return None
            
            # Find features within tolerance
            for cached_item in reversed(self.feature_cache[pair]):
                time_diff = abs((timestamp - cached_item['timestamp']).total_seconds())
                if time_diff <= tolerance_seconds:
                    return cached_item['features'].copy()
            
            return None
    
    def cache_indicator(self, pair: str, indicator_name: str, 
                       params: Dict, result: Any) -> None:
        """Cache technical indicator calculation result."""
        with self.lock:
            cache_key = f"{pair}:{indicator_name}:{hash(str(sorted(params.items())))}"
            self.indicator_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now(),
                'params': params.copy()
            }
    
    def get_cached_indicator(self, pair: str, indicator_name: str, 
                           params: Dict, max_age_seconds: int = 300) -> Optional[Any]:
        """Get cached indicator result if available and fresh."""
        with self.lock:
            cache_key = f"{pair}:{indicator_name}:{hash(str(sorted(params.items())))}"
            
            if cache_key in self.indicator_cache:
                cached_item = self.indicator_cache[cache_key]
                age = (datetime.now() - cached_item['timestamp']).total_seconds()
                
                if age <= max_age_seconds and cached_item['params'] == params:
                    return cached_item['result']
            
            return None
    
    def cleanup_old_data(self, max_age_hours: int = 24) -> int:
        """Clean up old cached data."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            removed_count = 0
            
            # Clean feature cache
            for pair in list(self.feature_cache.keys()):
                original_length = len(self.feature_cache[pair])
                self.feature_cache[pair] = deque(
                    [item for item in self.feature_cache[pair] 
                     if item['timestamp'] > cutoff_time],
                    maxlen=self.history_length
                )
                removed_count += original_length - len(self.feature_cache[pair])
            
            # Clean indicator cache
            expired_keys = [
                key for key, item in self.indicator_cache.items()
                if item['timestamp'] < cutoff_time
            ]
            
            for key in expired_keys:
                del self.indicator_cache[key]
                removed_count += 1
            
            return removed_count


def performance_monitor(monitor: PerformanceMonitor):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = f"{func.__module__}.{func.__name__}"
            monitor.start_timer(operation_name)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                monitor.end_timer(operation_name)
        
        return wrapper
    return decorator


def cached_function(cache: IntelligentCache, ttl: Optional[int] = None):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def memory_efficient_batch_processing(batch_size: int = 32):
    """Decorator for memory-efficient batch processing."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(data_list: List[Any], *args, **kwargs):
            if len(data_list) <= batch_size:
                return func(data_list, *args, **kwargs)
            
            # Process in batches
            results = []
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                batch_results = func(batch, *args, **kwargs)
                results.extend(batch_results)
                
                # Force garbage collection between batches
                gc.collect()
            
            return results
        
        return wrapper
    return decorator


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitor = PerformanceMonitor(
            window_size=config.get('monitor_window_size', 100)
        )
        self.cache = IntelligentCache(
            max_size=config.get('cache_max_size', 1000),
            default_ttl=config.get('cache_default_ttl', 300)
        )
        self.model_optimizer = ModelOptimizer()
        self.feature_cache = FeatureCache(
            max_pairs=config.get('max_trading_pairs', 10),
            history_length=config.get('feature_history_length', 200)
        )
        
        # Performance thresholds
        self.max_inference_time_ms = config.get('max_inference_time_ms', 100)
        self.max_memory_usage_mb = config.get('max_memory_usage_mb', 1024)
        self.min_cache_hit_rate = config.get('min_cache_hit_rate', 0.7)
        
        logger.info("PerformanceOptimizer initialized")
    
    def optimize_model_inference(self, model: torch.nn.Module, model_name: str) -> torch.nn.Module:
        """Optimize model for inference performance."""
        return self.model_optimizer.optimize_model(model, model_name)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        system_metrics = self.monitor.get_system_metrics()
        cache_stats = self.cache.get_stats()
        
        return PerformanceMetrics(
            inference_time_ms=self.monitor.get_average_metric('model_inference_time_ms'),
            memory_usage_mb=system_metrics.get('memory_usage_mb', 0),
            cache_hit_rate=cache_stats['hit_rate'],
            feature_calc_time_ms=self.monitor.get_average_metric('feature_calculation_time_ms'),
            risk_calc_time_ms=self.monitor.get_average_metric('risk_calculation_time_ms'),
            total_latency_ms=self.monitor.get_average_metric('total_latency_time_ms'),
            cpu_usage_percent=system_metrics.get('cpu_usage_percent', 0),
            gpu_usage_percent=system_metrics.get('gpu_memory_mb')
        )
    
    def check_performance_thresholds(self) -> Dict[str, Any]:
        """Check if performance is within acceptable thresholds."""
        metrics = self.get_performance_metrics()
        issues = []
        warnings = []
        
        # Check inference time
        if metrics.inference_time_ms > self.max_inference_time_ms:
            issues.append(f"Inference time {metrics.inference_time_ms:.2f}ms exceeds threshold {self.max_inference_time_ms}ms")
        
        # Check memory usage
        if metrics.memory_usage_mb > self.max_memory_usage_mb:
            warnings.append(f"Memory usage {metrics.memory_usage_mb:.2f}MB exceeds threshold {self.max_memory_usage_mb}MB")
        
        # Check cache hit rate
        if metrics.cache_hit_rate < self.min_cache_hit_rate:
            warnings.append(f"Cache hit rate {metrics.cache_hit_rate:.2f} below threshold {self.min_cache_hit_rate}")
        
        return {
            'performance_ok': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'metrics': metrics
        }
    
    def cleanup_resources(self) -> Dict[str, int]:
        """Clean up cached resources and free memory."""
        results = {}
        
        # Clean up feature cache
        results['feature_cache_cleaned'] = self.feature_cache.cleanup_old_data()
        
        # Clean up main cache
        results['main_cache_cleaned'] = self.cache.invalidate()
        
        # Force garbage collection
        collected = gc.collect()
        results['gc_collected'] = collected
        
        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            results['gpu_cache_cleared'] = True
        
        logger.info(f"Resource cleanup completed: {results}")
        return results
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        performance_check = self.check_performance_thresholds()
        cache_stats = self.cache.get_stats()
        system_metrics = self.monitor.get_system_metrics()
        performance_summary = self.monitor.get_performance_summary()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'performance_status': performance_check,
            'cache_statistics': cache_stats,
            'system_metrics': system_metrics,
            'performance_history': performance_summary,
            'recommendations': self._generate_recommendations(performance_check, cache_stats)
        }
    
    def _generate_recommendations(self, performance_check: Dict, cache_stats: Dict) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if performance_check['metrics'].inference_time_ms > 50:
            recommendations.append("Consider model quantization or pruning to reduce inference time")
        
        if cache_stats['hit_rate'] < 0.5:
            recommendations.append("Increase cache size or TTL to improve hit rate")
        
        if performance_check['metrics'].memory_usage_mb > 512:
            recommendations.append("Consider reducing batch sizes or implementing memory pooling")
        
        if performance_check['metrics'].cpu_usage_percent > 80:
            recommendations.append("Consider distributing workload or optimizing CPU-intensive operations")
        
        return recommendations