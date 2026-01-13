"""
Optimized Feature Engineering Pipeline with caching and performance enhancements.

This module extends the base feature engine with intelligent caching,
vectorized operations, and memory-efficient processing.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import threading
from collections import defaultdict, deque
import time

from data.feature_engineering import FeatureEngine, FeatureVector, TechnicalIndicators
from data.weex_fetcher import MarketData
from utils.performance_optimizer import (
    PerformanceOptimizer, performance_monitor, cached_function,
    FeatureCache, memory_efficient_batch_processing
)

logger = logging.getLogger(__name__)


class OptimizedTechnicalIndicators(TechnicalIndicators):
    """Optimized technical indicators with caching and vectorized operations."""
    
    def __init__(self, cache: FeatureCache):
        self.cache = cache
        self.indicator_cache = {}
        self.lock = threading.RLock()
    
    @performance_monitor
    def calculate_returns_vectorized(self, prices: pd.Series, periods: List[int] = [5, 15]) -> Dict[str, float]:
        """Vectorized returns calculation with caching."""
        cache_key = f"returns_{hash(tuple(prices.tail(max(periods)+1).values))}_{tuple(periods)}"
        
        with self.lock:
            if cache_key in self.indicator_cache:
                cached_item = self.indicator_cache[cache_key]
                if (datetime.now() - cached_item['timestamp']).seconds < 60:  # 1 minute cache
                    return cached_item['result']
        
        if len(prices) < max(periods):
            result = {f'returns_{p}m': 0.0 for p in periods}
        else:
            result = {}
            current_price = prices.iloc[-1]
            
            # Vectorized calculation for all periods at once
            for period in periods:
                if len(prices) >= period:
                    past_price = prices.iloc[-period]
                    if past_price != 0:
                        result[f'returns_{period}m'] = (current_price - past_price) / past_price
                    else:
                        result[f'returns_{period}m'] = 0.0
                else:
                    result[f'returns_{period}m'] = 0.0
        
        # Cache result
        with self.lock:
            self.indicator_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }
        
        return result
    
    @performance_monitor
    def calculate_indicators_batch(self, ohlcv_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate multiple indicators in a single pass for efficiency."""
        if len(ohlcv_data) < 26:  # Need at least 26 periods for MACD
            return {
                'rsi_14': 50.0,
                'macd': 0.0,
                'signal': 0.0,
                'histogram': 0.0,
                'atr_percentage': 0.0
            }
        
        try:
            # Pre-allocate result dictionary
            indicators = {}
            
            # Calculate RSI
            delta = ohlcv_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi_14'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            
            # Calculate MACD components
            ema_12 = ohlcv_data['close'].ewm(span=12).mean()
            ema_26 = ohlcv_data['close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            indicators['macd'] = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
            indicators['signal'] = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
            indicators['histogram'] = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
            
            # Calculate ATR
            high_low = ohlcv_data['high'] - ohlcv_data['low']
            high_close = np.abs(ohlcv_data['high'] - ohlcv_data['close'].shift())
            low_close = np.abs(ohlcv_data['low'] - ohlcv_data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean()
            current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
            current_price = float(ohlcv_data['close'].iloc[-1])
            
            indicators['atr_percentage'] = (current_atr / current_price) * 100 if current_price != 0 else 0.0
            
            return indicators
            
        except Exception as e:
            logger.warning(f"Batch indicator calculation failed: {e}")
            return {
                'rsi_14': 50.0,
                'macd': 0.0,
                'signal': 0.0,
                'histogram': 0.0,
                'atr_percentage': 0.0
            }


class OptimizedFeatureEngine(FeatureEngine):
    """
    Optimized feature engineering pipeline with:
    - Intelligent caching of computed features
    - Vectorized operations for better performance
    - Memory-efficient data processing
    - Batch processing capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 performance_optimizer: Optional[PerformanceOptimizer] = None):
        super().__init__(config)
        
        self.performance_optimizer = performance_optimizer
        self.config = config or {}
        
        # Enhanced caching
        self.feature_cache = FeatureCache(
            max_pairs=self.config.get('max_trading_pairs', 10),
            history_length=self.config.get('feature_history_length', 200)
        )
        
        # Optimized technical indicators
        self.technical_indicators = OptimizedTechnicalIndicators(self.feature_cache)
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Pre-computed feature templates
        self.feature_templates = {}
        self.lock = threading.RLock()
        
        # Memory-efficient data structures
        self.rolling_windows = {}
        self.indicator_buffers = {}
        
        logger.info("OptimizedFeatureEngine initialized with performance enhancements")
    
    @performance_monitor
    def process_data_optimized(self, market_data: MarketData, use_cache: bool = True) -> FeatureVector:
        """
        Optimized feature processing with caching and performance monitoring.
        
        Args:
            market_data: Raw market data from WEEX API
            use_cache: Whether to use feature caching
            
        Returns:
            FeatureVector: 9-dimensional state vector for PPO agent
        """
        start_time = time.perf_counter()
        
        # Check cache first
        if use_cache:
            cached_features = self.feature_cache.get_cached_features(
                market_data.pair, market_data.timestamp, tolerance_seconds=30
            )
            if cached_features is not None:
                self.cache_hits += 1
                return FeatureVector(**cached_features)
            
            self.cache_misses += 1
        
        try:
            # Update price history efficiently
            self._update_price_history_optimized(market_data.pair, market_data)
            
            # Get historical data
            history = self.price_history.get(market_data.pair)
            if history is None or len(history) < 2:
                logger.warning(f"Insufficient history for {market_data.pair}, returning zero vector")
                return self._get_zero_vector()
            
            # Batch calculate all indicators
            indicators = self._calculate_all_indicators_batch(history, market_data)
            
            # Create feature vector
            feature_vector = FeatureVector(
                returns_5m=indicators['returns_5m'],
                returns_15m=indicators['returns_15m'],
                rsi_14=indicators['rsi_14'] / 100.0,  # Normalize to [0, 1]
                macd_histogram=np.tanh(indicators['macd_histogram'] * 1000),
                atr_percentage=np.clip(indicators['atr_percentage'] / 10.0, 0, 1),
                volume_zscore=np.tanh(indicators['volume_zscore']),
                orderbook_imbalance=indicators['orderbook_imbalance'],
                funding_rate=np.tanh(indicators['funding_rate'] * 1000),
                volatility_regime=indicators['volatility_regime']
            )
            
            # Validate and cache
            if self._validate_features(feature_vector):
                if use_cache:
                    self.feature_cache.cache_features(
                        market_data.pair, market_data.timestamp, feature_vector.to_dict()
                    )
                
                # Record processing time
                processing_time = (time.perf_counter() - start_time) * 1000
                self.processing_times.append(processing_time)
                
                if self.performance_optimizer:
                    self.performance_optimizer.monitor.record_metric(
                        'feature_calculation_time_ms', processing_time
                    )
                
                return feature_vector
            else:
                logger.warning("Feature validation failed, returning zero vector")
                return self._get_zero_vector()
                
        except Exception as e:
            logger.error(f"Optimized feature processing failed for {market_data.pair}: {e}")
            return self._get_zero_vector()
    
    def _update_price_history_optimized(self, pair: str, market_data: MarketData) -> None:
        """Memory-efficient price history update."""
        with self.lock:
            if pair not in self.price_history:
                # Initialize with pre-allocated DataFrame
                self.price_history[pair] = pd.DataFrame(
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
            
            # Use efficient append method
            new_data = {
                'timestamp': market_data.timestamp,
                'open': market_data.open,
                'high': market_data.high,
                'low': market_data.low,
                'close': market_data.close,
                'volume': market_data.volume
            }
            
            # Append using loc for better performance
            idx = len(self.price_history[pair])
            self.price_history[pair].loc[idx] = new_data
            
            # Efficient memory management
            if len(self.price_history[pair]) > self.max_history_length:
                # Remove oldest 20% of data to avoid frequent trimming
                trim_size = int(self.max_history_length * 0.2)
                self.price_history[pair] = self.price_history[pair].iloc[trim_size:].reset_index(drop=True)
    
    def _calculate_all_indicators_batch(self, history: pd.DataFrame, market_data: MarketData) -> Dict[str, float]:
        """Calculate all indicators in a single batch operation."""
        indicators = {}
        
        # Returns calculation (vectorized)
        returns = self.technical_indicators.calculate_returns_vectorized(history['close'], [5, 15])
        indicators.update(returns)
        
        # Technical indicators (batch calculation)
        tech_indicators = self.technical_indicators.calculate_indicators_batch(history)
        indicators.update(tech_indicators)
        
        # Volume z-score (optimized)
        indicators['volume_zscore'] = self._calculate_volume_zscore_optimized(history['volume'])
        
        # Orderbook imbalance
        indicators['orderbook_imbalance'] = self.orderbook_analyzer.calculate_imbalance(
            market_data.orderbook_bids, market_data.orderbook_asks
        )
        
        # Volatility regime (cached)
        indicators['volatility_regime'] = self._detect_volatility_regime_cached(history)
        
        # Funding rate
        indicators['funding_rate'] = market_data.funding_rate if market_data.funding_rate is not None else 0.0
        
        return indicators
    
    def _calculate_volume_zscore_optimized(self, volumes: pd.Series, window: int = 20) -> float:
        """Optimized volume z-score calculation."""
        if len(volumes) < window:
            return 0.0
        
        try:
            # Use rolling operations for efficiency
            rolling_mean = volumes.rolling(window=window).mean()
            rolling_std = volumes.rolling(window=window).std()
            
            current_volume = volumes.iloc[-1]
            mean_volume = rolling_mean.iloc[-1]
            std_volume = rolling_std.iloc[-1]
            
            if std_volume == 0 or pd.isna(std_volume):
                return 0.0
            
            zscore = (current_volume - mean_volume) / std_volume
            return float(zscore)
            
        except Exception as e:
            logger.warning(f"Optimized volume z-score calculation failed: {e}")
            return 0.0
    
    def _detect_volatility_regime_cached(self, history: pd.DataFrame) -> int:
        """Cached volatility regime detection."""
        if len(history) < 25:
            return 0
        
        # Generate cache key based on recent price data
        recent_prices = history['close'].tail(25)
        cache_key = f"regime_{hash(tuple(recent_prices.values))}"
        
        # Check cache
        cached_regime = self.feature_cache.get_cached_indicator(
            'volatility_regime', 'regime_detection', {'key': cache_key}, max_age_seconds=120
        )
        
        if cached_regime is not None:
            return cached_regime
        
        # Calculate regime
        regime = self.regime_detector.detect_regime(
            history['close'], history['high'], history['low']
        )
        
        # Cache result
        self.feature_cache.cache_indicator(
            'volatility_regime', 'regime_detection', {'key': cache_key}, regime
        )
        
        return regime
    
    @memory_efficient_batch_processing(batch_size=16)
    def batch_process_features(self, market_data_list: List[MarketData]) -> List[FeatureVector]:
        """Memory-efficient batch processing of multiple market data points."""
        features = []
        
        for market_data in market_data_list:
            try:
                feature_vector = self.process_data_optimized(market_data, use_cache=True)
                features.append(feature_vector)
            except Exception as e:
                logger.error(f"Failed to process features for {market_data.pair}: {e}")
                features.append(self._get_zero_vector())
        
        return features
    
    def warm_up_cache(self, historical_data: Dict[str, List[MarketData]]) -> None:
        """Warm up feature cache with historical data."""
        logger.info("Warming up feature cache...")
        
        total_processed = 0
        for pair, data_list in historical_data.items():
            for market_data in data_list[-50:]:  # Use last 50 data points
                self.process_data_optimized(market_data, use_cache=True)
                total_processed += 1
        
        logger.info(f"Feature cache warmed up with {total_processed} data points")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the optimized feature engine."""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'avg_processing_time_ms': avg_processing_time,
            'feature_cache_size': len(self.feature_cache.feature_cache),
            'indicator_cache_size': len(self.feature_cache.indicator_cache),
            'recent_processing_times': list(self.processing_times)[-10:]
        }
    
    def optimize_memory_usage(self) -> Dict[str, int]:
        """Optimize memory usage by cleaning up old data."""
        results = {}
        
        # Clean up feature cache
        results['feature_cache_cleaned'] = self.feature_cache.cleanup_old_data(max_age_hours=12)
        
        # Clean up price history
        with self.lock:
            original_total = sum(len(df) for df in self.price_history.values())
            
            for pair in list(self.price_history.keys()):
                if len(self.price_history[pair]) > self.max_history_length:
                    # Keep only recent data
                    self.price_history[pair] = self.price_history[pair].tail(
                        int(self.max_history_length * 0.8)
                    ).reset_index(drop=True)
            
            new_total = sum(len(df) for df in self.price_history.values())
            results['price_history_cleaned'] = original_total - new_total
        
        # Clean up indicator cache
        if hasattr(self.technical_indicators, 'indicator_cache'):
            old_cache_size = len(self.technical_indicators.indicator_cache)
            # Remove entries older than 1 hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            with self.technical_indicators.lock:
                expired_keys = [
                    key for key, item in self.technical_indicators.indicator_cache.items()
                    if item['timestamp'] < cutoff_time
                ]
                
                for key in expired_keys:
                    del self.technical_indicators.indicator_cache[key]
            
            results['indicator_cache_cleaned'] = old_cache_size - len(self.technical_indicators.indicator_cache)
        
        logger.info(f"Memory optimization completed: {results}")
        return results
    
    def precompute_feature_templates(self, pairs: List[str]) -> None:
        """Precompute feature templates for common trading pairs."""
        logger.info(f"Precomputing feature templates for {len(pairs)} pairs...")
        
        with self.lock:
            for pair in pairs:
                # Create template with default values
                template = {
                    'returns_5m': 0.0,
                    'returns_15m': 0.0,
                    'rsi_14': 0.5,
                    'macd_histogram': 0.0,
                    'atr_percentage': 0.0,
                    'volume_zscore': 0.0,
                    'orderbook_imbalance': 0.0,
                    'funding_rate': 0.0,
                    'volatility_regime': 0
                }
                
                self.feature_templates[pair] = template
        
        logger.info("Feature templates precomputed")
    
    def get_feature_template(self, pair: str) -> Dict[str, float]:
        """Get precomputed feature template for a pair."""
        with self.lock:
            return self.feature_templates.get(pair, {}).copy()


def create_optimized_feature_engine(config: Dict[str, Any], 
                                   performance_optimizer: PerformanceOptimizer) -> OptimizedFeatureEngine:
    """Factory function to create optimized feature engine."""
    return OptimizedFeatureEngine(config, performance_optimizer)