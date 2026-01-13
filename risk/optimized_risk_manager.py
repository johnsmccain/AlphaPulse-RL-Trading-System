"""
Optimized Risk Manager with performance enhancements and caching.

This module extends the base risk manager with intelligent caching,
vectorized calculations, and optimized risk metric computations.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading
from collections import deque
import time

from risk.risk_manager import RiskManager, RiskMetrics
from trading.portfolio import PortfolioState, Position
from utils.performance_optimizer import (
    PerformanceOptimizer, performance_monitor, cached_function,
    IntelligentCache
)

logger = logging.getLogger(__name__)


@dataclass
class CachedRiskMetrics(RiskMetrics):
    """Extended risk metrics with caching metadata."""
    cache_timestamp: datetime
    cache_key: str
    computation_time_ms: float


class OptimizedRiskManager(RiskManager):
    """
    Optimized risk manager with:
    - Cached risk calculations
    - Vectorized portfolio metrics
    - Efficient validation algorithms
    - Performance monitoring
    """
    
    def __init__(self, config_path: str = "config/trading_params.yaml", 
                 enable_monitoring: bool = True,
                 performance_optimizer: Optional[PerformanceOptimizer] = None):
        super().__init__(config_path, enable_monitoring)
        
        self.performance_optimizer = performance_optimizer
        
        # Risk calculation cache
        self.risk_cache = IntelligentCache(
            max_size=500,
            default_ttl=30  # 30 seconds cache for risk metrics
        )
        
        # Validation cache
        self.validation_cache = IntelligentCache(
            max_size=1000,
            default_ttl=10  # 10 seconds cache for trade validation
        )
        
        # Performance tracking
        self.calculation_times = deque(maxlen=100)
        self.validation_times = deque(maxlen=100)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Pre-computed risk thresholds
        self.risk_thresholds = self._precompute_risk_thresholds()
        
        # Thread-safe operations
        self.lock = threading.RLock()
        
        logger.info("OptimizedRiskManager initialized with performance enhancements")
    
    def _precompute_risk_thresholds(self) -> Dict[str, float]:
        """Precompute commonly used risk thresholds."""
        return {
            'leverage_threshold': self.max_leverage,
            'position_size_threshold': self.max_position_size_percent / 100.0,
            'daily_loss_threshold': self.max_daily_loss_percent / 100.0,
            'drawdown_threshold': self.max_total_drawdown_percent / 100.0,
            'volatility_threshold': self.volatility_threshold,
            'margin_buffer': 0.05  # 5% margin buffer
        }
    
    @performance_monitor
    def validate_trade_optimized(self, action: List[float], portfolio: PortfolioState, 
                                current_price: float, volatility: float = 0.0, 
                                pair: str = "BTCUSDT", use_cache: bool = True) -> Tuple[bool, str]:
        """
        Optimized trade validation with caching and performance monitoring.
        
        Args:
            action: [direction, size, leverage] from PPO agent
            portfolio: Current portfolio state
            current_price: Current market price
            volatility: Current market volatility
            pair: Trading pair for enhanced monitoring
            use_cache: Whether to use validation caching
            
        Returns:
            Tuple of (is_valid, reason)
        """
        start_time = time.perf_counter()
        
        # Generate cache key if caching is enabled
        if use_cache:
            cache_key = self._generate_validation_cache_key(
                action, portfolio, current_price, volatility, pair
            )
            
            cached_result = self.validation_cache.get(cache_key)
            if cached_result is not None:
                self.cache_hits += 1
                return cached_result
            
            self.cache_misses += 1
        
        # Perform validation
        result = self._perform_optimized_validation(action, portfolio, current_price, volatility, pair)
        
        # Cache the result
        if use_cache:
            self.validation_cache.put(cache_key, result)
        
        # Record validation time
        validation_time = (time.perf_counter() - start_time) * 1000
        self.validation_times.append(validation_time)
        
        if self.performance_optimizer:
            self.performance_optimizer.monitor.record_metric('risk_validation_time_ms', validation_time)
        
        return result
    
    def _perform_optimized_validation(self, action: List[float], portfolio: PortfolioState,
                                    current_price: float, volatility: float, pair: str) -> Tuple[bool, str]:
        """Perform optimized validation with vectorized operations."""
        # Check emergency mode first (fastest check)
        if self.emergency_mode:
            return False, "Emergency mode active - all trading suspended"
        
        # Extract action components
        direction, size, leverage = action
        
        # Vectorized threshold checks
        thresholds = self.risk_thresholds
        
        # 1. Leverage check
        if leverage > thresholds['leverage_threshold']:
            return False, f"Leverage {leverage:.2f}x exceeds maximum {thresholds['leverage_threshold']}x"
        
        # 2. Position size check
        total_equity = portfolio.get_total_equity()
        position_size_percent = size
        
        if position_size_percent > thresholds['position_size_threshold']:
            return False, (f"Position size {position_size_percent*100:.2f}% exceeds "
                          f"maximum {thresholds['position_size_threshold']*100:.2f}%")
        
        # 3. Daily loss check (vectorized)
        if portfolio.daily_start_balance > 0:
            daily_loss_ratio = abs(portfolio.daily_pnl / portfolio.daily_start_balance)
            if portfolio.daily_pnl < 0 and daily_loss_ratio >= thresholds['daily_loss_threshold']:
                return False, (f"Daily loss {daily_loss_ratio*100:.2f}% at limit "
                              f"{thresholds['daily_loss_threshold']*100:.2f}%")
        
        # 4. Drawdown check
        if portfolio.max_drawdown >= thresholds['drawdown_threshold']:
            self._trigger_emergency_mode(portfolio)
            return False, (f"Total drawdown {portfolio.max_drawdown*100:.2f}% exceeds "
                          f"limit {thresholds['drawdown_threshold']*100:.2f}%")
        
        # 5. Enhanced volatility check
        if self.risk_monitor:
            is_safe, volatility_reason = self.risk_monitor.check_volatility_threshold(pair, current_price)
            if not is_safe:
                return False, f"Enhanced volatility check failed: {volatility_reason}"
        else:
            if volatility > thresholds['volatility_threshold']:
                return False, (f"Market volatility {volatility:.4f} exceeds "
                              f"threshold {thresholds['volatility_threshold']:.4f}")
        
        # 6. Margin check (optimized)
        position_value = size * total_equity
        margin_required = position_value
        available_margin = total_equity - portfolio.get_total_margin_used()
        margin_with_buffer = margin_required * (1 + thresholds['margin_buffer'])
        
        if margin_with_buffer > available_margin:
            return False, (f"Insufficient margin: required {margin_with_buffer:.2f}, "
                          f"available {available_margin:.2f}")
        
        return True, "Trade validated successfully"
    
    @performance_monitor
    def get_risk_metrics_optimized(self, portfolio: PortfolioState, volatility: float = 0.0,
                                  use_cache: bool = True) -> CachedRiskMetrics:
        """
        Optimized risk metrics calculation with caching.
        
        Args:
            portfolio: Current portfolio state
            volatility: Current market volatility
            use_cache: Whether to use risk metrics caching
            
        Returns:
            CachedRiskMetrics object with current risk statistics
        """
        start_time = time.perf_counter()
        
        # Generate cache key if caching is enabled
        if use_cache:
            cache_key = self._generate_risk_metrics_cache_key(portfolio, volatility)
            
            cached_metrics = self.risk_cache.get(cache_key)
            if cached_metrics is not None:
                self.cache_hits += 1
                return cached_metrics
            
            self.cache_misses += 1
        
        # Calculate metrics using vectorized operations
        metrics = self._calculate_risk_metrics_vectorized(portfolio, volatility)
        
        # Create cached metrics object
        computation_time = (time.perf_counter() - start_time) * 1000
        cached_metrics = CachedRiskMetrics(
            current_drawdown=metrics.current_drawdown,
            daily_pnl_percent=metrics.daily_pnl_percent,
            position_exposure_percent=metrics.position_exposure_percent,
            total_leverage=metrics.total_leverage,
            margin_utilization=metrics.margin_utilization,
            volatility_level=metrics.volatility_level,
            risk_score=metrics.risk_score,
            cache_timestamp=datetime.now(),
            cache_key=cache_key if use_cache else "",
            computation_time_ms=computation_time
        )
        
        # Cache the result
        if use_cache:
            self.risk_cache.put(cache_key, cached_metrics)
        
        # Record calculation time
        self.calculation_times.append(computation_time)
        
        if self.performance_optimizer:
            self.performance_optimizer.monitor.record_metric('risk_calculation_time_ms', computation_time)
        
        return cached_metrics
    
    def _calculate_risk_metrics_vectorized(self, portfolio: PortfolioState, volatility: float) -> RiskMetrics:
        """Calculate risk metrics using vectorized operations for better performance."""
        total_equity = portfolio.get_total_equity()
        
        # Vectorized calculations
        daily_pnl_percent = 0.0
        if portfolio.daily_start_balance > 0:
            daily_pnl_percent = (portfolio.daily_pnl / portfolio.daily_start_balance) * 100
        
        # Portfolio exposure calculations (vectorized)
        if portfolio.positions:
            position_values = np.array([
                pos.size * pos.current_price * pos.leverage 
                for pos in portfolio.positions.values()
            ])
            margin_values = np.array([
                pos.size * pos.current_price 
                for pos in portfolio.positions.values()
            ])
            
            total_exposure = np.sum(position_values)
            total_margin = np.sum(margin_values)
        else:
            total_exposure = 0.0
            total_margin = 0.0
        
        # Calculate derived metrics
        position_exposure_percent = (total_exposure / total_equity) * 100 if total_equity > 0 else 0.0
        total_leverage = total_exposure / total_margin if total_margin > 0 else 0.0
        margin_utilization = (total_margin / total_equity) * 100 if total_equity > 0 else 0.0
        
        # Vectorized risk score calculation
        risk_components = np.array([
            portfolio.max_drawdown * 100 / self.max_total_drawdown_percent,
            abs(daily_pnl_percent) / self.max_daily_loss_percent,
            position_exposure_percent / (self.max_position_size_percent * 10),
            volatility * 100 / (self.volatility_threshold * 100)
        ])
        
        # Clip to [0, 1] range
        risk_components = np.clip(risk_components, 0, 1)
        
        # Weighted risk score
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        risk_score = np.dot(risk_components, weights) * 100
        
        return RiskMetrics(
            current_drawdown=portfolio.max_drawdown * 100,
            daily_pnl_percent=daily_pnl_percent,
            position_exposure_percent=position_exposure_percent,
            total_leverage=total_leverage,
            margin_utilization=margin_utilization,
            volatility_level=volatility,
            risk_score=min(100, risk_score)
        )
    
    def batch_validate_trades(self, trade_requests: List[Dict[str, Any]]) -> List[Tuple[bool, str]]:
        """
        Batch validate multiple trades for improved performance.
        
        Args:
            trade_requests: List of trade request dictionaries
            
        Returns:
            List of validation results
        """
        results = []
        
        # Pre-compute common values
        emergency_active = self.emergency_mode
        thresholds = self.risk_thresholds
        
        for request in trade_requests:
            if emergency_active:
                results.append((False, "Emergency mode active"))
                continue
            
            try:
                result = self.validate_trade_optimized(
                    action=request['action'],
                    portfolio=request['portfolio'],
                    current_price=request['current_price'],
                    volatility=request.get('volatility', 0.0),
                    pair=request.get('pair', 'BTCUSDT'),
                    use_cache=request.get('use_cache', True)
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Batch validation failed for request: {e}")
                results.append((False, f"Validation error: {e}"))
        
        return results
    
    def _generate_validation_cache_key(self, action: List[float], portfolio: PortfolioState,
                                     current_price: float, volatility: float, pair: str) -> str:
        """Generate cache key for trade validation."""
        # Round values to reduce cache key variations
        rounded_action = [round(x, 4) for x in action]
        rounded_price = round(current_price, 2)
        rounded_volatility = round(volatility, 6)
        
        # Include key portfolio metrics
        portfolio_key = f"{portfolio.balance:.2f}_{portfolio.daily_pnl:.2f}_{portfolio.max_drawdown:.4f}"
        
        return f"validate_{rounded_action}_{portfolio_key}_{rounded_price}_{rounded_volatility}_{pair}"
    
    def _generate_risk_metrics_cache_key(self, portfolio: PortfolioState, volatility: float) -> str:
        """Generate cache key for risk metrics."""
        # Include key portfolio state
        portfolio_key = (f"{portfolio.balance:.2f}_{portfolio.daily_pnl:.2f}_"
                        f"{portfolio.max_drawdown:.4f}_{len(portfolio.positions)}")
        
        # Include position summary
        if portfolio.positions:
            position_summary = sum(pos.unrealized_pnl for pos in portfolio.positions.values())
            portfolio_key += f"_{position_summary:.2f}"
        
        rounded_volatility = round(volatility, 6)
        
        return f"risk_metrics_{portfolio_key}_{rounded_volatility}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the optimized risk manager."""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        avg_calculation_time = (sum(self.calculation_times) / len(self.calculation_times) 
                               if self.calculation_times else 0.0)
        avg_validation_time = (sum(self.validation_times) / len(self.validation_times) 
                              if self.validation_times else 0.0)
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'avg_calculation_time_ms': avg_calculation_time,
            'avg_validation_time_ms': avg_validation_time,
            'risk_cache_stats': self.risk_cache.get_stats(),
            'validation_cache_stats': self.validation_cache.get_stats(),
            'recent_calculation_times': list(self.calculation_times)[-10:],
            'recent_validation_times': list(self.validation_times)[-10:]
        }
    
    def optimize_cache_settings(self, validation_frequency: float, metrics_frequency: float) -> None:
        """
        Optimize cache settings based on usage patterns.
        
        Args:
            validation_frequency: Average validations per second
            metrics_frequency: Average metrics calculations per second
        """
        # Adjust cache sizes based on frequency
        if validation_frequency > 10:  # High frequency
            self.validation_cache.max_size = 2000
            self.validation_cache.default_ttl = 5
        elif validation_frequency > 1:  # Medium frequency
            self.validation_cache.max_size = 1000
            self.validation_cache.default_ttl = 10
        else:  # Low frequency
            self.validation_cache.max_size = 500
            self.validation_cache.default_ttl = 30
        
        if metrics_frequency > 5:  # High frequency
            self.risk_cache.max_size = 1000
            self.risk_cache.default_ttl = 15
        elif metrics_frequency > 0.5:  # Medium frequency
            self.risk_cache.max_size = 500
            self.risk_cache.default_ttl = 30
        else:  # Low frequency
            self.risk_cache.max_size = 200
            self.risk_cache.default_ttl = 60
        
        logger.info(f"Cache settings optimized for validation_freq={validation_frequency:.2f}, "
                   f"metrics_freq={metrics_frequency:.2f}")
    
    def clear_caches(self) -> Dict[str, int]:
        """Clear all caches and return statistics."""
        risk_cleared = self.risk_cache.invalidate()
        validation_cleared = self.validation_cache.invalidate()
        
        # Reset counters
        self.cache_hits = 0
        self.cache_misses = 0
        self.calculation_times.clear()
        self.validation_times.clear()
        
        return {
            'risk_cache_cleared': risk_cleared,
            'validation_cache_cleared': validation_cleared
        }
    
    def precompute_common_validations(self, common_scenarios: List[Dict[str, Any]]) -> None:
        """Precompute validations for common trading scenarios."""
        logger.info(f"Precomputing validations for {len(common_scenarios)} scenarios...")
        
        for scenario in common_scenarios:
            try:
                self.validate_trade_optimized(
                    action=scenario['action'],
                    portfolio=scenario['portfolio'],
                    current_price=scenario['current_price'],
                    volatility=scenario.get('volatility', 0.0),
                    pair=scenario.get('pair', 'BTCUSDT'),
                    use_cache=True
                )
            except Exception as e:
                logger.warning(f"Failed to precompute scenario: {e}")
        
        logger.info("Validation precomputation completed")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        performance_stats = self.get_performance_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'performance_stats': performance_stats,
            'cache_efficiency': {
                'risk_cache': self.risk_cache.get_stats(),
                'validation_cache': self.validation_cache.get_stats()
            },
            'optimization_status': {
                'vectorized_calculations': True,
                'caching_enabled': True,
                'batch_processing_enabled': True,
                'performance_monitoring': self.performance_optimizer is not None
            },
            'recommendations': self._generate_optimization_recommendations(performance_stats)
        }
    
    def _generate_optimization_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on performance stats."""
        recommendations = []
        
        if stats['cache_hit_rate'] < 0.5:
            recommendations.append("Consider increasing cache TTL or size to improve hit rate")
        
        if stats['avg_calculation_time_ms'] > 10:
            recommendations.append("Risk calculations taking too long, consider further vectorization")
        
        if stats['avg_validation_time_ms'] > 5:
            recommendations.append("Trade validation taking too long, consider caching optimization")
        
        if stats['risk_cache_stats']['utilization'] > 0.9:
            recommendations.append("Risk cache near capacity, consider increasing size")
        
        if stats['validation_cache_stats']['utilization'] > 0.9:
            recommendations.append("Validation cache near capacity, consider increasing size")
        
        return recommendations


def create_optimized_risk_manager(config_path: str, performance_optimizer: PerformanceOptimizer) -> OptimizedRiskManager:
    """Factory function to create optimized risk manager."""
    return OptimizedRiskManager(config_path, enable_monitoring=True, performance_optimizer=performance_optimizer)