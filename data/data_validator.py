"""
Data Validation and Quality Checks

This module implements comprehensive data validation to ensure feature quality,
detect anomalies, and provide fallback mechanisms for missing or corrupted data.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import os

from .weex_fetcher import MarketData

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of data validation with details about issues found"""
    is_valid: bool
    severity: ValidationSeverity
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, message: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        """Add a validation issue"""
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.issues.append(message)
            if severity == ValidationSeverity.CRITICAL:
                self.is_valid = False
                self.severity = ValidationSeverity.CRITICAL
            elif self.severity != ValidationSeverity.CRITICAL:
                self.is_valid = False
                self.severity = ValidationSeverity.ERROR
        else:
            self.warnings.append(message)
            if self.severity == ValidationSeverity.INFO:
                self.severity = ValidationSeverity.WARNING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'is_valid': self.is_valid,
            'severity': self.severity.value,
            'issues': self.issues,
            'warnings': self.warnings,
            'metadata': self.metadata
        }


@dataclass
class DataQualityMetrics:
    """Metrics for tracking data quality over time"""
    timestamp: datetime
    pair: str
    completeness_score: float  # [0, 1] - percentage of required fields present
    consistency_score: float   # [0, 1] - OHLC consistency, price reasonableness
    freshness_score: float     # [0, 1] - how recent the data is
    accuracy_score: float      # [0, 1] - based on anomaly detection
    overall_score: float       # [0, 1] - weighted average of all scores
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and storage"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'pair': self.pair,
            'completeness_score': self.completeness_score,
            'consistency_score': self.consistency_score,
            'freshness_score': self.freshness_score,
            'accuracy_score': self.accuracy_score,
            'overall_score': self.overall_score
        }


class AnomalyDetector:
    """Statistical anomaly detection for market data"""
    
    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
    
    def update_history(self, pair: str, price: float, volume: float):
        """Update historical data for anomaly detection"""
        if pair not in self.price_history:
            self.price_history[pair] = []
            self.volume_history[pair] = []
        
        self.price_history[pair].append(price)
        self.volume_history[pair].append(volume)
        
        # Keep only recent history
        if len(self.price_history[pair]) > self.window_size:
            self.price_history[pair] = self.price_history[pair][-self.window_size:]
            self.volume_history[pair] = self.volume_history[pair][-self.window_size:]
    
    def detect_price_anomaly(self, pair: str, current_price: float) -> Tuple[bool, float]:
        """
        Detect price anomalies using z-score analysis
        
        Returns:
            Tuple[bool, float]: (is_anomaly, z_score)
        """
        if pair not in self.price_history or len(self.price_history[pair]) < 10:
            return False, 0.0
        
        try:
            prices = np.array(self.price_history[pair])
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            
            if std_price == 0:
                return False, 0.0
            
            z_score = abs(current_price - mean_price) / std_price
            is_anomaly = z_score > self.z_threshold
            
            return is_anomaly, z_score
        except Exception as e:
            logger.warning(f"Price anomaly detection failed for {pair}: {e}")
            return False, 0.0
    
    def detect_volume_anomaly(self, pair: str, current_volume: float) -> Tuple[bool, float]:
        """
        Detect volume anomalies using z-score analysis
        
        Returns:
            Tuple[bool, float]: (is_anomaly, z_score)
        """
        if pair not in self.volume_history or len(self.volume_history[pair]) < 10:
            return False, 0.0
        
        try:
            volumes = np.array(self.volume_history[pair])
            mean_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            
            if std_volume == 0:
                return False, 0.0
            
            z_score = abs(current_volume - mean_volume) / std_volume
            is_anomaly = z_score > self.z_threshold
            
            return is_anomaly, z_score
        except Exception as e:
            logger.warning(f"Volume anomaly detection failed for {pair}: {e}")
            return False, 0.0
    
    def detect_return_anomaly(self, pair: str, current_return: float) -> Tuple[bool, float]:
        """
        Detect abnormal returns that might indicate data corruption
        
        Returns:
            Tuple[bool, float]: (is_anomaly, return_magnitude)
        """
        # Define reasonable return thresholds (e.g., ±50% in a single period is suspicious)
        max_reasonable_return = 0.5  # 50%
        
        return_magnitude = abs(current_return)
        is_anomaly = return_magnitude > max_reasonable_return
        
        return is_anomaly, return_magnitude


class DataValidator:
    """
    Comprehensive data validator for market data and features
    with quality scoring and anomaly detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.anomaly_detector = AnomalyDetector(
            window_size=self.config.get('anomaly_window_size', 100),
            z_threshold=self.config.get('anomaly_z_threshold', 3.0)
        )
        
        # Quality thresholds
        self.min_freshness_minutes = self.config.get('min_freshness_minutes', 5)
        self.min_overall_quality_score = self.config.get('min_overall_quality_score', 0.7)
        
        # Fallback data storage
        self.fallback_data: Dict[str, MarketData] = {}
        self.fallback_features: Dict[str, Any] = {}  # Generic feature storage
        
        # Quality metrics history
        self.quality_history: List[DataQualityMetrics] = []
        self.max_quality_history = self.config.get('max_quality_history', 1000)
    
    def validate_market_data(self, data: MarketData) -> ValidationResult:
        """
        Comprehensive validation of market data
        
        Args:
            data: MarketData object to validate
            
        Returns:
            ValidationResult: Detailed validation results
        """
        result = ValidationResult(is_valid=True, severity=ValidationSeverity.INFO)
        
        try:
            # 1. Completeness validation
            self._validate_completeness(data, result)
            
            # 2. Data type and range validation
            self._validate_data_types_and_ranges(data, result)
            
            # 3. OHLC consistency validation
            self._validate_ohlc_consistency(data, result)
            
            # 4. Freshness validation
            self._validate_freshness(data, result)
            
            # 5. Anomaly detection
            self._validate_anomalies(data, result)
            
            # 6. Orderbook validation
            self._validate_orderbook(data, result)
            
            # 7. Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(data, result)
            result.metadata['quality_metrics'] = quality_metrics.to_dict()
            
            # Store quality metrics
            self.quality_history.append(quality_metrics)
            if len(self.quality_history) > self.max_quality_history:
                self.quality_history = self.quality_history[-self.max_quality_history:]
            
            # Update anomaly detector history
            self.anomaly_detector.update_history(data.pair, data.close, data.volume)
            
            # Store as fallback if quality is good
            if quality_metrics.overall_score >= self.min_overall_quality_score:
                self.fallback_data[data.pair] = data
            
        except Exception as e:
            result.add_issue(f"Validation process failed: {e}", ValidationSeverity.CRITICAL)
            logger.error(f"Market data validation failed: {e}")
        
        return result
    
    def validate_feature_vector(self, features: Any, pair: str) -> ValidationResult:
        """
        Validate feature vector for quality and reasonableness
        
        Args:
            features: Feature vector object with to_array() method
            pair: Trading pair name
            
        Returns:
            ValidationResult: Detailed validation results
        """
        result = ValidationResult(is_valid=True, severity=ValidationSeverity.INFO)
        
        try:
            # Convert to array for numerical checks
            if hasattr(features, 'to_array'):
                feature_array = features.to_array()
            else:
                # Assume it's already an array-like object
                feature_array = np.array(features)
            
            # 1. Check for NaN/inf values
            if np.any(np.isnan(feature_array)):
                result.add_issue("Feature vector contains NaN values", ValidationSeverity.CRITICAL)
            
            if np.any(np.isinf(feature_array)):
                result.add_issue("Feature vector contains infinite values", ValidationSeverity.CRITICAL)
            
            # 2. Range validation for specific features (if features object has attributes)
            if hasattr(features, 'rsi_14') and hasattr(features, 'volatility_regime'):
                self._validate_feature_ranges(features, result)
                
                # 3. Consistency checks
                self._validate_feature_consistency(features, result)
                
                # 4. Anomaly detection for returns
                self._validate_feature_anomalies(features, pair, result)
            else:
                # Generic validation for array-like features
                if len(feature_array) >= 9:  # Expected 9-dimensional vector
                    # Check for extreme values
                    if np.any(np.abs(feature_array) > 100):  # Very large values
                        result.add_issue("Feature vector contains extremely large values", ValidationSeverity.WARNING)
            
            # Store as fallback if valid
            if result.is_valid:
                self.fallback_features[pair] = features
            
        except Exception as e:
            result.add_issue(f"Feature validation failed: {e}", ValidationSeverity.CRITICAL)
            logger.error(f"Feature vector validation failed: {e}")
        
        return result
    
    def get_fallback_market_data(self, pair: str) -> Optional[MarketData]:
        """
        Get fallback market data for a trading pair
        
        Args:
            pair: Trading pair name
            
        Returns:
            Optional[MarketData]: Fallback data if available
        """
        fallback = self.fallback_data.get(pair)
        if fallback:
            # Check if fallback data is not too old
            age = datetime.now() - fallback.timestamp
            if age.total_seconds() / 60 <= self.config.get('max_fallback_age_minutes', 30):
                logger.info(f"Using fallback market data for {pair} (age: {age})")
                return fallback
            else:
                logger.warning(f"Fallback data for {pair} is too old ({age}), discarding")
                del self.fallback_data[pair]
        
        return None
    
    def get_fallback_features(self, pair: str) -> Optional[Any]:
        """
        Get fallback feature vector for a trading pair
        
        Args:
            pair: Trading pair name
            
        Returns:
            Optional[Any]: Fallback features if available
        """
        return self.fallback_features.get(pair)
    
    def get_quality_summary(self, pair: Optional[str] = None, 
                           hours: int = 24) -> Dict[str, Any]:
        """
        Get data quality summary for monitoring
        
        Args:
            pair: Specific trading pair (None for all pairs)
            hours: Number of hours to look back
            
        Returns:
            Dict[str, Any]: Quality summary statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter quality history
        recent_metrics = [
            m for m in self.quality_history 
            if m.timestamp >= cutoff_time and (pair is None or m.pair == pair)
        ]
        
        if not recent_metrics:
            return {'message': 'No quality data available for the specified period'}
        
        # Calculate summary statistics
        pairs = list(set(m.pair for m in recent_metrics))
        summary = {
            'period_hours': hours,
            'pairs': pairs,
            'total_validations': len(recent_metrics),
            'average_scores': {},
            'min_scores': {},
            'quality_trends': {}
        }
        
        for pair_name in pairs:
            pair_metrics = [m for m in recent_metrics if m.pair == pair_name]
            
            if pair_metrics:
                scores = [m.overall_score for m in pair_metrics]
                summary['average_scores'][pair_name] = np.mean(scores)
                summary['min_scores'][pair_name] = np.min(scores)
                
                # Simple trend calculation (last 10 vs first 10)
                if len(pair_metrics) >= 20:
                    recent_scores = [m.overall_score for m in pair_metrics[-10:]]
                    older_scores = [m.overall_score for m in pair_metrics[:10]]
                    trend = np.mean(recent_scores) - np.mean(older_scores)
                    summary['quality_trends'][pair_name] = 'improving' if trend > 0.05 else 'declining' if trend < -0.05 else 'stable'
        
        return summary
    
    def _validate_completeness(self, data: MarketData, result: ValidationResult):
        """Validate data completeness"""
        required_fields = ['pair', 'timestamp', 'close', 'volume']
        missing_fields = []
        
        for field in required_fields:
            if not hasattr(data, field) or getattr(data, field) is None:
                missing_fields.append(field)
        
        if missing_fields:
            result.add_issue(f"Missing required fields: {missing_fields}", ValidationSeverity.CRITICAL)
        
        # Check OHLC completeness
        ohlc_fields = ['open', 'high', 'low', 'close']
        missing_ohlc = [f for f in ohlc_fields if not hasattr(data, f) or getattr(data, f) is None]
        
        if missing_ohlc:
            result.add_issue(f"Missing OHLC fields: {missing_ohlc}", ValidationSeverity.ERROR)
    
    def _validate_data_types_and_ranges(self, data: MarketData, result: ValidationResult):
        """Validate data types and reasonable ranges"""
        # Price validation
        price_fields = ['open', 'high', 'low', 'close']
        for field in price_fields:
            if hasattr(data, field) and getattr(data, field) is not None:
                value = getattr(data, field)
                if not isinstance(value, (int, float)) or value <= 0:
                    result.add_issue(f"Invalid {field} price: {value}", ValidationSeverity.ERROR)
                elif value > 1000000:  # Sanity check for extremely high prices
                    result.add_issue(f"Suspiciously high {field} price: {value}", ValidationSeverity.WARNING)
        
        # Volume validation
        if hasattr(data, 'volume') and data.volume is not None:
            if not isinstance(data.volume, (int, float)) or data.volume < 0:
                result.add_issue(f"Invalid volume: {data.volume}", ValidationSeverity.ERROR)
        
        # Funding rate validation (if present)
        if hasattr(data, 'funding_rate') and data.funding_rate is not None:
            if not isinstance(data.funding_rate, (int, float)):
                result.add_issue(f"Invalid funding rate type: {type(data.funding_rate)}", ValidationSeverity.ERROR)
            elif abs(data.funding_rate) > 0.01:  # 1% funding rate is extremely high
                result.add_issue(f"Extreme funding rate: {data.funding_rate}", ValidationSeverity.WARNING)
    
    def _validate_ohlc_consistency(self, data: MarketData, result: ValidationResult):
        """Validate OHLC price consistency"""
        try:
            if all(hasattr(data, f) and getattr(data, f) is not None for f in ['open', 'high', 'low', 'close']):
                o, h, l, c = data.open, data.high, data.low, data.close
                
                # High should be >= all other prices
                if not (h >= o and h >= l and h >= c):
                    result.add_issue(f"High price ({h}) is not the highest among OHLC", ValidationSeverity.ERROR)
                
                # Low should be <= all other prices
                if not (l <= o and l <= h and l <= c):
                    result.add_issue(f"Low price ({l}) is not the lowest among OHLC", ValidationSeverity.ERROR)
                
                # Check for zero spread (suspicious)
                if h == l and h > 0:
                    result.add_issue("Zero price spread detected (high == low)", ValidationSeverity.WARNING)
        except Exception as e:
            result.add_issue(f"OHLC consistency check failed: {e}", ValidationSeverity.WARNING)
    
    def _validate_freshness(self, data: MarketData, result: ValidationResult):
        """Validate data freshness"""
        if hasattr(data, 'timestamp') and data.timestamp:
            age = datetime.now() - data.timestamp
            age_minutes = age.total_seconds() / 60
            
            if age_minutes > self.min_freshness_minutes:
                severity = ValidationSeverity.WARNING if age_minutes < 15 else ValidationSeverity.ERROR
                result.add_issue(f"Data is stale ({age_minutes:.1f} minutes old)", severity)
    
    def _validate_anomalies(self, data: MarketData, result: ValidationResult):
        """Validate for statistical anomalies"""
        try:
            # Price anomaly detection
            is_price_anomaly, price_z_score = self.anomaly_detector.detect_price_anomaly(data.pair, data.close)
            if is_price_anomaly:
                result.add_issue(f"Price anomaly detected (z-score: {price_z_score:.2f})", ValidationSeverity.WARNING)
            
            # Volume anomaly detection
            is_volume_anomaly, volume_z_score = self.anomaly_detector.detect_volume_anomaly(data.pair, data.volume)
            if is_volume_anomaly:
                result.add_issue(f"Volume anomaly detected (z-score: {volume_z_score:.2f})", ValidationSeverity.WARNING)
            
            result.metadata['anomaly_scores'] = {
                'price_z_score': price_z_score,
                'volume_z_score': volume_z_score
            }
        except Exception as e:
            result.add_issue(f"Anomaly detection failed: {e}", ValidationSeverity.WARNING)
    
    def _validate_orderbook(self, data: MarketData, result: ValidationResult):
        """Validate orderbook data"""
        if hasattr(data, 'orderbook_bids') and hasattr(data, 'orderbook_asks'):
            # Check if orderbook data exists
            if not data.orderbook_bids and not data.orderbook_asks:
                result.add_issue("Empty orderbook data", ValidationSeverity.WARNING)
                return
            
            # Validate bid prices are descending
            if data.orderbook_bids:
                bid_prices = [bid[0] for bid in data.orderbook_bids]
                if bid_prices != sorted(bid_prices, reverse=True):
                    result.add_issue("Bid prices are not in descending order", ValidationSeverity.ERROR)
            
            # Validate ask prices are ascending
            if data.orderbook_asks:
                ask_prices = [ask[0] for ask in data.orderbook_asks]
                if ask_prices != sorted(ask_prices):
                    result.add_issue("Ask prices are not in ascending order", ValidationSeverity.ERROR)
            
            # Validate spread (best ask > best bid)
            if data.orderbook_bids and data.orderbook_asks:
                best_bid = data.orderbook_bids[0][0]
                best_ask = data.orderbook_asks[0][0]
                if best_ask <= best_bid:
                    result.add_issue(f"Invalid spread: best_ask ({best_ask}) <= best_bid ({best_bid})", ValidationSeverity.ERROR)
    
    def _validate_feature_ranges(self, features: Any, result: ValidationResult):
        """Validate feature value ranges"""
        # RSI should be normalized to [0, 1]
        if hasattr(features, 'rsi_14') and not (0 <= features.rsi_14 <= 1):
            result.add_issue(f"RSI out of normalized range [0,1]: {features.rsi_14}", ValidationSeverity.ERROR)
        
        # Volatility regime should be 0 or 1
        if hasattr(features, 'volatility_regime') and features.volatility_regime not in [0, 1]:
            result.add_issue(f"Invalid volatility regime: {features.volatility_regime}", ValidationSeverity.ERROR)
        
        # Orderbook imbalance should be in [-1, 1]
        if hasattr(features, 'orderbook_imbalance') and not (-1 <= features.orderbook_imbalance <= 1):
            result.add_issue(f"Orderbook imbalance out of range [-1,1]: {features.orderbook_imbalance}", ValidationSeverity.ERROR)
        
        # Check for extreme returns (might indicate data corruption)
        if hasattr(features, 'returns_5m') and hasattr(features, 'returns_15m'):
            for return_field, return_value in [('returns_5m', features.returns_5m), ('returns_15m', features.returns_15m)]:
                is_anomaly, magnitude = self.anomaly_detector.detect_return_anomaly('', return_value)
                if is_anomaly:
                    result.add_issue(f"Extreme {return_field}: {return_value} ({magnitude:.2%})", ValidationSeverity.WARNING)
    
    def _validate_feature_consistency(self, features: Any, result: ValidationResult):
        """Validate internal consistency of features"""
        # If both 5m and 15m returns are available, 15m should generally be larger in magnitude
        # (this is a soft check, not always true)
        if (hasattr(features, 'returns_5m') and hasattr(features, 'returns_15m') and
            abs(features.returns_5m) > 0.1 and abs(features.returns_15m) > 0.1):
            if abs(features.returns_5m) > abs(features.returns_15m) * 2:
                result.add_issue("5m returns significantly larger than 15m returns", ValidationSeverity.WARNING)
    
    def _validate_feature_anomalies(self, features: Any, pair: str, result: ValidationResult):
        """Detect anomalies in feature values"""
        # Check for extreme z-scores in normalized features
        extreme_threshold = 5.0
        
        if hasattr(features, 'volume_zscore') and abs(features.volume_zscore) > extreme_threshold:
            result.add_issue(f"Extreme volume z-score: {features.volume_zscore}", ValidationSeverity.WARNING)
        
        # Check for extreme MACD histogram values
        if hasattr(features, 'macd_histogram') and abs(features.macd_histogram) > 0.9:  # Since it's tanh-normalized, values near ±1 are extreme
            result.add_issue(f"Extreme MACD histogram: {features.macd_histogram}", ValidationSeverity.WARNING)
    
    def _calculate_quality_metrics(self, data: MarketData, validation_result: ValidationResult) -> DataQualityMetrics:
        """Calculate comprehensive quality metrics"""
        # Completeness score (0-1)
        required_fields = ['pair', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        present_fields = sum(1 for field in required_fields if hasattr(data, field) and getattr(data, field) is not None)
        completeness_score = present_fields / len(required_fields)
        
        # Consistency score (0-1) - based on validation issues
        error_count = len([issue for issue in validation_result.issues if 'consistency' in issue.lower() or 'ohlc' in issue.lower()])
        consistency_score = max(0, 1 - (error_count * 0.2))  # Each error reduces score by 0.2
        
        # Freshness score (0-1)
        if hasattr(data, 'timestamp') and data.timestamp:
            age_minutes = (datetime.now() - data.timestamp).total_seconds() / 60
            freshness_score = max(0, 1 - (age_minutes / 60))  # Linear decay over 1 hour
        else:
            freshness_score = 0
        
        # Accuracy score (0-1) - based on anomaly detection
        anomaly_count = len([issue for issue in validation_result.issues + validation_result.warnings if 'anomaly' in issue.lower()])
        accuracy_score = max(0, 1 - (anomaly_count * 0.1))  # Each anomaly reduces score by 0.1
        
        # Overall score (weighted average)
        weights = {'completeness': 0.3, 'consistency': 0.3, 'freshness': 0.2, 'accuracy': 0.2}
        overall_score = (
            weights['completeness'] * completeness_score +
            weights['consistency'] * consistency_score +
            weights['freshness'] * freshness_score +
            weights['accuracy'] * accuracy_score
        )
        
        return DataQualityMetrics(
            timestamp=datetime.now(),
            pair=data.pair,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            freshness_score=freshness_score,
            accuracy_score=accuracy_score,
            overall_score=overall_score
        )
    
    def save_quality_report(self, filepath: str):
        """Save quality metrics to file for analysis"""
        try:
            report_data = {
                'generated_at': datetime.now().isoformat(),
                'total_validations': len(self.quality_history),
                'quality_metrics': [metrics.to_dict() for metrics in self.quality_history],
                'summary': self.get_quality_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Quality report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save quality report: {e}")


# Utility functions
def create_data_validator(config: Optional[Dict[str, Any]] = None) -> DataValidator:
    """Factory function to create DataValidator with configuration"""
    return DataValidator(config)


def validate_data_batch(data_list: List[MarketData], 
                       validator: DataValidator) -> List[ValidationResult]:
    """Validate a batch of market data"""
    results = []
    for data in data_list:
        try:
            result = validator.validate_market_data(data)
            results.append(result)
        except Exception as e:
            error_result = ValidationResult(is_valid=False, severity=ValidationSeverity.CRITICAL)
            error_result.add_issue(f"Batch validation failed: {e}", ValidationSeverity.CRITICAL)
            results.append(error_result)
            logger.error(f"Batch validation failed for {data.pair if hasattr(data, 'pair') else 'unknown'}: {e}")
    
    return results