"""
Feature Engineering Pipeline

This module implements technical indicator calculations, volatility regime detection,
and generates the 9-dimensional state vector for the PPO agent.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import ta
from scipy import stats

from .weex_fetcher import MarketData
from .data_validator import DataValidator, ValidationResult, ValidationSeverity

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """9-dimensional state vector for the PPO agent"""
    returns_5m: float
    returns_15m: float
    rsi_14: float
    macd_histogram: float
    atr_percentage: float
    volume_zscore: float
    orderbook_imbalance: float
    funding_rate: float
    volatility_regime: int  # 0=range-bound, 1=trending
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([
            self.returns_5m,
            self.returns_15m,
            self.rsi_14,
            self.macd_histogram,
            self.atr_percentage,
            self.volume_zscore,
            self.orderbook_imbalance,
            self.funding_rate,
            self.volatility_regime
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        return {
            'returns_5m': self.returns_5m,
            'returns_15m': self.returns_15m,
            'rsi_14': self.rsi_14,
            'macd_histogram': self.macd_histogram,
            'atr_percentage': self.atr_percentage,
            'volume_zscore': self.volume_zscore,
            'orderbook_imbalance': self.orderbook_imbalance,
            'funding_rate': self.funding_rate,
            'volatility_regime': self.volatility_regime
        }


class TechnicalIndicators:
    """Technical indicator calculations with standard parameters"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series, periods: List[int] = [5, 15]) -> Dict[str, float]:
        """Calculate returns for multiple timeframes"""
        if len(prices) < max(periods):
            return {f'returns_{p}m': 0.0 for p in periods}
        
        returns = {}
        for period in periods:
            if len(prices) >= period:
                current_price = prices.iloc[-1]
                past_price = prices.iloc[-period]
                if past_price != 0:
                    returns[f'returns_{period}m'] = (current_price - past_price) / past_price
                else:
                    returns[f'returns_{period}m'] = 0.0
            else:
                returns[f'returns_{period}m'] = 0.0
        
        return returns
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        try:
            rsi = ta.momentum.RSIIndicator(close=prices, window=period).rsi()
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return 50.0
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow + signal:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        try:
            macd_indicator = ta.trend.MACD(close=prices, window_fast=fast, window_slow=slow, window_sign=signal)
            macd_line = macd_indicator.macd()
            signal_line = macd_indicator.macd_signal()
            histogram = macd_indicator.macd_diff()
            
            return {
                'macd': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
                'signal': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
                'histogram': float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
            }
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate ATR (Average True Range)"""
        if len(close) < period + 1:
            return 0.0
        
        try:
            atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()
            current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
            current_price = float(close.iloc[-1])
            
            # Return ATR as percentage of current price
            return (current_atr / current_price) * 100 if current_price != 0 else 0.0
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_volume_zscore(volumes: pd.Series, window: int = 20) -> float:
        """Calculate volume z-score for anomaly detection"""
        if len(volumes) < window:
            return 0.0
        
        try:
            recent_volumes = volumes.tail(window)
            current_volume = volumes.iloc[-1]
            
            mean_volume = recent_volumes.mean()
            std_volume = recent_volumes.std()
            
            if std_volume == 0:
                return 0.0
            
            zscore = (current_volume - mean_volume) / std_volume
            return float(zscore)
        except Exception as e:
            logger.warning(f"Volume z-score calculation failed: {e}")
            return 0.0


class VolatilityRegimeDetector:
    """Volatility regime detection using moving average slopes and ATR ratios"""
    
    def __init__(self, ma_period: int = 20, atr_period: int = 14, slope_threshold: float = 0.001):
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.slope_threshold = slope_threshold
    
    def detect_regime(self, prices: pd.Series, high: pd.Series, low: pd.Series) -> int:
        """
        Detect volatility regime: 0 = range-bound, 1 = trending
        
        Uses combination of:
        - Moving average slope
        - ATR trend
        - Price momentum
        """
        if len(prices) < max(self.ma_period, self.atr_period) + 5:
            return 0  # Default to range-bound
        
        try:
            # Calculate moving average and its slope
            ma = prices.rolling(window=self.ma_period).mean()
            ma_slope = self._calculate_slope(ma.tail(5))
            
            # Calculate ATR trend
            atr = ta.volatility.AverageTrueRange(
                high=high, low=low, close=prices, window=self.atr_period
            ).average_true_range()
            atr_trend = self._calculate_slope(atr.tail(5))
            
            # Calculate price momentum
            price_momentum = abs(prices.iloc[-1] - prices.iloc[-self.ma_period]) / prices.iloc[-self.ma_period]
            
            # Regime classification logic
            trending_signals = 0
            
            # Strong MA slope indicates trend
            if abs(ma_slope) > self.slope_threshold:
                trending_signals += 1
            
            # Increasing ATR indicates trending market
            if atr_trend > 0:
                trending_signals += 1
            
            # Strong price momentum indicates trend
            if price_momentum > 0.02:  # 2% momentum threshold
                trending_signals += 1
            
            # Require at least 2 out of 3 signals for trending regime
            return 1 if trending_signals >= 2 else 0
            
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return 0
    
    def _calculate_slope(self, series: pd.Series) -> float:
        """Calculate slope of a time series using linear regression"""
        if len(series) < 2:
            return 0.0
        
        try:
            x = np.arange(len(series))
            y = series.values
            
            # Remove NaN values
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                return 0.0
            
            slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
            return float(slope)
        except Exception as e:
            logger.warning(f"Slope calculation failed: {e}")
            return 0.0


class OrderbookAnalyzer:
    """Orderbook analysis for imbalance calculations"""
    
    @staticmethod
    def calculate_imbalance(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], 
                          depth: int = 10) -> float:
        """
        Calculate orderbook imbalance ratio
        
        Returns:
            float: Imbalance ratio [-1, 1] where:
                   -1 = heavily ask-sided (bearish)
                    0 = balanced
                   +1 = heavily bid-sided (bullish)
        """
        if not bids or not asks:
            return 0.0
        
        try:
            # Take top N levels for analysis
            top_bids = bids[:min(depth, len(bids))]
            top_asks = asks[:min(depth, len(asks))]
            
            # Calculate total volume on each side
            bid_volume = sum(size for _, size in top_bids)
            ask_volume = sum(size for _, size in top_asks)
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return 0.0
            
            # Calculate imbalance ratio
            imbalance = (bid_volume - ask_volume) / total_volume
            return float(np.clip(imbalance, -1.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Orderbook imbalance calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_spread(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> float:
        """Calculate bid-ask spread as percentage of mid price"""
        if not bids or not asks:
            return 0.0
        
        try:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            return (spread / mid_price) * 100 if mid_price != 0 else 0.0
        except Exception as e:
            logger.warning(f"Spread calculation failed: {e}")
            return 0.0


class FeatureEngine:
    """
    Main feature engineering pipeline that processes raw market data
    into structured features for the PPO agent
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.technical_indicators = TechnicalIndicators()
        self.regime_detector = VolatilityRegimeDetector()
        self.orderbook_analyzer = OrderbookAnalyzer()
        
        # Historical data storage for calculations
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.max_history_length = self.config.get('max_history_length', 200)
        
        # Initialize data validator for feature validation
        self.data_validator = DataValidator(self.config.get('validation', {}))
    
    def update_price_history(self, pair: str, market_data: MarketData):
        """Update historical price data for technical indicator calculations"""
        new_row = pd.DataFrame({
            'timestamp': [market_data.timestamp],
            'open': [market_data.open],
            'high': [market_data.high],
            'low': [market_data.low],
            'close': [market_data.close],
            'volume': [market_data.volume]
        })
        
        if pair not in self.price_history:
            self.price_history[pair] = new_row
        else:
            self.price_history[pair] = pd.concat([self.price_history[pair], new_row], ignore_index=True)
            
            # Keep only recent history to manage memory
            if len(self.price_history[pair]) > self.max_history_length:
                self.price_history[pair] = self.price_history[pair].tail(self.max_history_length)
    
    def process_data(self, market_data: MarketData) -> FeatureVector:
        """
        Process raw market data into 9-dimensional feature vector
        
        Args:
            market_data: Raw market data from WEEX API
            
        Returns:
            FeatureVector: 9-dimensional state vector for PPO agent
        """
        try:
            # Update price history
            self.update_price_history(market_data.pair, market_data)
            
            # Get historical data for calculations
            history = self.price_history.get(market_data.pair)
            if history is None or len(history) < 2:
                logger.warning(f"Insufficient history for {market_data.pair}, returning zero vector")
                return self._get_zero_vector()
            
            # Calculate returns for 5m and 15m periods
            returns = self.technical_indicators.calculate_returns(history['close'], [5, 15])
            
            # Calculate technical indicators
            rsi = self.technical_indicators.calculate_rsi(history['close'])
            macd = self.technical_indicators.calculate_macd(history['close'])
            atr_pct = self.technical_indicators.calculate_atr(
                history['high'], history['low'], history['close']
            )
            
            # Calculate volume z-score
            volume_zscore = self.technical_indicators.calculate_volume_zscore(history['volume'])
            
            # Calculate orderbook imbalance
            orderbook_imbalance = self.orderbook_analyzer.calculate_imbalance(
                market_data.orderbook_bids, market_data.orderbook_asks
            )
            
            # Detect volatility regime
            volatility_regime = self.regime_detector.detect_regime(
                history['close'], history['high'], history['low']
            )
            
            # Handle funding rate (may be None)
            funding_rate = market_data.funding_rate if market_data.funding_rate is not None else 0.0
            
            # Create feature vector
            feature_vector = FeatureVector(
                returns_5m=returns.get('returns_5m', 0.0),
                returns_15m=returns.get('returns_15m', 0.0),
                rsi_14=rsi / 100.0,  # Normalize RSI to [0, 1]
                macd_histogram=np.tanh(macd['histogram'] * 1000),  # Normalize MACD histogram
                atr_percentage=np.clip(atr_pct / 10.0, 0, 1),  # Normalize ATR percentage
                volume_zscore=np.tanh(volume_zscore),  # Normalize volume z-score
                orderbook_imbalance=orderbook_imbalance,  # Already normalized [-1, 1]
                funding_rate=np.tanh(funding_rate * 1000),  # Normalize funding rate
                volatility_regime=volatility_regime  # Binary: 0 or 1
            )
            
            # Validate feature vector
            validation_result = self.data_validator.validate_feature_vector(feature_vector, market_data.pair)
            
            if validation_result.is_valid:
                logger.debug(f"Feature validation passed for {market_data.pair}")
                return feature_vector
            else:
                logger.warning(f"Feature validation failed for {market_data.pair}: {validation_result.issues}")
                
                # Try to get fallback features
                fallback_features = self.data_validator.get_fallback_features(market_data.pair)
                if fallback_features:
                    logger.info(f"Using fallback features for {market_data.pair}")
                    return fallback_features
                else:
                    logger.warning("No fallback features available, returning zero vector")
                    return self._get_zero_vector()
                
        except Exception as e:
            logger.error(f"Feature processing failed for {market_data.pair}: {e}")
            return self._get_zero_vector()
    
    def _validate_features(self, features: FeatureVector) -> bool:
        """Validate feature vector for NaN/inf values and reasonable ranges"""
        try:
            feature_array = features.to_array()
            
            # Check for NaN or infinite values
            if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
                logger.warning("Feature vector contains NaN or infinite values")
                return False
            
            # Check reasonable ranges for normalized features
            if not (-10 <= features.returns_5m <= 10):  # ±1000% returns seem unreasonable
                logger.warning(f"Unreasonable 5m returns: {features.returns_5m}")
                return False
            
            if not (-10 <= features.returns_15m <= 10):
                logger.warning(f"Unreasonable 15m returns: {features.returns_15m}")
                return False
            
            if not (0 <= features.rsi_14 <= 1):
                logger.warning(f"RSI out of range: {features.rsi_14}")
                return False
            
            if features.volatility_regime not in [0, 1]:
                logger.warning(f"Invalid volatility regime: {features.volatility_regime}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Feature validation error: {e}")
            return False
    
    def _get_zero_vector(self) -> FeatureVector:
        """Return a zero-initialized feature vector for error cases"""
        return FeatureVector(
            returns_5m=0.0,
            returns_15m=0.0,
            rsi_14=0.5,  # Neutral RSI
            macd_histogram=0.0,
            atr_percentage=0.0,
            volume_zscore=0.0,
            orderbook_imbalance=0.0,
            funding_rate=0.0,
            volatility_regime=0  # Default to range-bound
        )
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for logging and analysis"""
        return [
            'returns_5m',
            'returns_15m', 
            'rsi_14',
            'macd_histogram',
            'atr_percentage',
            'volume_zscore',
            'orderbook_imbalance',
            'funding_rate',
            'volatility_regime'
        ]
    
    def get_feature_statistics(self, pair: str) -> Dict[str, Any]:
        """Get statistics about features for monitoring and debugging"""
        if pair not in self.price_history:
            return {}
        
        history = self.price_history[pair]
        
        return {
            'history_length': len(history),
            'price_range': {
                'min': float(history['close'].min()),
                'max': float(history['close'].max()),
                'current': float(history['close'].iloc[-1])
            },
            'volume_stats': {
                'mean': float(history['volume'].mean()),
                'std': float(history['volume'].std()),
                'current': float(history['volume'].iloc[-1])
            },
            'last_update': history['timestamp'].iloc[-1].isoformat() if len(history) > 0 else None
        }
    
    def get_feature_quality_summary(self, pair: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Get feature quality summary from the validator"""
        return self.data_validator.get_quality_summary(pair, hours)
    
    def save_feature_quality_report(self, filepath: str):
        """Save feature quality report to file"""
        self.data_validator.save_quality_report(filepath)


# Utility functions
def create_feature_engine(config: Optional[Dict[str, Any]] = None) -> FeatureEngine:
    """Factory function to create FeatureEngine with configuration"""
    return FeatureEngine(config)


def batch_process_features(market_data_list: List[MarketData], 
                          feature_engine: FeatureEngine) -> List[FeatureVector]:
    """Process multiple market data points into feature vectors"""
    features = []
    for market_data in market_data_list:
        try:
            feature_vector = feature_engine.process_data(market_data)
            features.append(feature_vector)
        except Exception as e:
            logger.error(f"Failed to process features for {market_data.pair}: {e}")
            features.append(feature_engine._get_zero_vector())
    
    return features