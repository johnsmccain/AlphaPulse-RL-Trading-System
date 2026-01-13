#!/usr/bin/env python3
"""
Data Validation Example

This example demonstrates how to use the data validation system
to ensure data quality and implement fallback mechanisms.
"""

import asyncio
import sys
import os
from datetime import datetime
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.weex_fetcher import MarketData, WeexDataFetcher
from data.feature_engineering import FeatureEngine
from data.data_validator import DataValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate data validation and fallback mechanisms"""
    logger.info("Data Validation Example")
    logger.info("=" * 50)
    
    # Configuration
    config = {
        'api_base_url': 'https://api.weex.com',
        'timeout_seconds': 30,
        'validation': {
            'min_freshness_minutes': 5,
            'min_overall_quality_score': 0.7,
            'anomaly_z_threshold': 3.0
        }
    }
    
    # Create components
    feature_engine = FeatureEngine(config)
    validator = DataValidator(config.get('validation', {}))
    
    # Example 1: Validate good market data
    logger.info("Example 1: Validating good market data")
    good_data = MarketData(
        timestamp=datetime.now(),
        pair="BTCUSDT",
        open=50000.0,
        high=50500.0,
        low=49500.0,
        close=50200.0,
        volume=1000.0,
        orderbook_bids=[(50100.0, 10.0), (50050.0, 5.0)],
        orderbook_asks=[(50150.0, 8.0), (50200.0, 12.0)],
        funding_rate=0.0001
    )
    
    result = validator.validate_market_data(good_data)
    logger.info(f"Validation result: {result.is_valid}")
    logger.info(f"Quality score: {result.metadata.get('quality_metrics', {}).get('overall_score', 'N/A')}")
    
    # Example 2: Process features with validation
    logger.info("\nExample 2: Processing features with validation")
    features = feature_engine.process_data(good_data)
    logger.info(f"Features processed successfully: {features is not None}")
    if features:
        logger.info(f"RSI: {features.rsi_14:.3f}, Volatility Regime: {features.volatility_regime}")
    
    # Example 3: Demonstrate fallback mechanism
    logger.info("\nExample 3: Demonstrating fallback mechanisms")
    
    # First, store some good data as fallback
    validator.validate_market_data(good_data)
    
    # Now try to get fallback data
    fallback_data = validator.get_fallback_market_data("BTCUSDT")
    logger.info(f"Fallback data available: {fallback_data is not None}")
    
    # Example 4: Quality monitoring
    logger.info("\nExample 4: Quality monitoring")
    
    # Validate several data points to build history
    for i in range(5):
        test_data = MarketData(
            timestamp=datetime.now(),
            pair="BTCUSDT",
            open=50000.0 + i * 10,
            high=50500.0 + i * 10,
            low=49500.0 + i * 10,
            close=50200.0 + i * 10,
            volume=1000.0 + i * 100,
            orderbook_bids=[(50100.0 + i * 10, 10.0)],
            orderbook_asks=[(50150.0 + i * 10, 8.0)],
            funding_rate=0.0001
        )
        validator.validate_market_data(test_data)
    
    # Get quality summary
    summary = validator.get_quality_summary("BTCUSDT", hours=1)
    logger.info(f"Quality summary: {summary}")
    
    logger.info("\nData validation example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())