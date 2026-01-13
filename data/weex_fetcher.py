"""
WEEX API Data Fetcher

This module implements the WeexDataFetcher class for real-time price data fetching
for BTC/ETH pairs with error handling, retry logic, and rate limiting.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass, asdict
import aiohttp
import pandas as pd
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Market data structure for OHLCV and orderbook data"""
    timestamp: datetime
    pair: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    orderbook_bids: List[Tuple[float, float]]  # [(price, size), ...]
    orderbook_asks: List[Tuple[float, float]]  # [(price, size), ...]
    funding_rate: Optional[float] = None


class RateLimiter:
    """Rate limiter to control API request frequency"""
    
    def __init__(self, max_requests_per_second: int = 10):
        self.max_requests_per_second = max_requests_per_second
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request"""
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 second
            self.requests = [req_time for req_time in self.requests if now - req_time < 1.0]
            
            if len(self.requests) >= self.max_requests_per_second:
                sleep_time = 1.0 - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    return await self.acquire()
            
            self.requests.append(now)


class DataCache:
    """Simple in-memory cache for market data"""
    
    def __init__(self, max_age_minutes: int = 60, max_size_mb: int = 100):
        self.max_age = timedelta(minutes=max_age_minutes)
        self.max_size_mb = max_size_mb
        self.cache: Dict[str, Tuple[datetime, Any]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data if not expired"""
        if key in self.cache:
            timestamp, data = self.cache[key]
            if datetime.now() - timestamp < self.max_age:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: Any):
        """Store data in cache"""
        self.cache[key] = (datetime.now(), data)
        self._cleanup_if_needed()
    
    def _cleanup_if_needed(self):
        """Remove old entries if cache is too large"""
        if len(self.cache) > 1000:  # Simple size check
            # Remove oldest 20% of entries
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1][0])
            items_to_remove = len(sorted_items) // 5
            for key, _ in sorted_items[:items_to_remove]:
                del self.cache[key]


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying failed API calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator


class WeexDataFetcher:
    """
    WEEX API data fetcher with real-time price data fetching for BTC/ETH pairs.
    Includes error handling, retry logic, rate limiting, caching, and data validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get('api_base_url', 'https://api.weex.com')
        self.timeout = config.get('timeout_seconds', 30)
        self.rate_limiter = RateLimiter(config.get('rate_limit_requests_per_second', 10))
        self.cache = DataCache(
            max_age_minutes=config.get('cache_duration_minutes', 60),
            max_size_mb=config.get('max_cache_size_mb', 100)
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.supported_pairs = ['BTCUSDT', 'ETHUSDT']
        
        # Initialize data validator
        from .data_validator import DataValidator
        self.data_validator = DataValidator(config.get('validation', {}))
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make HTTP request to WEEX API with rate limiting and error handling"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        await self.rate_limiter.acquire()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"API request failed: {error_text}"
                    )
        except asyncio.TimeoutError:
            raise Exception(f"Request timeout for {url}")
        except aiohttp.ClientError as e:
            raise Exception(f"Client error for {url}: {e}")
    
    async def get_latest_price(self, pair: str) -> Dict[str, float]:
        """Get latest price data for a trading pair"""
        if pair not in self.supported_pairs:
            raise ValueError(f"Unsupported trading pair: {pair}")
        
        cache_key = f"price_{pair}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # WEEX API endpoint for ticker data (mock structure)
            data = await self._make_request(f"/v1/ticker/24hr", {"symbol": pair})
            
            price_data = {
                'open': float(data.get('openPrice', 0)),
                'high': float(data.get('highPrice', 0)),
                'low': float(data.get('lowPrice', 0)),
                'close': float(data.get('lastPrice', 0)),
                'volume': float(data.get('volume', 0)),
                'timestamp': datetime.now()
            }
            
            self.cache.set(cache_key, price_data)
            return price_data
            
        except Exception as e:
            logger.error(f"Failed to fetch price data for {pair}: {e}")
            # Try to return cached data even if expired
            cached_data = self.cache.cache.get(cache_key)
            if cached_data:
                logger.warning(f"Using expired cached data for {pair}")
                return cached_data[1]
            raise
    
    async def get_kline_data(self, pair: str, interval: str = '5m', limit: int = 100) -> pd.DataFrame:
        """Get historical kline/candlestick data"""
        if pair not in self.supported_pairs:
            raise ValueError(f"Unsupported trading pair: {pair}")
        
        cache_key = f"kline_{pair}_{interval}_{limit}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            params = {
                'symbol': pair,
                'interval': interval,
                'limit': limit
            }
            
            data = await self._make_request("/v1/klines", params)
            
            # Convert to DataFrame (assuming WEEX returns standard kline format)
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert timestamp and numeric columns
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Keep only essential columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            self.cache.set(cache_key, df)
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch kline data for {pair}: {e}")
            raise
    
    async def get_orderbook(self, pair: str, limit: int = 20) -> Dict[str, List[Tuple[float, float]]]:
        """Get orderbook data (bids and asks)"""
        if pair not in self.supported_pairs:
            raise ValueError(f"Unsupported trading pair: {pair}")
        
        cache_key = f"orderbook_{pair}_{limit}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            params = {'symbol': pair, 'limit': limit}
            data = await self._make_request("/v1/depth", params)
            
            orderbook = {
                'bids': [(float(bid[0]), float(bid[1])) for bid in data.get('bids', [])],
                'asks': [(float(ask[0]), float(ask[1])) for ask in data.get('asks', [])]
            }
            
            self.cache.set(cache_key, orderbook)
            return orderbook
            
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {pair}: {e}")
            raise
    
    async def get_funding_rate(self, pair: str) -> Optional[float]:
        """Get current funding rate for perpetual futures"""
        if pair not in self.supported_pairs:
            raise ValueError(f"Unsupported trading pair: {pair}")
        
        cache_key = f"funding_{pair}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            params = {'symbol': pair}
            data = await self._make_request("/v1/premiumIndex", params)
            
            funding_rate = float(data.get('lastFundingRate', 0))
            self.cache.set(cache_key, funding_rate)
            return funding_rate
            
        except Exception as e:
            logger.error(f"Failed to fetch funding rate for {pair}: {e}")
            return None
    
    async def get_comprehensive_data(self, pair: str) -> MarketData:
        """Get comprehensive market data including price, orderbook, and funding rate"""
        if pair not in self.supported_pairs:
            raise ValueError(f"Unsupported trading pair: {pair}")
        
        try:
            # Fetch all data concurrently
            price_task = self.get_latest_price(pair)
            orderbook_task = self.get_orderbook(pair)
            funding_task = self.get_funding_rate(pair)
            
            price_data, orderbook_data, funding_rate = await asyncio.gather(
                price_task, orderbook_task, funding_task,
                return_exceptions=True
            )
            
            # Handle exceptions in individual tasks
            if isinstance(price_data, Exception):
                logger.error(f"Price data fetch failed: {price_data}")
                raise price_data
            
            if isinstance(orderbook_data, Exception):
                logger.warning(f"Orderbook fetch failed: {orderbook_data}")
                orderbook_data = {'bids': [], 'asks': []}
            
            if isinstance(funding_rate, Exception):
                logger.warning(f"Funding rate fetch failed: {funding_rate}")
                funding_rate = None
            
            return MarketData(
                timestamp=price_data['timestamp'],
                pair=pair,
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                volume=price_data['volume'],
                orderbook_bids=orderbook_data['bids'],
                orderbook_asks=orderbook_data['asks'],
                funding_rate=funding_rate
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch comprehensive data for {pair}: {e}")
            raise
    
    def validate_data(self, data: MarketData) -> bool:
        """Validate market data using comprehensive data validator"""
        try:
            validation_result = self.data_validator.validate_market_data(data)
            
            # Log validation results
            if not validation_result.is_valid:
                logger.warning(f"Data validation failed for {data.pair}: {validation_result.issues}")
                if validation_result.warnings:
                    logger.info(f"Data validation warnings for {data.pair}: {validation_result.warnings}")
            
            return validation_result.is_valid
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False
    
    async def get_latest_data(self, pair: str) -> Optional[MarketData]:
        """Get latest validated market data with comprehensive fallback mechanisms"""
        try:
            # Attempt to get fresh data
            data = await self.get_comprehensive_data(pair)
            
            # Validate the data
            if self.validate_data(data):
                logger.debug(f"Successfully validated fresh data for {pair}")
                return data
            else:
                logger.warning(f"Fresh data validation failed for {pair}, attempting fallback")
                
                # Try to get fallback data from validator
                fallback_data = self.data_validator.get_fallback_market_data(pair)
                if fallback_data:
                    logger.info(f"Using validator fallback data for {pair}")
                    return fallback_data
                
                # Try cached data as last resort
                cache_key = f"validated_data_{pair}"
                cached_data = self.cache.cache.get(cache_key)
                if cached_data:
                    logger.info(f"Using cached validated data for {pair}")
                    return cached_data[1]
                
                logger.error(f"No valid data available for {pair}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get latest data for {pair}: {e}")
            
            # Try fallback mechanisms on exception
            fallback_data = self.data_validator.get_fallback_market_data(pair)
            if fallback_data:
                logger.info(f"Using fallback data after exception for {pair}")
                return fallback_data
            
            return None
    
    def get_data_quality_summary(self, pair: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Get data quality summary from the validator"""
        return self.data_validator.get_quality_summary(pair, hours)
    
    def save_quality_report(self, filepath: str):
        """Save data quality report to file"""
        self.data_validator.save_quality_report(filepath)


# Utility functions for easy usage
async def create_weex_fetcher(config: Dict[str, Any]) -> WeexDataFetcher:
    """Factory function to create and initialize WeexDataFetcher"""
    fetcher = WeexDataFetcher(config)
    return fetcher


async def fetch_multiple_pairs(pairs: List[str], config: Dict[str, Any]) -> Dict[str, Optional[MarketData]]:
    """Fetch data for multiple pairs concurrently"""
    async with WeexDataFetcher(config) as fetcher:
        tasks = [fetcher.get_latest_data(pair) for pair in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            pair: result if not isinstance(result, Exception) else None
            for pair, result in zip(pairs, results)
        }