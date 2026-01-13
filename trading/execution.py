"""
Execution Engine for AlphaPulse-RL Trading System

This module implements the ExecutionEngine class that translates agent actions
into WEEX API orders with proper error handling and retry logic.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import json

from .portfolio import PortfolioState, Position, TradeRecord

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    """Order status enumeration"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class OrderRequest:
    """Order request structure"""
    pair: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    leverage: float = 1.0
    time_in_force: str = "GTC"  # Good Till Canceled


@dataclass
class OrderResponse:
    """Order response structure"""
    order_id: str
    pair: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    filled_quantity: float
    price: Optional[float]
    average_price: Optional[float]
    status: OrderStatus
    timestamp: datetime
    commission: float = 0.0
    commission_asset: str = "USDT"


@dataclass
class ExecutionResult:
    """Result of trade execution"""
    success: bool
    order_response: Optional[OrderResponse] = None
    position: Optional[Position] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class WeexAPIClient:
    """WEEX API client for order execution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get('api_base_url', 'https://api.weex.com')
        self.api_key = config.get('api_key', '')
        self.secret_key = config.get('secret_key', '')
        self.timeout = config.get('timeout_seconds', 30)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate API signature (mock implementation)"""
        # In real implementation, this would generate HMAC signature
        # For now, return a mock signature
        return "mock_signature"
    
    async def _make_authenticated_request(self, method: str, endpoint: str, 
                                        params: Optional[Dict] = None, 
                                        data: Optional[Dict] = None) -> Dict:
        """Make authenticated request to WEEX API"""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        url = f"{self.base_url}{endpoint}"
        
        # Add timestamp and signature for authentication
        if params is None:
            params = {}
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = self._generate_signature(params)
        
        try:
            if method.upper() == 'GET':
                async with self.session.get(url, params=params) as response:
                    return await self._handle_response(response)
            elif method.upper() == 'POST':
                async with self.session.post(url, params=params, json=data) as response:
                    return await self._handle_response(response)
            elif method.upper() == 'DELETE':
                async with self.session.delete(url, params=params) as response:
                    return await self._handle_response(response)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
        except asyncio.TimeoutError:
            raise Exception(f"Request timeout for {url}")
        except aiohttp.ClientError as e:
            raise Exception(f"Client error for {url}: {e}")
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict:
        """Handle API response"""
        if response.status == 200:
            return await response.json()
        else:
            error_text = await response.text()
            raise aiohttp.ClientResponseError(
                request_info=response.request_info,
                history=response.history,
                status=response.status,
                message=f"API request failed: {error_text}"
            )
    
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place a new order"""
        data = {
            'symbol': order_request.pair,
            'side': order_request.side.value,
            'type': order_request.order_type.value,
            'quantity': str(order_request.quantity),
            'timeInForce': order_request.time_in_force
        }
        
        if order_request.price is not None:
            data['price'] = str(order_request.price)
        
        if order_request.leverage != 1.0:
            data['leverage'] = str(order_request.leverage)
        
        try:
            response = await self._make_authenticated_request('POST', '/v1/order', data=data)
            
            return OrderResponse(
                order_id=response['orderId'],
                pair=response['symbol'],
                side=OrderSide(response['side']),
                order_type=OrderType(response['type']),
                quantity=float(response['origQty']),
                filled_quantity=float(response['executedQty']),
                price=float(response['price']) if response.get('price') else None,
                average_price=float(response['avgPrice']) if response.get('avgPrice') else None,
                status=OrderStatus(response['status']),
                timestamp=datetime.fromtimestamp(response['transactTime'] / 1000),
                commission=float(response.get('commission', 0)),
                commission_asset=response.get('commissionAsset', 'USDT')
            )
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    async def get_order_status(self, order_id: str, pair: str) -> OrderResponse:
        """Get order status"""
        params = {
            'symbol': pair,
            'orderId': order_id
        }
        
        try:
            response = await self._make_authenticated_request('GET', '/v1/order', params=params)
            
            return OrderResponse(
                order_id=response['orderId'],
                pair=response['symbol'],
                side=OrderSide(response['side']),
                order_type=OrderType(response['type']),
                quantity=float(response['origQty']),
                filled_quantity=float(response['executedQty']),
                price=float(response['price']) if response.get('price') else None,
                average_price=float(response['avgPrice']) if response.get('avgPrice') else None,
                status=OrderStatus(response['status']),
                timestamp=datetime.fromtimestamp(response['time'] / 1000),
                commission=float(response.get('commission', 0)),
                commission_asset=response.get('commissionAsset', 'USDT')
            )
            
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            raise
    
    async def cancel_order(self, order_id: str, pair: str) -> bool:
        """Cancel an order"""
        params = {
            'symbol': pair,
            'orderId': order_id
        }
        
        try:
            await self._make_authenticated_request('DELETE', '/v1/order', params=params)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, pair: str) -> Dict[str, Any]:
        """Cancel all open orders for a pair"""
        params = {'symbol': pair}
        
        try:
            response = await self._make_authenticated_request('DELETE', '/v1/openOrders', params=params)
            return response
        except Exception as e:
            logger.error(f"Failed to cancel all orders for {pair}: {e}")
            raise
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            response = await self._make_authenticated_request('GET', '/v1/account')
            return response
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise
    
    async def get_position_info(self, pair: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get position information"""
        params = {}
        if pair:
            params['symbol'] = pair
        
        try:
            response = await self._make_authenticated_request('GET', '/v1/positionRisk', params=params)
            return response
        except Exception as e:
            logger.error(f"Failed to get position info: {e}")
            raise


class ExecutionEngine:
    """
    Execution engine that translates agent actions into WEEX API orders
    with proper error handling and retry logic.
    """
    
    def __init__(self, config: Dict[str, Any], portfolio: PortfolioState):
        self.config = config
        self.portfolio = portfolio
        self.api_client = WeexAPIClient(config.get('exchange', {}))
        self.max_retries = config.get('execution', {}).get('max_retries', 3)
        self.retry_delay = config.get('execution', {}).get('retry_delay', 1.0)
        self.commission_rate = config.get('execution', {}).get('commission_rate', 0.001)  # 0.1%
        self.slippage_rate = config.get('execution', {}).get('slippage_rate', 0.0005)  # 0.05%
        
    def _action_to_order_params(self, action: List[float], pair: str, 
                               current_price: float) -> Tuple[OrderSide, float, float]:
        """Convert agent action to order parameters"""
        direction, size, leverage = action
        
        # Determine order side based on direction
        if direction > 0:
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL
        
        # Calculate position size in base currency
        equity = self.portfolio.get_total_equity()
        position_value = equity * abs(size)  # size is already 0-0.1 range
        quantity = position_value / current_price
        
        # Apply leverage
        leverage = max(1.0, min(12.0, leverage))  # Clamp to 1-12x range
        
        return side, quantity, leverage
    
    def _calculate_expected_price(self, side: OrderSide, current_price: float) -> float:
        """Calculate expected execution price including slippage"""
        if side == OrderSide.BUY:
            return current_price * (1 + self.slippage_rate)
        else:
            return current_price * (1 - self.slippage_rate)
    
    def _create_position_from_order(self, order_response: OrderResponse, 
                                   leverage: float) -> Position:
        """Create position object from order response"""
        side = 'long' if order_response.side == OrderSide.BUY else 'short'
        
        return Position(
            pair=order_response.pair,
            side=side,
            size=order_response.filled_quantity,
            leverage=leverage,
            entry_price=order_response.average_price or order_response.price,
            current_price=order_response.average_price or order_response.price,
            unrealized_pnl=0.0,
            timestamp=order_response.timestamp
        )
    
    async def execute_trade(self, action: List[float], pair: str, 
                          current_price: float) -> ExecutionResult:
        """
        Execute a trade based on agent action.
        
        Args:
            action: [direction, size, leverage] from PPO agent
            pair: Trading pair (e.g., 'BTCUSDT')
            current_price: Current market price
            
        Returns:
            ExecutionResult with success status and details
        """
        try:
            # Convert action to order parameters
            side, quantity, leverage = self._action_to_order_params(action, pair, current_price)
            
            # Create order request
            order_request = OrderRequest(
                pair=pair,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                leverage=leverage
            )
            
            # Execute order with retry logic
            order_response = await self._execute_order_with_retry(order_request)
            
            if order_response and order_response.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                # Create position from successful order
                position = self._create_position_from_order(order_response, leverage)
                
                # Update portfolio
                self.portfolio.add_position(position)
                
                logger.info(f"Trade executed successfully: {pair} {side.value} {quantity:.6f} @ {order_response.average_price}")
                
                return ExecutionResult(
                    success=True,
                    order_response=order_response,
                    position=position
                )
            else:
                error_msg = f"Order not filled: {order_response.status if order_response else 'No response'}"
                logger.warning(error_msg)
                return ExecutionResult(
                    success=False,
                    order_response=order_response,
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"Trade execution failed: {e}"
            logger.error(error_msg)
            return ExecutionResult(
                success=False,
                error_message=error_msg
            )
    
    async def _execute_order_with_retry(self, order_request: OrderRequest) -> Optional[OrderResponse]:
        """Execute order with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self.api_client as client:
                    order_response = await client.place_order(order_request)
                    
                    # Wait for order to be processed
                    if order_response.status == OrderStatus.NEW:
                        order_response = await self._wait_for_order_fill(
                            client, order_response.order_id, order_request.pair
                        )
                    
                    return order_response
                    
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Order execution attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries + 1} order execution attempts failed")
        
        if last_exception:
            raise last_exception
        return None
    
    async def _wait_for_order_fill(self, client: WeexAPIClient, order_id: str, 
                                  pair: str, timeout: int = 30) -> OrderResponse:
        """Wait for order to be filled with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                order_status = await client.get_order_status(order_id, pair)
                
                if order_status.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED, 
                                         OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                    return order_status
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                logger.warning(f"Error checking order status: {e}")
                await asyncio.sleep(1.0)
        
        # Timeout reached, try to cancel the order
        logger.warning(f"Order {order_id} timeout, attempting to cancel")
        await client.cancel_order(order_id, pair)
        
        # Return last known status
        try:
            return await client.get_order_status(order_id, pair)
        except Exception:
            # If we can't get status, create a timeout response
            return OrderResponse(
                order_id=order_id,
                pair=pair,
                side=OrderSide.BUY,  # Will be overridden
                order_type=OrderType.MARKET,
                quantity=0.0,
                filled_quantity=0.0,
                price=None,
                average_price=None,
                status=OrderStatus.EXPIRED,
                timestamp=datetime.now()
            )
    
    async def close_position(self, pair: str) -> ExecutionResult:
        """Close an existing position"""
        if pair not in self.portfolio.positions:
            return ExecutionResult(
                success=False,
                error_message=f"No position found for {pair}"
            )
        
        position = self.portfolio.positions[pair]
        
        # Determine opposite side to close position
        close_side = OrderSide.SELL if position.side == 'long' else OrderSide.BUY
        
        order_request = OrderRequest(
            pair=pair,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=position.size,
            leverage=position.leverage
        )
        
        try:
            order_response = await self._execute_order_with_retry(order_request)
            
            if order_response and order_response.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                # Calculate realized PnL
                if position.side == 'long':
                    pnl = (order_response.average_price - position.entry_price) * position.size * position.leverage
                else:
                    pnl = (position.entry_price - order_response.average_price) * position.size * position.leverage
                
                # Remove position from portfolio
                self.portfolio.remove_position(pair)
                
                # Update portfolio balance
                self.portfolio.balance += pnl
                self.portfolio.total_pnl += pnl
                
                logger.info(f"Position closed: {pair} PnL: {pnl:.2f}")
                
                return ExecutionResult(
                    success=True,
                    order_response=order_response
                )
            else:
                error_msg = f"Failed to close position: {order_response.status if order_response else 'No response'}"
                return ExecutionResult(
                    success=False,
                    order_response=order_response,
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"Position closure failed: {e}"
            logger.error(error_msg)
            return ExecutionResult(
                success=False,
                error_message=error_msg
            )
    
    async def cancel_all_orders(self, pair: str) -> Dict[str, Any]:
        """Cancel all open orders for a trading pair"""
        try:
            async with self.api_client as client:
                result = await client.cancel_all_orders(pair)
                logger.info(f"Cancelled all orders for {pair}")
                return result
        except Exception as e:
            logger.error(f"Failed to cancel all orders for {pair}: {e}")
            raise
    
    async def get_position_status(self, pair: str) -> Optional[Dict[str, Any]]:
        """Get current position status from exchange"""
        try:
            async with self.api_client as client:
                positions = await client.get_position_info(pair)
                
                for pos in positions:
                    if pos['symbol'] == pair and float(pos['positionAmt']) != 0:
                        return pos
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get position status for {pair}: {e}")
            return None
    
    async def sync_portfolio_with_exchange(self) -> bool:
        """Synchronize portfolio state with exchange positions"""
        try:
            async with self.api_client as client:
                # Get account info
                account_info = await client.get_account_info()
                
                # Update balance
                for balance in account_info.get('balances', []):
                    if balance['asset'] == 'USDT':
                        self.portfolio.balance = float(balance['free']) + float(balance['locked'])
                        break
                
                # Get all positions
                positions = await client.get_position_info()
                
                # Clear current positions
                self.portfolio.positions.clear()
                
                # Add active positions
                for pos_data in positions:
                    position_amt = float(pos_data['positionAmt'])
                    if position_amt != 0:
                        side = 'long' if position_amt > 0 else 'short'
                        
                        position = Position(
                            pair=pos_data['symbol'],
                            side=side,
                            size=abs(position_amt),
                            leverage=float(pos_data['leverage']),
                            entry_price=float(pos_data['entryPrice']),
                            current_price=float(pos_data['markPrice']),
                            unrealized_pnl=float(pos_data['unRealizedProfit']),
                            timestamp=datetime.now()
                        )
                        
                        self.portfolio.positions[position.pair] = position
                
                logger.info("Portfolio synchronized with exchange")
                return True
                
        except Exception as e:
            logger.error(f"Failed to sync portfolio with exchange: {e}")
            return False
    
    def create_trade_record(self, action: List[float], execution_result: ExecutionResult,
                          confidence: float, market_regime: int, 
                          risk_metrics: Dict[str, Any]) -> TradeRecord:
        """Create a trade record for logging"""
        return TradeRecord(
            timestamp=datetime.now(),
            pair=execution_result.order_response.pair if execution_result.order_response else "UNKNOWN",
            action=action,
            entry_price=execution_result.order_response.average_price if execution_result.order_response else 0.0,
            exit_price=None,
            pnl=None,
            confidence=confidence,
            market_regime=market_regime,
            risk_metrics=risk_metrics
        )