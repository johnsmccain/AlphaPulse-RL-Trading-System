#!/usr/bin/env python3
"""
Unit Tests for ExecutionEngine

Tests for order execution logic, error handling, portfolio state updates,
and PnL calculations with mocked WEEX API interactions.

Requirements tested: 4.4
"""

import unittest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import sys
import os
import pytest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.execution import (
    ExecutionEngine, WeexAPIClient, OrderRequest, OrderResponse, ExecutionResult,
    OrderSide, OrderType, OrderStatus
)
from trading.portfolio import PortfolioState, Position, TradeRecord


class TestWeexAPIClient(unittest.TestCase):
    """Test WEEX API client functionality with mocked responses"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'api_base_url': 'https://api.weex.com',
            'api_key': 'test_api_key',
            'secret_key': 'test_secret_key',
            'timeout_seconds': 30
        }
        self.client = WeexAPIClient(self.config)
    
    def test_client_initialization(self):
        """Test API client initialization"""
        self.assertEqual(self.client.base_url, 'https://api.weex.com')
        self.assertEqual(self.client.api_key, 'test_api_key')
        self.assertEqual(self.client.secret_key, 'test_secret_key')
        self.assertEqual(self.client.timeout, 30)
        self.assertIsNone(self.client.session)
    
    def test_signature_generation(self):
        """Test API signature generation"""
        params = {'symbol': 'BTCUSDT', 'side': 'BUY'}
        signature = self.client._generate_signature(params)
        
        # Should return a string (mock implementation)
        self.assertIsInstance(signature, str)
        self.assertEqual(signature, "mock_signature")
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_place_order_success(self):
        """Test successful order placement"""
        # Mock successful API response
        mock_response = {
            'orderId': '12345',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'type': 'MARKET',
            'origQty': '0.001',
            'executedQty': '0.001',
            'price': '50000.0',
            'avgPrice': '50000.0',
            'status': 'FILLED',
            'transactTime': 1640995200000,  # 2022-01-01 00:00:00
            'commission': '0.05',
            'commissionAsset': 'USDT'
        }
        
        # Mock session and response
        mock_session = AsyncMock()
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json.return_value = mock_response
        mock_session.post.return_value.__aenter__.return_value = mock_response_obj
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            async with self.client as client:
                order_request = OrderRequest(
                    pair='BTCUSDT',
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=0.001,
                    leverage=2.0
                )
                
                order_response = await client.place_order(order_request)
                
                # Verify response
                self.assertEqual(order_response.order_id, '12345')
                self.assertEqual(order_response.pair, 'BTCUSDT')
                self.assertEqual(order_response.side, OrderSide.BUY)
                self.assertEqual(order_response.order_type, OrderType.MARKET)
                self.assertEqual(order_response.quantity, 0.001)
                self.assertEqual(order_response.filled_quantity, 0.001)
                self.assertEqual(order_response.status, OrderStatus.FILLED)
                self.assertEqual(order_response.commission, 0.05)
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_place_order_failure(self):
        """Test order placement failure"""
        # Mock failed API response
        mock_session = AsyncMock()
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 400
        mock_response_obj.text.return_value = "Invalid order parameters"
        mock_session.post.return_value.__aenter__.return_value = mock_response_obj
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            async with self.client as client:
                order_request = OrderRequest(
                    pair='BTCUSDT',
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=0.001
                )
                
                with self.assertRaises(Exception):
                    await client.place_order(order_request)
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_get_order_status(self):
        """Test getting order status"""
        mock_response = {
            'orderId': '12345',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'type': 'MARKET',
            'origQty': '0.001',
            'executedQty': '0.001',
            'price': '50000.0',
            'avgPrice': '50000.0',
            'status': 'FILLED',
            'time': 1640995200000
        }
        
        mock_session = AsyncMock()
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json.return_value = mock_response
        mock_session.get.return_value.__aenter__.return_value = mock_response_obj
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            async with self.client as client:
                order_response = await client.get_order_status('12345', 'BTCUSDT')
                
                self.assertEqual(order_response.order_id, '12345')
                self.assertEqual(order_response.status, OrderStatus.FILLED)
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_cancel_order(self):
        """Test order cancellation"""
        mock_session = AsyncMock()
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json.return_value = {'orderId': '12345', 'status': 'CANCELED'}
        mock_session.delete.return_value.__aenter__.return_value = mock_response_obj
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            async with self.client as client:
                result = await client.cancel_order('12345', 'BTCUSDT')
                self.assertTrue(result)


class TestExecutionEngine(unittest.TestCase):
    """Test execution engine functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'exchange': {
                'api_base_url': 'https://api.weex.com',
                'api_key': 'test_key',
                'secret_key': 'test_secret'
            },
            'execution': {
                'max_retries': 3,
                'retry_delay': 1.0,
                'commission_rate': 0.001,
                'slippage_rate': 0.0005
            }
        }
        
        self.portfolio = PortfolioState(balance=1000.0)
        self.engine = ExecutionEngine(self.config, self.portfolio)
        
        # Mock the API client
        self.engine.api_client = Mock(spec=WeexAPIClient)
    
    def test_engine_initialization(self):
        """Test execution engine initialization"""
        self.assertEqual(self.engine.config, self.config)
        self.assertEqual(self.engine.portfolio, self.portfolio)
        self.assertEqual(self.engine.max_retries, 3)
        self.assertEqual(self.engine.retry_delay, 1.0)
        self.assertEqual(self.engine.commission_rate, 0.001)
        self.assertEqual(self.engine.slippage_rate, 0.0005)
    
    def test_action_to_order_params(self):
        """Test conversion of agent action to order parameters"""
        # Test long position
        action = [0.8, 0.05, 2.0]  # direction, size, leverage
        current_price = 50000.0
        
        side, quantity, leverage = self.engine._action_to_order_params(
            action, 'BTCUSDT', current_price
        )
        
        self.assertEqual(side, OrderSide.BUY)
        self.assertAlmostEqual(quantity, 0.001, places=6)  # (1000 * 0.05) / 50000
        self.assertEqual(leverage, 2.0)
        
        # Test short position
        action = [-0.8, 0.03, 1.5]
        side, quantity, leverage = self.engine._action_to_order_params(
            action, 'BTCUSDT', current_price
        )
        
        self.assertEqual(side, OrderSide.SELL)
        self.assertAlmostEqual(quantity, 0.0006, places=6)  # (1000 * 0.03) / 50000
        self.assertEqual(leverage, 1.5)
    
    def test_action_to_order_params_bounds(self):
        """Test action parameter bounds enforcement"""
        # Test leverage bounds
        action = [0.5, 0.05, 20.0]  # Leverage too high
        current_price = 50000.0
        
        side, quantity, leverage = self.engine._action_to_order_params(
            action, 'BTCUSDT', current_price
        )
        
        self.assertEqual(leverage, 12.0)  # Should be clamped to max
        
        # Test minimum leverage
        action = [0.5, 0.05, 0.5]  # Leverage too low
        side, quantity, leverage = self.engine._action_to_order_params(
            action, 'BTCUSDT', current_price
        )
        
        self.assertEqual(leverage, 1.0)  # Should be clamped to min
    
    def test_calculate_expected_price(self):
        """Test expected price calculation with slippage"""
        current_price = 50000.0
        
        # Test buy order (price should increase)
        buy_price = self.engine._calculate_expected_price(OrderSide.BUY, current_price)
        expected_buy_price = current_price * (1 + self.engine.slippage_rate)
        self.assertAlmostEqual(buy_price, expected_buy_price, places=2)
        
        # Test sell order (price should decrease)
        sell_price = self.engine._calculate_expected_price(OrderSide.SELL, current_price)
        expected_sell_price = current_price * (1 - self.engine.slippage_rate)
        self.assertAlmostEqual(sell_price, expected_sell_price, places=2)
    
    def test_create_position_from_order(self):
        """Test position creation from order response"""
        order_response = OrderResponse(
            order_id='12345',
            pair='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            filled_quantity=0.001,
            price=50000.0,
            average_price=50000.0,
            status=OrderStatus.FILLED,
            timestamp=datetime.now()
        )
        
        position = self.engine._create_position_from_order(order_response, 2.0)
        
        self.assertEqual(position.pair, 'BTCUSDT')
        self.assertEqual(position.side, 'long')
        self.assertEqual(position.size, 0.001)
        self.assertEqual(position.leverage, 2.0)
        self.assertEqual(position.entry_price, 50000.0)
        self.assertEqual(position.current_price, 50000.0)
        self.assertEqual(position.unrealized_pnl, 0.0)
    
    @pytest.mark.asyncio
    async def test_execute_trade_success(self):
        """Test successful trade execution"""
        # Mock successful order response
        order_response = OrderResponse(
            order_id='12345',
            pair='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            filled_quantity=0.001,
            price=50000.0,
            average_price=50000.0,
            status=OrderStatus.FILLED,
            timestamp=datetime.now()
        )
        
        # Mock the retry method to return successful response
        self.engine._execute_order_with_retry = AsyncMock(return_value=order_response)
        
        action = [0.8, 0.05, 2.0]
        result = await self.engine.execute_trade(action, 'BTCUSDT', 50000.0)
        
        # Verify successful execution
        self.assertTrue(result.success)
        self.assertIsNotNone(result.order_response)
        self.assertIsNotNone(result.position)
        self.assertIsNone(result.error_message)
        
        # Verify portfolio was updated
        self.assertIn('BTCUSDT', self.portfolio.positions)
        self.assertEqual(self.portfolio.trade_count, 1)
    
    @pytest.mark.asyncio
    async def test_execute_trade_failure(self):
        """Test trade execution failure"""
        # Mock failed order response
        order_response = OrderResponse(
            order_id='12345',
            pair='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            filled_quantity=0.0,
            price=50000.0,
            average_price=None,
            status=OrderStatus.REJECTED,
            timestamp=datetime.now()
        )
        
        self.engine._execute_order_with_retry = AsyncMock(return_value=order_response)
        
        action = [0.8, 0.05, 2.0]
        result = await self.engine.execute_trade(action, 'BTCUSDT', 50000.0)
        
        # Verify failed execution
        self.assertFalse(result.success)
        self.assertIsNotNone(result.order_response)
        self.assertIsNone(result.position)
        self.assertIsNotNone(result.error_message)
        
        # Verify portfolio was not updated
        self.assertNotIn('BTCUSDT', self.portfolio.positions)
        self.assertEqual(self.portfolio.trade_count, 0)
    
    @pytest.mark.asyncio
    async def test_execute_trade_exception(self):
        """Test trade execution with exception"""
        # Mock exception during execution
        self.engine._execute_order_with_retry = AsyncMock(
            side_effect=Exception("API connection failed")
        )
        
        action = [0.8, 0.05, 2.0]
        result = await self.engine.execute_trade(action, 'BTCUSDT', 50000.0)
        
        # Verify failed execution
        self.assertFalse(result.success)
        self.assertIsNone(result.order_response)
        self.assertIsNone(result.position)
        self.assertIn("API connection failed", result.error_message)
    
    @pytest.mark.asyncio
    async def test_close_position_success(self):
        """Test successful position closure"""
        # Add a position to portfolio
        position = Position(
            pair='BTCUSDT',
            side='long',
            size=0.001,
            leverage=2.0,
            entry_price=50000.0,
            current_price=51000.0,
            unrealized_pnl=2.0,
            timestamp=datetime.now()
        )
        self.portfolio.positions['BTCUSDT'] = position
        initial_balance = self.portfolio.balance
        
        # Mock successful close order
        close_order_response = OrderResponse(
            order_id='67890',
            pair='BTCUSDT',
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.001,
            filled_quantity=0.001,
            price=51000.0,
            average_price=51000.0,
            status=OrderStatus.FILLED,
            timestamp=datetime.now()
        )
        
        self.engine._execute_order_with_retry = AsyncMock(return_value=close_order_response)
        
        result = await self.engine.close_position('BTCUSDT')
        
        # Verify successful closure
        self.assertTrue(result.success)
        self.assertIsNotNone(result.order_response)
        
        # Verify position was removed and PnL realized
        self.assertNotIn('BTCUSDT', self.portfolio.positions)
        expected_pnl = (51000.0 - 50000.0) * 0.001 * 2.0  # 2.0 profit
        self.assertAlmostEqual(self.portfolio.balance, initial_balance + expected_pnl, places=2)
    
    @pytest.mark.asyncio
    async def test_close_position_no_position(self):
        """Test closing non-existent position"""
        result = await self.engine.close_position('ETHUSDT')
        
        self.assertFalse(result.success)
        self.assertIn("No position found", result.error_message)
    
    @pytest.mark.asyncio
    async def test_close_position_short(self):
        """Test closing short position"""
        # Add a short position
        position = Position(
            pair='BTCUSDT',
            side='short',
            size=0.001,
            leverage=2.0,
            entry_price=50000.0,
            current_price=49000.0,
            unrealized_pnl=2.0,
            timestamp=datetime.now()
        )
        self.portfolio.positions['BTCUSDT'] = position
        initial_balance = self.portfolio.balance
        
        # Mock successful close order (buy to close short)
        close_order_response = OrderResponse(
            order_id='67890',
            pair='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            filled_quantity=0.001,
            price=49000.0,
            average_price=49000.0,
            status=OrderStatus.FILLED,
            timestamp=datetime.now()
        )
        
        self.engine._execute_order_with_retry = AsyncMock(return_value=close_order_response)
        
        result = await self.engine.close_position('BTCUSDT')
        
        # Verify successful closure
        self.assertTrue(result.success)
        
        # Verify PnL calculation for short position
        expected_pnl = (50000.0 - 49000.0) * 0.001 * 2.0  # 2.0 profit
        self.assertAlmostEqual(self.portfolio.balance, initial_balance + expected_pnl, places=2)


class TestOrderRetryLogic(unittest.TestCase):
    """Test order retry logic and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'exchange': {'api_key': 'test', 'secret_key': 'test'},
            'execution': {'max_retries': 3, 'retry_delay': 0.1}  # Fast retry for testing
        }
        self.portfolio = PortfolioState(balance=1000.0)
        self.engine = ExecutionEngine(self.config, self.portfolio)
    
    @pytest.mark.asyncio
    async def test_retry_logic_success_after_failure(self):
        """Test successful execution after initial failures"""
        order_request = OrderRequest(
            pair='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001
        )
        
        # Mock API client that fails twice then succeeds
        mock_client = AsyncMock()
        call_count = 0
        
        async def mock_place_order(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary API error")
            return OrderResponse(
                order_id='12345',
                pair='BTCUSDT',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.001,
                filled_quantity=0.001,
                price=50000.0,
                average_price=50000.0,
                status=OrderStatus.FILLED,
                timestamp=datetime.now()
            )
        
        mock_client.place_order = mock_place_order
        
        with patch.object(self.engine, 'api_client', mock_client):
            result = await self.engine._execute_order_with_retry(order_request)
            
            # Should succeed after retries
            self.assertIsNotNone(result)
            self.assertEqual(result.status, OrderStatus.FILLED)
            self.assertEqual(call_count, 3)  # Failed twice, succeeded on third try
    
    @pytest.mark.asyncio
    async def test_retry_logic_max_retries_exceeded(self):
        """Test failure after max retries exceeded"""
        order_request = OrderRequest(
            pair='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001
        )
        
        # Mock API client that always fails
        mock_client = AsyncMock()
        mock_client.place_order = AsyncMock(side_effect=Exception("Persistent API error"))
        
        with patch.object(self.engine, 'api_client', mock_client):
            with self.assertRaises(Exception) as context:
                await self.engine._execute_order_with_retry(order_request)
            
            self.assertIn("Persistent API error", str(context.exception))
            # Should have tried max_retries + 1 times
            self.assertEqual(mock_client.place_order.call_count, 4)
    
    @pytest.mark.asyncio
    async def test_wait_for_order_fill_success(self):
        """Test waiting for order fill success"""
        mock_client = AsyncMock()
        
        # Mock order status progression: NEW -> FILLED
        status_responses = [
            OrderResponse(
                order_id='12345', pair='BTCUSDT', side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=0.001, filled_quantity=0.0,
                price=50000.0, average_price=None, status=OrderStatus.NEW,
                timestamp=datetime.now()
            ),
            OrderResponse(
                order_id='12345', pair='BTCUSDT', side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=0.001, filled_quantity=0.001,
                price=50000.0, average_price=50000.0, status=OrderStatus.FILLED,
                timestamp=datetime.now()
            )
        ]
        
        mock_client.get_order_status = AsyncMock(side_effect=status_responses)
        
        result = await self.engine._wait_for_order_fill(mock_client, '12345', 'BTCUSDT', timeout=5)
        
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(result.filled_quantity, 0.001)
    
    @pytest.mark.asyncio
    async def test_wait_for_order_fill_timeout(self):
        """Test order fill timeout handling"""
        mock_client = AsyncMock()
        
        # Mock order that stays NEW
        mock_client.get_order_status = AsyncMock(return_value=OrderResponse(
            order_id='12345', pair='BTCUSDT', side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=0.001, filled_quantity=0.0,
            price=50000.0, average_price=None, status=OrderStatus.NEW,
            timestamp=datetime.now()
        ))
        
        mock_client.cancel_order = AsyncMock(return_value=True)
        
        result = await self.engine._wait_for_order_fill(mock_client, '12345', 'BTCUSDT', timeout=1)
        
        # Should attempt to cancel and return expired status
        mock_client.cancel_order.assert_called_once_with('12345', 'BTCUSDT')
        self.assertEqual(result.status, OrderStatus.EXPIRED)


class TestPortfolioUpdates(unittest.TestCase):
    """Test portfolio state updates and PnL calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'exchange': {'api_key': 'test', 'secret_key': 'test'},
            'execution': {'max_retries': 1, 'retry_delay': 0.1}
        }
        self.portfolio = PortfolioState(balance=1000.0)
        self.engine = ExecutionEngine(self.config, self.portfolio)
    
    def test_portfolio_add_position(self):
        """Test adding position to portfolio"""
        initial_balance = self.portfolio.balance
        initial_trade_count = self.portfolio.trade_count
        
        position = Position(
            pair='BTCUSDT',
            side='long',
            size=0.001,
            leverage=2.0,
            entry_price=50000.0,
            current_price=50000.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now()
        )
        
        self.portfolio.add_position(position)
        
        # Verify position was added
        self.assertIn('BTCUSDT', self.portfolio.positions)
        self.assertEqual(self.portfolio.positions['BTCUSDT'], position)
        self.assertEqual(self.portfolio.trade_count, initial_trade_count + 1)
        self.assertIsNotNone(self.portfolio.last_trade_time)
    
    def test_portfolio_remove_position(self):
        """Test removing position from portfolio"""
        # Add position first
        position = Position(
            pair='BTCUSDT',
            side='long',
            size=0.001,
            leverage=2.0,
            entry_price=50000.0,
            current_price=50000.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now()
        )
        self.portfolio.positions['BTCUSDT'] = position
        
        # Remove position
        removed_position = self.portfolio.remove_position('BTCUSDT')
        
        # Verify position was removed
        self.assertNotIn('BTCUSDT', self.portfolio.positions)
        self.assertEqual(removed_position, position)
        
        # Test removing non-existent position
        result = self.portfolio.remove_position('ETHUSDT')
        self.assertIsNone(result)
    
    def test_portfolio_equity_calculation(self):
        """Test portfolio equity calculation with unrealized PnL"""
        initial_balance = 1000.0
        self.portfolio.balance = initial_balance
        
        # Add position with unrealized profit
        position = Position(
            pair='BTCUSDT',
            side='long',
            size=0.001,
            leverage=2.0,
            entry_price=50000.0,
            current_price=52000.0,  # Price increased
            unrealized_pnl=4.0,  # (52000-50000) * 0.001 * 2.0
            timestamp=datetime.now()
        )
        self.portfolio.positions['BTCUSDT'] = position
        
        # Test equity calculation
        expected_equity = initial_balance + 4.0
        self.assertAlmostEqual(self.portfolio.get_total_equity(), expected_equity, places=2)
    
    def test_position_pnl_calculation_long(self):
        """Test PnL calculation for long position"""
        position = Position(
            pair='BTCUSDT',
            side='long',
            size=0.001,
            leverage=2.0,
            entry_price=50000.0,
            current_price=50000.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now()
        )
        
        # Test profit scenario
        position.update_price(52000.0)
        expected_pnl = (52000.0 - 50000.0) * 0.001 * 2.0  # 4.0
        self.assertAlmostEqual(position.unrealized_pnl, expected_pnl, places=2)
        
        # Test loss scenario
        position.update_price(48000.0)
        expected_pnl = (48000.0 - 50000.0) * 0.001 * 2.0  # -4.0
        self.assertAlmostEqual(position.unrealized_pnl, expected_pnl, places=2)
    
    def test_position_pnl_calculation_short(self):
        """Test PnL calculation for short position"""
        position = Position(
            pair='BTCUSDT',
            side='short',
            size=0.001,
            leverage=2.0,
            entry_price=50000.0,
            current_price=50000.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now()
        )
        
        # Test profit scenario (price goes down)
        position.update_price(48000.0)
        expected_pnl = (50000.0 - 48000.0) * 0.001 * 2.0  # 4.0
        self.assertAlmostEqual(position.unrealized_pnl, expected_pnl, places=2)
        
        # Test loss scenario (price goes up)
        position.update_price(52000.0)
        expected_pnl = (50000.0 - 52000.0) * 0.001 * 2.0  # -4.0
        self.assertAlmostEqual(position.unrealized_pnl, expected_pnl, places=2)
    
    def test_portfolio_margin_calculations(self):
        """Test portfolio margin and exposure calculations"""
        # Add multiple positions
        positions = [
            Position(
                pair='BTCUSDT', side='long', size=0.001, leverage=2.0,
                entry_price=50000.0, current_price=50000.0, unrealized_pnl=0.0,
                timestamp=datetime.now()
            ),
            Position(
                pair='ETHUSDT', side='short', size=0.01, leverage=3.0,
                entry_price=3000.0, current_price=3000.0, unrealized_pnl=0.0,
                timestamp=datetime.now()
            )
        ]
        
        for pos in positions:
            self.portfolio.positions[pos.pair] = pos
        
        # Test margin calculation
        expected_margin = (0.001 * 50000.0) + (0.01 * 3000.0)  # 50 + 30 = 80
        self.assertAlmostEqual(self.portfolio.get_total_margin_used(), expected_margin, places=2)
        
        # Test notional exposure
        expected_exposure = (0.001 * 50000.0 * 2.0) + (0.01 * 3000.0 * 3.0)  # 100 + 90 = 190
        self.assertAlmostEqual(self.portfolio.get_total_notional_exposure(), expected_exposure, places=2)
    
    def test_portfolio_drawdown_calculation(self):
        """Test portfolio drawdown calculation"""
        self.portfolio.balance = 1000.0
        self.portfolio.peak_balance = 1000.0
        
        # Simulate loss
        self.portfolio.balance = 900.0
        self.portfolio.update_max_drawdown()
        
        expected_drawdown = (1000.0 - 900.0) / 1000.0  # 0.1 or 10%
        self.assertAlmostEqual(self.portfolio.max_drawdown, expected_drawdown, places=3)
        
        # Simulate recovery (drawdown should not decrease)
        self.portfolio.balance = 950.0
        self.portfolio.update_max_drawdown()
        self.assertAlmostEqual(self.portfolio.max_drawdown, expected_drawdown, places=3)
        
        # Simulate new peak
        self.portfolio.balance = 1100.0
        self.portfolio.update_max_drawdown()
        self.assertEqual(self.portfolio.peak_balance, 1100.0)


class TestTradeRecordCreation(unittest.TestCase):
    """Test trade record creation for logging"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'exchange': {'api_key': 'test', 'secret_key': 'test'},
            'execution': {'max_retries': 1, 'retry_delay': 0.1}
        }
        self.portfolio = PortfolioState(balance=1000.0)
        self.engine = ExecutionEngine(self.config, self.portfolio)
    
    def test_create_trade_record_success(self):
        """Test creating trade record for successful execution"""
        action = [0.8, 0.05, 2.0]
        order_response = OrderResponse(
            order_id='12345',
            pair='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            filled_quantity=0.001,
            price=50000.0,
            average_price=50000.0,
            status=OrderStatus.FILLED,
            timestamp=datetime.now()
        )
        
        execution_result = ExecutionResult(
            success=True,
            order_response=order_response
        )
        
        confidence = 0.85
        market_regime = 1
        risk_metrics = {'drawdown': 0.05, 'leverage': 2.0}
        
        trade_record = self.engine.create_trade_record(
            action, execution_result, confidence, market_regime, risk_metrics
        )
        
        # Verify trade record
        self.assertEqual(trade_record.pair, 'BTCUSDT')
        self.assertEqual(trade_record.action, action)
        self.assertEqual(trade_record.entry_price, 50000.0)
        self.assertIsNone(trade_record.exit_price)
        self.assertIsNone(trade_record.pnl)
        self.assertEqual(trade_record.confidence, confidence)
        self.assertEqual(trade_record.market_regime, market_regime)
        self.assertEqual(trade_record.risk_metrics, risk_metrics)
        self.assertIsInstance(trade_record.timestamp, datetime)
    
    def test_create_trade_record_failure(self):
        """Test creating trade record for failed execution"""
        action = [0.8, 0.05, 2.0]
        execution_result = ExecutionResult(
            success=False,
            error_message="Order rejected"
        )
        
        trade_record = self.engine.create_trade_record(
            action, execution_result, 0.75, 0, {}
        )
        
        # Verify trade record for failed execution
        self.assertEqual(trade_record.pair, "UNKNOWN")
        self.assertEqual(trade_record.entry_price, 0.0)


class TestSyncPortfolioWithExchange(unittest.TestCase):
    """Test portfolio synchronization with exchange"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'exchange': {'api_key': 'test', 'secret_key': 'test'},
            'execution': {'max_retries': 1, 'retry_delay': 0.1}
        }
        self.portfolio = PortfolioState(balance=1000.0)
        self.engine = ExecutionEngine(self.config, self.portfolio)
    
    @pytest.mark.asyncio
    async def test_sync_portfolio_success(self):
        """Test successful portfolio synchronization"""
        # Mock exchange responses
        mock_account_info = {
            'balances': [
                {'asset': 'USDT', 'free': '950.0', 'locked': '50.0'},
                {'asset': 'BTC', 'free': '0.0', 'locked': '0.0'}
            ]
        }
        
        mock_positions = [
            {
                'symbol': 'BTCUSDT',
                'positionAmt': '0.001',
                'leverage': '2',
                'entryPrice': '50000.0',
                'markPrice': '51000.0',
                'unRealizedProfit': '2.0'
            },
            {
                'symbol': 'ETHUSDT',
                'positionAmt': '0.0',
                'leverage': '1',
                'entryPrice': '0.0',
                'markPrice': '3000.0',
                'unRealizedProfit': '0.0'
            }
        ]
        
        # Mock API client
        mock_client = AsyncMock()
        mock_client.get_account_info = AsyncMock(return_value=mock_account_info)
        mock_client.get_position_info = AsyncMock(return_value=mock_positions)
        
        with patch.object(self.engine, 'api_client', mock_client):
            result = await self.engine.sync_portfolio_with_exchange()
            
            # Verify synchronization success
            self.assertTrue(result)
            
            # Verify balance update
            self.assertEqual(self.portfolio.balance, 1000.0)  # 950 + 50
            
            # Verify position update
            self.assertIn('BTCUSDT', self.portfolio.positions)
            btc_position = self.portfolio.positions['BTCUSDT']
            self.assertEqual(btc_position.side, 'long')
            self.assertEqual(btc_position.size, 0.001)
            self.assertEqual(btc_position.leverage, 2.0)
            self.assertEqual(btc_position.entry_price, 50000.0)
            self.assertEqual(btc_position.current_price, 51000.0)
            self.assertEqual(btc_position.unrealized_pnl, 2.0)
            
            # Verify empty position not added
            self.assertNotIn('ETHUSDT', self.portfolio.positions)
    
    @pytest.mark.asyncio
    async def test_sync_portfolio_failure(self):
        """Test portfolio synchronization failure"""
        # Mock API client that raises exception
        mock_client = AsyncMock()
        mock_client.get_account_info = AsyncMock(side_effect=Exception("API error"))
        
        with patch.object(self.engine, 'api_client', mock_client):
            result = await self.engine.sync_portfolio_with_exchange()
            
            # Verify synchronization failure
            self.assertFalse(result)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)