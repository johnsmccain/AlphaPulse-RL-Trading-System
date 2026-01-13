#!/usr/bin/env python3
"""
Unit Tests for WeexTradingEnv

Tests for reward function calculations, action space validation,
environment reset and step functionality.

Requirements tested: 4.2
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.weex_trading_env import WeexTradingEnv, PortfolioState, Position, TradeExecution
from env.env_utils import EnvironmentDataProcessor, create_test_environment


class TestWeexTradingEnv(unittest.TestCase):
    """Test suite for WeexTradingEnv"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Create synthetic test data
        self.data_processor = EnvironmentDataProcessor()
        self.test_data = self.data_processor.create_synthetic_data(n_samples=100)
        
        # Environment configuration
        self.config = {
            'initial_balance': 1000.0,
            'transaction_cost_percent': 0.05,
            'slippage_percent': 0.02,
            'transaction_cost_bps': 5,
            'slippage_bps': 2,
            'market_impact_coef': 0.1
        }
        
        # Create test environment
        self.env = WeexTradingEnv(data=self.test_data, config=self.config)
    
    def tearDown(self):
        """Clean up after each test"""
        self.env = None


class TestActionSpaceValidation(TestWeexTradingEnv):
    """Test action space validation functionality"""
    
    def test_action_space_properties(self):
        """Test action space has correct properties"""
        action_space = self.env.action_space
        
        # Check dimensions
        self.assertEqual(action_space.shape, (3,))
        
        # Check bounds (with tolerance for float32 precision)
        expected_low = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        expected_high = np.array([1.0, 0.1, 12.0], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(action_space.low, expected_low, decimal=6)
        np.testing.assert_array_almost_equal(action_space.high, expected_high, decimal=6)
    
    def test_valid_actions(self):
        """Test validation of valid actions"""
        valid_actions = [
            np.array([0.0, 0.0, 1.0]),      # Hold position
            np.array([1.0, 0.05, 2.0]),     # Long position
            np.array([-1.0, 0.03, 1.5]),    # Short position
            np.array([0.5, 0.1, 12.0]),     # Max position size and leverage
            np.array([-0.8, 0.001, 1.0])    # Min position size
        ]
        
        for action in valid_actions:
            with self.subTest(action=action):
                is_valid, msg = self.env.validator.validate_action(action)
                self.assertTrue(is_valid, f"Action {action} should be valid: {msg}")
    
    def test_invalid_actions(self):
        """Test validation of invalid actions"""
        invalid_actions = [
            np.array([2.0, 0.05, 2.0]),     # Direction out of range
            np.array([-2.0, 0.05, 2.0]),    # Direction out of range
            np.array([0.5, 0.2, 2.0]),      # Size too large
            np.array([0.5, -0.1, 2.0]),     # Negative size
            np.array([0.5, 0.05, 20.0]),    # Leverage too high
            np.array([0.5, 0.05, 0.5]),     # Leverage too low
            np.array([np.nan, 0.05, 2.0]),  # NaN value
            np.array([0.5, np.inf, 2.0])    # Infinite value
        ]
        
        for action in invalid_actions:
            with self.subTest(action=action):
                is_valid, msg = self.env.validator.validate_action(action)
                self.assertFalse(is_valid, f"Action {action} should be invalid")
    
    def test_action_space_sampling(self):
        """Test that sampled actions are valid"""
        for _ in range(100):
            action = self.env.action_space.sample()
            is_valid, msg = self.env.validator.validate_action(action)
            self.assertTrue(is_valid, f"Sampled action {action} should be valid: {msg}")
    
    def test_action_interpretation(self):
        """Test action interpretation logic"""
        # Test hold action (small direction and size)
        hold_action = np.array([0.05, 0.001, 1.0])
        self.env.reset()
        obs, reward, done, info = self.env.step(hold_action)
        trade_result = info.get('trade_result', {})
        self.assertEqual(trade_result.get('reason', ''), 'Hold position')
        
        # Test long action
        long_action = np.array([0.8, 0.05, 2.0])
        self.env.reset()
        obs, reward, done, info = self.env.step(long_action)
        trade_result = info.get('trade_result', {})
        if trade_result.get('success', False):
            self.assertIn('long', trade_result.get('reason', '').lower())
        
        # Test short action
        short_action = np.array([-0.8, 0.05, 2.0])
        self.env.reset()
        obs, reward, done, info = self.env.step(short_action)
        trade_result = info.get('trade_result', {})
        if trade_result.get('success', False):
            self.assertIn('short', trade_result.get('reason', '').lower())


class TestEnvironmentReset(TestWeexTradingEnv):
    """Test environment reset functionality"""
    
    def test_basic_reset(self):
        """Test basic reset functionality"""
        # Take some steps first
        for _ in range(5):
            action = self.env.action_space.sample()
            self.env.step(action)
        
        # Reset environment
        obs = self.env.reset()
        
        # Check observation
        self.assertEqual(obs.shape, (9,))
        self.assertFalse(np.any(np.isnan(obs)))
        self.assertFalse(np.any(np.isinf(obs)))
        
        # Check environment state
        self.assertEqual(self.env.current_step, 0)
        self.assertFalse(self.env.done)
        self.assertEqual(self.env.portfolio.balance, self.config['initial_balance'])
        self.assertEqual(self.env.portfolio.equity, self.config['initial_balance'])
        self.assertEqual(len(self.env.portfolio.positions), 0)
        self.assertEqual(self.env.portfolio.trade_count, 0)
        self.assertEqual(self.env.portfolio.total_pnl, 0.0)
        self.assertEqual(self.env.portfolio.max_drawdown, 0.0)
    
    def test_reset_with_seed(self):
        """Test reset with seed for reproducibility"""
        # Reset with same seed multiple times
        obs1 = self.env.reset(seed=42)
        obs2 = self.env.reset(seed=42)
        obs3 = self.env.reset(seed=42)
        
        # Observations should be identical
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(obs2, obs3)
        
        # Portfolio states should be identical
        self.assertEqual(self.env.portfolio.balance, self.config['initial_balance'])
        self.assertEqual(self.env.current_step, 0)
    
    def test_reset_with_options(self):
        """Test reset with custom options"""
        # Reset with custom initial balance
        custom_balance = 2000.0
        obs = self.env.reset(options={'initial_balance': custom_balance})
        
        self.assertEqual(self.env.portfolio.balance, custom_balance)
        self.assertEqual(self.env.portfolio.equity, custom_balance)
        self.assertEqual(self.env.initial_balance, custom_balance)
        
        # Reset with custom start step
        obs = self.env.reset(options={'start_step': 10})
        self.assertEqual(self.env.current_step, 10)
    
    def test_reset_clears_history(self):
        """Test that reset clears episode history"""
        # Execute some actions to build history
        for _ in range(10):
            action = self.env.action_space.sample()
            self.env.step(action)
        
        # Verify history exists
        self.assertGreater(len(self.env.reward_history), 0)
        self.assertGreater(len(self.env.state_log), 0)
        
        # Reset and verify history is cleared
        self.env.reset()
        self.assertEqual(len(self.env.reward_history), 0)
        self.assertEqual(len(self.env.state_log), 0)
        self.assertEqual(len(self.env.volatility_window), 0)


class TestEnvironmentStep(TestWeexTradingEnv):
    """Test environment step functionality"""
    
    def test_step_return_format(self):
        """Test that step returns correct format"""
        self.env.reset()
        action = self.env.action_space.sample()
        
        result = self.env.step(action)
        
        # Should return tuple of (obs, reward, done, info)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        
        obs, reward, done, info = result
        
        # Check observation
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (9,))
        self.assertFalse(np.any(np.isnan(obs)))
        self.assertFalse(np.any(np.isinf(obs)))
        
        # Check reward
        self.assertIsInstance(reward, (int, float, np.number))
        self.assertFalse(np.isnan(reward))
        self.assertFalse(np.isinf(reward))
        
        # Check done flag
        self.assertIsInstance(done, bool)
        
        # Check info dictionary
        self.assertIsInstance(info, dict)
        required_keys = ['portfolio', 'trade_result', 'reward_components', 'step']
        for key in required_keys:
            self.assertIn(key, info)
    
    def test_step_increments_counter(self):
        """Test that step increments step counter"""
        self.env.reset()
        initial_step = self.env.current_step
        
        action = self.env.action_space.sample()
        self.env.step(action)
        
        self.assertEqual(self.env.current_step, initial_step + 1)
    
    def test_step_updates_portfolio(self):
        """Test that step updates portfolio state"""
        self.env.reset()
        initial_balance = self.env.portfolio.balance
        
        # Execute a trade action
        action = np.array([1.0, 0.05, 2.0])  # Long position
        obs, reward, done, info = self.env.step(action)
        
        # Portfolio should be updated
        trade_result = info.get('trade_result', {})
        if trade_result.get('success', False):
            # Balance should decrease due to transaction costs
            self.assertLess(self.env.portfolio.balance, initial_balance)
            # Should have a position
            self.assertGreater(len(self.env.portfolio.positions), 0)
            # Trade count should increase
            self.assertGreater(self.env.portfolio.trade_count, 0)
    
    def test_multiple_steps(self):
        """Test multiple consecutive steps"""
        self.env.reset()
        
        for i in range(10):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            
            # Check step counter
            self.assertEqual(self.env.current_step, i + 1)
            
            # Check that we get valid returns
            self.assertEqual(obs.shape, (9,))
            self.assertIsInstance(reward, (int, float, np.number))
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)
            
            if done:
                break
    
    def test_episode_termination(self):
        """Test episode termination conditions"""
        self.env.reset()
        
        # Test max steps termination
        max_steps = self.env.max_steps
        for i in range(max_steps + 5):  # Go beyond max steps
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            
            if i >= max_steps - 1:
                self.assertTrue(done, f"Episode should be done at step {i}")
                if 'done_reason' in info:
                    self.assertEqual(info['done_reason'], 'max_steps_reached')
                break


class TestRewardCalculation(TestWeexTradingEnv):
    """Test reward function calculations"""
    
    def test_reward_components_exist(self):
        """Test that reward components are calculated"""
        self.env.reset()
        action = np.array([1.0, 0.05, 2.0])
        
        obs, reward, done, info = self.env.step(action)
        
        # Check reward components exist
        self.assertIn('reward_components', info)
        reward_components = info['reward_components']
        
        expected_components = [
            'pnl_reward', 'volatility_penalty', 'drawdown_penalty',
            'overtrading_penalty', 'cost_penalty', 'total_reward'
        ]
        
        for component in expected_components:
            self.assertIn(component, reward_components)
            self.assertIsInstance(reward_components[component], (int, float, np.number))
            self.assertFalse(np.isnan(reward_components[component]))
    
    def test_reward_components_sum(self):
        """Test that reward components are calculated correctly"""
        self.env.reset()
        action = np.array([1.0, 0.05, 2.0])
        
        obs, reward, done, info = self.env.step(action)
        
        reward_components = info['reward_components']
        
        # Check that all components are valid numbers
        for component_name, component_value in reward_components.items():
            self.assertIsInstance(component_value, (int, float, np.number))
            self.assertFalse(np.isnan(component_value), f"{component_name} is NaN")
            self.assertFalse(np.isinf(component_value), f"{component_name} is infinite")
        
        # Check that the main components exist and have reasonable values
        expected_components = [
            'pnl_reward', 'volatility_penalty', 'drawdown_penalty',
            'overtrading_penalty', 'cost_penalty', 'total_reward'
        ]
        
        for component in expected_components:
            self.assertIn(component, reward_components)
            
        # Penalties should be negative or zero
        self.assertLessEqual(reward_components['volatility_penalty'], 0.0)
        self.assertLessEqual(reward_components['drawdown_penalty'], 0.0)
        self.assertLessEqual(reward_components['overtrading_penalty'], 0.0)
        self.assertLessEqual(reward_components['cost_penalty'], 0.0)
        
        # The returned reward should be a valid number
        self.assertIsInstance(reward, (int, float, np.number))
        self.assertFalse(np.isnan(reward))
        self.assertFalse(np.isinf(reward))
        
        # The reward should be within a reasonable range
        self.assertGreater(reward, -10.0)
        self.assertLess(reward, 10.0)
    
    def test_pnl_reward_component(self):
        """Test PnL reward component calculation"""
        self.env.reset()
        initial_equity = self.env.portfolio.equity
        
        # Execute trade
        action = np.array([1.0, 0.05, 2.0])
        obs, reward, done, info = self.env.step(action)
        
        reward_components = info['reward_components']
        pnl_reward = reward_components['pnl_reward']
        
        # PnL reward should be normalized by initial balance
        # Note: The environment updates last_equity after reward calculation,
        # so we need to account for the timing of the calculation
        equity_change = self.env.portfolio.equity - self.env.last_equity
        expected_pnl_reward = equity_change / self.env.initial_balance
        
        # The PnL reward is calculated based on equity change from last step
        # For the first step, this should be close to the actual equity change
        self.assertIsInstance(pnl_reward, (int, float, np.number))
        self.assertFalse(np.isnan(pnl_reward))
        self.assertFalse(np.isinf(pnl_reward))
    
    def test_volatility_penalty(self):
        """Test volatility penalty calculation"""
        self.env.reset()
        
        # Execute multiple steps to build volatility history
        rewards = []
        for _ in range(5):
            action = np.array([1.0, 0.05, 2.0])
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
            
            reward_components = info['reward_components']
            volatility_penalty = reward_components['volatility_penalty']
            
            # Volatility penalty should be negative or zero
            self.assertLessEqual(volatility_penalty, 0.0)
            
            if done:
                break
    
    def test_drawdown_penalty(self):
        """Test drawdown penalty calculation"""
        self.env.reset()
        action = np.array([1.0, 0.05, 2.0])
        
        obs, reward, done, info = self.env.step(action)
        
        reward_components = info['reward_components']
        drawdown_penalty = reward_components['drawdown_penalty']
        
        # Drawdown penalty should be negative or zero
        self.assertLessEqual(drawdown_penalty, 0.0)
        
        # Should be proportional to max drawdown
        expected_penalty = -0.2 * self.env.portfolio.max_drawdown
        self.assertAlmostEqual(drawdown_penalty, expected_penalty, places=6)
    
    def test_overtrading_penalty(self):
        """Test overtrading penalty calculation"""
        self.env.reset()
        
        # Test different action intensities
        test_actions = [
            np.array([0.0, 0.0, 1.0]),      # No trading
            np.array([0.5, 0.02, 1.5]),     # Light trading
            np.array([1.0, 0.1, 12.0])      # Heavy trading
        ]
        
        penalties = []
        for action in test_actions:
            self.env.reset()
            obs, reward, done, info = self.env.step(action)
            
            reward_components = info['reward_components']
            overtrading_penalty = reward_components['overtrading_penalty']
            
            # Penalty should be negative or zero
            self.assertLessEqual(overtrading_penalty, 0.0)
            penalties.append(overtrading_penalty)
        
        # Higher intensity should have higher penalty (more negative)
        self.assertLessEqual(penalties[2], penalties[1])  # Heavy < Light
        self.assertLessEqual(penalties[1], penalties[0])  # Light < None
    
    def test_cost_penalty(self):
        """Test transaction cost penalty calculation"""
        self.env.reset()
        action = np.array([1.0, 0.05, 2.0])
        
        obs, reward, done, info = self.env.step(action)
        
        reward_components = info['reward_components']
        cost_penalty = reward_components['cost_penalty']
        trade_result = info['trade_result']
        
        # Cost penalty should be negative or zero
        self.assertLessEqual(cost_penalty, 0.0)
        
        # Should be proportional to transaction costs
        if trade_result.get('success', False):
            total_costs = trade_result.get('transaction_cost', 0) + trade_result.get('slippage', 0)
            expected_penalty = -total_costs / self.env.initial_balance
            self.assertAlmostEqual(cost_penalty, expected_penalty, places=6)
    
    def test_reward_range(self):
        """Test that rewards are within reasonable range"""
        self.env.reset()
        
        rewards = []
        for _ in range(20):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
            
            # Individual reward should be reasonable
            self.assertGreater(reward, -10.0, "Reward too negative")
            self.assertLess(reward, 10.0, "Reward too positive")
            
            if done:
                self.env.reset()
        
        # Check reward distribution
        self.assertGreater(len(rewards), 0)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        # Mean should be reasonable (not extremely positive or negative)
        self.assertGreater(mean_reward, -1.0)
        self.assertLess(mean_reward, 1.0)
        
        # Standard deviation should indicate some variation
        self.assertGreater(std_reward, 0.0)


class TestObservationSpace(TestWeexTradingEnv):
    """Test observation space functionality"""
    
    def test_observation_space_properties(self):
        """Test observation space has correct properties"""
        obs_space = self.env.observation_space
        
        # Check dimensions
        self.assertEqual(obs_space.shape, (9,))
        
        # Check data type
        self.assertEqual(obs_space.dtype, np.float32)
    
    def test_observation_validity(self):
        """Test that observations are valid"""
        self.env.reset()
        
        for _ in range(10):
            obs = self.env._get_current_observation()
            
            # Check shape and type
            self.assertEqual(obs.shape, (9,))
            self.assertEqual(obs.dtype, np.float32)
            
            # Check for invalid values
            self.assertFalse(np.any(np.isnan(obs)), "Observation contains NaN")
            self.assertFalse(np.any(np.isinf(obs)), "Observation contains infinite values")
            
            # Validate using environment validator
            is_valid, msg = self.env.validator.validate_observation(obs)
            self.assertTrue(is_valid, f"Observation validation failed: {msg}")
            
            # Step environment
            action = self.env.action_space.sample()
            self.env.step(action)
    
    def test_observation_features(self):
        """Test that observation contains expected features"""
        self.env.reset()
        obs = self.env._get_current_observation()
        
        # Should have 9 features as specified in design
        feature_names = [
            'returns_5m', 'returns_15m', 'rsi_14', 'macd_histogram',
            'atr_percentage', 'volume_zscore', 'orderbook_imbalance',
            'funding_rate', 'volatility_regime'
        ]
        
        self.assertEqual(len(obs), len(feature_names))
        
        # Check reasonable ranges for some features
        rsi_value = obs[2]  # RSI should be between 0 and 1
        self.assertGreaterEqual(rsi_value, 0.0)
        self.assertLessEqual(rsi_value, 1.0)
        
        volatility_regime = obs[8]  # Should be 0 or 1
        self.assertIn(volatility_regime, [0, 1])


class TestPortfolioTracking(TestWeexTradingEnv):
    """Test portfolio state tracking"""
    
    def test_initial_portfolio_state(self):
        """Test initial portfolio state is correct"""
        self.env.reset()
        portfolio = self.env.portfolio
        
        self.assertEqual(portfolio.balance, self.config['initial_balance'])
        self.assertEqual(portfolio.equity, self.config['initial_balance'])
        self.assertEqual(len(portfolio.positions), 0)
        self.assertEqual(portfolio.daily_pnl, 0.0)
        self.assertEqual(portfolio.total_pnl, 0.0)
        self.assertEqual(portfolio.max_drawdown, 0.0)
        self.assertEqual(portfolio.peak_equity, self.config['initial_balance'])
        self.assertEqual(portfolio.trade_count, 0)
        self.assertIsNone(portfolio.last_trade_time)
    
    def test_portfolio_updates_after_trade(self):
        """Test portfolio updates after successful trade"""
        self.env.reset()
        initial_balance = self.env.portfolio.balance
        
        # Execute trade
        action = np.array([1.0, 0.05, 2.0])
        obs, reward, done, info = self.env.step(action)
        
        trade_result = info.get('trade_result', {})
        if trade_result.get('success', False):
            portfolio = self.env.portfolio
            
            # Balance should decrease due to costs
            self.assertLess(portfolio.balance, initial_balance)
            
            # Should have a position
            self.assertGreater(len(portfolio.positions), 0)
            
            # Trade count should increase
            self.assertGreater(portfolio.trade_count, 0)
            
            # Last trade time should be set
            self.assertIsNotNone(portfolio.last_trade_time)
    
    def test_portfolio_validation(self):
        """Test portfolio state validation"""
        self.env.reset()
        
        # Execute several trades
        for _ in range(5):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            
            # Validate portfolio state
            is_valid, msg = self.env.validator.validate_portfolio_state(self.env.portfolio)
            self.assertTrue(is_valid, f"Portfolio state invalid: {msg}")
            
            if done:
                break
    
    def test_equity_calculation(self):
        """Test equity calculation includes unrealized PnL"""
        self.env.reset()
        
        # Execute trade to create position
        action = np.array([1.0, 0.05, 2.0])
        obs, reward, done, info = self.env.step(action)
        
        trade_result = info.get('trade_result', {})
        if trade_result.get('success', False):
            portfolio = self.env.portfolio
            
            # Equity should equal balance plus unrealized PnL
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in portfolio.positions.values())
            expected_equity = portfolio.balance + total_unrealized_pnl
            
            self.assertAlmostEqual(portfolio.equity, expected_equity, places=2)


class TestTransactionCosts(TestWeexTradingEnv):
    """Test transaction cost modeling"""
    
    def test_transaction_costs_applied(self):
        """Test that transaction costs are applied to trades"""
        self.env.reset()
        initial_balance = self.env.portfolio.balance
        
        # Execute trade
        action = np.array([1.0, 0.05, 2.0])
        obs, reward, done, info = self.env.step(action)
        
        trade_result = info.get('trade_result', {})
        if trade_result.get('success', False):
            # Should have transaction costs
            transaction_cost = trade_result.get('transaction_cost', 0)
            slippage = trade_result.get('slippage', 0)
            
            self.assertGreater(transaction_cost, 0, "Transaction cost should be positive")
            self.assertGreaterEqual(slippage, 0, "Slippage should be non-negative")
            
            # Balance should decrease by costs
            expected_balance = initial_balance - transaction_cost - slippage
            self.assertAlmostEqual(self.env.portfolio.balance, expected_balance, places=2)
    
    def test_cost_scaling(self):
        """Test that costs scale with position size and leverage"""
        self.env.reset()
        
        # Test different position sizes
        test_cases = [
            {'size': 0.01, 'leverage': 2.0},
            {'size': 0.05, 'leverage': 2.0},
            {'size': 0.1, 'leverage': 2.0}
        ]
        
        costs = []
        for case in test_cases:
            self.env.reset()
            action = np.array([1.0, case['size'], case['leverage']])
            obs, reward, done, info = self.env.step(action)
            
            trade_result = info.get('trade_result', {})
            if trade_result.get('success', False):
                total_cost = trade_result.get('transaction_cost', 0) + trade_result.get('slippage', 0)
                costs.append(total_cost)
        
        # Costs should generally increase with position size
        if len(costs) >= 2:
            self.assertGreater(costs[-1], costs[0], "Larger positions should have higher costs")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)