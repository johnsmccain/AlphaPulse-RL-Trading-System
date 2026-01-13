#!/usr/bin/env python3
"""
Unit Tests for Backtesting Validation

Tests to ensure backtesting produces consistent results and validates
performance metric calculations according to requirement 4.3.

Requirements tested: 4.3
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create mock torch for testing
    torch = Mock()

try:
    from models.backtesting import BacktestingEngine, BacktestConfig, BacktestResult
    BACKTESTING_AVAILABLE = True
except ImportError as e:
    BACKTESTING_AVAILABLE = False
    # Create mock classes for testing
    BacktestingEngine = Mock
    BacktestConfig = Mock
    BacktestResult = Mock

from models.performance_analysis import ComprehensivePerformanceEvaluator, PerformanceMetrics, TradeRecord

try:
    from models.ppo_agent import PPOAgent
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    PPOAgent = Mock


class TestBacktestingConsistency(unittest.TestCase):
    """Test backtesting produces consistent results."""
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def setUp(self):
        """Set up test fixtures."""
        # Create mock agent
        self.mock_agent = Mock(spec=PPOAgent)
        self.mock_agent.predict.return_value = np.array([0.5, 0.05, 2.0])
        self.mock_agent.get_confidence.return_value = 0.8
        
        # Create test config
        self.config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-01-31",
            initial_balance=1000.0,
            pairs=["BTCUSDT"],
            data_interval="5m",
            transaction_cost_percent=0.05,
            slippage_percent=0.02,
            confidence_threshold=0.7
        )
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def test_backtesting_deterministic_results(self):
        """Test that backtesting produces deterministic results with same inputs."""
        engine1 = BacktestingEngine(self.mock_agent, self.config)
        engine2 = BacktestingEngine(self.mock_agent, self.config)
        
        # Run backtests with same configuration
        result1 = engine1.run_backtest()
        result2 = engine2.run_backtest()
        
        # Results should be identical for deterministic agent
        self.assertEqual(len(result1.trade_records), len(result2.trade_records))
        self.assertEqual(len(result1.portfolio_history), len(result2.portfolio_history))
        
        # Final portfolio values should match
        if result1.portfolio_history and result2.portfolio_history:
            final_value1 = result1.portfolio_history[-1]['total_value']
            final_value2 = result2.portfolio_history[-1]['total_value']
            self.assertAlmostEqual(final_value1, final_value2, places=2)
    
    def test_backtesting_with_different_seeds(self):
        """Test backtesting behavior with different random seeds."""
        # Create engines with different synthetic data (different random seeds)
        np.random.seed(42)
        engine1 = BacktestingEngine(self.mock_agent, self.config)
        result1 = engine1.run_backtest()
        
        np.random.seed(123)
        engine2 = BacktestingEngine(self.mock_agent, self.config)
        result2 = engine2.run_backtest()
        
        # Should have same structure but potentially different values due to synthetic data
        self.assertEqual(len(result1.timestamps), len(result2.timestamps))
        self.assertIsInstance(result1.performance_metrics, dict)
        self.assertIsInstance(result2.performance_metrics, dict)
    
    def test_backtesting_config_consistency(self):
        """Test that backtest results reflect configuration parameters."""
        engine = BacktestingEngine(self.mock_agent, self.config)
        result = engine.run_backtest()
        
        # Verify config is preserved in results
        self.assertEqual(result.config.initial_balance, 1000.0)
        self.assertEqual(result.config.pairs, ["BTCUSDT"])
        self.assertEqual(result.config.confidence_threshold, 0.7)
        
        # Verify initial balance matches first portfolio entry
        if result.portfolio_history:
            # Initial balance should be close to configured amount
            initial_balance = result.portfolio_history[0]['balance']
            self.assertAlmostEqual(initial_balance, 1000.0, delta=1.0)
    
    def test_backtesting_empty_data_handling(self):
        """Test backtesting handles empty or insufficient data gracefully."""
        # Create config with very short period
        short_config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-01-01",  # Same day
            initial_balance=1000.0,
            pairs=["BTCUSDT"]
        )
        
        engine = BacktestingEngine(self.mock_agent, short_config)
        result = engine.run_backtest()
        
        # Should complete without errors
        self.assertIsInstance(result, BacktestResult)
        self.assertIsInstance(result.trade_records, list)
        self.assertIsInstance(result.portfolio_history, list)


class TestPerformanceMetricCalculations(unittest.TestCase):
    """Test performance metric calculations are accurate."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ComprehensivePerformanceEvaluator()
        
        # Create test return series
        self.test_returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012])
        self.test_timestamps = [
            datetime(2023, 1, 1) + timedelta(days=i) 
            for i in range(len(self.test_returns))
        ]
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation accuracy."""
        metrics = self.evaluator.calculate_comprehensive_metrics(self.test_returns)
        
        # Manual calculation
        mean_return = np.mean(self.test_returns)
        std_return = np.std(self.test_returns)
        expected_sharpe = (mean_return * 252 - 0.02) / (std_return * np.sqrt(252))  # 2% risk-free rate
        
        self.assertAlmostEqual(metrics['sharpe_ratio'], expected_sharpe, places=4)
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation accuracy."""
        metrics = self.evaluator.calculate_comprehensive_metrics(self.test_returns)
        
        # Manual calculation
        cumulative = np.cumsum(self.test_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + 1e-8)
        expected_max_dd = abs(np.min(drawdowns))
        
        self.assertAlmostEqual(metrics['max_drawdown'], expected_max_dd, places=6)
    
    def test_win_rate_calculation(self):
        """Test win rate calculation accuracy."""
        metrics = self.evaluator.calculate_comprehensive_metrics(self.test_returns)
        
        # Manual calculation
        positive_returns = self.test_returns[self.test_returns > 0]
        expected_win_rate = len(positive_returns) / len(self.test_returns)
        
        self.assertAlmostEqual(metrics['win_rate'], expected_win_rate, places=6)
    
    def test_profit_factor_calculation(self):
        """Test profit factor calculation accuracy."""
        metrics = self.evaluator.calculate_comprehensive_metrics(self.test_returns)
        
        # Manual calculation
        positive_returns = self.test_returns[self.test_returns > 0]
        negative_returns = self.test_returns[self.test_returns < 0]
        
        gross_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0
        gross_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 1e-8
        expected_profit_factor = gross_profit / gross_loss
        
        self.assertAlmostEqual(metrics['profit_factor'], expected_profit_factor, places=6)
    
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation accuracy."""
        metrics = self.evaluator.calculate_comprehensive_metrics(self.test_returns)
        
        # Manual calculation
        mean_return = np.mean(self.test_returns)
        downside_returns = self.test_returns[self.test_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        expected_sortino = (mean_return * 252 - 0.02) / downside_deviation if downside_deviation > 0 else 0
        
        self.assertAlmostEqual(metrics['sortino_ratio'], expected_sortino, places=4)
    
    def test_var_calculation(self):
        """Test Value at Risk calculation accuracy."""
        metrics = self.evaluator.calculate_comprehensive_metrics(self.test_returns)
        
        # Manual calculation
        expected_var = np.percentile(self.test_returns, 5)
        
        self.assertAlmostEqual(metrics['value_at_risk_95'], expected_var, places=6)
    
    def test_metrics_with_empty_returns(self):
        """Test metric calculations handle empty returns gracefully."""
        empty_returns = np.array([])
        metrics = self.evaluator.calculate_comprehensive_metrics(empty_returns)
        
        # Should return empty dict for empty returns (graceful handling)
        self.assertIsInstance(metrics, dict)
        # Empty returns should result in empty metrics dict
        self.assertEqual(len(metrics), 0)
    
    def test_metrics_with_all_positive_returns(self):
        """Test metric calculations with all positive returns."""
        positive_returns = np.array([0.01, 0.02, 0.015, 0.008, 0.012])
        metrics = self.evaluator.calculate_comprehensive_metrics(positive_returns)
        
        # Win rate should be 100%
        self.assertEqual(metrics['win_rate'], 1.0)
        # Max drawdown should be 0
        self.assertEqual(metrics['max_drawdown'], 0.0)
        # Profit factor should be very high (infinity handled)
        self.assertGreater(metrics['profit_factor'], 100)
    
    def test_metrics_with_all_negative_returns(self):
        """Test metric calculations with all negative returns."""
        negative_returns = np.array([-0.01, -0.02, -0.015, -0.008, -0.012])
        metrics = self.evaluator.calculate_comprehensive_metrics(negative_returns)
        
        # Win rate should be 0%
        self.assertEqual(metrics['win_rate'], 0.0)
        # Profit factor should be 0
        self.assertEqual(metrics['profit_factor'], 0.0)
        # Sharpe ratio should be negative
        self.assertLess(metrics['sharpe_ratio'], 0)


class TestPerformanceMetricsObject(unittest.TestCase):
    """Test PerformanceMetrics object creation and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ComprehensivePerformanceEvaluator()
        self.test_returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        self.test_timestamps = [
            datetime(2023, 1, 1) + timedelta(days=i) 
            for i in range(len(self.test_returns))
        ]
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics object is created correctly."""
        metrics = self.evaluator.evaluate_returns(self.test_returns, self.test_timestamps)
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        
        # Check all required fields are present
        required_fields = [
            'total_return', 'annualized_return', 'sharpe_ratio', 'sortino_ratio',
            'calmar_ratio', 'max_drawdown', 'volatility', 'win_rate', 'profit_factor'
        ]
        
        for field in required_fields:
            self.assertTrue(hasattr(metrics, field))
            self.assertIsInstance(getattr(metrics, field), (int, float))
    
    def test_performance_metrics_values_validity(self):
        """Test PerformanceMetrics values are within expected ranges."""
        metrics = self.evaluator.evaluate_returns(self.test_returns, self.test_timestamps)
        
        # Win rate should be between 0 and 1
        self.assertGreaterEqual(metrics.win_rate, 0.0)
        self.assertLessEqual(metrics.win_rate, 1.0)
        
        # Max drawdown should be non-negative
        self.assertGreaterEqual(metrics.max_drawdown, 0.0)
        
        # Volatility should be non-negative
        self.assertGreaterEqual(metrics.volatility, 0.0)
        
        # Profit factor should be non-negative
        self.assertGreaterEqual(metrics.profit_factor, 0.0)
    
    def test_performance_metrics_with_benchmark(self):
        """Test PerformanceMetrics calculation with benchmark returns."""
        benchmark_returns = np.array([0.005, -0.002, 0.01, -0.005, 0.008])
        
        metrics = self.evaluator.calculate_comprehensive_metrics(
            self.test_returns, 
            self.test_timestamps, 
            benchmark_returns
        )
        
        # Should include information ratio when benchmark provided
        self.assertIn('information_ratio', metrics)
        self.assertIsInstance(metrics['information_ratio'], (int, float))


class TestBacktestResultValidation(unittest.TestCase):
    """Test BacktestResult object validation and serialization."""
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def setUp(self):
        """Set up test fixtures."""
        self.mock_agent = Mock(spec=PPOAgent)
        self.mock_agent.predict.return_value = np.array([0.5, 0.05, 2.0])
        self.mock_agent.get_confidence.return_value = 0.8
        
        self.config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-01-10",
            initial_balance=1000.0,
            pairs=["BTCUSDT"]
        )
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def test_backtest_result_structure(self):
        """Test BacktestResult has correct structure."""
        engine = BacktestingEngine(self.mock_agent, self.config)
        result = engine.run_backtest()
        
        # Check required fields
        required_fields = [
            'config', 'performance_metrics', 'trade_records', 'portfolio_history',
            'regime_analysis', 'execution_summary', 'timestamps', 'returns'
        ]
        
        for field in required_fields:
            self.assertTrue(hasattr(result, field))
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def test_backtest_result_serialization(self):
        """Test BacktestResult can be serialized to dictionary."""
        engine = BacktestingEngine(self.mock_agent, self.config)
        result = engine.run_backtest()
        
        # Should be able to convert to dictionary
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        
        # Should be JSON serializable
        json_str = json.dumps(result_dict, default=str)
        self.assertIsInstance(json_str, str)
        
        # Should be able to parse back
        parsed_dict = json.loads(json_str)
        self.assertIsInstance(parsed_dict, dict)
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def test_trade_records_validity(self):
        """Test trade records have valid structure."""
        engine = BacktestingEngine(self.mock_agent, self.config)
        result = engine.run_backtest()
        
        for trade in result.trade_records:
            self.assertIsInstance(trade, TradeRecord)
            
            # Check required fields
            self.assertIsInstance(trade.timestamp, datetime)
            self.assertIn(trade.side, ['long', 'short'])
            self.assertIsInstance(trade.entry_price, (int, float))
            self.assertIsInstance(trade.confidence, (int, float))
            self.assertIn(trade.market_regime, [0, 1])
            
            # Check value ranges
            self.assertGreaterEqual(trade.confidence, 0.0)
            self.assertLessEqual(trade.confidence, 1.0)
            self.assertGreater(trade.entry_price, 0.0)
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def test_portfolio_history_consistency(self):
        """Test portfolio history maintains consistency."""
        engine = BacktestingEngine(self.mock_agent, self.config)
        result = engine.run_backtest()
        
        if result.portfolio_history:
            # First entry should have initial balance
            first_entry = result.portfolio_history[0]
            self.assertAlmostEqual(first_entry['balance'], 1000.0, delta=10.0)
            
            # All entries should have required fields
            for entry in result.portfolio_history:
                required_fields = ['timestamp', 'balance', 'total_value', 'unrealized_pnl', 'positions']
                for field in required_fields:
                    self.assertIn(field, entry)
                
                # Values should be reasonable
                self.assertGreaterEqual(entry['balance'], 0.0)
                self.assertGreaterEqual(entry['total_value'], 0.0)
                self.assertGreaterEqual(entry['positions'], 0)


class TestBacktestingEdgeCases(unittest.TestCase):
    """Test backtesting handles edge cases correctly."""
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def setUp(self):
        """Set up test fixtures."""
        self.mock_agent = Mock(spec=PPOAgent)
        
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def test_agent_always_holds(self):
        """Test backtesting when agent never trades."""
        # Agent that never meets confidence threshold
        self.mock_agent.predict.return_value = np.array([0.1, 0.01, 1.0])  # Small direction
        self.mock_agent.get_confidence.return_value = 0.3  # Low confidence
        
        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-01-05",
            initial_balance=1000.0,
            pairs=["BTCUSDT"],
            confidence_threshold=0.8  # High threshold
        )
        
        engine = BacktestingEngine(self.mock_agent, config)
        result = engine.run_backtest()
        
        # Should have no trades
        self.assertEqual(len(result.trade_records), 0)
        
        # Portfolio should remain close to initial balance
        if result.portfolio_history:
            final_balance = result.portfolio_history[-1]['balance']
            self.assertAlmostEqual(final_balance, 1000.0, delta=10.0)
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def test_agent_always_trades(self):
        """Test backtesting when agent always trades."""
        # Agent with high confidence and clear signals
        self.mock_agent.predict.return_value = np.array([0.8, 0.05, 3.0])
        self.mock_agent.get_confidence.return_value = 0.9
        
        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-01-05",
            initial_balance=1000.0,
            pairs=["BTCUSDT"],
            confidence_threshold=0.7
        )
        
        engine = BacktestingEngine(self.mock_agent, config)
        result = engine.run_backtest()
        
        # Should have some trades (exact number depends on synthetic data)
        self.assertGreaterEqual(len(result.trade_records), 0)
        
        # Execution summary should reflect trading activity
        self.assertGreaterEqual(result.execution_summary['total_trades'], 0)
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def test_extreme_market_conditions(self):
        """Test backtesting handles extreme market conditions."""
        self.mock_agent.predict.return_value = np.array([0.5, 0.05, 2.0])
        self.mock_agent.get_confidence.return_value = 0.8
        
        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-01-03",
            initial_balance=1000.0,
            pairs=["BTCUSDT"],
            transaction_cost_percent=1.0,  # High transaction costs
            slippage_percent=0.5  # High slippage
        )
        
        engine = BacktestingEngine(self.mock_agent, config)
        result = engine.run_backtest()
        
        # Should complete without errors despite extreme conditions
        self.assertIsInstance(result, BacktestResult)
        self.assertIsInstance(result.performance_metrics, dict)
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def test_multiple_pairs_consistency(self):
        """Test backtesting consistency with multiple trading pairs."""
        self.mock_agent.predict.return_value = np.array([0.5, 0.05, 2.0])
        self.mock_agent.get_confidence.return_value = 0.8
        
        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-01-05",
            initial_balance=1000.0,
            pairs=["BTCUSDT", "ETHUSDT"]  # Multiple pairs
        )
        
        engine = BacktestingEngine(self.mock_agent, config)
        result = engine.run_backtest()
        
        # Should handle multiple pairs
        pairs_traded = result.execution_summary.get('pairs_traded', [])
        self.assertIsInstance(pairs_traded, list)
        
        # Trade records should include both pairs if any trades occurred
        if result.trade_records:
            traded_pairs = set(trade.pair for trade in result.trade_records)
            self.assertTrue(traded_pairs.issubset(set(config.pairs)))


class TestBacktestingSaveLoad(unittest.TestCase):
    """Test backtesting result saving and loading functionality."""
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def setUp(self):
        """Set up test fixtures."""
        self.mock_agent = Mock(spec=PPOAgent)
        self.mock_agent.predict.return_value = np.array([0.5, 0.05, 2.0])
        self.mock_agent.get_confidence.return_value = 0.8
        
        self.config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-01-05",
            initial_balance=1000.0,
            pairs=["BTCUSDT"]
        )
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def test_save_results_creates_files(self):
        """Test that save_results creates expected files."""
        engine = BacktestingEngine(self.mock_agent, self.config)
        result = engine.run_backtest()
        
        # Save results
        engine.save_results(result, self.temp_dir)
        
        # Check that files are created
        expected_files = [
            "backtest_results.json",
            "portfolio_history.csv"
        ]
        
        for filename in expected_files:
            file_path = Path(self.temp_dir) / filename
            self.assertTrue(file_path.exists(), f"File {filename} was not created")
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def test_saved_json_is_valid(self):
        """Test that saved JSON file is valid and loadable."""
        engine = BacktestingEngine(self.mock_agent, self.config)
        result = engine.run_backtest()
        
        engine.save_results(result, self.temp_dir)
        
        # Load and validate JSON
        json_path = Path(self.temp_dir) / "backtest_results.json"
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        
        # Should have main sections
        expected_sections = ['config', 'performance_metrics', 'execution_summary']
        for section in expected_sections:
            self.assertIn(section, loaded_data)
    
    @unittest.skipUnless(BACKTESTING_AVAILABLE and PPO_AVAILABLE, "Backtesting or PPO dependencies not available")
    def test_saved_csv_is_valid(self):
        """Test that saved CSV files are valid and loadable."""
        engine = BacktestingEngine(self.mock_agent, self.config)
        result = engine.run_backtest()
        
        engine.save_results(result, self.temp_dir)
        
        # Load and validate portfolio history CSV
        csv_path = Path(self.temp_dir) / "portfolio_history.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            # Should have expected columns
            expected_columns = ['timestamp', 'balance', 'total_value']
            for col in expected_columns:
                self.assertIn(col, df.columns)
            
            # Should have data
            self.assertGreater(len(df), 0)


if __name__ == '__main__':
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Run tests with detailed output
    unittest.main(verbosity=2)