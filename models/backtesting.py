"""
Backtesting Engine for AlphaPulse-RL Trading System.

This module provides a comprehensive backtesting framework that uses identical
logic to live trading for accurate performance evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

# Import system components
from data.weex_fetcher import WeexDataFetcher
from data.feature_engineering import FeatureEngine
from models.ppo_agent import PPOAgent
from risk.risk_manager import RiskManager
from trading.portfolio import Portfolio, Position
from trading.execution import ExecutionEngine
from models.performance_analysis import ComprehensivePerformanceEvaluator, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    start_date: str
    end_date: str
    initial_balance: float = 1000.0
    pairs: List[str] = None
    data_interval: str = "5m"
    transaction_cost_percent: float = 0.05
    slippage_percent: float = 0.02
    confidence_threshold: float = 0.8
    max_positions: int = 2
    regime_threshold: float = 0.5
    
    def __post_init__(self):
        if self.pairs is None:
            self.pairs = ["BTCUSDT", "ETHUSDT"]


@dataclass
class BacktestResult:
    """Results from a backtesting run."""
    config: BacktestConfig
    performance_metrics: Dict[str, Any]
    trade_records: List[TradeRecord]
    portfolio_history: List[Dict[str, Any]]
    regime_analysis: Dict[str, Any]
    execution_summary: Dict[str, Any]
    timestamps: List[datetime]
    returns: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'config': asdict(self.config),
            'performance_metrics': self.performance_metrics,
            'trade_records': [asdict(trade) for trade in self.trade_records],
            'portfolio_history': self.portfolio_history,
            'regime_analysis': self.regime_analysis,
            'execution_summary': self.execution_summary,
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'returns': self.returns
        }


class BacktestingEngine:
    """
    Comprehensive backtesting engine that replicates live trading logic.
    
    This engine uses the same components as live trading to ensure
    accurate performance evaluation and regime-specific analysis.
    """
    
    def __init__(
        self,
        agent: PPOAgent,
        config: BacktestConfig,
        data_source: Optional[str] = None
    ):
        """
        Initialize backtesting engine.
        
        Args:
            agent: Trained PPO agent for trading decisions
            config: Backtesting configuration
            data_source: Optional path to historical data file
        """
        self.agent = agent
        self.config = config
        self.data_source = data_source
        
        # Initialize components (same as live trading)
        self.feature_engine = FeatureEngine()
        self.risk_manager = RiskManager()
        self.portfolio = Portfolio(initial_balance=config.initial_balance)
        
        # Mock execution engine for backtesting
        self.execution_engine = MockExecutionEngine(
            transaction_cost_percent=config.transaction_cost_percent,
            slippage_percent=config.slippage_percent
        )
        
        # Performance evaluator
        self.performance_evaluator = ComprehensivePerformanceEvaluator()
        
        # Tracking variables
        self.trade_records: List[TradeRecord] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.timestamps: List[datetime] = []
        self.returns: List[float] = []
        
        logger.info(f"Backtesting engine initialized for period {config.start_date} to {config.end_date}")
    
    def load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load historical market data for backtesting.
        
        Returns:
            Dictionary of DataFrames keyed by trading pair
        """
        historical_data = {}
        
        if self.data_source and Path(self.data_source).exists():
            # Load from file
            logger.info(f"Loading historical data from {self.data_source}")
            try:
                data = pd.read_csv(self.data_source)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                # Filter by date range
                start_date = pd.to_datetime(self.config.start_date)
                end_date = pd.to_datetime(self.config.end_date)
                data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
                
                # Split by pairs if multiple pairs in data
                if 'pair' in data.columns:
                    for pair in self.config.pairs:
                        pair_data = data[data['pair'] == pair].copy()
                        if not pair_data.empty:
                            historical_data[pair] = pair_data.sort_values('timestamp')
                else:
                    # Assume single pair data
                    historical_data[self.config.pairs[0]] = data.sort_values('timestamp')
                    
            except Exception as e:
                logger.error(f"Error loading historical data: {e}")
                historical_data = self._generate_synthetic_data()
        else:
            # Generate synthetic data for testing
            logger.warning("No historical data source provided, generating synthetic data")
            historical_data = self._generate_synthetic_data()
        
        return historical_data
    
    def _generate_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic market data for testing."""
        synthetic_data = {}
        
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        # Generate timestamps based on interval
        if self.config.data_interval == "5m":
            freq = "5T"
        elif self.config.data_interval == "15m":
            freq = "15T"
        elif self.config.data_interval == "1h":
            freq = "1H"
        else:
            freq = "5T"  # Default to 5 minutes
        
        timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        for pair in self.config.pairs:
            # Generate realistic price data with trends and volatility
            n_points = len(timestamps)
            
            # Base price
            if "BTC" in pair:
                base_price = 45000
                volatility = 0.03
            else:  # ETH
                base_price = 3000
                volatility = 0.04
            
            # Generate price series with trend and mean reversion
            returns = np.random.normal(0, volatility / np.sqrt(288), n_points)  # 5-min returns
            
            # Add some trend periods
            trend_periods = np.random.choice(n_points, size=n_points//10, replace=False)
            for period in trend_periods:
                trend_length = min(50, n_points - period)
                trend_direction = np.random.choice([-1, 1])
                trend_strength = np.random.uniform(0.001, 0.003)
                returns[period:period+trend_length] += trend_direction * trend_strength
            
            # Calculate prices
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Generate OHLCV data
            data = []
            for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
                # Generate realistic OHLC from close price
                volatility_factor = np.random.uniform(0.5, 1.5)
                high = close * (1 + abs(returns[i]) * volatility_factor)
                low = close * (1 - abs(returns[i]) * volatility_factor)
                
                if i == 0:
                    open_price = close
                else:
                    open_price = prices[i-1]
                
                volume = np.random.randint(1000, 10000)
                
                data.append({
                    'timestamp': timestamp,
                    'pair': pair,
                    'open': open_price,
                    'high': max(open_price, high, close),
                    'low': min(open_price, low, close),
                    'close': close,
                    'volume': volume
                })
            
            synthetic_data[pair] = pd.DataFrame(data)
        
        logger.info(f"Generated synthetic data for {len(self.config.pairs)} pairs")
        return synthetic_data
    
    def run_backtest(self) -> BacktestResult:
        """
        Run complete backtesting simulation.
        
        Returns:
            BacktestResult with comprehensive results
        """
        logger.info("Starting backtesting simulation")
        
        # Load historical data
        historical_data = self.load_historical_data()
        
        if not historical_data:
            raise ValueError("No historical data available for backtesting")
        
        # Get the primary pair for main loop (first pair in config)
        primary_pair = self.config.pairs[0]
        primary_data = historical_data[primary_pair]
        
        # Initialize tracking
        self.trade_records = []
        self.portfolio_history = []
        self.timestamps = []
        self.returns = []
        
        # Main backtesting loop
        for i, row in primary_data.iterrows():
            current_time = row['timestamp']
            self.timestamps.append(current_time)
            
            # Process each trading pair
            for pair in self.config.pairs:
                if pair not in historical_data:
                    continue
                
                pair_data = historical_data[pair]
                
                # Find current data point for this pair
                current_data = pair_data[pair_data['timestamp'] <= current_time]
                if current_data.empty:
                    continue
                
                current_row = current_data.iloc[-1]
                
                # Get recent data for feature calculation (last 100 points)
                recent_data = current_data.tail(100)
                
                if len(recent_data) < 20:  # Need minimum data for indicators
                    continue
                
                # Generate features
                try:
                    features = self._generate_features(recent_data, pair)
                    if features is None:
                        continue
                    
                    # Get agent decision
                    action = self.agent.predict(features, deterministic=True)
                    confidence = self.agent.get_confidence(features)
                    
                    # Apply confidence threshold
                    if confidence < self.config.confidence_threshold:
                        continue
                    
                    # Current market state
                    current_price = current_row['close']
                    market_regime = int(features[-1])  # Last feature is regime
                    
                    # Risk management check
                    portfolio_state = self.portfolio.get_state()
                    
                    # Create trade proposal
                    trade_proposal = {
                        'pair': pair,
                        'action': action,
                        'current_price': current_price,
                        'confidence': confidence,
                        'timestamp': current_time,
                        'market_regime': market_regime
                    }
                    
                    # Validate trade through risk manager
                    if self.risk_manager.validate_trade(action, portfolio_state):
                        # Execute trade
                        execution_result = self.execution_engine.execute_trade(
                            action=action,
                            pair=pair,
                            current_price=current_price,
                            portfolio=self.portfolio
                        )
                        
                        if execution_result['success']:
                            # Record trade
                            trade_record = self._create_trade_record(
                                trade_proposal, execution_result, current_time
                            )
                            self.trade_records.append(trade_record)
                
                except Exception as e:
                    logger.warning(f"Error processing {pair} at {current_time}: {e}")
                    continue
            
            # Update portfolio with current prices
            self._update_portfolio_values(historical_data, current_time)
            
            # Record portfolio state
            portfolio_state = self.portfolio.get_state()
            self.portfolio_history.append({
                'timestamp': current_time,
                'balance': portfolio_state['balance'],
                'total_value': portfolio_state['total_value'],
                'unrealized_pnl': portfolio_state['unrealized_pnl'],
                'positions': len(portfolio_state['positions'])
            })
            
            # Calculate return for this period
            if len(self.portfolio_history) > 1:
                prev_value = self.portfolio_history[-2]['total_value']
                current_value = portfolio_state['total_value']
                period_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
                self.returns.append(period_return)
            else:
                self.returns.append(0.0)
        
        # Generate comprehensive results
        result = self._generate_backtest_result()
        
        logger.info(f"Backtesting completed. Total trades: {len(self.trade_records)}")
        logger.info(f"Final portfolio value: {self.portfolio.get_state()['total_value']:.2f}")
        
        return result
    
    def _generate_features(self, data: pd.DataFrame, pair: str) -> Optional[np.ndarray]:
        """Generate feature vector from market data."""
        try:
            # Use the same feature engineering as live trading
            features = self.feature_engine.process_market_data(data, pair)
            return features
        except Exception as e:
            logger.warning(f"Error generating features for {pair}: {e}")
            return None
    
    def _update_portfolio_values(self, historical_data: Dict[str, pd.DataFrame], current_time: datetime) -> None:
        """Update portfolio positions with current market prices."""
        portfolio_state = self.portfolio.get_state()
        
        for position_id, position in portfolio_state['positions'].items():
            pair = position.pair
            
            if pair in historical_data:
                pair_data = historical_data[pair]
                current_data = pair_data[pair_data['timestamp'] <= current_time]
                
                if not current_data.empty:
                    current_price = current_data.iloc[-1]['close']
                    self.portfolio.update_position_price(position_id, current_price)
    
    def _create_trade_record(
        self,
        trade_proposal: Dict[str, Any],
        execution_result: Dict[str, Any],
        timestamp: datetime
    ) -> TradeRecord:
        """Create a trade record from execution results."""
        action = trade_proposal['action']
        
        # Determine trade side
        side = 'long' if action[0] > 0 else 'short'
        
        # Calculate position details
        position_size = action[1]  # Already scaled to [0, 0.1]
        leverage = action[2]       # Already scaled to [1, 12]
        
        return TradeRecord(
            timestamp=timestamp,
            pair=trade_proposal['pair'],
            side=side,
            entry_price=execution_result.get('execution_price', trade_proposal['current_price']),
            exit_price=0.0,  # Will be updated when position is closed
            position_size=position_size,
            leverage=leverage,
            pnl=0.0,  # Will be calculated when position is closed
            pnl_percent=0.0,
            duration=timedelta(0),
            confidence=trade_proposal['confidence'],
            market_regime=trade_proposal['market_regime'],
            entry_reason='model_signal',
            exit_reason='pending'
        )
    
    def _generate_backtest_result(self) -> BacktestResult:
        """Generate comprehensive backtest results."""
        # Calculate performance metrics
        returns_array = np.array(self.returns)
        performance_metrics = self.performance_evaluator.evaluate_returns(
            returns_array, self.timestamps
        )
        
        # Regime-specific analysis
        regime_analysis = self._analyze_regime_performance()
        
        # Execution summary
        execution_summary = {
            'total_trades': len(self.trade_records),
            'successful_trades': len([t for t in self.trade_records if t.pnl > 0]),
            'failed_trades': len([t for t in self.trade_records if t.pnl < 0]),
            'avg_confidence': np.mean([t.confidence for t in self.trade_records]) if self.trade_records else 0,
            'pairs_traded': list(set([t.pair for t in self.trade_records])),
            'regime_distribution': {
                'trending': len([t for t in self.trade_records if t.market_regime == 1]),
                'ranging': len([t for t in self.trade_records if t.market_regime == 0])
            }
        }
        
        return BacktestResult(
            config=self.config,
            performance_metrics=asdict(performance_metrics),
            trade_records=self.trade_records,
            portfolio_history=self.portfolio_history,
            regime_analysis=regime_analysis,
            execution_summary=execution_summary,
            timestamps=self.timestamps,
            returns=self.returns
        )
    
    def _analyze_regime_performance(self) -> Dict[str, Any]:
        """Analyze performance across different market regimes."""
        trending_trades = [t for t in self.trade_records if t.market_regime == 1]
        ranging_trades = [t for t in self.trade_records if t.market_regime == 0]
        
        def calculate_regime_metrics(trades: List[TradeRecord]) -> Dict[str, float]:
            if not trades:
                return {'count': 0, 'win_rate': 0, 'avg_pnl': 0, 'total_pnl': 0}
            
            winning_trades = [t for t in trades if t.pnl > 0]
            return {
                'count': len(trades),
                'win_rate': len(winning_trades) / len(trades),
                'avg_pnl': np.mean([t.pnl for t in trades]),
                'total_pnl': sum([t.pnl for t in trades]),
                'avg_confidence': np.mean([t.confidence for t in trades])
            }
        
        return {
            'trending_market': calculate_regime_metrics(trending_trades),
            'ranging_market': calculate_regime_metrics(ranging_trades),
            'regime_switching_frequency': self._calculate_regime_switches()
        }
    
    def _calculate_regime_switches(self) -> int:
        """Calculate how often market regime switched during backtest."""
        if len(self.trade_records) < 2:
            return 0
        
        switches = 0
        prev_regime = self.trade_records[0].market_regime
        
        for trade in self.trade_records[1:]:
            if trade.market_regime != prev_regime:
                switches += 1
                prev_regime = trade.market_regime
        
        return switches
    
    def save_results(self, result: BacktestResult, save_dir: str) -> None:
        """
        Save backtest results to files.
        
        Args:
            result: BacktestResult to save
            save_dir: Directory to save results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete results as JSON
        results_path = save_path / "backtest_results.json"
        with open(results_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Save trades as CSV
        if result.trade_records:
            trades_df = pd.DataFrame([asdict(trade) for trade in result.trade_records])
            trades_path = save_path / "backtest_trades.csv"
            trades_df.to_csv(trades_path, index=False)
        
        # Save portfolio history
        if result.portfolio_history:
            portfolio_df = pd.DataFrame(result.portfolio_history)
            portfolio_path = save_path / "portfolio_history.csv"
            portfolio_df.to_csv(portfolio_path, index=False)
        
        # Generate performance report
        self.performance_evaluator.generate_performance_report(
            returns=np.array(result.returns),
            timestamps=result.timestamps,
            trades_csv_path=str(save_path / "backtest_trades.csv") if result.trade_records else None,
            save_dir=str(save_path / "performance_analysis"),
            create_plots=True
        )
        
        logger.info(f"Backtest results saved to {save_path}")


class MockExecutionEngine:
    """Mock execution engine for backtesting that simulates real execution."""
    
    def __init__(self, transaction_cost_percent: float = 0.05, slippage_percent: float = 0.02):
        self.transaction_cost_percent = transaction_cost_percent / 100
        self.slippage_percent = slippage_percent / 100
    
    def execute_trade(
        self,
        action: np.ndarray,
        pair: str,
        current_price: float,
        portfolio: Portfolio
    ) -> Dict[str, Any]:
        """
        Simulate trade execution with realistic costs and slippage.
        
        Args:
            action: Agent action [direction, size, leverage]
            pair: Trading pair
            current_price: Current market price
            portfolio: Portfolio instance
            
        Returns:
            Execution result dictionary
        """
        try:
            direction = action[0]  # [-1, 1]
            position_size = action[1]  # [0, 0.1]
            leverage = action[2]  # [1, 12]
            
            # Determine trade side
            side = 'long' if direction > 0 else 'short'
            
            # Calculate position value
            portfolio_state = portfolio.get_state()
            position_value = portfolio_state['total_value'] * position_size
            
            # Apply leverage
            notional_value = position_value * leverage
            
            # Simulate slippage
            slippage_factor = np.random.uniform(-self.slippage_percent, self.slippage_percent)
            execution_price = current_price * (1 + slippage_factor)
            
            # Calculate transaction costs
            transaction_cost = notional_value * self.transaction_cost_percent
            
            # Create position
            position = Position(
                pair=pair,
                side=side,
                size=position_value,
                leverage=leverage,
                entry_price=execution_price,
                current_price=execution_price,
                unrealized_pnl=0.0,
                timestamp=datetime.now()
            )
            
            # Add position to portfolio
            position_id = portfolio.add_position(position)
            
            # Deduct transaction costs
            portfolio.update_balance(-transaction_cost)
            
            return {
                'success': True,
                'position_id': position_id,
                'execution_price': execution_price,
                'transaction_cost': transaction_cost,
                'slippage': slippage_factor,
                'notional_value': notional_value
            }
            
        except Exception as e:
            logger.error(f"Mock execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


if __name__ == "__main__":
    # Example usage
    from models.ppo_agent import PPOAgent
    
    # Initialize agent (would normally load trained model)
    agent = PPOAgent()
    
    # Configure backtest
    config = BacktestConfig(
        start_date="2023-01-01",
        end_date="2023-03-31",
        initial_balance=1000.0,
        pairs=["BTCUSDT", "ETHUSDT"]
    )
    
    # Run backtest
    engine = BacktestingEngine(agent, config)
    results = engine.run_backtest()
    
    # Save results
    engine.save_results(results, "logs/backtest_results")
    
    print(f"Backtest completed!")
    print(f"Total trades: {results.execution_summary['total_trades']}")
    print(f"Final portfolio value: {results.portfolio_history[-1]['total_value']:.2f}")