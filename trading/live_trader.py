"""
Live Trading Orchestrator for AlphaPulse-RL Trading System

This module implements the LiveTrader class that orchestrates the complete trading pipeline:
- Data fetching → feature generation → agent prediction → risk validation → execution
- Confidence-based trade filtering (only trade when confidence > 0.8)
- Comprehensive logging and monitoring
"""

import asyncio
import time
import logging
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import yaml

from data.weex_fetcher import WeexDataFetcher, MarketData
from data.feature_engineering import FeatureEngine, FeatureVector
from models.ppo_agent import PPOAgent
from risk.risk_manager import RiskManager, RiskMetrics
from trading.execution import ExecutionEngine, ExecutionResult
from trading.portfolio import PortfolioState, PortfolioManager, TradeRecord
from trading.logging_system import ComprehensiveLogger, setup_trading_logging

logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Main orchestrator for live trading operations.
    
    Implements the complete trading pipeline:
    1. Fetch latest market data
    2. Generate features
    3. Get agent prediction
    4. Apply trading filters (confidence > 0.8)
    5. Risk validation
    6. Execute trade
    7. Log decision and results
    
    Requirements implemented:
    - 1.5: Confidence-based trade filtering (only trade when confidence > 0.8)
    - 4.3: Complete trading pipeline orchestration
    - 4.4: Integration of all system components
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize LiveTrader with configuration."""
        self.config = self._load_config(config_path)
        
        # Trading parameters
        self.trading_pairs = self.config.get('trading', {}).get('pairs', ['BTCUSDT', 'ETHUSDT'])
        self.trading_interval = self.config.get('trading', {}).get('interval_seconds', 30)
        self.confidence_threshold = self.config.get('trading', {}).get('confidence_threshold', 0.8)
        self.max_positions = self.config.get('trading', {}).get('max_positions', 2)
        
        # Initialize components
        self.data_fetcher = WeexDataFetcher(self.config.get('data_fetcher', {}))
        self.feature_engine = FeatureEngine(self.config.get('feature_engine', {}))
        self.ppo_agent = self._initialize_agent()
        self.risk_manager = RiskManager(
            config_path=self.config.get('risk', {}).get('config_path', 'config/trading_params.yaml')
        )
        
        # Portfolio management
        self.portfolio_manager = PortfolioManager(self.config)
        self.portfolio = self.portfolio_manager.load_portfolio(
            initial_balance=self.config.get('portfolio', {}).get('initial_balance', 1000.0)
        )
        
        # Execution engine
        self.execution_engine = ExecutionEngine(self.config, self.portfolio)
        
        # Comprehensive logging system
        self.comprehensive_logger = ComprehensiveLogger(self.config.get('logging', {}))
        
        # Setup system logging
        setup_trading_logging(
            log_level=self.config.get('logging', {}).get('level', 'INFO'),
            log_file=self.config.get('logging', {}).get('system_log_file', 'logs/alphapulse.log')
        )
        
        # Trading state
        self.is_trading = False
        self.trading_task: Optional[asyncio.Task] = None
        self.last_trade_times: Dict[str, datetime] = {}
        self.trade_cooldown = timedelta(minutes=self.config.get('trading', {}).get('cooldown_minutes', 5))
        
        # Logging setup (legacy - now handled by comprehensive logger)
        self._setup_legacy_logging()
        
        logger.info(f"LiveTrader initialized for pairs: {self.trading_pairs}")
        logger.info(f"Trading interval: {self.trading_interval}s, Confidence threshold: {self.confidence_threshold}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file loading fails."""
        return {
            'trading': {
                'pairs': ['BTCUSDT', 'ETHUSDT'],
                'interval_seconds': 30,
                'confidence_threshold': 0.8,
                'max_positions': 2,
                'cooldown_minutes': 5
            },
            'portfolio': {
                'initial_balance': 1000.0
            },
            'logging': {
                'trade_history_file': 'logs/trades.csv',
                'ai_decisions_file': 'logs/ai_decisions.json',
                'portfolio_metrics_file': 'logs/portfolio_metrics.json'
            }
        }
    
    def _initialize_agent(self) -> PPOAgent:
        """Initialize PPO agent and load trained model if available."""
        agent = PPOAgent(
            state_dim=9,
            action_dim=3,
            lr_actor=self.config.get('agent', {}).get('lr_actor', 3e-4),
            lr_critic=self.config.get('agent', {}).get('lr_critic', 1e-3),
            device=self.config.get('agent', {}).get('device', 'cpu')
        )
        
        # Load trained model if available
        model_path = self.config.get('agent', {}).get('model_path', 'models/ppo_agent.pth')
        if Path(model_path).exists():
            try:
                agent.load_model(model_path)
                logger.info(f"Loaded trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
                logger.info("Using randomly initialized model")
        else:
            logger.info("No trained model found, using randomly initialized model")
        
        return agent
    
    def _setup_legacy_logging(self):
        """Setup legacy logging files and directories (for backward compatibility)."""
        log_config = self.config.get('logging', {})
        
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        # Legacy CSV trade history file
        self.trade_history_file = log_config.get('trade_history_file', 'logs/trades.csv')
        self._initialize_trade_history_csv()
        
        # Legacy AI decisions log file
        self.ai_decisions_file = log_config.get('ai_decisions_file', 'logs/ai_decisions.json')
        
        # Legacy portfolio metrics log file
        self.portfolio_metrics_file = log_config.get('portfolio_metrics_file', 'logs/portfolio_metrics.json')
    
    def _initialize_trade_history_csv(self):
        """Initialize CSV file for trade history if it doesn't exist."""
        if not Path(self.trade_history_file).exists():
            with open(self.trade_history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'pair', 'action_direction', 'action_size', 'action_leverage',
                    'entry_price', 'exit_price', 'pnl', 'confidence', 'market_regime',
                    'portfolio_balance', 'total_pnl', 'daily_pnl', 'max_drawdown',
                    'risk_score', 'execution_success', 'reason'
                ])
    
    async def start_trading(self):
        """Start the live trading loop."""
        if self.is_trading:
            logger.warning("Trading is already active")
            return
        
        self.is_trading = True
        logger.info("Starting live trading...")
        
        try:
            # Sync portfolio with exchange before starting
            await self.execution_engine.sync_portfolio_with_exchange()
            
            # Start the main trading loop
            self.trading_task = asyncio.create_task(self._trading_loop())
            await self.trading_task
            
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
        finally:
            self.is_trading = False
    
    async def stop_trading(self):
        """Stop the live trading loop."""
        if not self.is_trading:
            logger.warning("Trading is not active")
            return
        
        logger.info("Stopping live trading...")
        self.is_trading = False
        
        if self.trading_task:
            self.trading_task.cancel()
            try:
                await self.trading_task
            except asyncio.CancelledError:
                pass
        
        # Save portfolio state
        self.portfolio_manager.save_portfolio(self.portfolio)
        logger.info("Trading stopped and portfolio saved")
    
    async def _trading_loop(self):
        """Main trading loop that processes each trading pair."""
        while self.is_trading:
            try:
                # Process each trading pair
                for pair in self.trading_pairs:
                    if not self.is_trading:
                        break
                    
                    await self._process_trading_pair(pair)
                
                # Auto-save portfolio if needed
                if self.portfolio_manager.should_auto_save():
                    self.portfolio_manager.save_portfolio(self.portfolio)
                
                # Log system health periodically (every 10 iterations)
                if hasattr(self, '_health_check_counter'):
                    self._health_check_counter += 1
                else:
                    self._health_check_counter = 1
                
                if self._health_check_counter % 10 == 0:
                    await self._log_system_health()
                
                # Wait for next iteration
                await asyncio.sleep(self.trading_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(self.trading_interval)
    
    async def _process_trading_pair(self, pair: str):
        """Process a single trading pair through the complete pipeline."""
        try:
            # 1. Fetch latest market data
            async with self.data_fetcher as fetcher:
                market_data = await fetcher.get_latest_data(pair)
            
            if not market_data:
                logger.warning(f"No market data available for {pair}")
                return
            
            # 2. Generate features
            feature_vector = self.feature_engine.process_data(market_data)
            
            # 3. Get agent prediction and confidence
            state_array = feature_vector.to_array()
            action = self.ppo_agent.predict(state_array, deterministic=True)
            confidence = self.ppo_agent.get_confidence(state_array)
            
            # 4. Apply trading filters
            should_trade, filter_reason = self._apply_trading_filters(
                pair, action, confidence, feature_vector
            )
            
            if not should_trade:
                self._log_no_trade_decision(pair, action, confidence, feature_vector, filter_reason)
                return
            
            # 5. Risk validation
            current_price = market_data.close
            volatility = self._estimate_volatility(pair)
            
            is_valid, risk_reason = self.risk_manager.validate_trade(
                action.tolist(), self.portfolio, current_price, volatility, pair
            )
            
            if not is_valid:
                self._log_no_trade_decision(pair, action, confidence, feature_vector, f"Risk: {risk_reason}")
                return
            
            # 6. Execute trade
            execution_result = await self.execution_engine.execute_trade(
                action.tolist(), pair, current_price
            )
            
            # 7. Log decision and results
            await self._log_trade_decision(
                pair, action, confidence, feature_vector, execution_result, market_data
            )
            
            # Update last trade time
            if execution_result.success:
                self.last_trade_times[pair] = datetime.now()
                logger.info(f"Trade executed successfully for {pair}")
            else:
                logger.warning(f"Trade execution failed for {pair}: {execution_result.error_message}")
            
        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")
    
    def _apply_trading_filters(self, pair: str, action: np.ndarray, confidence: float, 
                             features: FeatureVector) -> Tuple[bool, str]:
        """
        Apply trading filters to determine if trade should be executed.
        
        Returns:
            Tuple of (should_trade, reason)
        """
        # 1. Confidence threshold filter (Requirement 1.5)
        if confidence < self.confidence_threshold:
            return False, f"Low confidence: {confidence:.3f} < {self.confidence_threshold}"
        
        # 2. Position size filter (must be meaningful)
        position_size = action[1]  # action[1] is position size
        if position_size < 0.001:  # Less than 0.1% position size
            return False, f"Position size too small: {position_size:.4f}"
        
        # 3. Cooldown filter
        if pair in self.last_trade_times:
            time_since_last = datetime.now() - self.last_trade_times[pair]
            if time_since_last < self.trade_cooldown:
                return False, f"Cooldown active: {time_since_last.total_seconds():.0f}s remaining"
        
        # 4. Maximum positions filter
        if len(self.portfolio.positions) >= self.max_positions and pair not in self.portfolio.positions:
            return False, f"Max positions reached: {len(self.portfolio.positions)}/{self.max_positions}"
        
        # 5. Market regime filter (optional - could skip trades in uncertain regimes)
        # For now, we trade in all regimes but log the regime
        
        return True, "All filters passed"
    
    def _estimate_volatility(self, pair: str) -> float:
        """Estimate current volatility for risk management."""
        try:
            if pair in self.feature_engine.price_history:
                history = self.feature_engine.price_history[pair]
                if len(history) >= 20:
                    returns = history['close'].pct_change().dropna()
                    if len(returns) > 0:
                        return float(returns.std())
            return 0.02  # Default volatility estimate
        except Exception as e:
            logger.warning(f"Volatility estimation failed for {pair}: {e}")
            return 0.02
    
    def _log_no_trade_decision(self, pair: str, action: np.ndarray, confidence: float,
                              features: FeatureVector, reason: str):
        """Log when a trade is not executed due to filters."""
        # Get current risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics(self.portfolio, self._estimate_volatility(pair))
        
        # Get current market data context
        market_data = {
            'pair': pair,
            'estimated_volatility': self._estimate_volatility(pair)
        }
        
        # Use comprehensive logger
        self.comprehensive_logger.log_trade_decision(
            pair=pair,
            action=action.tolist(),
            confidence=confidence,
            features=features,
            market_data=market_data,
            portfolio=self.portfolio,
            risk_metrics=risk_metrics,
            decision_type='NO_TRADE',
            reason=reason
        )
        
        # Legacy logging for backward compatibility
        decision_log = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'action': action.tolist(),
            'confidence': confidence,
            'features': features.to_dict(),
            'decision': 'NO_TRADE',
            'reason': reason,
            'portfolio_metrics': self.portfolio.calculate_portfolio_metrics()
        }
        
        self._append_to_json_log(self.ai_decisions_file, decision_log)
    
    async def _log_trade_decision(self, pair: str, action: np.ndarray, confidence: float,
                                features: FeatureVector, execution_result: ExecutionResult,
                                market_data: MarketData):
        """Log trade decision and execution results using comprehensive logging system."""
        # Get risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics(self.portfolio, self._estimate_volatility(pair))
        
        # Prepare market data for logging
        market_data_dict = {
            'pair': pair,
            'price': market_data.close,
            'volume': market_data.volume,
            'funding_rate': market_data.funding_rate,
            'orderbook_bids_count': len(market_data.orderbook_bids),
            'orderbook_asks_count': len(market_data.orderbook_asks),
            'timestamp': market_data.timestamp.isoformat()
        }
        
        # Prepare execution details
        execution_details = None
        if execution_result.order_response:
            execution_details = {
                'order_id': execution_result.order_response.order_id,
                'entry_price': execution_result.order_response.average_price or execution_result.order_response.price,
                'filled_quantity': execution_result.order_response.filled_quantity,
                'commission': execution_result.order_response.commission,
                'status': execution_result.order_response.status.value
            }
        
        # Determine decision type
        decision_type = 'TRADE_EXECUTED' if execution_result.success else 'TRADE_FAILED'
        reason = execution_result.error_message if not execution_result.success else 'Trade executed successfully'
        
        # Use comprehensive logger
        self.comprehensive_logger.log_trade_decision(
            pair=pair,
            action=action.tolist(),
            confidence=confidence,
            features=features,
            market_data=market_data_dict,
            portfolio=self.portfolio,
            risk_metrics=risk_metrics,
            decision_type=decision_type,
            reason=reason,
            execution_details=execution_details
        )
        
        # Log portfolio metrics
        self.comprehensive_logger.log_portfolio_metrics(
            portfolio=self.portfolio,
            risk_metrics=risk_metrics,
            pair=pair
        )
        
        # Legacy logging for backward compatibility
        await self._legacy_log_trade_decision(pair, action, confidence, features, execution_result, market_data)
    
    async def _legacy_log_trade_decision(self, pair: str, action: np.ndarray, confidence: float,
                                       features: FeatureVector, execution_result: ExecutionResult,
                                       market_data: MarketData):
        """Legacy logging method for backward compatibility."""
        # Get risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics(self.portfolio, self._estimate_volatility(pair))
        
        # Create trade record for CSV
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'action_direction': action[0],
            'action_size': action[1],
            'action_leverage': action[2],
            'entry_price': execution_result.order_response.average_price if execution_result.order_response else market_data.close,
            'exit_price': None,  # Will be filled when position is closed
            'pnl': None,  # Will be calculated when position is closed
            'confidence': confidence,
            'market_regime': features.volatility_regime,
            'portfolio_balance': self.portfolio.balance,
            'total_pnl': self.portfolio.total_pnl,
            'daily_pnl': self.portfolio.daily_pnl,
            'max_drawdown': self.portfolio.max_drawdown,
            'risk_score': risk_metrics.risk_score,
            'execution_success': execution_result.success,
            'reason': execution_result.error_message if not execution_result.success else 'SUCCESS'
        }
        
        # Append to CSV
        self._append_to_csv(self.trade_history_file, trade_record)
        
        # Create detailed AI decision log
        ai_decision_log = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'action': action.tolist(),
            'confidence': confidence,
            'features': features.to_dict(),
            'market_data': {
                'price': market_data.close,
                'volume': market_data.volume,
                'funding_rate': market_data.funding_rate
            },
            'risk_metrics': {
                'current_drawdown': risk_metrics.current_drawdown,
                'daily_pnl_percent': risk_metrics.daily_pnl_percent,
                'position_exposure_percent': risk_metrics.position_exposure_percent,
                'total_leverage': risk_metrics.total_leverage,
                'risk_score': risk_metrics.risk_score
            },
            'execution_result': {
                'success': execution_result.success,
                'order_id': execution_result.order_response.order_id if execution_result.order_response else None,
                'filled_quantity': execution_result.order_response.filled_quantity if execution_result.order_response else 0,
                'error_message': execution_result.error_message
            },
            'portfolio_state': self.portfolio.to_dict(),
            'decision': 'TRADE_EXECUTED' if execution_result.success else 'TRADE_FAILED'
        }
        
        # Append to AI decisions log
        self._append_to_json_log(self.ai_decisions_file, ai_decision_log)
        
        # Log portfolio metrics
        portfolio_metrics = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'metrics': self.portfolio.calculate_portfolio_metrics(),
            'risk_metrics': {
                'current_drawdown': risk_metrics.current_drawdown,
                'daily_pnl_percent': risk_metrics.daily_pnl_percent,
                'risk_score': risk_metrics.risk_score
            }
        }
        
        self._append_to_json_log(self.portfolio_metrics_file, portfolio_metrics)
    
    def _append_to_csv(self, filename: str, record: Dict[str, Any]):
        """Append record to CSV file."""
        try:
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    record['timestamp'], record['pair'], record['action_direction'],
                    record['action_size'], record['action_leverage'], record['entry_price'],
                    record['exit_price'], record['pnl'], record['confidence'],
                    record['market_regime'], record['portfolio_balance'], record['total_pnl'],
                    record['daily_pnl'], record['max_drawdown'], record['risk_score'],
                    record['execution_success'], record['reason']
                ])
        except Exception as e:
            logger.error(f"Failed to append to CSV {filename}: {e}")
    
    def _append_to_json_log(self, filename: str, record: Dict[str, Any]):
        """Append record to JSON log file."""
        try:
            # Read existing data
            if Path(filename).exists():
                with open(filename, 'r') as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = []
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []
            
            # Append new record
            data.append(record)
            
            # Keep only recent records to prevent file from growing too large
            max_records = self.config.get('logging', {}).get('max_json_records', 10000)
            if len(data) > max_records:
                data = data[-max_records:]
            
            # Write back to file
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to append to JSON log {filename}: {e}")
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status and metrics."""
        # Get comprehensive logging metrics
        real_time_metrics = self.comprehensive_logger.get_real_time_metrics()
        trade_statistics = self.comprehensive_logger.get_trade_statistics(24)
        
        return {
            'is_trading': self.is_trading,
            'trading_pairs': self.trading_pairs,
            'confidence_threshold': self.confidence_threshold,
            'portfolio_metrics': self.portfolio.calculate_portfolio_metrics(),
            'risk_metrics': self.risk_manager.get_risk_metrics(self.portfolio).__dict__,
            'active_positions': len(self.portfolio.positions),
            'last_trade_times': {pair: time.isoformat() for pair, time in self.last_trade_times.items()},
            'emergency_mode': self.risk_manager.emergency_mode,
            'real_time_metrics': real_time_metrics,
            'trade_statistics_24h': trade_statistics
        }
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trade records."""
        try:
            if not Path(self.ai_decisions_file).exists():
                return []
            
            with open(self.ai_decisions_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Filter for actual trades and return most recent
                    trades = [record for record in data if record.get('decision') in ['TRADE_EXECUTED', 'TRADE_FAILED']]
                    return trades[-limit:] if len(trades) > limit else trades
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []
    
    async def emergency_stop(self):
        """Emergency stop - close all positions and stop trading."""
        logger.critical("EMERGENCY STOP INITIATED")
        
        try:
            # Stop trading loop
            await self.stop_trading()
            
            # Close all positions
            positions_to_close = self.risk_manager.emergency_flatten_positions(self.portfolio)
            
            for pair in positions_to_close:
                try:
                    result = await self.execution_engine.close_position(pair)
                    if result.success:
                        logger.info(f"Emergency closed position: {pair}")
                    else:
                        logger.error(f"Failed to emergency close {pair}: {result.error_message}")
                except Exception as e:
                    logger.error(f"Error closing position {pair}: {e}")
            
            # Save portfolio state
            self.portfolio_manager.save_portfolio(self.portfolio)
            
            logger.critical("EMERGENCY STOP COMPLETED")
            
        except Exception as e:
            logger.critical(f"EMERGENCY STOP FAILED: {e}")
    
    def reset_daily_metrics(self):
        """Reset daily metrics at start of new trading day."""
        self.portfolio.reset_daily_metrics()
        self.portfolio_manager.save_portfolio(self.portfolio)
        logger.info("Daily metrics reset for new trading day")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'HEALTHY',
            'components': {}
        }
        
        try:
            # Check data fetcher
            async with self.data_fetcher as fetcher:
                test_data = await fetcher.get_latest_price('BTCUSDT')
                health_status['components']['data_fetcher'] = 'HEALTHY' if test_data else 'UNHEALTHY'
        except Exception as e:
            health_status['components']['data_fetcher'] = f'UNHEALTHY: {e}'
        
        # Check portfolio state
        try:
            validation = self.portfolio_manager.validate_portfolio_integrity(self.portfolio)
            health_status['components']['portfolio'] = 'HEALTHY' if validation['is_valid'] else f"UNHEALTHY: {validation['issues']}"
        except Exception as e:
            health_status['components']['portfolio'] = f'UNHEALTHY: {e}'
        
        # Check risk manager
        try:
            is_allowed, reason = self.risk_manager.is_trading_allowed(self.portfolio)
            health_status['components']['risk_manager'] = 'HEALTHY' if is_allowed else f'WARNING: {reason}'
        except Exception as e:
            health_status['components']['risk_manager'] = f'UNHEALTHY: {e}'
        
        # Check model
        try:
            test_state = np.zeros(9)
            test_action = self.ppo_agent.predict(test_state)
            test_confidence = self.ppo_agent.get_confidence(test_state)
            health_status['components']['ppo_agent'] = 'HEALTHY'
        except Exception as e:
            health_status['components']['ppo_agent'] = f'UNHEALTHY: {e}'
        
        # Determine overall status
        unhealthy_components = [comp for comp, status in health_status['components'].items() 
                              if status.startswith('UNHEALTHY')]
        
        if unhealthy_components:
            health_status['overall_status'] = 'UNHEALTHY'
        elif any(status.startswith('WARNING') for status in health_status['components'].values()):
            health_status['overall_status'] = 'WARNING'
        
        return health_status
    
    async def _log_system_health(self):
        """Log system health status for monitoring."""
        try:
            # Check data fetcher health
            try:
                async with self.data_fetcher as fetcher:
                    test_data = await fetcher.get_latest_price('BTCUSDT')
                    data_fetcher_status = 'HEALTHY' if test_data else 'UNHEALTHY'
                    data_fetcher_details = {'last_price_fetch': 'SUCCESS' if test_data else 'FAILED'}
            except Exception as e:
                data_fetcher_status = 'UNHEALTHY'
                data_fetcher_details = {'error': str(e)}
            
            self.comprehensive_logger.log_system_health(
                component='data_fetcher',
                status=data_fetcher_status,
                details=data_fetcher_details
            )
            
            # Check portfolio health
            try:
                validation = self.portfolio_manager.validate_portfolio_integrity(self.portfolio)
                portfolio_status = 'HEALTHY' if validation['is_valid'] else 'WARNING'
                portfolio_details = {
                    'validation_issues': validation.get('issues', []),
                    'validation_warnings': validation.get('warnings', [])
                }
                portfolio_metrics = {
                    'total_equity': self.portfolio.get_total_equity(),
                    'active_positions': len(self.portfolio.positions),
                    'daily_pnl_percent': (self.portfolio.daily_pnl / self.portfolio.daily_start_balance * 100) if self.portfolio.daily_start_balance > 0 else 0
                }
            except Exception as e:
                portfolio_status = 'UNHEALTHY'
                portfolio_details = {'error': str(e)}
                portfolio_metrics = {}
            
            self.comprehensive_logger.log_system_health(
                component='portfolio',
                status=portfolio_status,
                details=portfolio_details,
                metrics=portfolio_metrics
            )
            
            # Check risk manager health
            try:
                is_allowed, reason = self.risk_manager.is_trading_allowed(self.portfolio)
                risk_status = 'HEALTHY' if is_allowed else 'WARNING'
                risk_details = {'trading_allowed': is_allowed, 'reason': reason}
                risk_metrics_obj = self.risk_manager.get_risk_metrics(self.portfolio)
                risk_metrics = {
                    'risk_score': risk_metrics_obj.risk_score,
                    'current_drawdown': risk_metrics_obj.current_drawdown,
                    'daily_pnl_percent': risk_metrics_obj.daily_pnl_percent
                }
            except Exception as e:
                risk_status = 'UNHEALTHY'
                risk_details = {'error': str(e)}
                risk_metrics = {}
            
            self.comprehensive_logger.log_system_health(
                component='risk_manager',
                status=risk_status,
                details=risk_details,
                metrics=risk_metrics
            )
            
            # Check PPO agent health
            try:
                test_state = np.zeros(9)
                test_action = self.ppo_agent.predict(test_state)
                test_confidence = self.ppo_agent.get_confidence(test_state)
                agent_status = 'HEALTHY'
                agent_details = {'prediction_test': 'SUCCESS'}
                agent_metrics = {'test_confidence': test_confidence}
            except Exception as e:
                agent_status = 'UNHEALTHY'
                agent_details = {'error': str(e)}
                agent_metrics = {}
            
            self.comprehensive_logger.log_system_health(
                component='ppo_agent',
                status=agent_status,
                details=agent_details,
                metrics=agent_metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to log system health: {e}")


# Utility functions for easy usage
async def create_live_trader(config_path: str = "config/config.yaml") -> LiveTrader:
    """Factory function to create and initialize LiveTrader."""
    return LiveTrader(config_path)


async def run_live_trading(config_path: str = "config/config.yaml"):
    """Run live trading with proper error handling and cleanup."""
    trader = None
    try:
        trader = await create_live_trader(config_path)
        await trader.start_trading()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Live trading error: {e}")
    finally:
        if trader:
            await trader.stop_trading()


if __name__ == "__main__":
    import numpy as np
    
    # Run live trading
    asyncio.run(run_live_trading())