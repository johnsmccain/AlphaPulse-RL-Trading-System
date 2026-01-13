"""
Comprehensive Logging System for AlphaPulse-RL Trading System

This module implements detailed trade decision logging with timestamps, states, actions,
and reasoning. Creates CSV trade history and JSON AI decision logs as specified.
Provides real-time monitoring of portfolio metrics and system health.
"""

import json
import csv
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import deque, defaultdict

from trading.portfolio import PortfolioState, Position, TradeRecord
from risk.risk_manager import RiskMetrics
from data.feature_engineering import FeatureVector

logger = logging.getLogger(__name__)


@dataclass
class TradeDecisionLog:
    """Comprehensive trade decision log entry"""
    timestamp: datetime
    pair: str
    action: List[float]
    confidence: float
    features: Dict[str, float]
    market_data: Dict[str, Any]
    risk_metrics: Dict[str, float]
    portfolio_state: Dict[str, Any]
    decision_type: str  # 'TRADE_EXECUTED', 'TRADE_FAILED', 'NO_TRADE'
    reason: str
    execution_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'pair': self.pair,
            'action': self.action,
            'confidence': self.confidence,
            'features': self.features,
            'market_data': self.market_data,
            'risk_metrics': self.risk_metrics,
            'portfolio_state': self.portfolio_state,
            'decision_type': self.decision_type,
            'reason': self.reason,
            'execution_details': self.execution_details
        }


@dataclass
class PortfolioMetricsLog:
    """Portfolio metrics log entry"""
    timestamp: datetime
    pair: Optional[str]
    total_equity: float
    balance: float
    daily_pnl: float
    daily_pnl_percent: float
    total_pnl: float
    max_drawdown: float
    active_positions: int
    margin_utilization: float
    risk_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'pair': self.pair,
            'total_equity': self.total_equity,
            'balance': self.balance,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_percent': self.daily_pnl_percent,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'active_positions': self.active_positions,
            'margin_utilization': self.margin_utilization,
            'risk_score': self.risk_score
        }


@dataclass
class SystemHealthLog:
    """System health monitoring log entry"""
    timestamp: datetime
    component: str
    status: str  # 'HEALTHY', 'WARNING', 'UNHEALTHY'
    details: Dict[str, Any]
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'status': self.status,
            'details': self.details,
            'metrics': self.metrics
        }


class RealTimeMonitor:
    """Real-time monitoring of portfolio metrics and system health"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.portfolio_history = deque(maxlen=max_history)
        self.trade_history = deque(maxlen=max_history)
        self.health_history = deque(maxlen=max_history)
        self.performance_metrics = defaultdict(list)
        self._lock = threading.Lock()
    
    def add_portfolio_metrics(self, metrics: PortfolioMetricsLog):
        """Add portfolio metrics to real-time monitoring"""
        with self._lock:
            self.portfolio_history.append(metrics)
            
            # Update performance metrics
            self.performance_metrics['equity'].append(metrics.total_equity)
            self.performance_metrics['pnl'].append(metrics.total_pnl)
            self.performance_metrics['drawdown'].append(metrics.max_drawdown)
            self.performance_metrics['risk_score'].append(metrics.risk_score)
            
            # Keep only recent metrics
            for key in self.performance_metrics:
                if len(self.performance_metrics[key]) > self.max_history:
                    self.performance_metrics[key] = self.performance_metrics[key][-self.max_history:]
    
    def add_trade_decision(self, decision: TradeDecisionLog):
        """Add trade decision to real-time monitoring"""
        with self._lock:
            self.trade_history.append(decision)
    
    def add_health_status(self, health: SystemHealthLog):
        """Add system health status to real-time monitoring"""
        with self._lock:
            self.health_history.append(health)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        with self._lock:
            if not self.portfolio_history:
                return {}
            
            latest = self.portfolio_history[-1]
            
            # Calculate recent performance
            recent_equity = self.performance_metrics['equity'][-100:] if len(self.performance_metrics['equity']) >= 100 else self.performance_metrics['equity']
            recent_trades = list(self.trade_history)[-50:] if len(self.trade_history) >= 50 else list(self.trade_history)
            
            # Calculate win rate
            executed_trades = [t for t in recent_trades if t.decision_type == 'TRADE_EXECUTED']
            win_rate = 0.0
            if executed_trades:
                # This is a simplified win rate calculation
                # In practice, you'd need to track trade outcomes
                win_rate = len([t for t in executed_trades if t.confidence > 0.8]) / len(executed_trades)
            
            return {
                'current_equity': latest.total_equity,
                'daily_pnl_percent': latest.daily_pnl_percent,
                'max_drawdown': latest.max_drawdown,
                'risk_score': latest.risk_score,
                'active_positions': latest.active_positions,
                'recent_trades_count': len(recent_trades),
                'win_rate': win_rate,
                'equity_trend': self._calculate_trend(recent_equity),
                'last_update': latest.timestamp.isoformat()
            }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        if len(values) < 2:
            return 'NEUTRAL'
        
        recent_avg = np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
        older_avg = np.mean(values[-20:-10]) if len(values) >= 20 else np.mean(values[:-10]) if len(values) > 10 else recent_avg
        
        if recent_avg > older_avg * 1.01:
            return 'UPWARD'
        elif recent_avg < older_avg * 0.99:
            return 'DOWNWARD'
        else:
            return 'NEUTRAL'
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter recent portfolio metrics
            recent_metrics = [m for m in self.portfolio_history if m.timestamp >= cutoff_time]
            recent_trades = [t for t in self.trade_history if t.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {}
            
            start_equity = recent_metrics[0].total_equity
            end_equity = recent_metrics[-1].total_equity
            
            return {
                'period_hours': hours,
                'start_equity': start_equity,
                'end_equity': end_equity,
                'period_return': (end_equity - start_equity) / start_equity * 100 if start_equity > 0 else 0,
                'max_drawdown': max([m.max_drawdown for m in recent_metrics]),
                'avg_risk_score': np.mean([m.risk_score for m in recent_metrics]),
                'trades_executed': len([t for t in recent_trades if t.decision_type == 'TRADE_EXECUTED']),
                'trades_rejected': len([t for t in recent_trades if t.decision_type == 'NO_TRADE']),
                'avg_confidence': np.mean([t.confidence for t in recent_trades]) if recent_trades else 0
            }


class ComprehensiveLogger:
    """
    Comprehensive logging system for AlphaPulse-RL trading system.
    
    Implements:
    - Detailed trade decision logging with timestamps, states, actions, and reasoning
    - CSV trade history and JSON AI decision logs
    - Real-time monitoring of portfolio metrics and system health
    
    Requirements implemented:
    - 3.1: Log every trading decision with timestamp, market pair, state vector, and action details
    - 3.2: Record reasoning for each trade including model confidence and market regime
    - 3.3: Maintain trade history in CSV format with PnL tracking
    - 3.4: Store AI decision logs in JSON format for detailed analysis
    - 3.5: Provide real-time monitoring of portfolio metrics and risk parameters
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_dir = Path(config.get('log_directory', 'logs'))
        self.log_dir.mkdir(exist_ok=True)
        
        # Log file paths
        self.trade_history_csv = self.log_dir / config.get('trade_history_file', 'trades.csv')
        self.ai_decisions_json = self.log_dir / config.get('ai_decisions_file', 'ai_decisions.json')
        self.portfolio_metrics_json = self.log_dir / config.get('portfolio_metrics_file', 'portfolio_metrics.json')
        self.system_health_json = self.log_dir / config.get('system_health_file', 'system_health.json')
        
        # Configuration
        self.max_json_records = config.get('max_json_records', 10000)
        self.enable_real_time_monitoring = config.get('enable_real_time_monitoring', True)
        
        # Real-time monitor
        self.real_time_monitor = RealTimeMonitor() if self.enable_real_time_monitoring else None
        
        # Initialize log files
        self._initialize_csv_files()
        
        # Thread safety
        self._csv_lock = threading.Lock()
        self._json_lock = threading.Lock()
        
        logger.info(f"ComprehensiveLogger initialized with log directory: {self.log_dir}")
    
    def _initialize_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        # Trade history CSV
        if not self.trade_history_csv.exists():
            with open(self.trade_history_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'pair', 'action_direction', 'action_size', 'action_leverage',
                    'confidence', 'market_regime', 'entry_price', 'exit_price', 'pnl',
                    'portfolio_balance', 'total_pnl', 'daily_pnl', 'max_drawdown',
                    'risk_score', 'decision_type', 'reason', 'features_json'
                ])
    
    def log_trade_decision(self, pair: str, action: List[float], confidence: float,
                          features: FeatureVector, market_data: Dict[str, Any],
                          portfolio: PortfolioState, risk_metrics: RiskMetrics,
                          decision_type: str, reason: str,
                          execution_details: Optional[Dict[str, Any]] = None):
        """
        Log a comprehensive trade decision.
        
        Args:
            pair: Trading pair
            action: Agent action [direction, size, leverage]
            confidence: Model confidence score
            features: Feature vector used for decision
            market_data: Current market data
            portfolio: Current portfolio state
            risk_metrics: Current risk metrics
            decision_type: 'TRADE_EXECUTED', 'TRADE_FAILED', 'NO_TRADE'
            reason: Reason for the decision
            execution_details: Optional execution details
        """
        timestamp = datetime.now()
        
        # Create comprehensive log entry
        decision_log = TradeDecisionLog(
            timestamp=timestamp,
            pair=pair,
            action=action,
            confidence=confidence,
            features=features.to_dict(),
            market_data=market_data,
            risk_metrics={
                'current_drawdown': risk_metrics.current_drawdown,
                'daily_pnl_percent': risk_metrics.daily_pnl_percent,
                'position_exposure_percent': risk_metrics.position_exposure_percent,
                'total_leverage': risk_metrics.total_leverage,
                'margin_utilization': risk_metrics.margin_utilization,
                'volatility_level': risk_metrics.volatility_level,
                'risk_score': risk_metrics.risk_score
            },
            portfolio_state=portfolio.to_dict(),
            decision_type=decision_type,
            reason=reason,
            execution_details=execution_details
        )
        
        # Log to CSV
        self._log_to_csv(decision_log, portfolio)
        
        # Log to JSON
        self._log_to_json(self.ai_decisions_json, decision_log.to_dict())
        
        # Add to real-time monitoring
        if self.real_time_monitor:
            self.real_time_monitor.add_trade_decision(decision_log)
        
        logger.info(f"Trade decision logged: {pair} {decision_type} - {reason}")
    
    def _log_to_csv(self, decision_log: TradeDecisionLog, portfolio: PortfolioState):
        """Log trade decision to CSV file"""
        with self._csv_lock:
            try:
                with open(self.trade_history_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Extract execution details
                    entry_price = 0.0
                    exit_price = None
                    pnl = None
                    
                    if decision_log.execution_details:
                        entry_price = decision_log.execution_details.get('entry_price', 0.0)
                        exit_price = decision_log.execution_details.get('exit_price')
                        pnl = decision_log.execution_details.get('pnl')
                    
                    writer.writerow([
                        decision_log.timestamp.isoformat(),
                        decision_log.pair,
                        decision_log.action[0],  # direction
                        decision_log.action[1],  # size
                        decision_log.action[2],  # leverage
                        decision_log.confidence,
                        decision_log.features.get('volatility_regime', 0),
                        entry_price,
                        exit_price,
                        pnl,
                        portfolio.balance,
                        portfolio.total_pnl,
                        portfolio.daily_pnl,
                        portfolio.max_drawdown,
                        decision_log.risk_metrics.get('risk_score', 0),
                        decision_log.decision_type,
                        decision_log.reason,
                        json.dumps(decision_log.features)
                    ])
                    
            except Exception as e:
                logger.error(f"Failed to log to CSV: {e}")
    
    def _log_to_json(self, filename: Path, record: Dict[str, Any]):
        """Log record to JSON file with rotation"""
        with self._json_lock:
            try:
                # Read existing data
                if filename.exists():
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
                
                # Rotate if too many records
                if len(data) > self.max_json_records:
                    data = data[-self.max_json_records:]
                
                # Write back to file
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
            except Exception as e:
                logger.error(f"Failed to log to JSON {filename}: {e}")
    
    def log_portfolio_metrics(self, portfolio: PortfolioState, risk_metrics: RiskMetrics,
                            pair: Optional[str] = None):
        """
        Log portfolio metrics for monitoring.
        
        Args:
            portfolio: Current portfolio state
            risk_metrics: Current risk metrics
            pair: Optional trading pair context
        """
        timestamp = datetime.now()
        portfolio_metrics = portfolio.calculate_portfolio_metrics()
        
        metrics_log = PortfolioMetricsLog(
            timestamp=timestamp,
            pair=pair,
            total_equity=portfolio_metrics['total_equity'],
            balance=portfolio_metrics['balance'],
            daily_pnl=portfolio_metrics['daily_pnl'],
            daily_pnl_percent=portfolio_metrics['daily_pnl_pct'],
            total_pnl=portfolio_metrics['total_pnl'],
            max_drawdown=portfolio_metrics['max_drawdown_pct'],
            active_positions=portfolio_metrics['active_positions'],
            margin_utilization=portfolio_metrics['margin_utilization_pct'],
            risk_score=risk_metrics.risk_score
        )
        
        # Log to JSON
        self._log_to_json(self.portfolio_metrics_json, metrics_log.to_dict())
        
        # Add to real-time monitoring
        if self.real_time_monitor:
            self.real_time_monitor.add_portfolio_metrics(metrics_log)
    
    def log_system_health(self, component: str, status: str, details: Dict[str, Any],
                         metrics: Optional[Dict[str, float]] = None):
        """
        Log system health status.
        
        Args:
            component: System component name
            status: Health status ('HEALTHY', 'WARNING', 'UNHEALTHY')
            details: Detailed status information
            metrics: Optional performance metrics
        """
        timestamp = datetime.now()
        
        health_log = SystemHealthLog(
            timestamp=timestamp,
            component=component,
            status=status,
            details=details,
            metrics=metrics or {}
        )
        
        # Log to JSON
        self._log_to_json(self.system_health_json, health_log.to_dict())
        
        # Add to real-time monitoring
        if self.real_time_monitor:
            self.real_time_monitor.add_health_status(health_log)
        
        # Log critical health issues
        if status == 'UNHEALTHY':
            logger.error(f"System health issue - {component}: {details}")
        elif status == 'WARNING':
            logger.warning(f"System health warning - {component}: {details}")
    
    def get_trade_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get trade statistics for the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Read AI decisions log
            if not self.ai_decisions_json.exists():
                return {}
            
            with open(self.ai_decisions_json, 'r') as f:
                data = json.load(f)
            
            # Filter recent decisions
            recent_decisions = [
                record for record in data
                if datetime.fromisoformat(record['timestamp']) >= cutoff_time
            ]
            
            if not recent_decisions:
                return {}
            
            # Calculate statistics
            total_decisions = len(recent_decisions)
            executed_trades = [d for d in recent_decisions if d['decision_type'] == 'TRADE_EXECUTED']
            failed_trades = [d for d in recent_decisions if d['decision_type'] == 'TRADE_FAILED']
            no_trades = [d for d in recent_decisions if d['decision_type'] == 'NO_TRADE']
            
            avg_confidence = np.mean([d['confidence'] for d in recent_decisions])
            
            # Confidence distribution
            high_confidence = len([d for d in recent_decisions if d['confidence'] > 0.8])
            medium_confidence = len([d for d in recent_decisions if 0.5 < d['confidence'] <= 0.8])
            low_confidence = len([d for d in recent_decisions if d['confidence'] <= 0.5])
            
            return {
                'period_hours': hours,
                'total_decisions': total_decisions,
                'executed_trades': len(executed_trades),
                'failed_trades': len(failed_trades),
                'no_trades': len(no_trades),
                'execution_rate': len(executed_trades) / total_decisions if total_decisions > 0 else 0,
                'avg_confidence': avg_confidence,
                'confidence_distribution': {
                    'high': high_confidence,
                    'medium': medium_confidence,
                    'low': low_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get trade statistics: {e}")
            return {}
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time monitoring metrics"""
        if not self.real_time_monitor:
            return {'monitoring_enabled': False}
        
        return {
            'monitoring_enabled': True,
            'current_metrics': self.real_time_monitor.get_current_metrics(),
            'performance_summary_24h': self.real_time_monitor.get_performance_summary(24),
            'performance_summary_1h': self.real_time_monitor.get_performance_summary(1)
        }
    
    def export_trade_history(self, output_file: str, format: str = 'csv',
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> bool:
        """
        Export trade history to file.
        
        Args:
            output_file: Output file path
            format: Export format ('csv' or 'json')
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            bool: True if export was successful
        """
        try:
            if format.lower() == 'csv':
                # CSV export - copy existing CSV with date filtering if needed
                if start_date or end_date:
                    df = pd.read_csv(self.trade_history_csv)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    if start_date:
                        df = df[df['timestamp'] >= start_date]
                    if end_date:
                        df = df[df['timestamp'] <= end_date]
                    
                    df.to_csv(output_file, index=False)
                else:
                    # Simple copy
                    import shutil
                    shutil.copy2(self.trade_history_csv, output_file)
                    
            elif format.lower() == 'json':
                # JSON export from AI decisions log
                with open(self.ai_decisions_json, 'r') as f:
                    data = json.load(f)
                
                # Filter by date if specified
                if start_date or end_date:
                    filtered_data = []
                    for record in data:
                        record_time = datetime.fromisoformat(record['timestamp'])
                        if start_date and record_time < start_date:
                            continue
                        if end_date and record_time > end_date:
                            continue
                        filtered_data.append(record)
                    data = filtered_data
                
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Trade history exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export trade history: {e}")
            return False
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files to manage disk space"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up JSON logs
            for json_file in [self.ai_decisions_json, self.portfolio_metrics_json, self.system_health_json]:
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Filter recent records
                    recent_data = [
                        record for record in data
                        if datetime.fromisoformat(record['timestamp']) >= cutoff_date
                    ]
                    
                    # Write back filtered data
                    with open(json_file, 'w') as f:
                        json.dump(recent_data, f, indent=2, default=str)
            
            logger.info(f"Cleaned up logs older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")


# Utility functions
def create_comprehensive_logger(config: Dict[str, Any]) -> ComprehensiveLogger:
    """Factory function to create ComprehensiveLogger"""
    return ComprehensiveLogger(config)


def setup_trading_logging(log_level: str = 'INFO', log_file: str = 'logs/alphapulse.log') -> None:
    """Setup comprehensive logging configuration for the trading system"""
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set specific log levels for components
    logging.getLogger('trading').setLevel(logging.INFO)
    logging.getLogger('risk').setLevel(logging.INFO)
    logging.getLogger('data').setLevel(logging.WARNING)
    logging.getLogger('models').setLevel(logging.INFO)