"""
Portfolio and Position data models for AlphaPulse-RL trading system.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import logging
import json
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position."""
    pair: str
    side: str  # 'long' or 'short'
    size: float
    leverage: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime
    
    def update_price(self, new_price: float) -> None:
        """Update current price and recalculate unrealized PnL."""
        self.current_price = new_price
        if self.side == 'long':
            self.unrealized_pnl = (new_price - self.entry_price) * self.size * self.leverage
        else:  # short
            self.unrealized_pnl = (self.entry_price - new_price) * self.size * self.leverage
    
    def get_notional_value(self) -> float:
        """Get the notional value of the position."""
        return self.size * self.current_price * self.leverage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for serialization."""
        return {
            'pair': self.pair,
            'side': self.side,
            'size': self.size,
            'leverage': self.leverage,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'timestamp': self.timestamp.isoformat()
        }
    
    def get_margin_used(self) -> float:
        """Get the margin used by this position."""
        return self.size * self.current_price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for serialization."""
        return {
            'pair': self.pair,
            'side': self.side,
            'size': self.size,
            'leverage': self.leverage,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create position from dictionary."""
        return cls(
            pair=data['pair'],
            side=data['side'],
            size=data['size'],
            leverage=data['leverage'],
            entry_price=data['entry_price'],
            current_price=data['current_price'],
            unrealized_pnl=data['unrealized_pnl'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


@dataclass
class PortfolioState:
    """Represents the current state of the trading portfolio."""
    balance: float
    positions: Dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0
    last_trade_time: Optional[datetime] = None
    daily_start_balance: float = 0.0
    peak_balance: float = 0.0
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.daily_start_balance == 0.0:
            self.daily_start_balance = self.balance
        if self.peak_balance == 0.0:
            self.peak_balance = self.balance
    
    def get_total_equity(self) -> float:
        """Get total equity including unrealized PnL."""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.balance + unrealized_pnl
    
    def get_total_margin_used(self) -> float:
        """Get total margin used across all positions."""
        return sum(pos.get_margin_used() for pos in self.positions.values())
    
    def get_total_notional_exposure(self) -> float:
        """Get total notional exposure across all positions."""
        return sum(pos.get_notional_value() for pos in self.positions.values())
    
    def update_daily_pnl(self) -> None:
        """Update daily PnL calculation."""
        current_equity = self.get_total_equity()
        self.daily_pnl = current_equity - self.daily_start_balance
    
    def update_max_drawdown(self) -> None:
        """Update maximum drawdown calculation."""
        current_equity = self.get_total_equity()
        
        # Update peak balance
        if current_equity > self.peak_balance:
            self.peak_balance = current_equity
        
        # Calculate current drawdown
        current_drawdown = (self.peak_balance - current_equity) / self.peak_balance
        
        # Update max drawdown
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def add_position(self, position: Position) -> None:
        """Add a new position to the portfolio."""
        self.positions[position.pair] = position
        self.trade_count += 1
        self.last_trade_time = datetime.now()
        logger.info(f"Added position: {position.pair} {position.side} {position.size}")
    
    def remove_position(self, pair: str) -> Optional[Position]:
        """Remove a position from the portfolio."""
        position = self.positions.pop(pair, None)
        if position:
            logger.info(f"Removed position: {pair}")
        return position
    
    def update_position_prices(self, price_data: Dict[str, float]) -> None:
        """Update all position prices with new market data."""
        for pair, position in self.positions.items():
            if pair in price_data:
                position.update_price(price_data[pair])
        
        # Update derived metrics
        self.update_daily_pnl()
        self.update_max_drawdown()
    
    def reset_daily_metrics(self) -> None:
        """Reset daily metrics at start of new trading day."""
        self.daily_start_balance = self.get_total_equity()
        self.daily_pnl = 0.0
        logger.info("Reset daily metrics for new trading day")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio state to dictionary for serialization."""
        return {
            'balance': self.balance,
            'positions': {pair: pos.to_dict() for pair, pos in self.positions.items()},
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'trade_count': self.trade_count,
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'daily_start_balance': self.daily_start_balance,
            'peak_balance': self.peak_balance,
            'timestamp': datetime.now().isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioState':
        """Create portfolio state from dictionary."""
        positions = {}
        for pair, pos_data in data.get('positions', {}).items():
            positions[pair] = Position.from_dict(pos_data)
        
        last_trade_time = None
        if data.get('last_trade_time'):
            last_trade_time = datetime.fromisoformat(data['last_trade_time'])
        
        return cls(
            balance=data['balance'],
            positions=positions,
            daily_pnl=data.get('daily_pnl', 0.0),
            total_pnl=data.get('total_pnl', 0.0),
            max_drawdown=data.get('max_drawdown', 0.0),
            trade_count=data.get('trade_count', 0),
            last_trade_time=last_trade_time,
            daily_start_balance=data.get('daily_start_balance', data['balance']),
            peak_balance=data.get('peak_balance', data['balance'])
        )
    
    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive portfolio performance metrics."""
        current_equity = self.get_total_equity()
        
        # Basic metrics
        metrics = {
            'total_equity': current_equity,
            'balance': self.balance,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (self.daily_pnl / self.daily_start_balance * 100) if self.daily_start_balance > 0 else 0.0,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': (self.total_pnl / (current_equity - self.total_pnl) * 100) if (current_equity - self.total_pnl) > 0 else 0.0,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown * 100,
            'trade_count': self.trade_count,
            'total_margin_used': self.get_total_margin_used(),
            'total_notional_exposure': self.get_total_notional_exposure(),
            'margin_utilization_pct': (self.get_total_margin_used() / current_equity * 100) if current_equity > 0 else 0.0
        }
        
        # Position-specific metrics
        metrics['active_positions'] = len(self.positions)
        metrics['unrealized_pnl'] = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Risk metrics
        if current_equity > 0:
            metrics['leverage_ratio'] = self.get_total_notional_exposure() / current_equity
        else:
            metrics['leverage_ratio'] = 0.0
        
        return metrics


@dataclass
class TradeRecord:
    """Represents a completed trade record."""
    timestamp: datetime
    pair: str
    action: List[float]  # [direction, size, leverage]
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    confidence: float = 0.0
    market_regime: int = 0
    risk_metrics: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert trade record to dictionary for logging."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'pair': self.pair,
            'action': self.action,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'confidence': self.confidence,
            'market_regime': self.market_regime,
            'risk_metrics': self.risk_metrics
        }


class PortfolioManager:
    """
    Portfolio manager with persistence and recovery mechanisms.
    Handles saving/loading portfolio state and provides backup/recovery functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = Path(config.get('portfolio', {}).get('data_dir', 'data/portfolio'))
        self.backup_dir = Path(config.get('portfolio', {}).get('backup_dir', 'data/portfolio/backups'))
        self.max_backups = config.get('portfolio', {}).get('max_backups', 10)
        self.auto_save_interval = config.get('portfolio', {}).get('auto_save_interval_minutes', 5)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.portfolio_file = self.data_dir / 'portfolio_state.json'
        self.backup_file_pattern = 'portfolio_backup_{timestamp}.json'
        
        self._last_save_time = datetime.now()
    
    def save_portfolio(self, portfolio: PortfolioState, create_backup: bool = True) -> bool:
        """
        Save portfolio state to disk with optional backup creation.
        
        Args:
            portfolio: Portfolio state to save
            create_backup: Whether to create a backup before saving
            
        Returns:
            bool: True if save was successful
        """
        try:
            # Create backup if requested and file exists
            if create_backup and self.portfolio_file.exists():
                self._create_backup()
            
            # Save current state
            portfolio_data = portfolio.to_dict()
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.portfolio_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(portfolio_data, f, indent=2, default=str)
            
            # Atomic rename
            temp_file.rename(self.portfolio_file)
            
            self._last_save_time = datetime.now()
            logger.info(f"Portfolio state saved to {self.portfolio_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save portfolio state: {e}")
            return False
    
    def load_portfolio(self, initial_balance: float = 1000.0) -> PortfolioState:
        """
        Load portfolio state from disk or create new one if not found.
        
        Args:
            initial_balance: Initial balance for new portfolio
            
        Returns:
            PortfolioState: Loaded or new portfolio state
        """
        try:
            if self.portfolio_file.exists():
                with open(self.portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
                
                portfolio = PortfolioState.from_dict(portfolio_data)
                logger.info(f"Portfolio state loaded from {self.portfolio_file}")
                return portfolio
            else:
                logger.info("No existing portfolio state found, creating new portfolio")
                return PortfolioState(balance=initial_balance)
                
        except Exception as e:
            logger.error(f"Failed to load portfolio state: {e}")
            logger.info("Creating new portfolio state")
            return PortfolioState(balance=initial_balance)
    
    def _create_backup(self) -> bool:
        """Create a backup of the current portfolio state."""
        try:
            if not self.portfolio_file.exists():
                return True
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = self.backup_file_pattern.format(timestamp=timestamp)
            backup_path = self.backup_dir / backup_filename
            
            # Copy current file to backup
            with open(self.portfolio_file, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            
            logger.info(f"Portfolio backup created: {backup_path}")
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create portfolio backup: {e}")
            return False
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backup files to maintain max_backups limit."""
        try:
            backup_files = list(self.backup_dir.glob('portfolio_backup_*.json'))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove excess backups
            for backup_file in backup_files[self.max_backups:]:
                backup_file.unlink()
                logger.debug(f"Removed old backup: {backup_file}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
    
    def restore_from_backup(self, backup_timestamp: Optional[str] = None) -> Optional[PortfolioState]:
        """
        Restore portfolio from backup.
        
        Args:
            backup_timestamp: Specific backup timestamp to restore from.
                            If None, restores from most recent backup.
            
        Returns:
            PortfolioState or None if restoration failed
        """
        try:
            if backup_timestamp:
                backup_filename = self.backup_file_pattern.format(timestamp=backup_timestamp)
                backup_path = self.backup_dir / backup_filename
            else:
                # Find most recent backup
                backup_files = list(self.backup_dir.glob('portfolio_backup_*.json'))
                if not backup_files:
                    logger.error("No backup files found")
                    return None
                
                backup_path = max(backup_files, key=lambda x: x.stat().st_mtime)
            
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return None
            
            with open(backup_path, 'r') as f:
                portfolio_data = json.load(f)
            
            portfolio = PortfolioState.from_dict(portfolio_data)
            logger.info(f"Portfolio restored from backup: {backup_path}")
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return None
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backup files with metadata."""
        backups = []
        
        try:
            backup_files = list(self.backup_dir.glob('portfolio_backup_*.json'))
            
            for backup_file in backup_files:
                stat = backup_file.stat()
                
                # Extract timestamp from filename
                timestamp_str = backup_file.stem.replace('portfolio_backup_', '')
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                except ValueError:
                    timestamp = datetime.fromtimestamp(stat.st_mtime)
                
                backups.append({
                    'filename': backup_file.name,
                    'timestamp': timestamp,
                    'size_bytes': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime)
                })
            
            # Sort by timestamp, most recent first
            backups.sort(key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
        
        return backups
    
    def should_auto_save(self) -> bool:
        """Check if portfolio should be auto-saved based on time interval."""
        time_since_save = datetime.now() - self._last_save_time
        return time_since_save.total_seconds() >= (self.auto_save_interval * 60)
    
    def validate_portfolio_integrity(self, portfolio: PortfolioState) -> Dict[str, Any]:
        """
        Validate portfolio state integrity and return validation results.
        
        Args:
            portfolio: Portfolio state to validate
            
        Returns:
            Dict with validation results and any issues found
        """
        issues = []
        warnings = []
        
        try:
            # Check basic data integrity
            if portfolio.balance < 0:
                issues.append("Negative balance detected")
            
            if portfolio.get_total_equity() <= 0:
                issues.append("Zero or negative total equity")
            
            # Validate positions
            for pair, position in portfolio.positions.items():
                if position.size <= 0:
                    issues.append(f"Invalid position size for {pair}: {position.size}")
                
                if position.leverage < 1 or position.leverage > 12:
                    issues.append(f"Invalid leverage for {pair}: {position.leverage}")
                
                if position.entry_price <= 0:
                    issues.append(f"Invalid entry price for {pair}: {position.entry_price}")
                
                # Check for stale positions (older than 24 hours without price update)
                if datetime.now() - position.timestamp > timedelta(hours=24):
                    warnings.append(f"Stale position detected for {pair}")
            
            # Check portfolio metrics consistency
            calculated_equity = portfolio.balance + sum(pos.unrealized_pnl for pos in portfolio.positions.values())
            reported_equity = portfolio.get_total_equity()
            
            if abs(calculated_equity - reported_equity) > 0.01:  # Allow small floating point differences
                warnings.append(f"Equity calculation mismatch: calculated={calculated_equity}, reported={reported_equity}")
            
            # Check drawdown calculation
            if portfolio.max_drawdown < 0 or portfolio.max_drawdown > 1:
                warnings.append(f"Unusual max drawdown value: {portfolio.max_drawdown}")
            
        except Exception as e:
            issues.append(f"Validation error: {e}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def export_portfolio_history(self, output_file: str, format: str = 'json') -> bool:
        """
        Export portfolio history from backups to a single file.
        
        Args:
            output_file: Output file path
            format: Export format ('json' or 'csv')
            
        Returns:
            bool: True if export was successful
        """
        try:
            backups = self.list_backups()
            
            if format.lower() == 'json':
                history_data = []
                
                for backup in backups:
                    backup_path = self.backup_dir / backup['filename']
                    with open(backup_path, 'r') as f:
                        portfolio_data = json.load(f)
                    
                    # Add metadata
                    portfolio_data['backup_timestamp'] = backup['timestamp'].isoformat()
                    history_data.append(portfolio_data)
                
                with open(output_file, 'w') as f:
                    json.dump(history_data, f, indent=2, default=str)
                    
            elif format.lower() == 'csv':
                import pandas as pd
                
                history_records = []
                
                for backup in backups:
                    backup_path = self.backup_dir / backup['filename']
                    with open(backup_path, 'r') as f:
                        portfolio_data = json.load(f)
                    
                    # Extract key metrics for CSV
                    record = {
                        'timestamp': backup['timestamp'],
                        'balance': portfolio_data['balance'],
                        'total_pnl': portfolio_data.get('total_pnl', 0),
                        'daily_pnl': portfolio_data.get('daily_pnl', 0),
                        'max_drawdown': portfolio_data.get('max_drawdown', 0),
                        'trade_count': portfolio_data.get('trade_count', 0),
                        'active_positions': len(portfolio_data.get('positions', {}))
                    }
                    
                    history_records.append(record)
                
                df = pd.DataFrame(history_records)
                df.to_csv(output_file, index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Portfolio history exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export portfolio history: {e}")
            return False