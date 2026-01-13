"""
Core Risk Manager for AlphaPulse-RL trading system.

Implements all risk limits and trade validation logic according to requirements:
- Maximum leverage: 12x
- Maximum position size: 10% of equity
- Maximum daily loss: 3% of portfolio
- Maximum total drawdown: 12% (triggers emergency flatten)
- Volatility threshold checking
"""

import logging
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from trading.portfolio import PortfolioState, Position

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for current risk metrics."""
    current_drawdown: float
    daily_pnl_percent: float
    position_exposure_percent: float
    total_leverage: float
    margin_utilization: float
    volatility_level: float
    risk_score: float


class RiskManager:
    """
    Core risk management system that enforces all trading limits.
    
    Requirements implemented:
    - 2.1: Maximum leverage of 12x on all positions
    - 2.2: Maximum position size of 10% of total equity
    - 2.3: Maximum daily loss limit of 3% of portfolio value
    - 2.4: Emergency position flattening at 12% total drawdown
    - 2.5: Volatility threshold checking to prevent trades during extreme market conditions
    """
    
    def __init__(self, config_path: str = "config/trading_params.yaml", enable_monitoring: bool = True):
        """Initialize risk manager with configuration."""
        self.config = self._load_config(config_path)
        
        # Risk limits from config
        self.max_leverage = self.config['risk']['max_leverage']
        self.max_position_size_percent = self.config['risk']['max_position_size_percent']
        self.max_daily_loss_percent = self.config['risk']['max_daily_loss_percent']
        self.max_total_drawdown_percent = self.config['risk']['max_total_drawdown_percent']
        self.volatility_threshold = self.config['risk']['volatility_threshold']
        
        # Emergency state tracking
        self.emergency_mode = False
        self.last_risk_check = datetime.now()
        
        # Initialize risk monitor for enhanced monitoring capabilities
        self.risk_monitor = None
        if enable_monitoring:
            from .risk_metrics import RiskMonitor
            self.risk_monitor = RiskMonitor(
                volatility_threshold=self.volatility_threshold,
                alert_log_path="logs/risk_alerts.json"
            )
        
        logger.info(f"RiskManager initialized with limits: "
                   f"leverage={self.max_leverage}x, "
                   f"position_size={self.max_position_size_percent}%, "
                   f"daily_loss={self.max_daily_loss_percent}%, "
                   f"drawdown={self.max_total_drawdown_percent}%, "
                   f"monitoring={'enabled' if self.risk_monitor else 'disabled'}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load trading parameters configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            # Return default values if config loading fails
            return {
                'risk': {
                    'max_leverage': 12.0,
                    'max_position_size_percent': 10.0,
                    'max_daily_loss_percent': 3.0,
                    'max_total_drawdown_percent': 12.0,
                    'volatility_threshold': 0.05
                }
            }
    
    def validate_trade(self, action: List[float], portfolio: PortfolioState, 
                      current_price: float, volatility: float = 0.0, pair: str = "BTCUSDT") -> Tuple[bool, str]:
        """
        Validate a proposed trade against all risk limits.
        
        Args:
            action: [direction, size, leverage] from PPO agent
            portfolio: Current portfolio state
            current_price: Current market price
            volatility: Current market volatility
            pair: Trading pair for enhanced monitoring
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check if in emergency mode
        if self.emergency_mode:
            return False, "Emergency mode active - all trading suspended"
        
        # Extract action components
        direction, size, leverage = action
        
        # 1. Validate leverage limit (Requirement 2.1)
        if leverage > self.max_leverage:
            return False, f"Leverage {leverage:.2f}x exceeds maximum {self.max_leverage}x"
        
        # 2. Validate position size limit (Requirement 2.2)
        total_equity = portfolio.get_total_equity()
        position_value = size * total_equity
        position_size_percent = (position_value / total_equity) * 100
        
        if position_size_percent > self.max_position_size_percent:
            return False, (f"Position size {position_size_percent:.2f}% exceeds "
                          f"maximum {self.max_position_size_percent}%")
        
        # 3. Check daily loss limit (Requirement 2.3)
        if portfolio.daily_start_balance > 0:
            daily_loss_percent = abs(portfolio.daily_pnl / portfolio.daily_start_balance) * 100
            if portfolio.daily_pnl < 0 and daily_loss_percent >= self.max_daily_loss_percent:
                return False, (f"Daily loss {daily_loss_percent:.2f}% at limit "
                              f"{self.max_daily_loss_percent}%")
        
        # 4. Check total drawdown (Requirement 2.4)
        if portfolio.max_drawdown * 100 >= self.max_total_drawdown_percent:
            self._trigger_emergency_mode(portfolio)
            return False, (f"Total drawdown {portfolio.max_drawdown*100:.2f}% exceeds "
                          f"limit {self.max_total_drawdown_percent}%")
        
        # 5. Enhanced volatility checking using risk monitor (Requirement 2.5)
        if self.risk_monitor:
            # Update price data for volatility calculations
            self.risk_monitor.update_price_data(pair, current_price)
            
            # Check volatility threshold with enhanced monitoring
            is_safe, volatility_reason = self.risk_monitor.check_volatility_threshold(pair, current_price)
            if not is_safe:
                return False, f"Enhanced volatility check failed: {volatility_reason}"
        else:
            # Fallback to basic volatility check
            if volatility > self.volatility_threshold:
                return False, (f"Market volatility {volatility:.4f} exceeds "
                              f"threshold {self.volatility_threshold:.4f}")
        
        # 6. Check margin requirements
        notional_value = position_value * leverage
        margin_required = position_value
        available_margin = total_equity - portfolio.get_total_margin_used()
        
        if margin_required > available_margin:
            return False, (f"Insufficient margin: required {margin_required:.2f}, "
                          f"available {available_margin:.2f}")
        
        # All checks passed
        return True, "Trade validated successfully"
    
    def calculate_position_size(self, direction: float, size: float, leverage: float, 
                              balance: float) -> float:
        """
        Calculate actual position size based on risk limits.
        
        Args:
            direction: Position direction (-1 to 1)
            size: Requested position size (0 to 0.1)
            leverage: Requested leverage (1 to 12)
            balance: Current account balance
            
        Returns:
            Actual position size in base currency
        """
        # Ensure size is within bounds
        size = max(0.0, min(size, self.max_position_size_percent / 100))
        
        # Ensure leverage is within bounds
        leverage = max(1.0, min(leverage, self.max_leverage))
        
        # Calculate position size
        position_size = size * balance
        
        logger.debug(f"Calculated position size: {position_size:.2f} "
                    f"(size={size:.4f}, leverage={leverage:.2f}x)")
        
        return position_size
    
    def check_daily_loss_limit(self, portfolio: PortfolioState) -> bool:
        """
        Check if daily loss limit has been reached.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            True if within limits, False if limit exceeded
        """
        if portfolio.daily_pnl >= 0:
            return True  # No loss, within limits
        
        daily_loss_percent = abs(portfolio.daily_pnl / portfolio.daily_start_balance) * 100
        within_limit = daily_loss_percent < self.max_daily_loss_percent
        
        if not within_limit:
            logger.warning(f"Daily loss limit exceeded: {daily_loss_percent:.2f}% "
                          f"(limit: {self.max_daily_loss_percent}%)")
        
        return within_limit
    
    def emergency_flatten_positions(self, portfolio: PortfolioState) -> List[str]:
        """
        Force close all positions due to emergency conditions.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            List of position pairs that need to be closed
        """
        positions_to_close = list(portfolio.positions.keys())
        
        if positions_to_close:
            logger.critical(f"EMERGENCY FLATTEN: Closing {len(positions_to_close)} positions")
            for pair in positions_to_close:
                logger.critical(f"Emergency close position: {pair}")
        
        self.emergency_mode = True
        return positions_to_close
    
    def _trigger_emergency_mode(self, portfolio: PortfolioState) -> None:
        """Trigger emergency mode due to risk limit breach."""
        self.emergency_mode = True
        logger.critical(f"EMERGENCY MODE ACTIVATED - Drawdown: {portfolio.max_drawdown*100:.2f}%")
    
    def reset_emergency_mode(self) -> None:
        """Reset emergency mode (manual intervention required)."""
        self.emergency_mode = False
        logger.info("Emergency mode reset by manual intervention")
    
    def get_risk_metrics(self, portfolio: PortfolioState, volatility: float = 0.0) -> RiskMetrics:
        """
        Calculate current risk metrics for monitoring.
        
        Args:
            portfolio: Current portfolio state
            volatility: Current market volatility
            
        Returns:
            RiskMetrics object with current risk statistics
        """
        total_equity = portfolio.get_total_equity()
        
        # Calculate daily PnL percentage
        daily_pnl_percent = 0.0
        if portfolio.daily_start_balance > 0:
            daily_pnl_percent = (portfolio.daily_pnl / portfolio.daily_start_balance) * 100
        
        # Calculate position exposure percentage
        total_exposure = portfolio.get_total_notional_exposure()
        position_exposure_percent = (total_exposure / total_equity) * 100 if total_equity > 0 else 0.0
        
        # Calculate total leverage
        total_margin = portfolio.get_total_margin_used()
        total_leverage = total_exposure / total_margin if total_margin > 0 else 0.0
        
        # Calculate margin utilization
        margin_utilization = (total_margin / total_equity) * 100 if total_equity > 0 else 0.0
        
        # Calculate risk score (0-100, higher is riskier)
        risk_score = self._calculate_risk_score(
            portfolio.max_drawdown * 100,
            abs(daily_pnl_percent),
            position_exposure_percent,
            volatility * 100
        )
        
        return RiskMetrics(
            current_drawdown=portfolio.max_drawdown * 100,
            daily_pnl_percent=daily_pnl_percent,
            position_exposure_percent=position_exposure_percent,
            total_leverage=total_leverage,
            margin_utilization=margin_utilization,
            volatility_level=volatility,
            risk_score=risk_score
        )
    
    def _calculate_risk_score(self, drawdown: float, daily_loss: float, 
                            exposure: float, volatility: float) -> float:
        """Calculate composite risk score (0-100)."""
        # Normalize each component to 0-100 scale
        drawdown_score = min(100, (drawdown / self.max_total_drawdown_percent) * 100)
        daily_loss_score = min(100, (daily_loss / self.max_daily_loss_percent) * 100)
        exposure_score = min(100, (exposure / (self.max_position_size_percent * 10)) * 100)
        volatility_score = min(100, (volatility / (self.volatility_threshold * 100)) * 100)
        
        # Weighted average (drawdown and daily loss are most important)
        risk_score = (
            drawdown_score * 0.4 +
            daily_loss_score * 0.3 +
            exposure_score * 0.2 +
            volatility_score * 0.1
        )
        
        return min(100, risk_score)
    
    def is_trading_allowed(self, portfolio: PortfolioState, volatility: float = 0.0) -> Tuple[bool, str]:
        """
        Check if trading is currently allowed based on all risk conditions.
        
        Args:
            portfolio: Current portfolio state
            volatility: Current market volatility
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        # Check emergency mode
        if self.emergency_mode:
            return False, "Emergency mode active"
        
        # Check daily loss limit
        if not self.check_daily_loss_limit(portfolio):
            return False, "Daily loss limit exceeded"
        
        # Check drawdown limit
        if portfolio.max_drawdown * 100 >= self.max_total_drawdown_percent:
            return False, "Total drawdown limit exceeded"
        
        # Check volatility
        if volatility > self.volatility_threshold:
            return False, f"Volatility too high: {volatility:.4f}"
        
        return True, "Trading allowed"
    
    def monitor_portfolio_risk(self, portfolio: PortfolioState, price_data: Dict[str, float]) -> Dict:
        """
        Comprehensive portfolio risk monitoring with alerting.
        
        Args:
            portfolio: Current portfolio state
            price_data: Current market prices for all pairs
            
        Returns:
            Dictionary containing risk monitoring results and alerts
        """
        if not self.risk_monitor:
            # Fallback to basic risk metrics if monitoring is disabled
            risk_metrics = self.get_risk_metrics(portfolio)
            return {
                'risk_metrics': risk_metrics,
                'alerts': [],
                'volatility_metrics': {},
                'monitoring_enabled': False
            }
        
        # Update price data for all pairs
        for pair, price in price_data.items():
            self.risk_monitor.update_price_data(pair, price)
        
        # Get comprehensive risk metrics
        risk_metrics = self.get_risk_metrics(portfolio)
        
        # Monitor risk metrics and generate alerts
        new_alerts = self.risk_monitor.monitor_risk_metrics(portfolio, risk_metrics)
        
        # Get volatility metrics for primary pairs
        volatility_metrics = {}
        for pair, price in price_data.items():
            volatility_metrics[pair] = self.risk_monitor.calculate_volatility_metrics(pair, price)
        
        # Generate comprehensive risk report
        risk_report = self.risk_monitor.get_risk_report(portfolio, risk_metrics)
        
        return {
            'risk_metrics': risk_metrics,
            'alerts': new_alerts,
            'volatility_metrics': volatility_metrics,
            'risk_report': risk_report,
            'monitoring_enabled': True
        }
    
    def get_risk_alerts(self) -> List:
        """Get current active risk alerts."""
        if self.risk_monitor:
            return self.risk_monitor.active_alerts
        return []
    
    def update_market_data(self, pair: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update market data for risk monitoring.
        
        Args:
            pair: Trading pair symbol
            price: Current market price
            timestamp: Optional timestamp (defaults to now)
        """
        if self.risk_monitor:
            self.risk_monitor.update_price_data(pair, price, timestamp)
    
    def check_volatility_regime(self, pair: str, current_price: float) -> Dict:
        """
        Check current volatility regime for a trading pair.
        
        Args:
            pair: Trading pair symbol
            current_price: Current market price
            
        Returns:
            Dictionary with volatility regime information
        """
        if not self.risk_monitor:
            return {
                'regime': 'unknown',
                'volatility': 0.0,
                'safe_to_trade': True,
                'reason': 'Monitoring disabled'
            }
        
        volatility_metrics = self.risk_monitor.calculate_volatility_metrics(pair, current_price)
        is_safe, reason = self.risk_monitor.check_volatility_threshold(pair, current_price)
        
        return {
            'regime': volatility_metrics.volatility_regime,
            'volatility': volatility_metrics.current_volatility,
            'volatility_percentile': volatility_metrics.volatility_percentile,
            'safe_to_trade': is_safe,
            'reason': reason
        }