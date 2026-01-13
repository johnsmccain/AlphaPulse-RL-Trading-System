"""
Risk metrics calculation and monitoring utilities for AlphaPulse-RL trading system.

Implements real-time risk metric calculations, volatility threshold checking,
and risk reporting capabilities according to requirements 2.5 and 3.5.
"""

import logging
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from trading.portfolio import PortfolioState, Position
from risk.risk_manager import RiskMetrics

logger = logging.getLogger(__name__)


@dataclass
class VolatilityMetrics:
    """Container for volatility-related metrics."""
    current_volatility: float
    volatility_percentile: float
    volatility_regime: str  # 'low', 'normal', 'high', 'extreme'
    atr_percentage: float
    price_change_1h: float
    price_change_24h: float


@dataclass
class RiskAlert:
    """Container for risk alerts."""
    timestamp: datetime
    alert_type: str
    severity: str  # 'info', 'warning', 'critical'
    message: str
    metrics: Dict
    action_required: bool


class RiskMonitor:
    """
    Real-time risk monitoring and alerting system.
    
    Requirements implemented:
    - 2.5: Volatility threshold checking to prevent trades during extreme market conditions
    - 3.5: Risk reporting and alerting capabilities
    """
    
    def __init__(self, volatility_threshold: float = 0.05, 
                 alert_log_path: str = "logs/risk_alerts.json"):
        """Initialize risk monitor."""
        self.volatility_threshold = volatility_threshold
        self.alert_log_path = alert_log_path
        
        # Historical data for calculations
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.volatility_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.risk_metrics_history: List[Tuple[datetime, RiskMetrics]] = []
        
        # Alert tracking
        self.active_alerts: List[RiskAlert] = []
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Thresholds for different alert levels
        self.warning_thresholds = {
            'drawdown': 8.0,  # 8% drawdown warning
            'daily_loss': 2.0,  # 2% daily loss warning
            'volatility': 0.04,  # 4% volatility warning
            'exposure': 80.0,  # 80% exposure warning
        }
        
        self.critical_thresholds = {
            'drawdown': 10.0,  # 10% drawdown critical
            'daily_loss': 2.5,  # 2.5% daily loss critical
            'volatility': 0.05,  # 5% volatility critical
            'exposure': 90.0,  # 90% exposure critical
        }
        
        logger.info(f"RiskMonitor initialized with volatility threshold: {volatility_threshold}")
    
    def update_price_data(self, pair: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """Update price history for volatility calculations."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if pair not in self.price_history:
            self.price_history[pair] = []
        
        self.price_history[pair].append((timestamp, price))
        
        # Keep only last 24 hours of data
        cutoff_time = timestamp - timedelta(hours=24)
        self.price_history[pair] = [
            (ts, p) for ts, p in self.price_history[pair] if ts > cutoff_time
        ]
    
    def calculate_volatility_metrics(self, pair: str, current_price: float) -> VolatilityMetrics:
        """
        Calculate comprehensive volatility metrics for a trading pair.
        
        Args:
            pair: Trading pair symbol
            current_price: Current market price
            
        Returns:
            VolatilityMetrics object with volatility analysis
        """
        if pair not in self.price_history or len(self.price_history[pair]) < 2:
            return VolatilityMetrics(
                current_volatility=0.0,
                volatility_percentile=0.0,
                volatility_regime='unknown',
                atr_percentage=0.0,
                price_change_1h=0.0,
                price_change_24h=0.0
            )
        
        prices = [p for _, p in self.price_history[pair]]
        timestamps = [ts for ts, _ in self.price_history[pair]]
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Current volatility (annualized)
        if len(returns) > 0:
            current_volatility = np.std(returns) * np.sqrt(365 * 24 * 12)  # 5-minute intervals
        else:
            current_volatility = 0.0
        
        # Volatility percentile (last 100 observations)
        recent_volatilities = []
        for i in range(max(0, len(returns) - 100), len(returns)):
            if i >= 20:  # Need at least 20 observations for volatility
                vol_window = returns[max(0, i-20):i]
                vol = np.std(vol_window) * np.sqrt(365 * 24 * 12)
                recent_volatilities.append(vol)
        
        if recent_volatilities:
            volatility_percentile = (np.sum(np.array(recent_volatilities) <= current_volatility) / 
                                   len(recent_volatilities)) * 100
        else:
            volatility_percentile = 50.0
        
        # Volatility regime classification
        volatility_regime = self._classify_volatility_regime(current_volatility, volatility_percentile)
        
        # ATR percentage (simplified using price range)
        if len(prices) >= 14:
            recent_prices = prices[-14:]
            atr = np.mean([max(recent_prices[i:i+1]) - min(recent_prices[i:i+1]) 
                          for i in range(len(recent_prices)-1)])
            atr_percentage = (atr / current_price) * 100
        else:
            atr_percentage = 0.0
        
        # Price changes
        price_change_1h = self._calculate_price_change(timestamps, prices, current_price, hours=1)
        price_change_24h = self._calculate_price_change(timestamps, prices, current_price, hours=24)
        
        return VolatilityMetrics(
            current_volatility=current_volatility,
            volatility_percentile=volatility_percentile,
            volatility_regime=volatility_regime,
            atr_percentage=atr_percentage,
            price_change_1h=price_change_1h,
            price_change_24h=price_change_24h
        )
    
    def _classify_volatility_regime(self, volatility: float, percentile: float) -> str:
        """Classify volatility regime based on current level and percentile."""
        if volatility > self.volatility_threshold * 2:
            return 'extreme'
        elif volatility > self.volatility_threshold:
            return 'high'
        elif percentile > 75:
            return 'high'
        elif percentile < 25:
            return 'low'
        else:
            return 'normal'
    
    def _calculate_price_change(self, timestamps: List[datetime], prices: List[float], 
                              current_price: float, hours: int) -> float:
        """Calculate price change over specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Find price closest to cutoff time
        past_price = None
        for ts, price in zip(timestamps, prices):
            if ts >= cutoff_time:
                past_price = price
                break
        
        if past_price is None and prices:
            past_price = prices[0]  # Use oldest available price
        
        if past_price is not None:
            return ((current_price - past_price) / past_price) * 100
        else:
            return 0.0
    
    def check_volatility_threshold(self, pair: str, current_price: float) -> Tuple[bool, str]:
        """
        Check if current volatility exceeds trading threshold.
        
        Args:
            pair: Trading pair symbol
            current_price: Current market price
            
        Returns:
            Tuple of (is_safe_to_trade, reason)
        """
        volatility_metrics = self.calculate_volatility_metrics(pair, current_price)
        
        if volatility_metrics.current_volatility > self.volatility_threshold:
            return False, (f"Volatility {volatility_metrics.current_volatility:.4f} exceeds "
                          f"threshold {self.volatility_threshold:.4f}")
        
        if volatility_metrics.volatility_regime == 'extreme':
            return False, "Extreme volatility regime detected"
        
        return True, "Volatility within acceptable range"
    
    def monitor_risk_metrics(self, portfolio: PortfolioState, risk_metrics: RiskMetrics) -> List[RiskAlert]:
        """
        Monitor risk metrics and generate alerts when thresholds are exceeded.
        
        Args:
            portfolio: Current portfolio state
            risk_metrics: Current risk metrics
            
        Returns:
            List of new risk alerts generated
        """
        new_alerts = []
        current_time = datetime.now()
        
        # Store metrics history
        self.risk_metrics_history.append((current_time, risk_metrics))
        
        # Keep only last 24 hours of history
        cutoff_time = current_time - timedelta(hours=24)
        self.risk_metrics_history = [
            (ts, metrics) for ts, metrics in self.risk_metrics_history if ts > cutoff_time
        ]
        
        # Check drawdown alerts
        new_alerts.extend(self._check_drawdown_alerts(risk_metrics, current_time))
        
        # Check daily loss alerts
        new_alerts.extend(self._check_daily_loss_alerts(risk_metrics, current_time))
        
        # Check volatility alerts
        new_alerts.extend(self._check_volatility_alerts(risk_metrics, current_time))
        
        # Check exposure alerts
        new_alerts.extend(self._check_exposure_alerts(risk_metrics, current_time))
        
        # Log new alerts
        for alert in new_alerts:
            self._log_alert(alert)
            self.active_alerts.append(alert)
        
        # Clean up old alerts
        self._cleanup_old_alerts()
        
        return new_alerts
    
    def _check_drawdown_alerts(self, risk_metrics: RiskMetrics, timestamp: datetime) -> List[RiskAlert]:
        """Check for drawdown-related alerts."""
        alerts = []
        drawdown = risk_metrics.current_drawdown
        
        # Critical drawdown alert
        if (drawdown >= self.critical_thresholds['drawdown'] and 
            self._should_send_alert('drawdown_critical', timestamp)):
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='drawdown_critical',
                severity='critical',
                message=f"Critical drawdown level reached: {drawdown:.2f}%",
                metrics={'drawdown': drawdown},
                action_required=True
            ))
        
        # Warning drawdown alert
        elif (drawdown >= self.warning_thresholds['drawdown'] and 
              self._should_send_alert('drawdown_warning', timestamp)):
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='drawdown_warning',
                severity='warning',
                message=f"Drawdown warning level reached: {drawdown:.2f}%",
                metrics={'drawdown': drawdown},
                action_required=False
            ))
        
        return alerts
    
    def _check_daily_loss_alerts(self, risk_metrics: RiskMetrics, timestamp: datetime) -> List[RiskAlert]:
        """Check for daily loss alerts."""
        alerts = []
        daily_loss = abs(risk_metrics.daily_pnl_percent) if risk_metrics.daily_pnl_percent < 0 else 0
        
        # Critical daily loss alert
        if (daily_loss >= self.critical_thresholds['daily_loss'] and 
            self._should_send_alert('daily_loss_critical', timestamp)):
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='daily_loss_critical',
                severity='critical',
                message=f"Critical daily loss level reached: {daily_loss:.2f}%",
                metrics={'daily_loss': daily_loss},
                action_required=True
            ))
        
        # Warning daily loss alert
        elif (daily_loss >= self.warning_thresholds['daily_loss'] and 
              self._should_send_alert('daily_loss_warning', timestamp)):
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='daily_loss_warning',
                severity='warning',
                message=f"Daily loss warning level reached: {daily_loss:.2f}%",
                metrics={'daily_loss': daily_loss},
                action_required=False
            ))
        
        return alerts
    
    def _check_volatility_alerts(self, risk_metrics: RiskMetrics, timestamp: datetime) -> List[RiskAlert]:
        """Check for volatility alerts."""
        alerts = []
        volatility = risk_metrics.volatility_level
        
        # Critical volatility alert
        if (volatility >= self.critical_thresholds['volatility'] and 
            self._should_send_alert('volatility_critical', timestamp)):
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='volatility_critical',
                severity='critical',
                message=f"Critical volatility level reached: {volatility:.4f}",
                metrics={'volatility': volatility},
                action_required=True
            ))
        
        # Warning volatility alert
        elif (volatility >= self.warning_thresholds['volatility'] and 
              self._should_send_alert('volatility_warning', timestamp)):
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='volatility_warning',
                severity='warning',
                message=f"Volatility warning level reached: {volatility:.4f}",
                metrics={'volatility': volatility},
                action_required=False
            ))
        
        return alerts
    
    def _check_exposure_alerts(self, risk_metrics: RiskMetrics, timestamp: datetime) -> List[RiskAlert]:
        """Check for position exposure alerts."""
        alerts = []
        exposure = risk_metrics.position_exposure_percent
        
        # Critical exposure alert
        if (exposure >= self.critical_thresholds['exposure'] and 
            self._should_send_alert('exposure_critical', timestamp)):
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='exposure_critical',
                severity='critical',
                message=f"Critical position exposure reached: {exposure:.2f}%",
                metrics={'exposure': exposure},
                action_required=True
            ))
        
        # Warning exposure alert
        elif (exposure >= self.warning_thresholds['exposure'] and 
              self._should_send_alert('exposure_warning', timestamp)):
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='exposure_warning',
                severity='warning',
                message=f"Position exposure warning: {exposure:.2f}%",
                metrics={'exposure': exposure},
                action_required=False
            ))
        
        return alerts
    
    def _should_send_alert(self, alert_type: str, timestamp: datetime) -> bool:
        """Check if alert should be sent based on cooldown period."""
        cooldown_minutes = {
            'drawdown_warning': 15,
            'drawdown_critical': 5,
            'daily_loss_warning': 15,
            'daily_loss_critical': 5,
            'volatility_warning': 10,
            'volatility_critical': 5,
            'exposure_warning': 10,
            'exposure_critical': 5,
        }
        
        if alert_type in self.alert_cooldowns:
            last_alert_time = self.alert_cooldowns[alert_type]
            cooldown_period = timedelta(minutes=cooldown_minutes.get(alert_type, 10))
            
            if timestamp - last_alert_time < cooldown_period:
                return False
        
        self.alert_cooldowns[alert_type] = timestamp
        return True
    
    def _log_alert(self, alert: RiskAlert) -> None:
        """Log alert to file and console."""
        alert_data = {
            'timestamp': alert.timestamp.isoformat(),
            'alert_type': alert.alert_type,
            'severity': alert.severity,
            'message': alert.message,
            'metrics': alert.metrics,
            'action_required': alert.action_required
        }
        
        # Log to file
        try:
            with open(self.alert_log_path, 'a') as f:
                f.write(json.dumps(alert_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to log alert to file: {e}")
        
        # Log to console
        if alert.severity == 'critical':
            logger.critical(f"RISK ALERT: {alert.message}")
        elif alert.severity == 'warning':
            logger.warning(f"RISK ALERT: {alert.message}")
        else:
            logger.info(f"RISK ALERT: {alert.message}")
    
    def _cleanup_old_alerts(self) -> None:
        """Remove old alerts from active list."""
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.active_alerts = [
            alert for alert in self.active_alerts if alert.timestamp > cutoff_time
        ]
    
    def get_risk_report(self, portfolio: PortfolioState, risk_metrics: RiskMetrics) -> Dict:
        """
        Generate comprehensive risk report.
        
        Args:
            portfolio: Current portfolio state
            risk_metrics: Current risk metrics
            
        Returns:
            Dictionary containing comprehensive risk report
        """
        current_time = datetime.now()
        
        # Calculate trend metrics
        trend_metrics = self._calculate_trend_metrics()
        
        report = {
            'timestamp': current_time.isoformat(),
            'portfolio_summary': {
                'total_equity': portfolio.get_total_equity(),
                'balance': portfolio.balance,
                'positions_count': len(portfolio.positions),
                'daily_pnl': portfolio.daily_pnl,
                'total_pnl': portfolio.total_pnl,
                'max_drawdown': portfolio.max_drawdown * 100,
                'trade_count': portfolio.trade_count
            },
            'risk_metrics': asdict(risk_metrics),
            'trend_analysis': trend_metrics,
            'active_alerts': [
                {
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.active_alerts
            ],
            'risk_assessment': self._assess_overall_risk(risk_metrics)
        }
        
        return report
    
    def _calculate_trend_metrics(self) -> Dict:
        """Calculate trend analysis from historical risk metrics."""
        if len(self.risk_metrics_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Get recent metrics
        recent_metrics = [metrics for _, metrics in self.risk_metrics_history[-10:]]
        
        # Calculate trends
        drawdown_trend = self._calculate_metric_trend([m.current_drawdown for m in recent_metrics])
        pnl_trend = self._calculate_metric_trend([m.daily_pnl_percent for m in recent_metrics])
        risk_score_trend = self._calculate_metric_trend([m.risk_score for m in recent_metrics])
        
        return {
            'drawdown_trend': drawdown_trend,
            'pnl_trend': pnl_trend,
            'risk_score_trend': risk_score_trend,
            'overall_trend': self._determine_overall_trend(drawdown_trend, pnl_trend, risk_score_trend)
        }
    
    def _calculate_metric_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a metric."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _determine_overall_trend(self, drawdown_trend: str, pnl_trend: str, risk_score_trend: str) -> str:
        """Determine overall risk trend."""
        if drawdown_trend == 'increasing' or risk_score_trend == 'increasing':
            return 'deteriorating'
        elif drawdown_trend == 'decreasing' and pnl_trend == 'increasing':
            return 'improving'
        else:
            return 'stable'
    
    def _assess_overall_risk(self, risk_metrics: RiskMetrics) -> Dict:
        """Assess overall risk level."""
        risk_level = 'low'
        
        if risk_metrics.risk_score > 75:
            risk_level = 'critical'
        elif risk_metrics.risk_score > 50:
            risk_level = 'high'
        elif risk_metrics.risk_score > 25:
            risk_level = 'medium'
        
        recommendations = []
        
        if risk_metrics.current_drawdown > 8:
            recommendations.append("Consider reducing position sizes")
        
        if risk_metrics.volatility_level > 0.04:
            recommendations.append("Monitor market volatility closely")
        
        if risk_metrics.position_exposure_percent > 80:
            recommendations.append("High position exposure - consider diversification")
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_metrics.risk_score,
            'recommendations': recommendations
        }