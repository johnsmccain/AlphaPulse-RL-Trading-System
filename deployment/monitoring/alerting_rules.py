#!/usr/bin/env python3
"""
AlphaPulse-RL Alerting Rules

Defines alerting rules, thresholds, and severity levels for the trading system.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class AlertCategory(Enum):
    """Alert categories."""
    SYSTEM = "SYSTEM"
    TRADING = "TRADING"
    RISK = "RISK"
    PERFORMANCE = "PERFORMANCE"
    CONNECTIVITY = "CONNECTIVITY"

@dataclass
class AlertRule:
    """Represents an alerting rule."""
    name: str
    rule_name: str
    threshold: float
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    severity: AlertSeverity
    category: AlertCategory
    cooldown_minutes: int = 15
    description: str = ""

class AlertingRules:
    """Manages all alerting rules for the system."""
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict[str, AlertRule]:
        """Initialize all alerting rules."""
        rules = {}
        
        # System Resource Alerts
        rules['high_cpu_warning'] = AlertRule(
            name="High CPU Usage Warning",
            rule_name="high_cpu_warning",
            threshold=80.0,
            operator=">=",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            cooldown_minutes=10,
            description="CPU usage is above 80%"
        )
        
        rules['high_cpu_critical'] = AlertRule(
            name="High CPU Usage Critical",
            rule_name="high_cpu_critical",
            threshold=90.0,
            operator=">=",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.SYSTEM,
            cooldown_minutes=5,
            description="CPU usage is above 90%"
        )
        
        rules['high_memory_warning'] = AlertRule(
            name="High Memory Usage Warning",
            rule_name="high_memory_warning",
            threshold=80.0,
            operator=">=",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            cooldown_minutes=10,
            description="Memory usage is above 80%"
        )
        
        rules['high_memory_critical'] = AlertRule(
            name="High Memory Usage Critical",
            rule_name="high_memory_critical",
            threshold=90.0,
            operator=">=",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.SYSTEM,
            cooldown_minutes=5,
            description="Memory usage is above 90%"
        )
        
        rules['high_disk_warning'] = AlertRule(
            name="High Disk Usage Warning",
            rule_name="high_disk_warning",
            threshold=85.0,
            operator=">=",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            cooldown_minutes=30,
            description="Disk usage is above 85%"
        )
        
        rules['high_disk_critical'] = AlertRule(
            name="High Disk Usage Critical",
            rule_name="high_disk_critical",
            threshold=95.0,
            operator=">=",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.SYSTEM,
            cooldown_minutes=15,
            description="Disk usage is above 95%"
        )
        
        # Trading System Alerts
        rules['trading_system_down'] = AlertRule(
            name="Trading System Down",
            rule_name="trading_system_down",
            threshold=1.0,
            operator=">=",
            severity=AlertSeverity.EMERGENCY,
            category=AlertCategory.TRADING,
            cooldown_minutes=5,
            description="Trading system is unavailable"
        )
        
        rules['api_error_rate_high'] = AlertRule(
            name="High API Error Rate",
            rule_name="api_error_rate_high",
            threshold=5.0,
            operator=">=",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.CONNECTIVITY,
            cooldown_minutes=10,
            description="API error rate is above 5%"
        )
        
        rules['api_error_rate_critical'] = AlertRule(
            name="Critical API Error Rate",
            rule_name="api_error_rate_critical",
            threshold=15.0,
            operator=">=",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.CONNECTIVITY,
            cooldown_minutes=5,
            description="API error rate is above 15%"
        )
        
        # Risk Management Alerts
        rules['daily_loss_warning'] = AlertRule(
            name="Daily Loss Warning",
            rule_name="daily_loss_warning",
            threshold=2.0,
            operator=">=",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            cooldown_minutes=30,
            description="Daily loss exceeds 2% of portfolio"
        )
        
        rules['daily_loss_critical'] = AlertRule(
            name="Daily Loss Critical",
            rule_name="daily_loss_critical",
            threshold=3.0,
            operator=">=",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            cooldown_minutes=15,
            description="Daily loss exceeds 3% of portfolio"
        )
        
        rules['drawdown_warning'] = AlertRule(
            name="Drawdown Warning",
            rule_name="drawdown_warning",
            threshold=8.0,
            operator=">=",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            cooldown_minutes=60,
            description="Portfolio drawdown exceeds 8%"
        )
        
        rules['drawdown_critical'] = AlertRule(
            name="Drawdown Critical",
            rule_name="drawdown_critical",
            threshold=12.0,
            operator=">=",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            cooldown_minutes=30,
            description="Portfolio drawdown exceeds 12%"
        )
        
        rules['position_size_violation'] = AlertRule(
            name="Position Size Violation",
            rule_name="position_size_violation",
            threshold=10.0,
            operator=">",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            cooldown_minutes=5,
            description="Position size exceeds 10% of portfolio"
        )
        
        rules['leverage_violation'] = AlertRule(
            name="Leverage Violation",
            rule_name="leverage_violation",
            threshold=12.0,
            operator=">",
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            cooldown_minutes=5,
            description="Leverage exceeds maximum allowed (12x)"
        )
        
        # Performance Alerts
        rules['model_inference_slow'] = AlertRule(
            name="Slow Model Inference",
            rule_name="model_inference_slow",
            threshold=5.0,
            operator=">=",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.PERFORMANCE,
            cooldown_minutes=20,
            description="Model inference time exceeds 5 seconds"
        )
        
        rules['low_confidence_trades'] = AlertRule(
            name="Low Confidence Trades",
            rule_name="low_confidence_trades",
            threshold=0.6,
            operator="<=",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.PERFORMANCE,
            cooldown_minutes=30,
            description="Average model confidence is below 60%"
        )
        
        rules['high_trade_frequency'] = AlertRule(
            name="High Trade Frequency",
            rule_name="high_trade_frequency",
            threshold=100.0,
            operator=">=",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.TRADING,
            cooldown_minutes=60,
            description="Trade frequency exceeds 100 trades per hour"
        )
        
        return rules
    
    def get_rule(self, rule_name: str) -> Optional[AlertRule]:
        """Get a specific alerting rule."""
        return self.rules.get(rule_name)
    
    def evaluate_rule(self, rule_name: str, value: float) -> bool:
        """Evaluate a rule against a value."""
        rule = self.get_rule(rule_name)
        if not rule:
            return False
        
        if rule.operator == ">=":
            return value >= rule.threshold
        elif rule.operator == ">":
            return value > rule.threshold
        elif rule.operator == "<=":
            return value <= rule.threshold
        elif rule.operator == "<":
            return value < rule.threshold
        elif rule.operator == "==":
            return value == rule.threshold
        elif rule.operator == "!=":
            return value != rule.threshold
        else:
            return False
    
    def get_alert_message(self, rule_name: str, value: float) -> str:
        """Generate alert message for a rule."""
        rule = self.get_rule(rule_name)
        if not rule:
            return f"Unknown rule: {rule_name}"
        
        # Generate contextual messages based on rule type
        if 'cpu' in rule_name.lower():
            return f"CPU usage is {value:.1f}% (threshold: {rule.threshold}%)"
        elif 'memory' in rule_name.lower():
            return f"Memory usage is {value:.1f}% (threshold: {rule.threshold}%)"
        elif 'disk' in rule_name.lower():
            return f"Disk usage is {value:.1f}% (threshold: {rule.threshold}%)"
        elif 'daily_loss' in rule_name.lower():
            return f"Daily loss is {value:.2f}% (threshold: {rule.threshold}%)"
        elif 'drawdown' in rule_name.lower():
            return f"Portfolio drawdown is {value:.2f}% (threshold: {rule.threshold}%)"
        elif 'api_error' in rule_name.lower():
            return f"API error rate is {value:.2f}% (threshold: {rule.threshold}%)"
        elif 'inference' in rule_name.lower():
            return f"Model inference time is {value:.2f}s (threshold: {rule.threshold}s)"
        elif 'confidence' in rule_name.lower():
            return f"Average model confidence is {value:.2f} (threshold: {rule.threshold})"
        elif 'trading_system_down' in rule_name.lower():
            return "Trading system is currently unavailable"
        elif 'position_size' in rule_name.lower():
            return f"Position size is {value:.2f}% (max allowed: {rule.threshold}%)"
        elif 'leverage' in rule_name.lower():
            return f"Leverage is {value:.1f}x (max allowed: {rule.threshold}x)"
        elif 'trade_frequency' in rule_name.lower():
            return f"Trade frequency is {value:.0f} trades/hour (threshold: {rule.threshold})"
        else:
            return f"{rule.name}: {value} {rule.operator} {rule.threshold}"
    
    def get_rules_by_category(self, category: AlertCategory) -> Dict[str, AlertRule]:
        """Get all rules for a specific category."""
        return {name: rule for name, rule in self.rules.items() 
                if rule.category == category}
    
    def get_rules_by_severity(self, severity: AlertSeverity) -> Dict[str, AlertRule]:
        """Get all rules for a specific severity level."""
        return {name: rule for name, rule in self.rules.items() 
                if rule.severity == severity}
    
    def get_all_rules(self) -> Dict[str, AlertRule]:
        """Get all alerting rules."""
        return self.rules.copy()

# Global instance
_alerting_rules = None

def get_alerting_rules() -> AlertingRules:
    """Get the global alerting rules instance."""
    global _alerting_rules
    if _alerting_rules is None:
        _alerting_rules = AlertingRules()
    return _alerting_rules