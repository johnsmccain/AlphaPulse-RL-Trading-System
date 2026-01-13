#!/usr/bin/env python3
"""
AlphaPulse-RL Monitoring Service

Provides real-time monitoring dashboard and alerting for the trading system.
"""

import os
import json
import time
import logging
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from flask import Flask, render_template, jsonify, request
import redis
import psutil
import requests

# Import alerting rules
from alerting_rules import get_alerting_rules, AlertSeverity, AlertCategory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
MONITORING_INTERVAL = int(os.getenv('MONITORING_INTERVAL', 30))
ALERT_EMAIL = os.getenv('ALERT_EMAIL')
ALERT_WEBHOOK = os.getenv('ALERT_WEBHOOK')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'localhost')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')

# Initialize alerting rules
alerting_rules = get_alerting_rules()

# Initialize Redis connection
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    logger.info("Connected to Redis")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

class AlertManager:
    """Manages alerting for the trading system."""
    
    def __init__(self):
        self.last_alerts = {}
        self.active_alerts = {}
        self.alert_history = []
    
    def evaluate_metrics(self, metrics: Dict[str, Any]):
        """Evaluate metrics against all alerting rules."""
        current_time = datetime.now()
        
        # System metrics evaluation
        system_metrics = metrics.get('system', {})
        self._check_system_alerts(system_metrics, current_time)
        
        # Trading metrics evaluation
        trading_metrics = metrics.get('trading', {})
        self._check_trading_alerts(trading_metrics, current_time)
    
    def _check_system_alerts(self, system_metrics: Dict[str, Any], current_time: datetime):
        """Check system-related alerts."""
        cpu_percent = system_metrics.get('cpu_percent', 0)
        memory_percent = system_metrics.get('memory_percent', 0)
        disk_percent = system_metrics.get('disk_percent', 0)
        
        # CPU alerts
        if alerting_rules.evaluate_rule('high_cpu_critical', cpu_percent):
            self._trigger_alert('high_cpu_critical', cpu_percent, current_time)
        elif alerting_rules.evaluate_rule('high_cpu_warning', cpu_percent):
            self._trigger_alert('high_cpu_warning', cpu_percent, current_time)
        
        # Memory alerts
        if alerting_rules.evaluate_rule('high_memory_critical', memory_percent):
            self._trigger_alert('high_memory_critical', memory_percent, current_time)
        elif alerting_rules.evaluate_rule('high_memory_warning', memory_percent):
            self._trigger_alert('high_memory_warning', memory_percent, current_time)
        
        # Disk alerts
        if alerting_rules.evaluate_rule('high_disk_critical', disk_percent):
            self._trigger_alert('high_disk_critical', disk_percent, current_time)
        elif alerting_rules.evaluate_rule('high_disk_warning', disk_percent):
            self._trigger_alert('high_disk_warning', disk_percent, current_time)
    
    def _check_trading_alerts(self, trading_metrics: Dict[str, Any], current_time: datetime):
        """Check trading-related alerts."""
        # Trading system availability
        if trading_metrics.get('status') == 'unavailable':
            self._trigger_alert('trading_system_down', 1, current_time)
        
        # API error rate
        api_error_rate = trading_metrics.get('api_error_rate', 0)
        if alerting_rules.evaluate_rule('api_error_rate_critical', api_error_rate):
            self._trigger_alert('api_error_rate_critical', api_error_rate, current_time)
        elif alerting_rules.evaluate_rule('api_error_rate_high', api_error_rate):
            self._trigger_alert('api_error_rate_high', api_error_rate, current_time)
        
        # Risk metrics
        portfolio = trading_metrics.get('portfolio', {})
        daily_pnl_percent = abs(portfolio.get('daily_pnl_percent', 0))
        drawdown_percent = portfolio.get('max_drawdown_percent', 0)
        
        # Daily loss alerts
        if alerting_rules.evaluate_rule('daily_loss_critical', daily_pnl_percent):
            self._trigger_alert('daily_loss_critical', daily_pnl_percent, current_time)
        elif alerting_rules.evaluate_rule('daily_loss_warning', daily_pnl_percent):
            self._trigger_alert('daily_loss_warning', daily_pnl_percent, current_time)
        
        # Drawdown alerts
        if alerting_rules.evaluate_rule('drawdown_critical', drawdown_percent):
            self._trigger_alert('drawdown_critical', drawdown_percent, current_time)
        elif alerting_rules.evaluate_rule('drawdown_warning', drawdown_percent):
            self._trigger_alert('drawdown_warning', drawdown_percent, current_time)
        
        # Performance alerts
        avg_inference_time = trading_metrics.get('avg_inference_time_ms', 0) / 1000  # Convert to seconds
        if alerting_rules.evaluate_rule('model_inference_slow', avg_inference_time):
            self._trigger_alert('model_inference_slow', avg_inference_time, current_time)
        
        avg_confidence = trading_metrics.get('avg_model_confidence', 1.0)
        if alerting_rules.evaluate_rule('low_confidence_trades', avg_confidence):
            self._trigger_alert('low_confidence_trades', avg_confidence, current_time)
    
    def _trigger_alert(self, rule_name: str, value: float, current_time: datetime):
        """Trigger an alert if cooldown period has passed."""
        rule = alerting_rules.get_rule(rule_name)
        if not rule:
            return
        
        # Check cooldown
        last_alert_time = self.last_alerts.get(rule_name)
        if last_alert_time:
            time_diff = (current_time - last_alert_time).total_seconds() / 60
            if time_diff < rule.cooldown_minutes:
                return
        
        # Create alert
        alert_data = {
            'rule_name': rule_name,
            'rule': rule.name,
            'severity': rule.severity.value,
            'category': rule.category.value,
            'message': alerting_rules.get_alert_message(rule_name, value),
            'value': value,
            'threshold': rule.threshold,
            'timestamp': current_time.isoformat(),
            'acknowledged': False
        }
        
        # Store alert
        self.active_alerts[rule_name] = alert_data
        self.alert_history.append(alert_data)
        self.last_alerts[rule_name] = current_time
        
        # Store in Redis
        if redis_client:
            try:
                redis_client.lpush('alphapulse_alerts', json.dumps(alert_data))
                redis_client.ltrim('alphapulse_alerts', 0, 99)  # Keep last 100 alerts
            except Exception as e:
                logger.error(f"Failed to store alert in Redis: {e}")
        
        # Send notifications
        self._send_alert_notifications(alert_data)
        
        logger.warning(f"ALERT TRIGGERED [{rule.severity.value}] {rule.name}: {alert_data['message']}")
    
    def _send_alert_notifications(self, alert_data: Dict[str, Any]):
        """Send alert notifications via configured channels."""
        
        # Send email notification
        if ALERT_EMAIL and SMTP_USERNAME and SMTP_PASSWORD:
            try:
                self._send_email_notification(alert_data)
            except Exception as e:
                logger.error(f"Failed to send email notification: {str(e)}")
        
        # Send webhook notification
        if ALERT_WEBHOOK:
            try:
                self._send_webhook_notification(alert_data)
            except Exception as e:
                logger.error(f"Failed to send webhook notification: {str(e)}")
    
    def _send_email_notification(self, alert_data: Dict[str, Any]):
        """Send email notification."""
        msg = MimeMultipart()
        msg['From'] = f"AlphaPulse Alert System <{SMTP_USERNAME}>"
        msg['To'] = ALERT_EMAIL
        msg['Subject'] = f"[{alert_data['severity']}] AlphaPulse Alert: {alert_data['rule']}"
        
        # Create email body
        body = f"""
AlphaPulse-RL Trading System Alert

Alert Details:
- Rule: {alert_data['rule']}
- Category: {alert_data['category']}
- Severity: {alert_data['severity']}
- Message: {alert_data['message']}
- Current Value: {alert_data['value']}
- Threshold: {alert_data['threshold']}
- Timestamp: {alert_data['timestamp']}

Please check the monitoring dashboard for more details and take appropriate action.

This is an automated alert from the AlphaPulse-RL trading system.
"""
        
        msg.attach(MimeText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email notification sent for alert: {alert_data['rule_name']}")
    
    def _send_webhook_notification(self, alert_data: Dict[str, Any]):
        """Send webhook notification (e.g., to Slack, Discord, etc.)."""
        
        # Format for Slack webhook
        color_map = {
            'INFO': '#36a64f',      # Green
            'WARNING': '#ff9500',   # Orange
            'CRITICAL': '#ff0000',  # Red
            'EMERGENCY': '#8b0000'  # Dark Red
        }
        
        webhook_payload = {
            "text": f"AlphaPulse Alert: {alert_data['rule']}",
            "attachments": [
                {
                    "color": color_map.get(alert_data['severity'], '#808080'),
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert_data['severity'],
                            "short": True
                        },
                        {
                            "title": "Category",
                            "value": alert_data['category'],
                            "short": True
                        },
                        {
                            "title": "Message",
                            "value": alert_data['message'],
                            "short": False
                        },
                        {
                            "title": "Value / Threshold",
                            "value": f"{alert_data['value']} / {alert_data['threshold']}",
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": alert_data['timestamp'],
                            "short": True
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(
            ALERT_WEBHOOK,
            json=webhook_payload,
            timeout=10
        )
        response.raise_for_status()
        
        logger.info(f"Webhook notification sent for alert: {alert_data['rule_name']}")
    
    def acknowledge_alert(self, rule_name: str, acknowledged_by: str):
        """Acknowledge an active alert."""
        if rule_name in self.active_alerts:
            self.active_alerts[rule_name]['acknowledged'] = True
            self.active_alerts[rule_name]['acknowledged_by'] = acknowledged_by
            self.active_alerts[rule_name]['acknowledged_at'] = datetime.now().isoformat()
            
            logger.info(f"Alert acknowledged by {acknowledged_by}: {rule_name}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if datetime.fromisoformat(alert['timestamp']) > last_24h
        ]
        
        stats = {
            'active_alerts_count': len(self.active_alerts),
            'alerts_last_24h': len(recent_alerts),
            'severity_breakdown': {},
            'category_breakdown': {},
            'most_frequent_alerts': {}
        }
        
        # Breakdown by severity and category
        for alert in recent_alerts:
            severity = alert['severity']
            category = alert['category']
            rule_name = alert['rule_name']
            
            stats['severity_breakdown'][severity] = stats['severity_breakdown'].get(severity, 0) + 1
            stats['category_breakdown'][category] = stats['category_breakdown'].get(category, 0) + 1
            stats['most_frequent_alerts'][rule_name] = stats['most_frequent_alerts'].get(rule_name, 0) + 1
        
        return stats

# Initialize alert manager
alert_manager = AlertManager()

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'alphapulse-monitor'
    })

@app.route('/')
def dashboard():
    """Main monitoring dashboard."""
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    """Get current system metrics."""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3)
            }
        }
        
        # Try to get trading system metrics
        try:
            response = requests.get('http://alphapulse-trading:8080/metrics', timeout=5)
            if response.status_code == 200:
                trading_metrics = response.json()
                metrics['trading'] = trading_metrics
            else:
                metrics['trading'] = {'status': 'unavailable'}
        except Exception as e:
            logger.warning(f"Failed to get trading metrics: {e}")
            metrics['trading'] = {'status': 'unavailable'}
        
        # Evaluate alerts
        alert_manager.evaluate_metrics(metrics)
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts."""
    try:
        active_alerts = alert_manager.get_active_alerts()
        alert_stats = alert_manager.get_alert_statistics()
        
        # Also get alerts from Redis for historical data
        redis_alerts = []
        if redis_client:
            try:
                alert_data = redis_client.lrange('alphapulse_alerts', 0, 49)  # Last 50 alerts
                for alert_json in alert_data:
                    try:
                        alert = json.loads(alert_json)
                        redis_alerts.append(alert)
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                logger.warning(f"Failed to get alerts from Redis: {e}")
        
        return jsonify({
            'active_alerts': active_alerts,
            'recent_alerts': redis_alerts,
            'statistics': alert_stats
        })
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/acknowledge', methods=['POST'])
def acknowledge_alert():
    """Acknowledge an alert."""
    try:
        data = request.get_json()
        rule_name = data.get('rule_name')
        acknowledged_by = data.get('acknowledged_by', 'unknown')
        
        if not rule_name:
            return jsonify({'error': 'rule_name is required'}), 400
        
        alert_manager.acknowledge_alert(rule_name, acknowledged_by)
        
        return jsonify({'success': True, 'message': f'Alert {rule_name} acknowledged'})
        
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs')
def get_logs():
    """Get recent log entries."""
    try:
        log_entries = []
        
        # Try to get logs from Redis first
        if redis_client:
            try:
                logs_data = redis_client.lrange('alphapulse_logs', 0, 99)
                for log_json in logs_data:
                    try:
                        log_entry = json.loads(log_json)
                        log_entries.append(log_entry)
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                logger.warning(f"Failed to get logs from Redis: {e}")
        
        # Fallback to file-based logs
        if not log_entries:
            log_file = '/app/logs/alphapulse.log'
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Get last 100 lines
                    for line in lines[-100:]:
                        if line.strip():
                            log_entries.append({
                                'timestamp': datetime.now().isoformat(),
                                'message': line.strip()
                            })
        
        return jsonify({'logs': log_entries})
        
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    """Get performance analytics."""
    try:
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() - psutil.boot_time(),
            'process_count': len(psutil.pids()),
            'network_io': psutil.net_io_counters()._asdict(),
            'disk_io': psutil.disk_io_counters()._asdict()
        }
        
        return jsonify(performance_data)
        
    except Exception as e:
        logger.error(f"Failed to get performance data: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=False)