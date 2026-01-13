"""
Logging configuration for AlphaPulse-RL Trading System
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
import yaml


def setup_logging(config_path: str = "config/config.yaml"):
    """
    Set up logging configuration based on config file
    
    Args:
        config_path: Path to the configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging_config = config.get('logging', {})
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, logging_config.get('level', 'INFO')))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if logging_config.get('console_output', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if logging_config.get('file_output', True):
        log_file = logs_dir / f"alphapulse_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=logging_config.get('max_file_size_mb', 50) * 1024 * 1024,
            backupCount=logging_config.get('backup_count', 10)
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Create specialized loggers
    setup_trade_logger(logging_config)
    setup_ai_decision_logger(logging_config)
    
    logging.info("Logging system initialized")


def setup_trade_logger(logging_config: dict):
    """Set up specialized logger for trade history"""
    trade_logger = logging.getLogger('trade_history')
    trade_logger.setLevel(logging.INFO)
    
    # Create CSV file handler for trades
    trade_file = logging_config.get('trade_history_file', 'logs/trades.csv')
    trade_handler = logging.FileHandler(trade_file)
    
    # CSV format for trades
    csv_formatter = logging.Formatter('%(message)s')
    trade_handler.setFormatter(csv_formatter)
    trade_logger.addHandler(trade_handler)
    
    # Write CSV header if file is new
    if not os.path.exists(trade_file) or os.path.getsize(trade_file) == 0:
        trade_logger.info("timestamp,pair,action_direction,action_size,action_leverage,entry_price,exit_price,pnl,confidence,market_regime,portfolio_balance")


def setup_ai_decision_logger(logging_config: dict):
    """Set up specialized logger for AI decisions"""
    ai_logger = logging.getLogger('ai_decisions')
    ai_logger.setLevel(logging.INFO)
    
    # Create JSON file handler for AI decisions
    ai_file = logging_config.get('ai_decisions_file', 'logs/ai_decisions.json')
    ai_handler = logging.FileHandler(ai_file)
    
    # JSON format for AI decisions
    json_formatter = logging.Formatter('%(message)s')
    ai_handler.setFormatter(json_formatter)
    ai_logger.addHandler(ai_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_trade(timestamp: str, pair: str, action: list, entry_price: float, 
              exit_price: float = None, pnl: float = None, confidence: float = None,
              market_regime: int = None, portfolio_balance: float = None):
    """
    Log a trade to the trade history CSV
    
    Args:
        timestamp: Trade timestamp
        pair: Trading pair
        action: Action vector [direction, size, leverage]
        entry_price: Entry price
        exit_price: Exit price (optional)
        pnl: Profit/loss (optional)
        confidence: Model confidence
        market_regime: Market regime (0=range, 1=trend)
        portfolio_balance: Current portfolio balance
    """
    trade_logger = logging.getLogger('trade_history')
    
    trade_record = f"{timestamp},{pair},{action[0]:.4f},{action[1]:.4f},{action[2]:.4f},{entry_price:.4f},{exit_price or ''},{pnl or ''},{confidence or ''},{market_regime or ''},{portfolio_balance or ''}"
    trade_logger.info(trade_record)


def log_ai_decision(decision_data: dict):
    """
    Log an AI decision to the JSON file
    
    Args:
        decision_data: Dictionary containing decision details
    """
    import json
    
    ai_logger = logging.getLogger('ai_decisions')
    ai_logger.info(json.dumps(decision_data))