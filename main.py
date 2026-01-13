#!/usr/bin/env python3
"""
AlphaPulse-RL Trading System
Main entry point for the trading system
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.logging_config import setup_logging
from config.config_loader import get_config, get_trading_params


def main():
    """Main entry point for the AlphaPulse-RL trading system"""
    
    parser = argparse.ArgumentParser(description="AlphaPulse-RL Trading System")
    parser.add_argument(
        "--mode", 
        choices=["train", "backtest", "live", "paper"],
        required=True,
        help="Operating mode: train model, run backtest, live trading, or paper trading"
    )
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model", 
        help="Path to trained model file"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Initialize logging
    setup_logging(args.config)
    logger = logging.getLogger(__name__)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = get_config()
        trading_params = get_trading_params()
        
        logger.info(f"Starting AlphaPulse-RL in {args.mode} mode")
        logger.info(f"System version: {config['system']['version']}")
        logger.info(f"Environment: {config['system']['environment']}")
        
        if args.mode == "train":
            logger.info("Training mode - model training will be implemented in task 4")
            print("Training mode selected. Model training functionality will be implemented in task 4.")
            
        elif args.mode == "backtest":
            logger.info("Backtesting mode - backtesting will be implemented in task 8")
            print("Backtesting mode selected. Backtesting functionality will be implemented in task 8.")
            
        elif args.mode == "live":
            logger.info("Live trading mode - live trading will be implemented in task 7")
            print("Live trading mode selected. Live trading functionality will be implemented in task 7.")
            
        elif args.mode == "paper":
            logger.info("Paper trading mode - paper trading will be implemented in task 7")
            print("Paper trading mode selected. Paper trading functionality will be implemented in task 7.")
            
        logger.info("System initialization completed successfully")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()