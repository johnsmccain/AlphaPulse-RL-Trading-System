"""
Configuration loader for AlphaPulse-RL Trading System
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging


class ConfigLoader:
    """Handles loading and validation of configuration files"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._config_cache = {}
        
    def load_config(self, config_file: str = "config.yaml") -> Dict[str, Any]:
        """
        Load main system configuration
        
        Args:
            config_file: Configuration file name
            
        Returns:
            Configuration dictionary
        """
        if config_file not in self._config_cache:
            config_path = self.config_dir / config_file
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Apply environment variable overrides
            config = self._apply_env_overrides(config)
            self._config_cache[config_file] = config
            
        return self._config_cache[config_file]
    
    def load_trading_params(self, params_file: str = "trading_params.yaml") -> Dict[str, Any]:
        """
        Load trading parameters configuration
        
        Args:
            params_file: Trading parameters file name
            
        Returns:
            Trading parameters dictionary
        """
        return self.load_config(params_file)
    
    def get_api_credentials(self) -> Dict[str, str]:
        """
        Get API credentials from environment variables
        
        Returns:
            Dictionary with API credentials
        """
        credentials = {
            'api_key': os.getenv('WEEX_API_KEY'),
            'api_secret': os.getenv('WEEX_API_SECRET'),
            'passphrase': os.getenv('WEEX_PASSPHRASE'),
            'sandbox': os.getenv('WEEX_SANDBOX', 'false').lower() == 'true'
        }
        
        # Validate required credentials
        required_fields = ['api_key', 'api_secret', 'passphrase']
        missing_fields = [field for field in required_fields if not credentials[field]]
        
        if missing_fields:
            raise ValueError(f"Missing required API credentials: {missing_fields}")
            
        return credentials
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            Configuration with environment overrides applied
        """
        # Override logging level if set
        if os.getenv('LOG_LEVEL'):
            config.setdefault('logging', {})['level'] = os.getenv('LOG_LEVEL')
            
        # Override environment if set
        if os.getenv('ENVIRONMENT'):
            config.setdefault('system', {})['environment'] = os.getenv('ENVIRONMENT')
            
        # Override exchange sandbox mode
        if os.getenv('WEEX_SANDBOX'):
            config.setdefault('exchange', {})['sandbox'] = os.getenv('WEEX_SANDBOX').lower() == 'true'
            
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure and values
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, raises exception if invalid
        """
        required_sections = ['system', 'exchange', 'trading_pairs', 'data', 'model', 'logging']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
                
        # Validate trading pairs
        if not config['trading_pairs'] or not isinstance(config['trading_pairs'], list):
            raise ValueError("trading_pairs must be a non-empty list")
            
        # Validate logging configuration
        logging_config = config.get('logging', {})
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if logging_config.get('level') not in valid_log_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_log_levels}")
            
        return True


# Global configuration loader instance
config_loader = ConfigLoader()


def get_config() -> Dict[str, Any]:
    """Get main system configuration"""
    return config_loader.load_config()


def get_trading_params() -> Dict[str, Any]:
    """Get trading parameters configuration"""
    return config_loader.load_trading_params()


def get_api_credentials() -> Dict[str, str]:
    """Get API credentials"""
    return config_loader.get_api_credentials()