"""
Utility functions for PPO agent and model management in AlphaPulse-RL Trading System.

This module provides helper functions for model operations, configuration management,
and common utilities used across the training pipeline.
"""

import numpy as np
import yaml
import json
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import logging
from datetime import datetime
import pickle

# Conditional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration management for PPO models."""
    
    DEFAULT_CONFIG = {
        'agent': {
            'state_dim': 9,
            'action_dim': 3,
            'lr_actor': 3e-4,
            'lr_critic': 1e-3,
            'gamma': 0.99,
            'clip_epsilon': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'device': 'cpu'
        },
        'training': {
            'total_timesteps': 1000000,
            'buffer_size': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'max_grad_norm': 0.5,
            'target_kl': 0.01,
            'eval_freq': 10000,
            'save_freq': 50000,
            'log_freq': 1000
        },
        'environment': {
            'initial_balance': 10000,
            'transaction_cost': 0.001,
            'slippage': 0.0005,
            'max_position_size': 0.1,
            'max_leverage': 12.0
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    loaded_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    loaded_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # Deep merge with default config
            self.config = self._deep_merge(self.config, loaded_config)
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Using default configuration")
    
    def save_config(self, save_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                if save_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif save_path.suffix.lower() == '.json':
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported save format: {save_path.suffix}")
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {save_path}: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated key path (e.g., 'agent.lr_actor')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


class ModelCheckpoint:
    """Model checkpoint management utilities."""
    
    def __init__(self, checkpoint_dir: str = "models/checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        agent,
        trainer_state: Dict[str, Any],
        metrics: Dict[str, float],
        timestep: int,
        is_best: bool = False
    ) -> str:
        """
        Save model checkpoint with metadata.
        
        Args:
            agent: PPO agent to save
            trainer_state: Training state information
            metrics: Performance metrics
            timestep: Current training timestep
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            checkpoint_name = f"best_model_{timestamp}.pt"
        else:
            checkpoint_name = f"checkpoint_{timestep}_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint_data = {
            'agent_state': {
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
            },
            'agent_config': {
                'state_dim': agent.state_dim,
                'action_dim': agent.action_dim,
                'gamma': agent.gamma,
                'clip_epsilon': agent.clip_epsilon,
                'entropy_coef': agent.entropy_coef,
                'value_coef': agent.value_coef
            },
            'trainer_state': trainer_state,
            'metrics': metrics,
            'timestep': timestep,
            'timestamp': timestamp,
            'is_best': is_best
        }
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available, cannot save checkpoint")
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint data.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data dictionary
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available, cannot load checkpoint")
            
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint_data
    
    def get_best_checkpoint(self, metric: str = 'sharpe_ratio') -> Optional[str]:
        """
        Find the best checkpoint based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Path to best checkpoint or None if no checkpoints found
        """
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        
        if not checkpoints:
            return None
        
        best_checkpoint = None
        best_value = float('-inf')
        
        for checkpoint_path in checkpoints:
            try:
                if not TORCH_AVAILABLE:
                    logger.warning("PyTorch not available, cannot load checkpoints")
                    break
                    
                data = torch.load(checkpoint_path, map_location='cpu')
                value = data.get('metrics', {}).get(metric, float('-inf'))
                
                if value > best_value:
                    best_value = value
                    best_checkpoint = str(checkpoint_path)
                    
            except Exception as e:
                logger.warning(f"Error loading checkpoint {checkpoint_path}: {e}")
        
        return best_checkpoint
    
    def cleanup_old_checkpoints(self, keep_n: int = 5) -> None:
        """
        Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_n: Number of checkpoints to keep
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for checkpoint in checkpoints[keep_n:]:
            try:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                logger.warning(f"Error removing checkpoint {checkpoint}: {e}")


class ActionSpaceUtils:
    """Utilities for handling continuous action spaces."""
    
    @staticmethod
    def validate_action(action: np.ndarray) -> bool:
        """
        Validate if action is within expected bounds.
        
        Args:
            action: Action array [direction, size, leverage]
            
        Returns:
            True if action is valid
        """
        if len(action) != 3:
            return False
        
        direction, size, leverage = action
        
        # Check bounds
        if not (-1.0 <= direction <= 1.0):
            return False
        if not (0.0 <= size <= 0.1):
            return False
        if not (1.0 <= leverage <= 12.0):
            return False
        
        return True
    
    @staticmethod
    def clip_action(action: np.ndarray) -> np.ndarray:
        """
        Clip action to valid bounds.
        
        Args:
            action: Action array [direction, size, leverage]
            
        Returns:
            Clipped action array
        """
        clipped_action = action.copy()
        
        # Clip to bounds
        clipped_action[0] = np.clip(clipped_action[0], -1.0, 1.0)  # direction
        clipped_action[1] = np.clip(clipped_action[1], 0.0, 0.1)   # size
        clipped_action[2] = np.clip(clipped_action[2], 1.0, 12.0)  # leverage
        
        return clipped_action
    
    @staticmethod
    def interpret_action(action: np.ndarray) -> Dict[str, Any]:
        """
        Interpret raw action values into trading decisions.
        
        Args:
            action: Action array [direction, size, leverage]
            
        Returns:
            Dictionary with interpreted action
        """
        direction, size, leverage = action
        
        # Determine trade side
        if abs(direction) < 0.1:  # Dead zone for no action
            side = 'hold'
        elif direction > 0:
            side = 'long'
        else:
            side = 'short'
        
        # Size interpretation
        if size < 0.01:  # Minimum position size
            side = 'hold'
            size = 0.0
        
        return {
            'side': side,
            'direction': direction,
            'position_size': size,
            'leverage': leverage,
            'should_trade': side != 'hold'
        }


class StateNormalizer:
    """Normalize state vectors for better training stability."""
    
    def __init__(self, state_dim: int = 9):
        """
        Initialize state normalizer.
        
        Args:
            state_dim: Dimension of state vector
        """
        self.state_dim = state_dim
        self.running_mean = np.zeros(state_dim)
        self.running_var = np.ones(state_dim)
        self.count = 0
        self.epsilon = 1e-8
    
    def update(self, state: np.ndarray) -> None:
        """
        Update running statistics with new state.
        
        Args:
            state: New state observation
        """
        self.count += 1
        delta = state - self.running_mean
        self.running_mean += delta / self.count
        delta2 = state - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count
    
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state using running statistics.
        
        Args:
            state: State to normalize
            
        Returns:
            Normalized state
        """
        if self.count == 0:
            return state
        
        return (state - self.running_mean) / (np.sqrt(self.running_var) + self.epsilon)
    
    def denormalize(self, normalized_state: np.ndarray) -> np.ndarray:
        """
        Denormalize state back to original scale.
        
        Args:
            normalized_state: Normalized state
            
        Returns:
            Original scale state
        """
        if self.count == 0:
            return normalized_state
        
        return normalized_state * np.sqrt(self.running_var) + self.running_mean
    
    def save(self, path: str) -> None:
        """Save normalizer state."""
        state = {
            'running_mean': self.running_mean,
            'running_var': self.running_var,
            'count': self.count,
            'state_dim': self.state_dim
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str) -> None:
        """Load normalizer state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.running_mean = state['running_mean']
        self.running_var = state['running_var']
        self.count = state['count']
        self.state_dim = state['state_dim']


def setup_device(prefer_gpu: bool = True) -> Optional[object]:
    """
    Setup computation device (CPU/GPU).
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        PyTorch device or None if torch not available
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, returning None for device")
        return None
        
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def count_parameters(model) -> int:
    """
    Count total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, cannot count parameters")
        return 0
        
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Random seed set to {seed}")


def get_model_summary(agent) -> Dict[str, Any]:
    """
    Get summary information about the model.
    
    Args:
        agent: PPO agent
        
    Returns:
        Model summary dictionary
    """
    if not TORCH_AVAILABLE:
        return {
            'error': 'PyTorch not available',
            'total_parameters': 0
        }
        
    actor_params = count_parameters(agent.actor)
    critic_params = count_parameters(agent.critic)
    total_params = actor_params + critic_params
    
    return {
        'actor_parameters': actor_params,
        'critic_parameters': critic_params,
        'total_parameters': total_params,
        'state_dim': agent.state_dim,
        'action_dim': agent.action_dim,
        'device': str(agent.device),
        'gamma': agent.gamma,
        'clip_epsilon': agent.clip_epsilon,
        'entropy_coef': agent.entropy_coef,
        'value_coef': agent.value_coef
    }


if __name__ == "__main__":
    # Example usage
    try:
        from models.ppo_agent import PPOAgent
        
        # Test configuration management
        config = ModelConfig()
        print("Default config:", config.get('agent.lr_actor'))
        
        # Test model utilities
        agent = PPOAgent()
        summary = get_model_summary(agent)
        print("Model summary:", summary)
        
        # Test action utilities
        test_action = np.array([0.5, 0.05, 8.0])
        print("Action valid:", ActionSpaceUtils.validate_action(test_action))
        print("Action interpretation:", ActionSpaceUtils.interpret_action(test_action))
        
    except ImportError as e:
        print(f"PyTorch not available: {e}")
        
        # Test configuration management without PyTorch
        config = ModelConfig()
        print("Default config:", config.get('agent.lr_actor'))
        
        # Test action utilities
        test_action = np.array([0.5, 0.05, 8.0])
        print("Action valid:", ActionSpaceUtils.validate_action(test_action))
        print("Action interpretation:", ActionSpaceUtils.interpret_action(test_action))