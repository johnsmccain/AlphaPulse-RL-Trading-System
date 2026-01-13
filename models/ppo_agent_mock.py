"""
Mock PPO Agent implementation for testing without PyTorch dependencies.

This module provides a lightweight mock of the PPO agent that maintains
the same interface but uses numpy instead of torch for basic testing.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MockActorNetwork:
    """Mock actor network using numpy."""
    
    def __init__(self, state_dim: int = 9, action_dim: int = 3, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize random weights
        self.w1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, 32) * 0.1
        self.b2 = np.zeros(32)
        self.w3 = np.random.randn(32, action_dim) * 0.1
        self.b3 = np.zeros(action_dim)
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through mock actor network."""
        x = np.maximum(0, np.dot(state, self.w1) + self.b1)  # ReLU
        x = np.maximum(0, np.dot(x, self.w2) + self.b2)      # ReLU
        x = np.tanh(np.dot(x, self.w3) + self.b3)           # Tanh
        return x
    
    def __call__(self, state):
        """Make the network callable like PyTorch modules."""
        return self.forward(state)


class MockCriticNetwork:
    """Mock critic network using numpy."""
    
    def __init__(self, state_dim: int = 9, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Initialize random weights
        self.w1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, 32) * 0.1
        self.b2 = np.zeros(32)
        self.w3 = np.random.randn(32, 1) * 0.1
        self.b3 = np.zeros(1)
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through mock critic network."""
        x = np.maximum(0, np.dot(state, self.w1) + self.b1)  # ReLU
        x = np.maximum(0, np.dot(x, self.w2) + self.b2)      # ReLU
        value = np.dot(x, self.w3) + self.b3
        return value
    
    def __call__(self, state):
        """Make the network callable like PyTorch modules."""
        return self.forward(state)


class PPOAgent:
    """
    Mock PPO Agent for testing without PyTorch dependencies.
    
    Maintains the same interface as the real PPO agent but uses numpy
    for computations to enable testing without full ML dependencies.
    """
    
    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 3,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu"
    ):
        """Initialize mock PPO agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device
        
        # Initialize mock networks
        self.actor = MockActorNetwork(state_dim, action_dim)
        self.critic = MockCriticNetwork(state_dim)
        
        # Action bounds for continuous action space
        self.action_bounds = {
            'position_direction': (-1.0, 1.0),
            'position_size': (0.0, 0.1),
            'leverage_multiplier': (1.0, 12.0)
        }
        
        logger.info(f"Mock PPO Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def _scale_actions(self, raw_actions) -> np.ndarray:
        """Scale raw network outputs to proper action ranges."""
        # Handle both numpy arrays and mock tensors
        if hasattr(raw_actions, 'data'):
            raw_actions = raw_actions.data
        
        raw_actions = np.array(raw_actions)
        if raw_actions.ndim == 1:
            raw_actions = raw_actions.reshape(1, -1)
        
        scaled_actions = np.zeros_like(raw_actions)
        
        # Position direction: [-1, 1] → [-1, 1] (no scaling needed)
        scaled_actions[:, 0] = raw_actions[:, 0]
        
        # Position size: [-1, 1] → [0, 0.1]
        scaled_actions[:, 1] = (raw_actions[:, 1] + 1) * 0.05
        
        # Leverage multiplier: [-1, 1] → [1, 12]
        scaled_actions[:, 2] = (raw_actions[:, 2] + 1) * 5.5 + 1
        
        return scaled_actions
    
    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given state."""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        raw_action = self.actor.forward(state)
        
        if not deterministic:
            # Add noise for exploration
            noise = np.random.normal(0, 0.1, size=raw_action.shape)
            raw_action = np.clip(raw_action + noise, -1, 1)
        
        scaled_action = self._scale_actions(raw_action)
        
        return scaled_action.flatten()
    
    def get_confidence(self, state: np.ndarray) -> float:
        """Calculate mock confidence for the given state."""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Get value estimate from critic
        value = self.critic.forward(state)[0]
        
        # Get action from actor
        raw_action = self.actor.forward(state)
        
        # Calculate mock confidence
        value_confidence = 1.0 / (1.0 + np.exp(-np.abs(value)))  # Sigmoid
        action_confidence = np.mean(np.abs(raw_action))
        
        # Combine confidences
        confidence = 0.6 * value_confidence + 0.4 * action_confidence
        
        return float(np.clip(confidence, 0.0, 1.0).item())
    
    def get_action_and_value(self, state) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get action, log probability, and value for given state."""
        # Handle both numpy arrays and mock tensors
        if hasattr(state, 'data'):
            state = state.data
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Get raw actions from actor
        raw_actions = self.actor.forward(state)
        
        # Scale actions to proper ranges
        actions = self._scale_actions(raw_actions)
        
        # Calculate mock log probabilities
        log_probs = -0.5 * np.sum(raw_actions**2, axis=1)
        
        # Get value estimates from critic
        values = self.critic.forward(state).flatten()
        
        return actions, log_probs, values
    
    def evaluate_actions(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate actions for given states."""
        if states.ndim == 1:
            states = states.reshape(1, -1)
        
        # Get raw actions from actor
        raw_actions = self.actor.forward(states)
        
        # Calculate log probabilities
        log_probs = -0.5 * np.sum(raw_actions**2, axis=1)
        
        # Get value estimates
        values = self.critic.forward(states).flatten()
        
        return log_probs, values
    
    def save_model(self, path: str) -> None:
        """Save mock model parameters to file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'actor_weights': {
                'w1': self.actor.w1.tolist(),
                'b1': self.actor.b1.tolist(),
                'w2': self.actor.w2.tolist(),
                'b2': self.actor.b2.tolist(),
                'w3': self.actor.w3.tolist(),
                'b3': self.actor.b3.tolist()
            },
            'critic_weights': {
                'w1': self.critic.w1.tolist(),
                'b1': self.critic.b1.tolist(),
                'w2': self.critic.w2.tolist(),
                'b2': self.critic.b2.tolist(),
                'w3': self.critic.w3.tolist(),
                'b3': self.critic.b3.tolist()
            },
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'clip_epsilon': self.clip_epsilon,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Mock model saved to {save_path}")
    
    def load_model(self, path: str) -> None:
        """Load mock model parameters from file."""
        with open(path, 'r') as f:
            model_data = json.load(f)
        
        # Load actor weights
        actor_weights = model_data['actor_weights']
        self.actor.w1 = np.array(actor_weights['w1'])
        self.actor.b1 = np.array(actor_weights['b1'])
        self.actor.w2 = np.array(actor_weights['w2'])
        self.actor.b2 = np.array(actor_weights['b2'])
        self.actor.w3 = np.array(actor_weights['w3'])
        self.actor.b3 = np.array(actor_weights['b3'])
        
        # Load critic weights
        critic_weights = model_data['critic_weights']
        self.critic.w1 = np.array(critic_weights['w1'])
        self.critic.b1 = np.array(critic_weights['b1'])
        self.critic.w2 = np.array(critic_weights['w2'])
        self.critic.b2 = np.array(critic_weights['b2'])
        self.critic.w3 = np.array(critic_weights['w3'])
        self.critic.b3 = np.array(critic_weights['b3'])
        
        # Load configuration
        config = model_data['config']
        self.gamma = config['gamma']
        self.clip_epsilon = config['clip_epsilon']
        self.entropy_coef = config['entropy_coef']
        self.value_coef = config['value_coef']
        
        logger.info(f"Mock model loaded from {path}")
    
    def set_training_mode(self, training: bool = True) -> None:
        """Mock training mode setter (no-op for numpy implementation)."""
        pass