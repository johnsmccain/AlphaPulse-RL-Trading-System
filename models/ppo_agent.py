"""
PPO Agent implementation for AlphaPulse-RL Trading System.

This module implements the Proximal Policy Optimization agent with actor-critic
architecture specifically designed for cryptocurrency trading decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ActorNetwork(nn.Module):
    """
    Actor network for PPO agent.
    Architecture: 9 → 64 → 32 → 3 (state_dim → hidden → hidden → action_dim)
    """
    
    def __init__(self, state_dim: int = 9, action_dim: int = 3, hidden_dim: int = 64):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network.
        
        Args:
            state: Input state tensor of shape (batch_size, 9)
            
        Returns:
            Raw action logits of shape (batch_size, 3)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Use tanh activation for final layer to bound outputs
        x = torch.tanh(self.fc3(x))
        return x


class CriticNetwork(nn.Module):
    """
    Critic network for PPO agent.
    Architecture: 9 → 64 → 32 → 1 (state_dim → hidden → hidden → value)
    """
    
    def __init__(self, state_dim: int = 9, hidden_dim: int = 64):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network.
        
        Args:
            state: Input state tensor of shape (batch_size, 9)
            
        Returns:
            State value estimate of shape (batch_size, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class PPOAgent:
    """
    Proximal Policy Optimization agent for cryptocurrency trading.
    
    The agent outputs continuous actions for:
    - position_direction: [-1, 1] (short to long)
    - position_size: [0, 0.1] (0% to 10% of equity)
    - leverage_multiplier: [1, 12] (1x to 12x leverage)
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
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space (9 for our feature vector)
            action_dim: Dimension of action space (3 for direction, size, leverage)
            lr_actor: Learning rate for actor network
            lr_critic: Learning rate for critic network
            gamma: Discount factor for rewards
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
            device: Device to run computations on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Action bounds for continuous action space
        self.action_bounds = {
            'position_direction': (-1.0, 1.0),
            'position_size': (0.0, 0.1),
            'leverage_multiplier': (1.0, 12.0)
        }
        
        logger.info(f"PPO Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def _scale_actions(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """
        Scale raw network outputs to proper action ranges.
        
        Args:
            raw_actions: Raw tanh outputs from actor network [-1, 1]
            
        Returns:
            Scaled actions with proper bounds
        """
        scaled_actions = torch.zeros_like(raw_actions)
        
        # Position direction: [-1, 1] → [-1, 1] (no scaling needed)
        scaled_actions[:, 0] = raw_actions[:, 0]
        
        # Position size: [-1, 1] → [0, 0.1]
        scaled_actions[:, 1] = (raw_actions[:, 1] + 1) * 0.05  # Maps [-1,1] to [0, 0.1]
        
        # Leverage multiplier: [-1, 1] → [1, 12]
        scaled_actions[:, 2] = (raw_actions[:, 2] + 1) * 5.5 + 1  # Maps [-1,1] to [1, 12]
        
        return scaled_actions
    
    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict action for given state.
        
        Args:
            state: Current market state (9-dimensional vector)
            deterministic: If True, use mean action; if False, sample from distribution
            
        Returns:
            Action array [position_direction, position_size, leverage_multiplier]
        """
        self.actor.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            raw_action = self.actor(state_tensor)
            
            if not deterministic:
                # Add noise for exploration during training
                noise = torch.normal(0, 0.1, size=raw_action.shape).to(self.device)
                raw_action = torch.clamp(raw_action + noise, -1, 1)
            
            scaled_action = self._scale_actions(raw_action)
            
        return scaled_action.cpu().numpy().flatten()
    
    def get_confidence(self, state: np.ndarray) -> float:
        """
        Calculate model confidence for the given state.
        
        Confidence is based on the magnitude of the critic's value estimate
        and the consistency of actor outputs.
        
        Args:
            state: Current market state (9-dimensional vector)
            
        Returns:
            Confidence score between 0 and 1
        """
        self.actor.eval()
        self.critic.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get value estimate from critic
            value = self.critic(state_tensor).item()
            
            # Get action from actor
            raw_action = self.actor(state_tensor)
            
            # Calculate confidence based on:
            # 1. Magnitude of value estimate (higher absolute value = more confident)
            # 2. Action magnitude (stronger actions = more confident)
            value_confidence = torch.sigmoid(torch.abs(torch.tensor(value))).item()
            action_confidence = torch.mean(torch.abs(raw_action)).item()
            
            # Combine confidences with weighted average
            confidence = 0.6 * value_confidence + 0.4 * action_confidence
            
        return float(np.clip(confidence, 0.0, 1.0))
    
    def get_action_and_value(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value for given state.
        Used during training for PPO updates.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Tuple of (actions, log_probs, values)
        """
        # Get raw actions from actor
        raw_actions = self.actor(state)
        
        # Scale actions to proper ranges
        actions = self._scale_actions(raw_actions)
        
        # Calculate log probabilities (assuming Gaussian distribution)
        # For continuous actions, we use the raw actions for log prob calculation
        log_probs = -0.5 * torch.sum(raw_actions**2, dim=1)
        
        # Get value estimates from critic
        values = self.critic(state).squeeze()
        
        return actions, log_probs, values
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states.
        Used during PPO policy updates.
        
        Args:
            states: State tensor of shape (batch_size, state_dim)
            actions: Action tensor of shape (batch_size, action_dim)
            
        Returns:
            Tuple of (log_probs, values)
        """
        # Get raw actions from actor
        raw_actions = self.actor(states)
        
        # Calculate log probabilities
        log_probs = -0.5 * torch.sum(raw_actions**2, dim=1)
        
        # Get value estimates
        values = self.critic(states).squeeze()
        
        return log_probs, values
    
    def save_model(self, path: str) -> None:
        """
        Save model parameters to file.
        
        Args:
            path: Path to save the model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'clip_epsilon': self.clip_epsilon,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef
            }
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str) -> None:
        """
        Load model parameters from file.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Load configuration
        config = checkpoint['config']
        self.gamma = config['gamma']
        self.clip_epsilon = config['clip_epsilon']
        self.entropy_coef = config['entropy_coef']
        self.value_coef = config['value_coef']
        
        logger.info(f"Model loaded from {path}")
    
    def set_training_mode(self, training: bool = True) -> None:
        """Set networks to training or evaluation mode."""
        if training:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()