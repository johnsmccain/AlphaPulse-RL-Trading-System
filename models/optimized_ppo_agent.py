"""
Optimized PPO Agent with performance enhancements for production deployment.

This module extends the base PPO agent with caching, batch processing,
and memory optimization for improved inference speed.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import threading
from collections import deque
import time

from models.ppo_agent import PPOAgent, ActorNetwork, CriticNetwork
from utils.performance_optimizer import (
    PerformanceOptimizer, performance_monitor, cached_function,
    IntelligentCache, ModelOptimizer
)

logger = logging.getLogger(__name__)


class OptimizedActorNetwork(ActorNetwork):
    """Optimized actor network with inference optimizations."""
    
    def __init__(self, state_dim: int = 9, action_dim: int = 3, hidden_dim: int = 64):
        super().__init__(state_dim, action_dim, hidden_dim)
        
        # Pre-allocate tensors for common batch sizes
        self.tensor_cache = {}
        self.common_batch_sizes = [1, 4, 8, 16, 32]
        
        # Initialize tensor cache
        self._init_tensor_cache()
    
    def _init_tensor_cache(self):
        """Pre-allocate tensors for common operations."""
        device = next(self.parameters()).device
        
        for batch_size in self.common_batch_sizes:
            self.tensor_cache[batch_size] = {
                'input_buffer': torch.zeros(batch_size, 9, device=device),
                'output_buffer': torch.zeros(batch_size, 3, device=device)
            }
    
    def forward_optimized(self, state: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass using pre-allocated tensors."""
        batch_size = state.size(0)
        
        # Use cached tensors if available
        if batch_size in self.tensor_cache:
            input_buffer = self.tensor_cache[batch_size]['input_buffer']
            input_buffer[:batch_size] = state
            
            # Forward pass
            x = torch.relu(self.fc1(input_buffer[:batch_size]))
            x = torch.relu(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            
            return x
        else:
            # Fallback to regular forward
            return self.forward(state)


class OptimizedCriticNetwork(CriticNetwork):
    """Optimized critic network with inference optimizations."""
    
    def __init__(self, state_dim: int = 9, hidden_dim: int = 64):
        super().__init__(state_dim, hidden_dim)
        
        # Pre-allocate tensors
        self.tensor_cache = {}
        self.common_batch_sizes = [1, 4, 8, 16, 32]
        self._init_tensor_cache()
    
    def _init_tensor_cache(self):
        """Pre-allocate tensors for common operations."""
        device = next(self.parameters()).device
        
        for batch_size in self.common_batch_sizes:
            self.tensor_cache[batch_size] = {
                'input_buffer': torch.zeros(batch_size, 9, device=device),
                'output_buffer': torch.zeros(batch_size, 1, device=device)
            }
    
    def forward_optimized(self, state: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass using pre-allocated tensors."""
        batch_size = state.size(0)
        
        if batch_size in self.tensor_cache:
            input_buffer = self.tensor_cache[batch_size]['input_buffer']
            input_buffer[:batch_size] = state
            
            # Forward pass
            x = torch.relu(self.fc1(input_buffer[:batch_size]))
            x = torch.relu(self.fc2(x))
            value = self.fc3(x)
            
            return value
        else:
            return self.forward(state)


class OptimizedPPOAgent(PPOAgent):
    """
    Optimized PPO Agent with performance enhancements:
    - Prediction caching
    - Batch inference optimization
    - Memory-efficient operations
    - Confidence calculation caching
    """
    
    def __init__(self, config: Dict[str, Any], performance_optimizer: PerformanceOptimizer):
        # Initialize base agent
        super().__init__(
            state_dim=config.get('state_dim', 9),
            action_dim=config.get('action_dim', 3),
            lr_actor=config.get('lr_actor', 3e-4),
            lr_critic=config.get('lr_critic', 1e-3),
            gamma=config.get('gamma', 0.99),
            clip_epsilon=config.get('clip_epsilon', 0.2),
            entropy_coef=config.get('entropy_coef', 0.01),
            value_coef=config.get('value_coef', 0.5),
            device=config.get('device', 'cpu')
        )
        
        self.performance_optimizer = performance_optimizer
        self.config = config
        
        # Replace networks with optimized versions
        self.actor = OptimizedActorNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic = OptimizedCriticNetwork(self.state_dim).to(self.device)
        
        # Optimize models
        self.actor = self.performance_optimizer.optimize_model_inference(self.actor, 'actor')
        self.critic = self.performance_optimizer.optimize_model_inference(self.critic, 'critic')
        
        # Prediction cache
        self.prediction_cache = IntelligentCache(
            max_size=config.get('prediction_cache_size', 100),
            default_ttl=config.get('prediction_cache_ttl', 30)
        )
        
        # Confidence cache
        self.confidence_cache = IntelligentCache(
            max_size=config.get('confidence_cache_size', 100),
            default_ttl=config.get('confidence_cache_ttl', 30)
        )
        
        # Batch processing
        self.batch_size = config.get('inference_batch_size', 8)
        self.pending_predictions = deque()
        self.batch_lock = threading.Lock()
        
        # Performance monitoring
        self.inference_times = deque(maxlen=100)
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info("OptimizedPPOAgent initialized with performance enhancements")
    
    @performance_monitor
    def predict_optimized(self, state: np.ndarray, deterministic: bool = True, 
                         use_cache: bool = True) -> np.ndarray:
        """
        Optimized prediction with caching and performance monitoring.
        
        Args:
            state: Current market state (9-dimensional vector)
            deterministic: If True, use mean action; if False, sample from distribution
            use_cache: Whether to use prediction caching
            
        Returns:
            Action array [position_direction, position_size, leverage_multiplier]
        """
        start_time = time.perf_counter()
        
        # Generate cache key if caching is enabled
        if use_cache:
            state_key = self._generate_state_key(state, deterministic)
            cached_prediction = self.prediction_cache.get(state_key)
            
            if cached_prediction is not None:
                self.cache_hits += 1
                return cached_prediction
            
            self.cache_misses += 1
        
        # Perform prediction
        self.actor.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Use optimized forward pass
            if hasattr(self.actor, 'forward_optimized'):
                raw_action = self.actor.forward_optimized(state_tensor)
            else:
                raw_action = self.actor(state_tensor)
            
            if not deterministic:
                # Add noise for exploration
                noise_std = self.config.get('exploration_noise_std', 0.1)
                noise = torch.normal(0, noise_std, size=raw_action.shape).to(self.device)
                raw_action = torch.clamp(raw_action + noise, -1, 1)
            
            scaled_action = self._scale_actions(raw_action)
        
        prediction = scaled_action.cpu().numpy().flatten()
        
        # Cache the prediction
        if use_cache:
            self.prediction_cache.put(state_key, prediction.copy())
        
        # Record inference time
        inference_time = (time.perf_counter() - start_time) * 1000
        self.inference_times.append(inference_time)
        self.performance_optimizer.monitor.record_metric('model_inference_time_ms', inference_time)
        
        return prediction
    
    @performance_monitor
    def get_confidence_optimized(self, state: np.ndarray, use_cache: bool = True) -> float:
        """
        Optimized confidence calculation with caching.
        
        Args:
            state: Current market state (9-dimensional vector)
            use_cache: Whether to use confidence caching
            
        Returns:
            Confidence score between 0 and 1
        """
        # Generate cache key if caching is enabled
        if use_cache:
            state_key = self._generate_state_key(state, deterministic=True, suffix='_confidence')
            cached_confidence = self.confidence_cache.get(state_key)
            
            if cached_confidence is not None:
                return cached_confidence
        
        # Calculate confidence
        self.actor.eval()
        self.critic.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get value estimate from critic
            if hasattr(self.critic, 'forward_optimized'):
                value = self.critic.forward_optimized(state_tensor).item()
            else:
                value = self.critic(state_tensor).item()
            
            # Get action from actor
            if hasattr(self.actor, 'forward_optimized'):
                raw_action = self.actor.forward_optimized(state_tensor)
            else:
                raw_action = self.actor(state_tensor)
            
            # Calculate confidence components
            value_confidence = torch.sigmoid(torch.abs(torch.tensor(value))).item()
            action_confidence = torch.mean(torch.abs(raw_action)).item()
            
            # Combine confidences
            confidence = 0.6 * value_confidence + 0.4 * action_confidence
        
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        # Cache the confidence
        if use_cache:
            self.confidence_cache.put(state_key, confidence)
        
        return confidence
    
    def batch_predict(self, states: List[np.ndarray], deterministic: bool = True) -> List[np.ndarray]:
        """
        Batch prediction for multiple states to improve GPU utilization.
        
        Args:
            states: List of market states
            deterministic: If True, use mean actions
            
        Returns:
            List of action arrays
        """
        if not states:
            return []
        
        if len(states) == 1:
            return [self.predict_optimized(states[0], deterministic)]
        
        start_time = time.perf_counter()
        
        self.actor.eval()
        
        with torch.no_grad():
            # Stack states into batch tensor
            state_batch = torch.FloatTensor(np.stack(states)).to(self.device)
            
            # Batch forward pass
            if hasattr(self.actor, 'forward_optimized'):
                raw_actions = self.actor.forward_optimized(state_batch)
            else:
                raw_actions = self.actor(state_batch)
            
            if not deterministic:
                # Add exploration noise
                noise_std = self.config.get('exploration_noise_std', 0.1)
                noise = torch.normal(0, noise_std, size=raw_actions.shape).to(self.device)
                raw_actions = torch.clamp(raw_actions + noise, -1, 1)
            
            # Scale actions
            scaled_actions = self._scale_actions(raw_actions)
        
        # Convert to list of numpy arrays
        predictions = [scaled_actions[i].cpu().numpy() for i in range(len(states))]
        
        # Record batch inference time
        batch_time = (time.perf_counter() - start_time) * 1000
        avg_time_per_prediction = batch_time / len(states)
        self.performance_optimizer.monitor.record_metric('batch_inference_time_ms', batch_time)
        self.performance_optimizer.monitor.record_metric('avg_batch_prediction_time_ms', avg_time_per_prediction)
        
        return predictions
    
    def _generate_state_key(self, state: np.ndarray, deterministic: bool, suffix: str = '') -> str:
        """Generate cache key for state."""
        # Round state values to reduce cache key variations
        rounded_state = np.round(state, decimals=4)
        state_str = ','.join(f'{x:.4f}' for x in rounded_state)
        return f"{state_str}_{deterministic}{suffix}"
    
    def warm_up_cache(self, sample_states: List[np.ndarray]) -> None:
        """Warm up prediction cache with sample states."""
        logger.info(f"Warming up cache with {len(sample_states)} sample states")
        
        for state in sample_states:
            # Cache both deterministic and non-deterministic predictions
            self.predict_optimized(state, deterministic=True, use_cache=True)
            self.predict_optimized(state, deterministic=False, use_cache=True)
            self.get_confidence_optimized(state, use_cache=True)
        
        logger.info("Cache warm-up completed")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the optimized agent."""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0.0
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'avg_inference_time_ms': avg_inference_time,
            'prediction_cache_stats': self.prediction_cache.get_stats(),
            'confidence_cache_stats': self.confidence_cache.get_stats(),
            'recent_inference_times': list(self.inference_times)[-10:]  # Last 10 times
        }
    
    def optimize_for_deployment(self) -> None:
        """Apply deployment-specific optimizations."""
        logger.info("Applying deployment optimizations...")
        
        # Set models to evaluation mode
        self.actor.eval()
        self.critic.eval()
        
        # Disable gradient computation
        for param in self.actor.parameters():
            param.requires_grad = False
        for param in self.critic.parameters():
            param.requires_grad = False
        
        # Try to use torch.jit.script for additional optimization
        try:
            self.actor = torch.jit.script(self.actor)
            self.critic = torch.jit.script(self.critic)
            logger.info("Models optimized with TorchScript")
        except Exception as e:
            logger.warning(f"TorchScript optimization failed: {e}")
        
        # Pre-allocate common tensors
        self._preallocate_tensors()
        
        logger.info("Deployment optimizations applied")
    
    def _preallocate_tensors(self) -> None:
        """Pre-allocate commonly used tensors."""
        # Pre-allocate tensors for single predictions (most common case)
        self.single_state_tensor = torch.zeros(1, self.state_dim, device=self.device)
        self.single_action_tensor = torch.zeros(1, self.action_dim, device=self.device)
        
        # Pre-allocate for small batches
        for batch_size in [2, 4, 8]:
            setattr(self, f'batch_{batch_size}_state_tensor', 
                   torch.zeros(batch_size, self.state_dim, device=self.device))
            setattr(self, f'batch_{batch_size}_action_tensor', 
                   torch.zeros(batch_size, self.action_dim, device=self.device))
    
    def clear_caches(self) -> Dict[str, int]:
        """Clear all caches and return statistics."""
        prediction_cleared = self.prediction_cache.invalidate()
        confidence_cleared = self.confidence_cache.invalidate()
        
        # Reset counters
        self.cache_hits = 0
        self.cache_misses = 0
        self.inference_times.clear()
        
        return {
            'prediction_cache_cleared': prediction_cleared,
            'confidence_cache_cleared': confidence_cleared
        }
    
    def save_optimized_model(self, path: str) -> None:
        """Save optimized model with performance metadata."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get performance stats
        perf_stats = self.get_performance_stats()
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'config': self.config,
            'performance_stats': perf_stats,
            'optimization_metadata': {
                'optimized': True,
                'cache_enabled': True,
                'batch_inference_enabled': True,
                'deployment_ready': True
            }
        }, save_path)
        
        logger.info(f"Optimized model saved to {save_path}")
    
    def load_optimized_model(self, path: str) -> None:
        """Load optimized model and restore performance settings."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Restore configuration
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        # Apply deployment optimizations if this was an optimized model
        if checkpoint.get('optimization_metadata', {}).get('deployment_ready', False):
            self.optimize_for_deployment()
        
        logger.info(f"Optimized model loaded from {path}")


def create_optimized_agent(config: Dict[str, Any], performance_optimizer: PerformanceOptimizer) -> OptimizedPPOAgent:
    """Factory function to create optimized PPO agent."""
    return OptimizedPPOAgent(config, performance_optimizer)