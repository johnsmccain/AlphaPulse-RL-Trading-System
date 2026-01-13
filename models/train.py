"""
Training pipeline for PPO agent in AlphaPulse-RL Trading System.

This module implements the complete training loop with experience collection,
policy updates, and evaluation metrics calculation.
"""

# Conditional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

# Conditional matplotlib import
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    mdates = None

from collections import defaultdict

from .ppo_agent import PPOAgent

logger = logging.getLogger(__name__)

# Import will be done at runtime to avoid circular imports
# from env.weex_trading_env import WeexTradingEnv


class ExperienceBuffer:
    """Buffer to store training experiences for PPO updates."""
    
    def __init__(self, buffer_size: int = 2048):
        self.buffer_size = buffer_size
        self.clear()
    
    def clear(self):
        """Clear all stored experiences."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []
    
    def add(self, state, action, reward, value, log_prob, done, next_state):
        """Add a single experience to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.next_states.append(next_state)
    
    def get_batch(self) -> Dict[str, Any]:
        """Get all experiences as tensors for training."""
        if not TORCH_AVAILABLE:
            # Return numpy arrays if torch not available
            return {
                'states': np.array(self.states),
                'actions': np.array(self.actions),
                'rewards': np.array(self.rewards),
                'values': np.array(self.values),
                'log_probs': np.array(self.log_probs),
                'dones': np.array(self.dones),
                'next_states': np.array(self.next_states)
            }
        
        return {
            'states': torch.FloatTensor(np.array(self.states)),
            'actions': torch.FloatTensor(np.array(self.actions)),
            'rewards': torch.FloatTensor(np.array(self.rewards)),
            'values': torch.FloatTensor(np.array(self.values)),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)),
            'dones': torch.BoolTensor(np.array(self.dones)),
            'next_states': torch.FloatTensor(np.array(self.next_states))
        }
    
    def size(self) -> int:
        """Return current buffer size."""
        return len(self.states)
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.states) >= self.buffer_size


class PPOTrainer:
    """
    PPO training pipeline with experience collection and policy updates.
    """
    
    def __init__(
        self,
        agent: PPOAgent,
        env,
        buffer_size: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
        save_dir: str = "models/checkpoints"
    ):
        """
        Initialize PPO trainer.
        
        Args:
            agent: PPO agent to train
            env: Trading environment
            buffer_size: Size of experience buffer
            batch_size: Batch size for training updates
            n_epochs: Number of epochs per update
            max_grad_norm: Maximum gradient norm for clipping
            target_kl: Target KL divergence for early stopping
            save_dir: Directory to save model checkpoints
        """
        self.agent = agent
        self.env = env
        self.buffer = ExperienceBuffer(buffer_size)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'total_losses': [],
            'kl_divergences': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'win_rates': []
        }
        
        logger.info("PPO Trainer initialized")
    
    def collect_experiences(self, n_steps: int) -> None:
        """
        Collect experiences by running the agent in the environment.
        
        Args:
            n_steps: Number of steps to collect
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available, cannot collect experiences")
            return
            
        self.buffer.clear()
        state = self.env.reset()
        
        for step in range(n_steps):
            # Get action and value from agent
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            action, log_prob, value = self.agent.get_action_and_value(state_tensor)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action.cpu().numpy().flatten())
            
            # Store experience
            self.buffer.add(
                state=state,
                action=action.cpu().numpy().flatten(),
                reward=reward,
                value=value.item(),
                log_prob=log_prob.item(),
                done=done,
                next_state=next_state
            )
            
            state = next_state
            
            if done:
                state = self.env.reset()
        
        logger.debug(f"Collected {self.buffer.size()} experiences")
    
    def compute_advantages(self, batch: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Tuple of (advantages, returns)
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available, cannot compute advantages")
            return None, None
            
        rewards = batch['rewards']
        values = batch['values']
        dones = batch['dones']
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute returns and advantages
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.agent.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.agent.gamma * 0.95 * (1 - dones[t]) * gae  # GAE lambda = 0.95
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.
        
        Returns:
            Dictionary of training metrics
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available, cannot update policy")
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'total_loss': 0.0,
                'kl_divergence': 0.0
            }
            
        batch = self.buffer.get_batch()
        batch = {k: v.to(self.agent.device) for k, v in batch.items()}
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(batch)
        
        # Store old log probs for PPO clipping
        old_log_probs = batch['log_probs'].detach()
        
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'total_loss': 0.0,
            'kl_divergence': 0.0
        }
        
        # PPO update epochs
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = torch.randperm(len(batch['states']))
            
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_kl_div = 0.0
            n_batches = 0
            
            # Mini-batch updates
            for start_idx in range(0, len(indices), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # Get current policy outputs
                current_log_probs, current_values = self.agent.evaluate_actions(
                    batch['states'][batch_indices],
                    batch['actions'][batch_indices]
                )
                
                # Calculate policy loss
                ratio = torch.exp(current_log_probs - old_log_probs[batch_indices])
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1 - self.agent.clip_epsilon, 1 + self.agent.clip_epsilon) * advantages[batch_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = torch.nn.functional.mse_loss(current_values, returns[batch_indices])
                
                # Calculate entropy bonus (for exploration)
                entropy = -current_log_probs.mean()
                
                # Total loss
                total_loss = policy_loss + self.agent.value_coef * value_loss - self.agent.entropy_coef * entropy
                
                # Backward pass
                self.agent.actor_optimizer.zero_grad()
                self.agent.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
                
                self.agent.actor_optimizer.step()
                self.agent.critic_optimizer.step()
                
                # Calculate KL divergence for early stopping
                kl_div = (old_log_probs[batch_indices] - current_log_probs).mean()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_kl_div += kl_div.item()
                n_batches += 1
            
            # Average losses for this epoch
            epoch_policy_loss /= n_batches
            epoch_value_loss /= n_batches
            epoch_kl_div /= n_batches
            
            # Early stopping if KL divergence is too high
            if epoch_kl_div > self.target_kl:
                logger.debug(f"Early stopping at epoch {epoch} due to high KL divergence: {epoch_kl_div:.4f}")
                break
            
            metrics['policy_loss'] = epoch_policy_loss
            metrics['value_loss'] = epoch_value_loss
            metrics['total_loss'] = epoch_policy_loss + epoch_value_loss
            metrics['kl_divergence'] = epoch_kl_div
        
        return metrics
    
    def evaluate_agent(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate agent performance over multiple episodes.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.agent.set_training_mode(False)
        
        episode_rewards = []
        episode_lengths = []
        episode_returns = []
        episode_trades = []
        episode_win_rates = []
        episode_sharpe_ratios = []
        episode_max_drawdowns = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_return = []
            trade_count = 0
            winning_trades = 0
            
            done = False
            while not done:
                action = self.agent.predict(state, deterministic=True)
                state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                episode_return.append(reward)
                
                # Track trades
                if info['trade_result']['success'] and info['trade_result']['position_size'] > 0:
                    trade_count += 1
                    if reward > 0:
                        winning_trades += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_returns.append(episode_return)
            episode_trades.append(trade_count)
            
            # Calculate episode-specific metrics
            if trade_count > 0:
                episode_win_rate = winning_trades / trade_count
            else:
                episode_win_rate = 0.0
            episode_win_rates.append(episode_win_rate)
            
            # Calculate Sharpe ratio for this episode
            if len(episode_return) > 1:
                returns_std = np.std(episode_return)
                if returns_std > 0:
                    episode_sharpe = np.mean(episode_return) / returns_std * np.sqrt(252)
                else:
                    episode_sharpe = 0.0
            else:
                episode_sharpe = 0.0
            episode_sharpe_ratios.append(episode_sharpe)
            
            # Get max drawdown from environment
            episode_stats = self.env.get_episode_stats()
            episode_max_drawdowns.append(episode_stats['max_drawdown'])
        
        # Calculate aggregate metrics
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        avg_trades = np.mean(episode_trades)
        
        # Calculate overall Sharpe ratio
        all_returns = np.concatenate(episode_returns)
        if len(all_returns) > 1:
            returns_std = np.std(all_returns)
            if returns_std > 0:
                overall_sharpe = np.mean(all_returns) / returns_std * np.sqrt(252)
            else:
                overall_sharpe = 0.0
        else:
            overall_sharpe = 0.0
        
        # Calculate maximum drawdown across all episodes
        cumulative_returns = np.cumsum(all_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / (running_max + 1e-8)
        max_drawdown = abs(np.min(drawdowns))
        
        # Calculate overall win rate
        total_winning_trades = sum(episode_win_rates[i] * episode_trades[i] for i in range(len(episode_win_rates)))
        total_trades = sum(episode_trades)
        overall_win_rate = total_winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit factor
        positive_returns = [r for r in all_returns if r > 0]
        negative_returns = [r for r in all_returns if r < 0]
        
        gross_profit = sum(positive_returns) if positive_returns else 0
        gross_loss = abs(sum(negative_returns)) if negative_returns else 1e-8
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        metrics = {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'avg_trades_per_episode': avg_trades,
            'sharpe_ratio': overall_sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': overall_win_rate,
            'profit_factor': profit_factor,
            'total_episodes': n_episodes,
            'total_returns': len(all_returns),
            'avg_episode_sharpe': np.mean(episode_sharpe_ratios),
            'std_episode_rewards': np.std(episode_rewards),
            'consistency_score': 1.0 - (np.std(episode_rewards) / (abs(np.mean(episode_rewards)) + 1e-8))
        }
        
        self.agent.set_training_mode(True)
        return metrics
    
    def train(
        self,
        total_timesteps: int,
        eval_freq: int = 10000,
        save_freq: int = 50000,
        log_freq: int = 1000
    ) -> None:
        """
        Main training loop.
        
        Args:
            total_timesteps: Total number of timesteps to train
            eval_freq: Frequency of evaluation (in timesteps)
            save_freq: Frequency of model saving (in timesteps)
            log_freq: Frequency of logging (in timesteps)
        """
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        timesteps_collected = 0
        update_count = 0
        
        while timesteps_collected < total_timesteps:
            # Collect experiences
            self.collect_experiences(self.buffer.buffer_size)
            timesteps_collected += self.buffer.size()
            
            # Update policy
            update_metrics = self.update_policy()
            update_count += 1
            
            # Store training metrics
            self.training_history['policy_losses'].append(update_metrics['policy_loss'])
            self.training_history['value_losses'].append(update_metrics['value_loss'])
            self.training_history['total_losses'].append(update_metrics['total_loss'])
            self.training_history['kl_divergences'].append(update_metrics['kl_divergence'])
            
            # Logging
            if timesteps_collected % log_freq == 0:
                logger.info(
                    f"Timesteps: {timesteps_collected}/{total_timesteps}, "
                    f"Updates: {update_count}, "
                    f"Policy Loss: {update_metrics['policy_loss']:.4f}, "
                    f"Value Loss: {update_metrics['value_loss']:.4f}, "
                    f"KL Div: {update_metrics['kl_divergence']:.4f}"
                )
            
            # Evaluation
            if timesteps_collected % eval_freq == 0:
                eval_metrics = self.evaluate_agent()
                
                # Store evaluation metrics
                self.training_history['sharpe_ratios'].append(eval_metrics['sharpe_ratio'])
                self.training_history['max_drawdowns'].append(eval_metrics['max_drawdown'])
                self.training_history['win_rates'].append(eval_metrics['win_rate'])
                
                logger.info(
                    f"Evaluation - Sharpe: {eval_metrics['sharpe_ratio']:.4f}, "
                    f"Max DD: {eval_metrics['max_drawdown']:.4f}, "
                    f"Win Rate: {eval_metrics['win_rate']:.4f}"
                )
            
            # Save model
            if timesteps_collected % save_freq == 0:
                save_path = self.save_dir / f"ppo_agent_{timesteps_collected}.pt"
                self.agent.save_model(str(save_path))
                
                # Save training checkpoint
                self.save_training_checkpoint(timesteps_collected)
                
                # Save training history
                history_path = self.save_dir / f"training_history_{timesteps_collected}.json"
                with open(history_path, 'w') as f:
                    json.dump(self.training_history, f, indent=2)
        
        # Final save
        final_save_path = self.save_dir / "ppo_agent_final.pt"
        self.agent.save_model(str(final_save_path))
        
        # Save final training history
        final_history_path = self.save_dir / "training_history_final.json"
        with open(final_history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Generate and save training report
        self._generate_training_report()
        
        logger.info("Training completed successfully")
    
    def _generate_training_report(self) -> None:
        """Generate comprehensive training report"""
        report_path = self.save_dir / "training_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# PPO Training Report - AlphaPulse-RL Trading System\n\n")
            f.write(f"**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Training Configuration
            f.write("## Training Configuration\n\n")
            f.write(f"- **Buffer Size:** {self.buffer.buffer_size}\n")
            f.write(f"- **Batch Size:** {self.batch_size}\n")
            f.write(f"- **Training Epochs:** {self.n_epochs}\n")
            f.write(f"- **Max Grad Norm:** {self.max_grad_norm}\n")
            f.write(f"- **Target KL:** {self.target_kl}\n")
            f.write(f"- **Agent Device:** {self.agent.device}\n\n")
            
            # Agent Configuration
            f.write("## Agent Configuration\n\n")
            f.write(f"- **State Dimension:** {self.agent.state_dim}\n")
            f.write(f"- **Action Dimension:** {self.agent.action_dim}\n")
            f.write(f"- **Gamma (Discount):** {self.agent.gamma}\n")
            f.write(f"- **Clip Epsilon:** {self.agent.clip_epsilon}\n")
            f.write(f"- **Entropy Coefficient:** {self.agent.entropy_coef}\n")
            f.write(f"- **Value Coefficient:** {self.agent.value_coef}\n\n")
            
            # Training Results
            f.write("## Training Results\n\n")
            if self.training_history['policy_losses']:
                f.write(f"- **Total Updates:** {len(self.training_history['policy_losses'])}\n")
                f.write(f"- **Final Policy Loss:** {self.training_history['policy_losses'][-1]:.6f}\n")
                f.write(f"- **Final Value Loss:** {self.training_history['value_losses'][-1]:.6f}\n")
                f.write(f"- **Final Total Loss:** {self.training_history['total_losses'][-1]:.6f}\n")
                f.write(f"- **Final KL Divergence:** {self.training_history['kl_divergences'][-1]:.6f}\n\n")
                
                # Loss trends
                policy_trend = np.polyfit(range(len(self.training_history['policy_losses'])), 
                                        self.training_history['policy_losses'], 1)[0]
                value_trend = np.polyfit(range(len(self.training_history['value_losses'])), 
                                       self.training_history['value_losses'], 1)[0]
                
                f.write(f"- **Policy Loss Trend:** {'Decreasing' if policy_trend < 0 else 'Increasing'} "
                       f"({policy_trend:.8f} per update)\n")
                f.write(f"- **Value Loss Trend:** {'Decreasing' if value_trend < 0 else 'Increasing'} "
                       f"({value_trend:.8f} per update)\n\n")
            
            # Performance Metrics
            if self.training_history['sharpe_ratios']:
                f.write("## Performance Metrics\n\n")
                f.write(f"- **Final Sharpe Ratio:** {self.training_history['sharpe_ratios'][-1]:.4f}\n")
                f.write(f"- **Best Sharpe Ratio:** {max(self.training_history['sharpe_ratios']):.4f}\n")
                f.write(f"- **Average Sharpe Ratio:** {np.mean(self.training_history['sharpe_ratios']):.4f}\n")
                f.write(f"- **Sharpe Ratio Std:** {np.std(self.training_history['sharpe_ratios']):.4f}\n\n")
                
                f.write(f"- **Final Max Drawdown:** {self.training_history['max_drawdowns'][-1]*100:.2f}%\n")
                f.write(f"- **Worst Drawdown:** {max(self.training_history['max_drawdowns'])*100:.2f}%\n")
                f.write(f"- **Average Drawdown:** {np.mean(self.training_history['max_drawdowns'])*100:.2f}%\n\n")
                
                f.write(f"- **Final Win Rate:** {self.training_history['win_rates'][-1]*100:.1f}%\n")
                f.write(f"- **Best Win Rate:** {max(self.training_history['win_rates'])*100:.1f}%\n")
                f.write(f"- **Average Win Rate:** {np.mean(self.training_history['win_rates'])*100:.1f}%\n\n")
                
                # Performance trends
                sharpe_trend = np.polyfit(range(len(self.training_history['sharpe_ratios'])), 
                                        self.training_history['sharpe_ratios'], 1)[0]
                f.write(f"- **Sharpe Ratio Trend:** {'Improving' if sharpe_trend > 0 else 'Declining'} "
                       f"({sharpe_trend:.6f} per evaluation)\n\n")
            
            # Training Stability
            f.write("## Training Stability\n\n")
            if len(self.training_history['policy_losses']) > 10:
                recent_policy_losses = self.training_history['policy_losses'][-10:]
                policy_stability = 1.0 - (np.std(recent_policy_losses) / (abs(np.mean(recent_policy_losses)) + 1e-8))
                f.write(f"- **Policy Loss Stability (last 10 updates):** {policy_stability:.4f}\n")
                
                recent_value_losses = self.training_history['value_losses'][-10:]
                value_stability = 1.0 - (np.std(recent_value_losses) / (abs(np.mean(recent_value_losses)) + 1e-8))
                f.write(f"- **Value Loss Stability (last 10 updates):** {value_stability:.4f}\n\n")
            
            if len(self.training_history['sharpe_ratios']) > 5:
                recent_sharpe = self.training_history['sharpe_ratios'][-5:]
                sharpe_consistency = 1.0 - (np.std(recent_sharpe) / (abs(np.mean(recent_sharpe)) + 1e-8))
                f.write(f"- **Performance Consistency (last 5 evaluations):** {sharpe_consistency:.4f}\n\n")
            
            # Recommendations
            f.write("## Training Assessment\n\n")
            
            # Assess final performance
            if self.training_history['sharpe_ratios']:
                final_sharpe = self.training_history['sharpe_ratios'][-1]
                final_drawdown = self.training_history['max_drawdowns'][-1]
                final_winrate = self.training_history['win_rates'][-1]
                
                if final_sharpe > 1.0:
                    f.write("✅ **Sharpe Ratio:** Excellent (>1.0)\n")
                elif final_sharpe > 0.5:
                    f.write("⚠️ **Sharpe Ratio:** Good (0.5-1.0)\n")
                else:
                    f.write("❌ **Sharpe Ratio:** Needs Improvement (<0.5)\n")
                
                if final_drawdown < 0.05:
                    f.write("✅ **Risk Control:** Excellent (<5% drawdown)\n")
                elif final_drawdown < 0.12:
                    f.write("⚠️ **Risk Control:** Acceptable (<12% drawdown)\n")
                else:
                    f.write("❌ **Risk Control:** Poor (>12% drawdown)\n")
                
                if final_winrate > 0.6:
                    f.write("✅ **Win Rate:** Excellent (>60%)\n")
                elif final_winrate > 0.5:
                    f.write("⚠️ **Win Rate:** Good (>50%)\n")
                else:
                    f.write("❌ **Win Rate:** Needs Improvement (<50%)\n")
                
                f.write("\n")
            
            # Training recommendations
            f.write("## Recommendations\n\n")
            
            if self.training_history['policy_losses']:
                if self.training_history['policy_losses'][-1] > self.training_history['policy_losses'][0]:
                    f.write("- Consider reducing learning rate - policy loss is increasing\n")
                
                avg_kl = np.mean(self.training_history['kl_divergences'][-10:]) if len(self.training_history['kl_divergences']) >= 10 else 0
                if avg_kl > self.target_kl * 2:
                    f.write("- Consider reducing clip epsilon - KL divergence is high\n")
                elif avg_kl < self.target_kl * 0.1:
                    f.write("- Consider increasing clip epsilon - KL divergence is very low\n")
            
            if self.training_history['sharpe_ratios']:
                if len(self.training_history['sharpe_ratios']) > 5:
                    recent_trend = np.polyfit(range(-5, 0), self.training_history['sharpe_ratios'][-5:], 1)[0]
                    if recent_trend < -0.01:
                        f.write("- Performance is declining - consider early stopping or hyperparameter adjustment\n")
                    elif recent_trend < 0.001:
                        f.write("- Performance has plateaued - consider longer training or different approach\n")
            
            f.write("\n---\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        logger.info(f"Training report saved to {report_path}")
    
    def save_training_checkpoint(self, timesteps: int) -> None:
        """Save training checkpoint with full state"""
        checkpoint_path = self.save_dir / f"training_checkpoint_{timesteps}.pt"
        
        checkpoint = {
            'timesteps': timesteps,
            'agent_state': {
                'actor_state_dict': self.agent.actor.state_dict(),
                'critic_state_dict': self.agent.critic.state_dict(),
                'actor_optimizer_state_dict': self.agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
            },
            'training_history': self.training_history,
            'trainer_config': {
                'buffer_size': self.buffer.buffer_size,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'max_grad_norm': self.max_grad_norm,
                'target_kl': self.target_kl
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Training checkpoint saved to {checkpoint_path}")
    
    def load_training_checkpoint(self, checkpoint_path: str) -> int:
        """Load training checkpoint and resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.agent.device)
        
        # Restore agent state
        self.agent.actor.load_state_dict(checkpoint['agent_state']['actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['agent_state']['critic_state_dict'])
        self.agent.actor_optimizer.load_state_dict(checkpoint['agent_state']['actor_optimizer_state_dict'])
        self.agent.critic_optimizer.load_state_dict(checkpoint['agent_state']['critic_optimizer_state_dict'])
        
        # Restore training history
        self.training_history = checkpoint['training_history']
        
        # Restore trainer config
        trainer_config = checkpoint['trainer_config']
        self.batch_size = trainer_config['batch_size']
        self.n_epochs = trainer_config['n_epochs']
        self.max_grad_norm = trainer_config['max_grad_norm']
        self.target_kl = trainer_config['target_kl']
        
        timesteps = checkpoint['timesteps']
        logger.info(f"Training checkpoint loaded from {checkpoint_path}, resuming from timestep {timesteps}")
        
        return timesteps
    
    def plot_training_progress(self, save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive training progress metrics.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Create a 3x4 grid for comprehensive metrics
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Training losses
        ax1 = fig.add_subplot(gs[0, 0])
        if self.training_history['policy_losses']:
            ax1.plot(self.training_history['policy_losses'], label='Policy Loss', color='red', alpha=0.7)
            ax1.set_title('Policy Loss')
            ax1.set_xlabel('Update')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        if self.training_history['value_losses']:
            ax2.plot(self.training_history['value_losses'], label='Value Loss', color='blue', alpha=0.7)
            ax2.set_title('Value Loss')
            ax2.set_xlabel('Update')
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        if self.training_history['total_losses']:
            ax3.plot(self.training_history['total_losses'], label='Total Loss', color='purple', alpha=0.7)
            ax3.set_title('Total Loss')
            ax3.set_xlabel('Update')
            ax3.set_ylabel('Loss')
            ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[0, 3])
        if self.training_history['kl_divergences']:
            ax4.plot(self.training_history['kl_divergences'], label='KL Divergence', color='orange', alpha=0.7)
            ax4.axhline(y=self.target_kl, color='red', linestyle='--', alpha=0.5, label=f'Target KL ({self.target_kl})')
            ax4.set_title('KL Divergence')
            ax4.set_xlabel('Update')
            ax4.set_ylabel('KL Div')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Performance metrics
        ax5 = fig.add_subplot(gs[1, 0])
        if self.training_history['sharpe_ratios']:
            ax5.plot(self.training_history['sharpe_ratios'], label='Sharpe Ratio', color='green', linewidth=2)
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax5.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Target (1.0)')
            ax5.set_title('Sharpe Ratio')
            ax5.set_xlabel('Evaluation')
            ax5.set_ylabel('Sharpe Ratio')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 1])
        if self.training_history['max_drawdowns']:
            ax6.plot(self.training_history['max_drawdowns'], label='Max Drawdown', color='red', linewidth=2)
            ax6.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Warning (5%)')
            ax6.axhline(y=0.12, color='red', linestyle='--', alpha=0.5, label='Limit (12%)')
            ax6.set_title('Maximum Drawdown')
            ax6.set_xlabel('Evaluation')
            ax6.set_ylabel('Drawdown')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        ax7 = fig.add_subplot(gs[1, 2])
        if self.training_history['win_rates']:
            ax7.plot(self.training_history['win_rates'], label='Win Rate', color='blue', linewidth=2)
            ax7.axhline(y=0.5, color='black', linestyle='-', alpha=0.3, label='Break-even')
            ax7.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Target (60%)')
            ax7.set_title('Win Rate')
            ax7.set_xlabel('Evaluation')
            ax7.set_ylabel('Win Rate')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # Combined performance view
        ax8 = fig.add_subplot(gs[1, 3])
        if (self.training_history['sharpe_ratios'] and 
            self.training_history['max_drawdowns'] and 
            self.training_history['win_rates']):
            
            # Normalize metrics for comparison
            sharpe_norm = np.array(self.training_history['sharpe_ratios']) / 2.0  # Normalize by target of 2.0
            drawdown_norm = 1.0 - np.array(self.training_history['max_drawdowns']) / 0.12  # Invert and normalize by 12% limit
            winrate_norm = np.array(self.training_history['win_rates'])
            
            ax8.plot(sharpe_norm, label='Sharpe (norm)', alpha=0.7)
            ax8.plot(drawdown_norm, label='1-Drawdown (norm)', alpha=0.7)
            ax8.plot(winrate_norm, label='Win Rate', alpha=0.7)
            ax8.set_title('Normalized Performance Metrics')
            ax8.set_xlabel('Evaluation')
            ax8.set_ylabel('Normalized Score')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # Training stability metrics
        ax9 = fig.add_subplot(gs[2, 0])
        if len(self.training_history['policy_losses']) > 10:
            # Calculate rolling standard deviation of losses
            window = 10
            policy_loss_rolling_std = []
            for i in range(window, len(self.training_history['policy_losses'])):
                std = np.std(self.training_history['policy_losses'][i-window:i])
                policy_loss_rolling_std.append(std)
            
            ax9.plot(policy_loss_rolling_std, label='Policy Loss Volatility', color='red', alpha=0.7)
            ax9.set_title('Training Stability (Loss Volatility)')
            ax9.set_xlabel('Update')
            ax9.set_ylabel('Rolling Std')
            ax9.grid(True, alpha=0.3)
        
        # Learning progress
        ax10 = fig.add_subplot(gs[2, 1])
        if self.training_history['sharpe_ratios']:
            # Calculate moving average of Sharpe ratio
            window = min(5, len(self.training_history['sharpe_ratios']))
            if window > 1:
                sharpe_ma = []
                for i in range(window-1, len(self.training_history['sharpe_ratios'])):
                    ma = np.mean(self.training_history['sharpe_ratios'][i-window+1:i+1])
                    sharpe_ma.append(ma)
                
                ax10.plot(self.training_history['sharpe_ratios'], alpha=0.3, color='green', label='Raw Sharpe')
                ax10.plot(range(window-1, len(self.training_history['sharpe_ratios'])), 
                         sharpe_ma, color='green', linewidth=2, label=f'MA({window})')
                ax10.set_title('Learning Progress (Sharpe Ratio)')
                ax10.set_xlabel('Evaluation')
                ax10.set_ylabel('Sharpe Ratio')
                ax10.legend()
                ax10.grid(True, alpha=0.3)
        
        # Performance distribution
        ax11 = fig.add_subplot(gs[2, 2])
        if self.training_history['sharpe_ratios']:
            ax11.hist(self.training_history['sharpe_ratios'], bins=min(10, len(self.training_history['sharpe_ratios'])//2), 
                     alpha=0.7, color='green', edgecolor='black')
            ax11.axvline(x=np.mean(self.training_history['sharpe_ratios']), color='red', 
                        linestyle='--', label=f'Mean: {np.mean(self.training_history["sharpe_ratios"]):.3f}')
            ax11.set_title('Sharpe Ratio Distribution')
            ax11.set_xlabel('Sharpe Ratio')
            ax11.set_ylabel('Frequency')
            ax11.legend()
            ax11.grid(True, alpha=0.3)
        
        # Training summary
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.axis('off')
        
        # Create summary text
        summary_text = "Training Summary\n" + "="*20 + "\n"
        
        if self.training_history['policy_losses']:
            summary_text += f"Total Updates: {len(self.training_history['policy_losses'])}\n"
            summary_text += f"Final Policy Loss: {self.training_history['policy_losses'][-1]:.4f}\n"
            summary_text += f"Final Value Loss: {self.training_history['value_losses'][-1]:.4f}\n"
        
        if self.training_history['sharpe_ratios']:
            summary_text += f"Final Sharpe Ratio: {self.training_history['sharpe_ratios'][-1]:.3f}\n"
            summary_text += f"Best Sharpe Ratio: {max(self.training_history['sharpe_ratios']):.3f}\n"
            summary_text += f"Final Max Drawdown: {self.training_history['max_drawdowns'][-1]*100:.1f}%\n"
            summary_text += f"Final Win Rate: {self.training_history['win_rates'][-1]*100:.1f}%\n"
        
        # Add training stability metrics
        if len(self.training_history['policy_losses']) > 1:
            policy_loss_trend = np.polyfit(range(len(self.training_history['policy_losses'])), 
                                         self.training_history['policy_losses'], 1)[0]
            summary_text += f"Policy Loss Trend: {'↓' if policy_loss_trend < 0 else '↑'} {abs(policy_loss_trend):.6f}\n"
        
        if len(self.training_history['sharpe_ratios']) > 1:
            sharpe_trend = np.polyfit(range(len(self.training_history['sharpe_ratios'])), 
                                    self.training_history['sharpe_ratios'], 1)[0]
            summary_text += f"Sharpe Trend: {'↑' if sharpe_trend > 0 else '↓'} {abs(sharpe_trend):.4f}\n"
        
        ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Main title
        fig.suptitle('PPO Training Progress - AlphaPulse-RL Trading System', fontsize=16, fontweight='bold')
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training progress plot saved to {save_path}")
        
        plt.show()


def create_training_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        'agent_config': {
            'lr_actor': 3e-4,
            'lr_critic': 1e-3,
            'gamma': 0.99,
            'clip_epsilon': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5
        },
        'training_config': {
            'total_timesteps': 1000000,
            'buffer_size': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'eval_freq': 10000,
            'save_freq': 50000,
            'log_freq': 1000
        }
    }


def train_ppo_agent(
    data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    save_dir: str = "models/checkpoints",
    device: str = "cpu"
) -> Tuple[PPOAgent, PPOTrainer]:
    """
    Main training function for PPO agent.
    
    Args:
        data: Training data with OHLCV and features
        config: Training configuration
        save_dir: Directory to save models and logs
        device: Device to run training on
        
    Returns:
        Tuple of (trained_agent, trainer)
    """
    # Import here to avoid circular imports
    from env.weex_trading_env import WeexTradingEnv
    
    # Use default config if none provided
    if config is None:
        config = create_training_config()
    
    # Initialize environment
    env_config = {
        'initial_balance': 10000.0,
        'transaction_cost_bps': 5,
        'slippage_bps': 2,
        'market_impact_coef': 0.1
    }
    env = WeexTradingEnv(data, env_config)
    
    # Initialize agent
    agent_config = config['agent_config']
    agent = PPOAgent(
        lr_actor=agent_config['lr_actor'],
        lr_critic=agent_config['lr_critic'],
        gamma=agent_config['gamma'],
        clip_epsilon=agent_config['clip_epsilon'],
        entropy_coef=agent_config['entropy_coef'],
        value_coef=agent_config['value_coef'],
        device=device
    )
    
    # Initialize trainer
    training_config = config['training_config']
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        buffer_size=training_config['buffer_size'],
        batch_size=training_config['batch_size'],
        n_epochs=training_config['n_epochs'],
        save_dir=save_dir
    )
    
    # Train the agent
    trainer.train(
        total_timesteps=training_config['total_timesteps'],
        eval_freq=training_config['eval_freq'],
        save_freq=training_config['save_freq'],
        log_freq=training_config['log_freq']
    )
    
    return agent, trainer


def evaluate_trained_agent(
    agent: PPOAgent,
    test_data: pd.DataFrame,
    n_episodes: int = 10,
    render: bool = False
) -> Dict[str, float]:
    """
    Evaluate trained agent on test data.
    
    Args:
        agent: Trained PPO agent
        test_data: Test data for evaluation
        n_episodes: Number of episodes to evaluate
        render: Whether to render environment during evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Import here to avoid circular imports
    from env.weex_trading_env import WeexTradingEnv
    
    # Create test environment
    env_config = {
        'initial_balance': 10000.0,
        'transaction_cost_bps': 5,
        'slippage_bps': 2
    }
    test_env = WeexTradingEnv(test_data, env_config)
    
    agent.set_training_mode(False)
    
    episode_metrics = []
    
    for episode in range(n_episodes):
        state = test_env.reset()
        episode_reward = 0
        episode_trades = 0
        episode_returns = []
        
        done = False
        while not done:
            # Get agent action
            action = agent.predict(state, deterministic=True)
            
            # Take step
            state, reward, done, info = test_env.step(action)
            
            episode_reward += reward
            episode_returns.append(reward)
            
            if info['trade_result']['success'] and info['trade_result']['position_size'] > 0:
                episode_trades += 1
            
            if render:
                test_env.render()
        
        # Calculate episode metrics
        episode_stats = test_env.get_episode_stats()
        episode_metrics.append({
            'total_reward': episode_reward,
            'total_return': episode_stats['total_return'],
            'sharpe_ratio': episode_stats['sharpe_ratio'],
            'max_drawdown': episode_stats['max_drawdown'],
            'win_rate': episode_stats['win_rate'],
            'total_trades': episode_trades
        })
    
    # Aggregate metrics across episodes
    aggregated_metrics = {}
    for key in episode_metrics[0].keys():
        values = [ep[key] for ep in episode_metrics]
        aggregated_metrics[f'avg_{key}'] = np.mean(values)
        aggregated_metrics[f'std_{key}'] = np.std(values)
        aggregated_metrics[f'min_{key}'] = np.min(values)
        aggregated_metrics[f'max_{key}'] = np.max(values)
    
    return aggregated_metrics


if __name__ == "__main__":
    # Example usage with enhanced training pipeline
    import sys
    import os
    
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from env.weex_trading_env import WeexTradingEnv
    from data.feature_engineering import FeatureEngine
    import pandas as pd
    
    # Create realistic dummy data with features
    np.random.seed(42)
    n_samples = 2000
    
    # Generate price data with realistic patterns
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, n_samples)  # 2% daily volatility
    prices = [base_price]
    
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices[1:])  # Remove initial price
    
    # Create OHLCV data
    dummy_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='5T'),
        'open': prices * np.random.uniform(0.999, 1.001, n_samples),
        'high': prices * np.random.uniform(1.001, 1.005, n_samples),
        'low': prices * np.random.uniform(0.995, 0.999, n_samples),
        'close': prices,
        'volume': np.random.lognormal(8, 1, n_samples),  # Log-normal volume distribution
        'pair': ['BTCUSDT'] * n_samples
    })
    
    # Add basic features for testing
    dummy_data['returns_5m'] = dummy_data['close'].pct_change().fillna(0)
    dummy_data['returns_15m'] = dummy_data['close'].pct_change(3).fillna(0)
    dummy_data['rsi_14'] = np.random.uniform(0.2, 0.8, n_samples)  # RSI between 20-80
    dummy_data['macd_histogram'] = np.random.normal(0, 0.001, n_samples)
    dummy_data['atr_percentage'] = np.random.uniform(0.01, 0.05, n_samples)
    dummy_data['volume_zscore'] = np.random.normal(0, 1, n_samples)
    dummy_data['orderbook_imbalance'] = np.random.normal(0, 0.1, n_samples)
    dummy_data['funding_rate'] = np.random.normal(0.0001, 0.0005, n_samples)
    dummy_data['volatility_regime'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    
    # Split data into train/test
    train_size = int(0.8 * len(dummy_data))
    train_data = dummy_data[:train_size].copy()
    test_data = dummy_data[train_size:].copy()
    
    print(f"Training data: {len(train_data)} samples")
    print(f"Test data: {len(test_data)} samples")
    
    # Create training configuration
    config = create_training_config()
    config['training_config']['total_timesteps'] = 50000  # Reduced for demo
    config['training_config']['eval_freq'] = 5000
    config['training_config']['save_freq'] = 10000
    
    # Train the agent
    print("Starting PPO training...")
    trained_agent, trainer = train_ppo_agent(
        data=train_data,
        config=config,
        save_dir="models/checkpoints",
        device="cpu"
    )
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    eval_metrics = evaluate_trained_agent(
        agent=trained_agent,
        test_data=test_data,
        n_episodes=5,
        render=False
    )
    
    print("\nEvaluation Results:")
    for metric, value in eval_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot training progress
    print("\nGenerating training progress plots...")
    trainer.plot_training_progress("models/checkpoints/training_progress.png")
    
    print("\nTraining completed successfully!")