"""
Model evaluation utilities for PPO agent in AlphaPulse-RL Trading System.

This module provides comprehensive evaluation metrics and analysis tools
for assessing trading agent performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta

# Conditional imports for plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

logger = logging.getLogger(__name__)


class TradingMetrics:
    """Calculate comprehensive trading performance metrics."""
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def max_drawdown(returns: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown and its duration.
        
        Args:
            returns: Array of returns
            
        Returns:
            Tuple of (max_drawdown, start_idx, end_idx)
        """
        if len(returns) == 0:
            return 0.0, 0, 0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + 1e-8)
        
        max_dd = np.min(drawdowns)
        max_dd_idx = np.argmin(drawdowns)
        
        # Find start of drawdown period
        start_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdowns[i] == 0:
                start_idx = i
                break
        
        return abs(max_dd), start_idx, max_dd_idx
    
    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """Calculate win rate (percentage of positive returns)."""
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns)
    
    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(returns) == 0:
            return 0.0
        
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (focuses on downside deviation).
        
        Args:
            returns: Array of returns
            target_return: Target return threshold
            
        Returns:
            Annualized Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(returns) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        max_dd, _, _ = TradingMetrics.max_drawdown(returns)
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_dd
    
    @staticmethod
    def value_at_risk(returns: np.ndarray, confidence: float = 0.05) -> float:
        """Calculate Value at Risk at given confidence level."""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, confidence * 100)
    
    @staticmethod
    def expected_shortfall(returns: np.ndarray, confidence: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(returns) == 0:
            return 0.0
        
        var = TradingMetrics.value_at_risk(returns, confidence)
        return np.mean(returns[returns <= var])


class PerformanceEvaluator:
    """Comprehensive performance evaluation for trading agents."""
    
    def __init__(self, agent, env):
        """
        Initialize performance evaluator.
        
        Args:
            agent: Trained PPO agent
            env: Trading environment
        """
        self.agent = agent
        self.env = env
        self.metrics = TradingMetrics()
    
    def evaluate_episode(self, deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate agent performance for a single episode.
        
        Args:
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary containing episode results
        """
        state = self.env.reset()
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'confidences': [],
            'portfolio_values': [],
            'positions': [],
            'trades': []
        }
        
        done = False
        step = 0
        
        while not done:
            # Get action and confidence
            action = self.agent.predict(state, deterministic=deterministic)
            confidence = self.agent.get_confidence(state)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Store data
            episode_data['states'].append(state.copy())
            episode_data['actions'].append(action.copy())
            episode_data['rewards'].append(reward)
            episode_data['confidences'].append(confidence)
            episode_data['portfolio_values'].append(info.get('portfolio_value', 0))
            episode_data['positions'].append(info.get('position', {}))
            
            if info.get('trade_executed', False):
                episode_data['trades'].append({
                    'step': step,
                    'action': action.copy(),
                    'price': info.get('current_price', 0),
                    'confidence': confidence
                })
            
            state = next_state
            step += 1
        
        return episode_data
    
    def evaluate_multiple_episodes(
        self,
        n_episodes: int = 100,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate agent over multiple episodes.
        
        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Evaluating agent over {n_episodes} episodes")
        
        all_returns = []
        all_rewards = []
        episode_metrics = []
        
        for episode in range(n_episodes):
            episode_data = self.evaluate_episode(deterministic)
            
            # Calculate episode metrics
            returns = np.array(episode_data['rewards'])
            
            episode_metric = {
                'episode': episode,
                'total_return': np.sum(returns),
                'episode_length': len(returns),
                'sharpe_ratio': self.metrics.sharpe_ratio(returns),
                'max_drawdown': self.metrics.max_drawdown(returns)[0],
                'win_rate': self.metrics.win_rate(returns),
                'profit_factor': self.metrics.profit_factor(returns),
                'n_trades': len(episode_data['trades']),
                'avg_confidence': np.mean(episode_data['confidences']),
                'final_portfolio_value': episode_data['portfolio_values'][-1] if episode_data['portfolio_values'] else 0
            }
            
            episode_metrics.append(episode_metric)
            all_returns.extend(returns)
            all_rewards.extend(returns)
            
            if (episode + 1) % 10 == 0:
                logger.debug(f"Completed {episode + 1}/{n_episodes} episodes")
        
        # Calculate aggregate metrics
        all_returns = np.array(all_returns)
        
        aggregate_metrics = {
            'sharpe_ratio': self.metrics.sharpe_ratio(all_returns),
            'sortino_ratio': self.metrics.sortino_ratio(all_returns),
            'calmar_ratio': self.metrics.calmar_ratio(all_returns),
            'max_drawdown': self.metrics.max_drawdown(all_returns)[0],
            'win_rate': self.metrics.win_rate(all_returns),
            'profit_factor': self.metrics.profit_factor(all_returns),
            'value_at_risk_5': self.metrics.value_at_risk(all_returns, 0.05),
            'expected_shortfall_5': self.metrics.expected_shortfall(all_returns, 0.05),
            'total_return': np.sum(all_returns),
            'avg_return': np.mean(all_returns),
            'volatility': np.std(all_returns) * np.sqrt(252),  # Annualized
            'avg_episode_length': np.mean([ep['episode_length'] for ep in episode_metrics]),
            'avg_trades_per_episode': np.mean([ep['n_trades'] for ep in episode_metrics]),
            'avg_confidence': np.mean([ep['avg_confidence'] for ep in episode_metrics])
        }
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'episode_metrics': episode_metrics,
            'all_returns': all_returns.tolist()
        }
    
    def regime_analysis(
        self,
        n_episodes: int = 50,
        regime_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Analyze performance across different market regimes.
        
        Args:
            n_episodes: Number of episodes to analyze
            regime_threshold: Threshold for regime classification
            
        Returns:
            Regime-specific performance metrics
        """
        logger.info("Performing regime-based performance analysis")
        
        trending_returns = []
        ranging_returns = []
        
        for episode in range(n_episodes):
            episode_data = self.evaluate_episode()
            
            for i, state in enumerate(episode_data['states']):
                # Assume last element of state is volatility regime
                regime = state[-1] if len(state) > 8 else 0
                reward = episode_data['rewards'][i]
                
                if regime > regime_threshold:
                    trending_returns.append(reward)
                else:
                    ranging_returns.append(reward)
        
        trending_returns = np.array(trending_returns)
        ranging_returns = np.array(ranging_returns)
        
        regime_metrics = {
            'trending_market': {
                'n_observations': len(trending_returns),
                'sharpe_ratio': self.metrics.sharpe_ratio(trending_returns),
                'win_rate': self.metrics.win_rate(trending_returns),
                'avg_return': np.mean(trending_returns) if len(trending_returns) > 0 else 0,
                'volatility': np.std(trending_returns) if len(trending_returns) > 0 else 0
            },
            'ranging_market': {
                'n_observations': len(ranging_returns),
                'sharpe_ratio': self.metrics.sharpe_ratio(ranging_returns),
                'win_rate': self.metrics.win_rate(ranging_returns),
                'avg_return': np.mean(ranging_returns) if len(ranging_returns) > 0 else 0,
                'volatility': np.std(ranging_returns) if len(ranging_returns) > 0 else 0
            }
        }
        
        return regime_metrics
    
    def confidence_analysis(self, n_episodes: int = 50) -> Dict[str, Any]:
        """
        Analyze relationship between model confidence and performance.
        
        Args:
            n_episodes: Number of episodes to analyze
            
        Returns:
            Confidence-based performance analysis
        """
        logger.info("Performing confidence-based performance analysis")
        
        confidence_bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        binned_returns = {f"bin_{i}": [] for i in range(len(confidence_bins) - 1)}
        
        for episode in range(n_episodes):
            episode_data = self.evaluate_episode()
            
            for confidence, reward in zip(episode_data['confidences'], episode_data['rewards']):
                bin_idx = np.digitize(confidence, confidence_bins) - 1
                bin_idx = max(0, min(bin_idx, len(confidence_bins) - 2))
                binned_returns[f"bin_{bin_idx}"].append(reward)
        
        confidence_metrics = {}
        for bin_name, returns in binned_returns.items():
            if len(returns) > 0:
                returns_array = np.array(returns)
                confidence_metrics[bin_name] = {
                    'n_observations': len(returns),
                    'avg_return': np.mean(returns_array),
                    'sharpe_ratio': self.metrics.sharpe_ratio(returns_array),
                    'win_rate': self.metrics.win_rate(returns_array),
                    'confidence_range': (
                        confidence_bins[int(bin_name.split('_')[1])],
                        confidence_bins[int(bin_name.split('_')[1]) + 1]
                    )
                }
        
        return confidence_metrics
    
    def generate_report(
        self,
        n_episodes: int = 100,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            n_episodes: Number of episodes for evaluation
            save_path: Path to save the report
            
        Returns:
            Complete evaluation report
        """
        logger.info("Generating comprehensive evaluation report")
        
        # Main evaluation
        main_results = self.evaluate_multiple_episodes(n_episodes)
        
        # Regime analysis
        regime_results = self.regime_analysis(n_episodes // 2)
        
        # Confidence analysis
        confidence_results = self.confidence_analysis(n_episodes // 2)
        
        # Compile report
        report = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'n_episodes': n_episodes,
                'agent_type': 'PPO',
                'environment': 'WeexTradingEnv'
            },
            'aggregate_metrics': main_results['aggregate_metrics'],
            'regime_analysis': regime_results,
            'confidence_analysis': confidence_results,
            'episode_summary': {
                'best_episode': max(main_results['episode_metrics'], key=lambda x: x['total_return']),
                'worst_episode': min(main_results['episode_metrics'], key=lambda x: x['total_return']),
                'median_performance': np.median([ep['total_return'] for ep in main_results['episode_metrics']])
            }
        }
        
        # Save report if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def plot_performance_analysis(
        self,
        evaluation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create comprehensive performance visualization.
        
        Args:
            evaluation_results: Results from evaluate_multiple_episodes
            save_path: Path to save the plots
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available, cannot create plots")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Agent Performance Analysis', fontsize=16)
        
        # Returns distribution
        returns = evaluation_results['all_returns']
        axes[0, 0].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Returns Distribution')
        axes[0, 0].set_xlabel('Return')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(returns), color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # Cumulative returns
        cumulative_returns = np.cumsum(returns)
        axes[0, 1].plot(cumulative_returns)
        axes[0, 1].set_title('Cumulative Returns')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Cumulative Return')
        axes[0, 1].grid(True)
        
        # Drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / (running_max + 1e-8)
        axes[0, 2].fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.3, color='red')
        axes[0, 2].set_title('Drawdown')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Drawdown')
        axes[0, 2].grid(True)
        
        # Episode performance
        episode_returns = [ep['total_return'] for ep in evaluation_results['episode_metrics']]
        axes[1, 0].plot(episode_returns)
        axes[1, 0].set_title('Episode Returns')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Return')
        axes[1, 0].grid(True)
        
        # Win rate by episode
        episode_win_rates = [ep['win_rate'] for ep in evaluation_results['episode_metrics']]
        axes[1, 1].plot(episode_win_rates)
        axes[1, 1].set_title('Win Rate by Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].grid(True)
        
        # Confidence distribution
        episode_confidences = [ep['avg_confidence'] for ep in evaluation_results['episode_metrics']]
        axes[1, 2].hist(episode_confidences, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Average Confidence Distribution')
        axes[1, 2].set_xlabel('Confidence')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    from models.ppo_agent import PPOAgent
    from env.weex_trading_env import WeexTradingEnv
    import pandas as pd
    
    # Create dummy data
    dummy_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='5T'),
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Initialize environment and agent
    env = WeexTradingEnv(dummy_data)
    agent = PPOAgent()
    
    # Create evaluator and run evaluation
    evaluator = PerformanceEvaluator(agent, env)
    results = evaluator.evaluate_multiple_episodes(n_episodes=10)
    
    # Generate report
    report = evaluator.generate_report(n_episodes=10)
    print(json.dumps(report['aggregate_metrics'], indent=2))
    
    # Plot results
    evaluator.plot_performance_analysis(results)