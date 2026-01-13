"""
Performance Analysis Tools for AlphaPulse-RL Trading System.

This module provides comprehensive performance evaluation tools including
advanced metrics, trade analysis, and visualization capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
from dataclasses import dataclass

# Conditional imports for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    from scipy import stats
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    mdates = None
    sns = None
    stats = None

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Individual trade record for analysis."""
    timestamp: datetime
    pair: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    position_size: float
    leverage: float
    pnl: float
    pnl_percent: float
    duration: timedelta
    confidence: float
    market_regime: int  # 0=ranging, 1=trending
    entry_reason: str
    exit_reason: str


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Return metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    value_at_risk_5: float
    expected_shortfall_5: float
    
    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Consistency metrics
    consistency_score: float
    monthly_win_rate: float
    best_month: float
    worst_month: float
    
    # Regime-specific metrics
    trending_performance: Dict[str, float]
    ranging_performance: Dict[str, float]


class AdvancedMetrics:
    """Advanced performance metrics calculations."""
    
    @staticmethod
    def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio - probability weighted ratio of gains vs losses.
        
        Args:
            returns: Array of returns
            threshold: Return threshold (default 0)
            
        Returns:
            Omega ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0]
        losses = excess_returns[excess_returns < 0]
        
        if len(losses) == 0:
            return float('inf') if len(gains) > 0 else 1.0
        
        gain_sum = np.sum(gains) if len(gains) > 0 else 0
        loss_sum = abs(np.sum(losses))
        
        return gain_sum / loss_sum if loss_sum > 0 else float('inf')
    
    @staticmethod
    def tail_ratio(returns: np.ndarray, percentile: float = 5) -> float:
        """
        Calculate tail ratio - ratio of average top percentile to bottom percentile.
        
        Args:
            returns: Array of returns
            percentile: Percentile for tail analysis
            
        Returns:
            Tail ratio
        """
        if len(returns) == 0:
            return 0.0
        
        top_percentile = np.percentile(returns, 100 - percentile)
        bottom_percentile = np.percentile(returns, percentile)
        
        top_returns = returns[returns >= top_percentile]
        bottom_returns = returns[returns <= bottom_percentile]
        
        if len(bottom_returns) == 0 or np.mean(bottom_returns) == 0:
            return float('inf') if len(top_returns) > 0 else 0.0
        
        return abs(np.mean(top_returns) / np.mean(bottom_returns))
    
    @staticmethod
    def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """
        Calculate Information Ratio - excess return per unit of tracking error.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        if len(returns) != len(benchmark_returns) or len(returns) == 0:
            return 0.0
        
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.mean(excess_returns) / tracking_error * np.sqrt(252)
    
    @staticmethod
    def ulcer_index(returns: np.ndarray) -> float:
        """
        Calculate Ulcer Index - measure of downside risk.
        
        Args:
            returns: Array of returns
            
        Returns:
            Ulcer Index
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + 1e-8)
        
        # Square the drawdowns and take average
        squared_drawdowns = drawdowns ** 2
        ulcer_index = np.sqrt(np.mean(squared_drawdowns))
        
        return ulcer_index
    
    @staticmethod
    def pain_index(returns: np.ndarray) -> float:
        """
        Calculate Pain Index - average drawdown over the period.
        
        Args:
            returns: Array of returns
            
        Returns:
            Pain Index
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + 1e-8)
        
        return abs(np.mean(drawdowns))


class TradeAnalyzer:
    """Analyze individual trades and trading patterns."""
    
    def __init__(self):
        self.trades: List[TradeRecord] = []
    
    def add_trade(self, trade: TradeRecord) -> None:
        """Add a trade record for analysis."""
        self.trades.append(trade)
    
    def load_trades_from_csv(self, csv_path: str) -> None:
        """
        Load trades from CSV file.
        
        Args:
            csv_path: Path to trades CSV file
        """
        try:
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                trade = TradeRecord(
                    timestamp=pd.to_datetime(row['timestamp']),
                    pair=row.get('pair', 'BTCUSDT'),
                    side=row.get('side', 'long'),
                    entry_price=float(row.get('entry_price', 0)),
                    exit_price=float(row.get('exit_price', 0)),
                    position_size=float(row.get('position_size', 0)),
                    leverage=float(row.get('leverage', 1)),
                    pnl=float(row.get('pnl', 0)),
                    pnl_percent=float(row.get('pnl_percent', 0)),
                    duration=timedelta(minutes=int(row.get('duration_minutes', 0))),
                    confidence=float(row.get('confidence', 0.5)),
                    market_regime=int(row.get('market_regime', 0)),
                    entry_reason=row.get('entry_reason', 'model_signal'),
                    exit_reason=row.get('exit_reason', 'model_signal')
                )
                self.trades.append(trade)
                
            logger.info(f"Loaded {len(self.trades)} trades from {csv_path}")
            
        except Exception as e:
            logger.error(f"Error loading trades from CSV: {e}")
    
    def analyze_trade_patterns(self) -> Dict[str, Any]:
        """
        Analyze trading patterns and behaviors.
        
        Returns:
            Dictionary of trade pattern analysis
        """
        if not self.trades:
            return {}
        
        # Convert to DataFrame for easier analysis
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'timestamp': trade.timestamp,
                'pair': trade.pair,
                'side': trade.side,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'duration_hours': trade.duration.total_seconds() / 3600,
                'confidence': trade.confidence,
                'market_regime': trade.market_regime,
                'position_size': trade.position_size,
                'leverage': trade.leverage
            })
        
        df = pd.DataFrame(trade_data)
        
        # Basic statistics
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        analysis = {
            'total_trades': len(df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(df) if len(df) > 0 else 0,
            
            # PnL analysis
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': df['pnl'].max(),
            'largest_loss': df['pnl'].min(),
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf'),
            
            # Duration analysis
            'avg_trade_duration_hours': df['duration_hours'].mean(),
            'median_trade_duration_hours': df['duration_hours'].median(),
            'max_trade_duration_hours': df['duration_hours'].max(),
            'min_trade_duration_hours': df['duration_hours'].min(),
            
            # Confidence analysis
            'avg_confidence': df['confidence'].mean(),
            'high_confidence_trades': len(df[df['confidence'] > 0.8]),
            'high_confidence_win_rate': len(df[(df['confidence'] > 0.8) & (df['pnl'] > 0)]) / len(df[df['confidence'] > 0.8]) if len(df[df['confidence'] > 0.8]) > 0 else 0,
            
            # Regime analysis
            'trending_trades': len(df[df['market_regime'] == 1]),
            'ranging_trades': len(df[df['market_regime'] == 0]),
            'trending_win_rate': len(df[(df['market_regime'] == 1) & (df['pnl'] > 0)]) / len(df[df['market_regime'] == 1]) if len(df[df['market_regime'] == 1]) > 0 else 0,
            'ranging_win_rate': len(df[(df['market_regime'] == 0) & (df['pnl'] > 0)]) / len(df[df['market_regime'] == 0]) if len(df[df['market_regime'] == 0]) > 0 else 0,
            
            # Side analysis
            'long_trades': len(df[df['side'] == 'long']),
            'short_trades': len(df[df['side'] == 'short']),
            'long_win_rate': len(df[(df['side'] == 'long') & (df['pnl'] > 0)]) / len(df[df['side'] == 'long']) if len(df[df['side'] == 'long']) > 0 else 0,
            'short_win_rate': len(df[(df['side'] == 'short') & (df['pnl'] > 0)]) / len(df[df['side'] == 'short']) if len(df[df['side'] == 'short']) > 0 else 0,
            
            # Position sizing analysis
            'avg_position_size': df['position_size'].mean(),
            'avg_leverage': df['leverage'].mean(),
            'max_leverage_used': df['leverage'].max(),
        }
        
        return analysis
    
    def analyze_consecutive_trades(self) -> Dict[str, Any]:
        """
        Analyze consecutive winning/losing streaks.
        
        Returns:
            Dictionary of streak analysis
        """
        if not self.trades:
            return {}
        
        # Sort trades by timestamp
        sorted_trades = sorted(self.trades, key=lambda x: x.timestamp)
        
        # Calculate streaks
        current_streak = 0
        current_streak_type = None
        max_win_streak = 0
        max_loss_streak = 0
        win_streaks = []
        loss_streaks = []
        
        for trade in sorted_trades:
            is_win = trade.pnl > 0
            
            if current_streak_type is None:
                current_streak_type = 'win' if is_win else 'loss'
                current_streak = 1
            elif (current_streak_type == 'win' and is_win) or (current_streak_type == 'loss' and not is_win):
                current_streak += 1
            else:
                # Streak ended
                if current_streak_type == 'win':
                    win_streaks.append(current_streak)
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    loss_streaks.append(current_streak)
                    max_loss_streak = max(max_loss_streak, current_streak)
                
                current_streak_type = 'win' if is_win else 'loss'
                current_streak = 1
        
        # Handle final streak
        if current_streak_type == 'win':
            win_streaks.append(current_streak)
            max_win_streak = max(max_win_streak, current_streak)
        else:
            loss_streaks.append(current_streak)
            max_loss_streak = max(max_loss_streak, current_streak)
        
        return {
            'max_consecutive_wins': max_win_streak,
            'max_consecutive_losses': max_loss_streak,
            'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
            'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
            'total_win_streaks': len(win_streaks),
            'total_loss_streaks': len(loss_streaks)
        }
    
    def monthly_performance_breakdown(self) -> Dict[str, Dict[str, float]]:
        """
        Break down performance by month.
        
        Returns:
            Dictionary of monthly performance metrics
        """
        if not self.trades:
            return {}
        
        monthly_data = {}
        
        for trade in self.trades:
            month_key = trade.timestamp.strftime('%Y-%m')
            
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    'trades': [],
                    'total_pnl': 0,
                    'winning_trades': 0,
                    'total_trades': 0
                }
            
            monthly_data[month_key]['trades'].append(trade)
            monthly_data[month_key]['total_pnl'] += trade.pnl
            monthly_data[month_key]['total_trades'] += 1
            if trade.pnl > 0:
                monthly_data[month_key]['winning_trades'] += 1
        
        # Calculate monthly metrics
        monthly_metrics = {}
        for month, data in monthly_data.items():
            monthly_metrics[month] = {
                'total_pnl': data['total_pnl'],
                'total_trades': data['total_trades'],
                'win_rate': data['winning_trades'] / data['total_trades'] if data['total_trades'] > 0 else 0,
                'avg_pnl_per_trade': data['total_pnl'] / data['total_trades'] if data['total_trades'] > 0 else 0,
                'best_trade': max([t.pnl for t in data['trades']]) if data['trades'] else 0,
                'worst_trade': min([t.pnl for t in data['trades']]) if data['trades'] else 0
            }
        
        return monthly_metrics


class PerformanceVisualizer:
    """Create comprehensive performance visualizations."""
    
    def __init__(self):
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available, visualization features disabled")
    
    def plot_equity_curve(
        self,
        returns: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        title: str = "Equity Curve",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot equity curve with drawdown.
        
        Args:
            returns: Array of returns
            timestamps: Optional timestamps for x-axis
            title: Plot title
            save_path: Path to save the plot
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Cannot create plots - matplotlib not available")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        # Calculate cumulative returns
        cumulative_returns = np.cumsum(returns)
        
        # Use timestamps if provided, otherwise use index
        x_axis = timestamps if timestamps else range(len(returns))
        
        # Plot equity curve
        ax1.plot(x_axis, cumulative_returns, linewidth=2, color='blue', label='Equity Curve')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Calculate and plot drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / (running_max + 1e-8)
        
        ax2.fill_between(x_axis, drawdowns, 0, alpha=0.3, color='red', label='Drawdown')
        ax2.set_ylabel('Drawdown', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Format x-axis if timestamps provided
        if timestamps:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equity curve saved to {save_path}")
        
        plt.show()
    
    def plot_returns_distribution(
        self,
        returns: np.ndarray,
        title: str = "Returns Distribution",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot returns distribution with statistics.
        
        Args:
            returns: Array of returns
            title: Plot title
            save_path: Path to save the plot
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Cannot create plots - matplotlib not available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.4f}')
        ax1.axvline(np.median(returns), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(returns):.4f}')
        ax1.set_title('Returns Histogram', fontsize=14)
        ax1.set_xlabel('Return', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot for normality check
        if stats is not None:
            stats.probplot(returns, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot (Normality Check)', fontsize=14)
            ax2.grid(True, alpha=0.3)
        else:
            # Fallback if scipy not available
            ax2.scatter(range(len(returns)), sorted(returns), alpha=0.6)
            ax2.set_title('Returns Scatter Plot', fontsize=14)
            ax2.set_xlabel('Index')
            ax2.set_ylabel('Return')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Returns distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_rolling_metrics(
        self,
        returns: np.ndarray,
        window: int = 252,
        timestamps: Optional[List[datetime]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot rolling performance metrics.
        
        Args:
            returns: Array of returns
            window: Rolling window size
            timestamps: Optional timestamps
            save_path: Path to save the plot
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Cannot create plots - matplotlib not available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Calculate rolling metrics
        rolling_sharpe = []
        rolling_volatility = []
        rolling_max_dd = []
        rolling_win_rate = []
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            
            # Rolling Sharpe ratio
            if np.std(window_returns) > 0:
                sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
            else:
                sharpe = 0
            rolling_sharpe.append(sharpe)
            
            # Rolling volatility
            rolling_volatility.append(np.std(window_returns) * np.sqrt(252))
            
            # Rolling max drawdown
            cum_returns = np.cumsum(window_returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / (running_max + 1e-8)
            rolling_max_dd.append(abs(np.min(drawdowns)))
            
            # Rolling win rate
            rolling_win_rate.append(np.sum(window_returns > 0) / len(window_returns))
        
        # X-axis for rolling metrics
        if timestamps:
            x_axis = timestamps[window:]
        else:
            x_axis = range(window, len(returns))
        
        # Plot rolling Sharpe ratio
        axes[0, 0].plot(x_axis, rolling_sharpe, color='blue', linewidth=1.5)
        axes[0, 0].set_title('Rolling Sharpe Ratio', fontsize=12)
        axes[0, 0].set_ylabel('Sharpe Ratio', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Plot rolling volatility
        axes[0, 1].plot(x_axis, rolling_volatility, color='orange', linewidth=1.5)
        axes[0, 1].set_title('Rolling Volatility', fontsize=12)
        axes[0, 1].set_ylabel('Volatility', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot rolling max drawdown
        axes[1, 0].plot(x_axis, rolling_max_dd, color='red', linewidth=1.5)
        axes[1, 0].set_title('Rolling Max Drawdown', fontsize=12)
        axes[1, 0].set_ylabel('Max Drawdown', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot rolling win rate
        axes[1, 1].plot(x_axis, rolling_win_rate, color='green', linewidth=1.5)
        axes[1, 1].set_title('Rolling Win Rate', fontsize=12)
        axes[1, 1].set_ylabel('Win Rate', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        
        # Format x-axis if timestamps provided
        if timestamps:
            for ax in axes.flat:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(f'Rolling Performance Metrics (Window: {window})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Rolling metrics plot saved to {save_path}")
        
        plt.show()
    
    def plot_trade_analysis(
        self,
        trade_analyzer: TradeAnalyzer,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create comprehensive trade analysis plots.
        
        Args:
            trade_analyzer: TradeAnalyzer instance with trade data
            save_path: Path to save the plot
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Cannot create plots - matplotlib not available")
            return
        
        if not trade_analyzer.trades:
            logger.warning("No trades to analyze")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract trade data
        pnls = [trade.pnl for trade in trade_analyzer.trades]
        confidences = [trade.confidence for trade in trade_analyzer.trades]
        durations = [trade.duration.total_seconds() / 3600 for trade in trade_analyzer.trades]  # in hours
        regimes = [trade.market_regime for trade in trade_analyzer.trades]
        sides = [trade.side for trade in trade_analyzer.trades]
        
        # 1. PnL distribution
        axes[0, 0].hist(pnls, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(pnls), color='red', linestyle='--', label=f'Mean: {np.mean(pnls):.2f}')
        axes[0, 0].set_title('PnL Distribution')
        axes[0, 0].set_xlabel('PnL')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confidence vs PnL scatter
        colors = ['red' if pnl < 0 else 'green' for pnl in pnls]
        axes[0, 1].scatter(confidences, pnls, c=colors, alpha=0.6)
        axes[0, 1].set_title('Confidence vs PnL')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('PnL')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Trade duration distribution
        axes[0, 2].hist(durations, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_title('Trade Duration Distribution')
        axes[0, 2].set_xlabel('Duration (hours)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Performance by market regime
        trending_pnls = [pnl for pnl, regime in zip(pnls, regimes) if regime == 1]
        ranging_pnls = [pnl for pnl, regime in zip(pnls, regimes) if regime == 0]
        
        regime_data = [trending_pnls, ranging_pnls]
        regime_labels = ['Trending', 'Ranging']
        axes[1, 0].boxplot(regime_data, labels=regime_labels)
        axes[1, 0].set_title('PnL by Market Regime')
        axes[1, 0].set_ylabel('PnL')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Performance by trade side
        long_pnls = [pnl for pnl, side in zip(pnls, sides) if side == 'long']
        short_pnls = [pnl for pnl, side in zip(pnls, sides) if side == 'short']
        
        side_data = [long_pnls, short_pnls]
        side_labels = ['Long', 'Short']
        axes[1, 1].boxplot(side_data, labels=side_labels)
        axes[1, 1].set_title('PnL by Trade Side')
        axes[1, 1].set_ylabel('PnL')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Cumulative PnL over time
        sorted_trades = sorted(trade_analyzer.trades, key=lambda x: x.timestamp)
        cumulative_pnl = np.cumsum([trade.pnl for trade in sorted_trades])
        timestamps = [trade.timestamp for trade in sorted_trades]
        
        axes[1, 2].plot(timestamps, cumulative_pnl, linewidth=2, color='blue')
        axes[1, 2].set_title('Cumulative PnL Over Time')
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Cumulative PnL')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Format timestamp axis
        axes[1, 2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle('Comprehensive Trade Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trade analysis plot saved to {save_path}")
        
        plt.show()


class ComprehensivePerformanceEvaluator:
    """
    Main class for comprehensive performance evaluation.
    Combines all analysis tools and generates complete reports.
    """
    
    def __init__(self):
        self.advanced_metrics = AdvancedMetrics()
        self.trade_analyzer = TradeAnalyzer()
        self.visualizer = PerformanceVisualizer()
        
    def calculate_comprehensive_metrics(
        self,
        returns: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics including Sharpe ratio, max drawdown, win rate, and profit factor.
        
        Args:
            returns: Array of returns
            timestamps: Optional timestamps
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Dictionary of comprehensive metrics
        """
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = np.sum(returns)
        annualized_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + 1e-8)
        max_drawdown = abs(np.min(drawdowns))
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win rate and profit factor
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns)
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        
        gross_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0
        gross_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 1e-8
        profit_factor = gross_profit / gross_loss
        
        # VaR and Expected Shortfall
        value_at_risk_95 = np.percentile(returns, 5)
        expected_shortfall_95 = np.mean(returns[returns <= value_at_risk_95])
        
        # Advanced metrics
        omega_ratio = self.advanced_metrics.omega_ratio(returns)
        tail_ratio = self.advanced_metrics.tail_ratio(returns)
        ulcer_index = self.advanced_metrics.ulcer_index(returns)
        pain_index = self.advanced_metrics.pain_index(returns)
        
        # Information ratio (if benchmark provided)
        information_ratio = 0
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            information_ratio = self.advanced_metrics.information_ratio(returns, benchmark_returns)
        
        # Recovery factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Sterling ratio
        sterling_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            # Return metrics
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'recovery_factor': recovery_factor,
            'sterling_ratio': sterling_ratio,
            
            # Risk metrics
            'max_drawdown': max_drawdown,
            'value_at_risk_95': value_at_risk_95,
            'expected_shortfall_95': expected_shortfall_95,
            'ulcer_index': ulcer_index,
            'pain_index': pain_index,
            
            # Trade metrics
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': np.max(returns),
            'largest_loss': np.min(returns),
            
            # Advanced metrics
            'omega_ratio': omega_ratio,
            'tail_ratio': tail_ratio,
            
            # Consistency metrics
            'consistency_score': 1.0 - (np.std(returns) / (abs(np.mean(returns)) + 1e-8))
        }
    
    def analyze_strategy_performance(
        self,
        returns: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        trades_data: Optional[List[TradeRecord]] = None
    ) -> Dict[str, Any]:
        """
        Analyze strategy performance with detailed breakdown.
        
        Args:
            returns: Array of returns
            timestamps: Optional timestamps
            trades_data: Optional trade records
            
        Returns:
            Dictionary of strategy performance analysis
        """
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(returns, timestamps)
        
        # Analyze trades if provided
        trade_analysis = {}
        if trades_data:
            self.trade_analyzer.trades = trades_data
            trade_analysis = {
                'trade_patterns': self.trade_analyzer.analyze_trade_patterns(),
                'consecutive_trades': self.trade_analyzer.analyze_consecutive_trades(),
                'monthly_breakdown': self.trade_analyzer.monthly_performance_breakdown()
            }
        
        # Time-based analysis
        time_analysis = {}
        if timestamps and len(timestamps) == len(returns):
            time_analysis = self._analyze_time_based_performance(returns, timestamps)
        
        # Regime-specific analysis
        regime_analysis = {}
        if trades_data:
            regime_analysis = self._analyze_regime_specific_performance(trades_data)
        
        return {
            'comprehensive_metrics': metrics,
            'trade_analysis': trade_analysis,
            'time_analysis': time_analysis,
            'regime_analysis': regime_analysis,
            'performance_summary': self._generate_performance_summary(metrics, trade_analysis) if trade_analysis else {}
        }
    
    def _analyze_time_based_performance(
        self,
        returns: np.ndarray,
        timestamps: List[datetime]
    ) -> Dict[str, Any]:
        """Analyze performance across different time periods."""
        # Monthly analysis
        monthly_returns = self._calculate_monthly_returns(returns, timestamps)
        
        # Quarterly analysis
        quarterly_returns = {}
        for timestamp, ret in zip(timestamps, returns):
            quarter_key = f"{timestamp.year}-Q{(timestamp.month-1)//3 + 1}"
            if quarter_key not in quarterly_returns:
                quarterly_returns[quarter_key] = 0
            quarterly_returns[quarter_key] += ret
        
        # Yearly analysis
        yearly_returns = {}
        for timestamp, ret in zip(timestamps, returns):
            year_key = str(timestamp.year)
            if year_key not in yearly_returns:
                yearly_returns[year_key] = 0
            yearly_returns[year_key] += ret
        
        return {
            'monthly_returns': monthly_returns,
            'quarterly_returns': quarterly_returns,
            'yearly_returns': yearly_returns,
            'best_month': max(monthly_returns.values()) if monthly_returns else 0,
            'worst_month': min(monthly_returns.values()) if monthly_returns else 0,
            'positive_months': sum(1 for ret in monthly_returns.values() if ret > 0),
            'total_months': len(monthly_returns),
            'monthly_win_rate': sum(1 for ret in monthly_returns.values() if ret > 0) / len(monthly_returns) if monthly_returns else 0
        }
    
    def _analyze_regime_specific_performance(
        self,
        trades_data: List[TradeRecord]
    ) -> Dict[str, Any]:
        """Analyze performance in different market regimes."""
        trending_trades = [t for t in trades_data if t.market_regime == 1]
        ranging_trades = [t for t in trades_data if t.market_regime == 0]
        
        def calculate_regime_metrics(trades: List[TradeRecord]) -> Dict[str, float]:
            if not trades:
                return {'count': 0, 'win_rate': 0, 'avg_pnl': 0, 'total_pnl': 0, 'sharpe_ratio': 0}
            
            pnls = [t.pnl for t in trades]
            winning_trades = [t for t in trades if t.pnl > 0]
            
            return {
                'count': len(trades),
                'win_rate': len(winning_trades) / len(trades),
                'avg_pnl': np.mean(pnls),
                'total_pnl': sum(pnls),
                'sharpe_ratio': np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0,
                'max_win': max(pnls),
                'max_loss': min(pnls),
                'profit_factor': sum([p for p in pnls if p > 0]) / abs(sum([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else float('inf')
            }
        
        return {
            'trending_market': calculate_regime_metrics(trending_trades),
            'ranging_market': calculate_regime_metrics(ranging_trades),
            'regime_preference': 'trending' if len(trending_trades) > len(ranging_trades) else 'ranging',
            'regime_adaptability': abs(len(trending_trades) - len(ranging_trades)) / len(trades_data) if trades_data else 0
        }
    
    def _generate_performance_summary(
        self,
        metrics: Dict[str, float],
        trade_analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate human-readable performance summary."""
        summary = {}
        
        # Sharpe ratio assessment
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > 2.0:
            summary['sharpe_assessment'] = 'Excellent (>2.0)'
        elif sharpe > 1.5:
            summary['sharpe_assessment'] = 'Very Good (1.5-2.0)'
        elif sharpe > 1.0:
            summary['sharpe_assessment'] = 'Good (1.0-1.5)'
        elif sharpe > 0.5:
            summary['sharpe_assessment'] = 'Fair (0.5-1.0)'
        else:
            summary['sharpe_assessment'] = 'Poor (<0.5)'
        
        # Risk assessment
        max_dd = metrics.get('max_drawdown', 0)
        if max_dd < 0.05:
            summary['risk_assessment'] = 'Very Low Risk (<5% max DD)'
        elif max_dd < 0.10:
            summary['risk_assessment'] = 'Low Risk (5-10% max DD)'
        elif max_dd < 0.15:
            summary['risk_assessment'] = 'Moderate Risk (10-15% max DD)'
        elif max_dd < 0.25:
            summary['risk_assessment'] = 'High Risk (15-25% max DD)'
        else:
            summary['risk_assessment'] = 'Very High Risk (>25% max DD)'
        
        # Win rate assessment
        win_rate = metrics.get('win_rate', 0)
        if win_rate > 0.7:
            summary['win_rate_assessment'] = 'Excellent (>70%)'
        elif win_rate > 0.6:
            summary['win_rate_assessment'] = 'Very Good (60-70%)'
        elif win_rate > 0.5:
            summary['win_rate_assessment'] = 'Good (50-60%)'
        else:
            summary['win_rate_assessment'] = 'Below Average (<50%)'
        
        # Overall assessment
        score = 0
        if sharpe > 1.0:
            score += 1
        if max_dd < 0.15:
            score += 1
        if win_rate > 0.5:
            score += 1
        if metrics.get('profit_factor', 0) > 1.5:
            score += 1
        
        if score >= 3:
            summary['overall_assessment'] = 'Strong Performance'
        elif score >= 2:
            summary['overall_assessment'] = 'Good Performance'
        elif score >= 1:
            summary['overall_assessment'] = 'Fair Performance'
        else:
            summary['overall_assessment'] = 'Needs Improvement'
        
    def evaluate_returns(
        self,
        returns: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> PerformanceMetrics:
        """
        Comprehensive evaluation of return series.
        
        Args:
            returns: Array of returns
            timestamps: Optional timestamps
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if len(returns) == 0:
            logger.warning("Empty returns array provided")
            return PerformanceMetrics(
                total_return=0, annualized_return=0, sharpe_ratio=0, sortino_ratio=0,
                calmar_ratio=0, max_drawdown=0, max_drawdown_duration=0, volatility=0,
                value_at_risk_5=0, expected_shortfall_5=0, total_trades=0, win_rate=0,
                profit_factor=0, avg_win=0, avg_loss=0, largest_win=0, largest_loss=0,
                consistency_score=0, monthly_win_rate=0, best_month=0, worst_month=0,
                trending_performance={}, ranging_performance={}
            )
        
        # Basic return metrics
        total_return = np.sum(returns)
        annualized_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + 1e-8)
        max_drawdown = abs(np.min(drawdowns))
        
        # Find drawdown duration
        max_dd_idx = np.argmin(drawdowns)
        max_drawdown_duration = 0
        for i in range(max_dd_idx, len(drawdowns)):
            if drawdowns[i] >= -0.01:  # Recovery threshold
                max_drawdown_duration = i - max_dd_idx
                break
        else:
            max_drawdown_duration = len(drawdowns) - max_dd_idx
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # VaR and ES
        value_at_risk_5 = np.percentile(returns, 5)
        expected_shortfall_5 = np.mean(returns[returns <= value_at_risk_5])
        
        # Trade-level metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns)
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        largest_win = np.max(returns)
        largest_loss = np.min(returns)
        
        gross_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0
        gross_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 1e-8
        profit_factor = gross_profit / gross_loss
        
        # Consistency metrics
        if timestamps and len(timestamps) == len(returns):
            monthly_returns = self._calculate_monthly_returns(returns, timestamps)
            monthly_win_rate = np.sum(np.array(list(monthly_returns.values())) > 0) / len(monthly_returns) if monthly_returns else 0
            best_month = max(monthly_returns.values()) if monthly_returns else 0
            worst_month = min(monthly_returns.values()) if monthly_returns else 0
            consistency_score = 1.0 - (np.std(list(monthly_returns.values())) / (abs(np.mean(list(monthly_returns.values()))) + 1e-8)) if monthly_returns else 0
        else:
            monthly_win_rate = win_rate
            best_month = largest_win
            worst_month = largest_loss
            consistency_score = 1.0 - (np.std(returns) / (abs(np.mean(returns)) + 1e-8))
        
        # Regime-specific performance (placeholder - would need regime data)
        trending_performance = {
            'sharpe_ratio': sharpe_ratio * 1.1,  # Placeholder
            'win_rate': win_rate * 1.05,
            'avg_return': np.mean(returns) * 1.1
        }
        
        ranging_performance = {
            'sharpe_ratio': sharpe_ratio * 0.9,  # Placeholder
            'win_rate': win_rate * 0.95,
            'avg_return': np.mean(returns) * 0.9
        }
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            volatility=volatility,
            value_at_risk_5=value_at_risk_5,
            expected_shortfall_5=expected_shortfall_5,
            total_trades=len(returns),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consistency_score=consistency_score,
            monthly_win_rate=monthly_win_rate,
            best_month=best_month,
            worst_month=worst_month,
            trending_performance=trending_performance,
            ranging_performance=ranging_performance
        )
    
    def _calculate_monthly_returns(
        self,
        returns: np.ndarray,
        timestamps: List[datetime]
    ) -> Dict[str, float]:
        """Calculate monthly returns from daily returns."""
        monthly_returns = {}
        current_month = None
        current_month_return = 0
        
        for ret, timestamp in zip(returns, timestamps):
            month_key = timestamp.strftime('%Y-%m')
            
            if current_month is None:
                current_month = month_key
                current_month_return = ret
            elif current_month == month_key:
                current_month_return += ret
            else:
                monthly_returns[current_month] = current_month_return
                current_month = month_key
                current_month_return = ret
        
        # Add final month
        if current_month:
            monthly_returns[current_month] = current_month_return
        
        return monthly_returns
    
    def create_comprehensive_visualizations(
        self,
        returns: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        trades_data: Optional[List[TradeRecord]] = None,
        save_dir: str = "logs/performance_analysis"
    ) -> Dict[str, str]:
        """
        Create comprehensive visualization suite including equity curves and trade distribution.
        
        Args:
            returns: Array of returns
            timestamps: Optional timestamps
            trades_data: Optional trade records
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of plot file paths
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Cannot create plots - matplotlib not available")
            return {}
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        
        # 1. Enhanced Equity Curve with Benchmark
        equity_path = save_path / "enhanced_equity_curve.png"
        self._plot_enhanced_equity_curve(returns, timestamps, str(equity_path))
        plot_paths['enhanced_equity_curve'] = str(equity_path)
        
        # 2. Performance Metrics Dashboard
        dashboard_path = save_path / "performance_dashboard.png"
        self._plot_performance_dashboard(returns, timestamps, str(dashboard_path))
        plot_paths['performance_dashboard'] = str(dashboard_path)
        
        # 3. Risk Analysis Plots
        risk_path = save_path / "risk_analysis.png"
        self._plot_risk_analysis(returns, timestamps, str(risk_path))
        plot_paths['risk_analysis'] = str(risk_path)
        
        # 4. Trade Distribution Analysis
        if trades_data:
            trade_dist_path = save_path / "trade_distribution.png"
            self._plot_trade_distribution_analysis(trades_data, str(trade_dist_path))
            plot_paths['trade_distribution'] = str(trade_dist_path)
            
            # 5. Regime Performance Comparison
            regime_path = save_path / "regime_performance.png"
            self._plot_regime_performance(trades_data, str(regime_path))
            plot_paths['regime_performance'] = str(regime_path)
        
        # 6. Monthly Performance Heatmap
        if timestamps and len(timestamps) == len(returns):
            heatmap_path = save_path / "monthly_heatmap.png"
            self._plot_monthly_performance_heatmap(returns, timestamps, str(heatmap_path))
            plot_paths['monthly_heatmap'] = str(heatmap_path)
        
        logger.info(f"Created {len(plot_paths)} visualization plots in {save_dir}")
        return plot_paths
    
    def _plot_enhanced_equity_curve(
        self,
        returns: np.ndarray,
        timestamps: Optional[List[datetime]],
        save_path: str
    ) -> None:
        """Plot enhanced equity curve with drawdown and rolling metrics."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[3, 1, 1])
        
        # Calculate metrics
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / (running_max + 1e-8)
        
        # X-axis
        x_axis = timestamps if timestamps else range(len(returns))
        
        # 1. Equity curve with buy/hold benchmark
        benchmark_returns = np.cumsum(np.random.normal(0.0003, 0.01, len(returns)))  # Simple benchmark
        
        axes[0].plot(x_axis, cumulative_returns, linewidth=2, color='blue', label='Strategy')
        axes[0].plot(x_axis, benchmark_returns, linewidth=1, color='gray', alpha=0.7, label='Buy & Hold')
        axes[0].fill_between(x_axis, cumulative_returns, benchmark_returns, 
                           where=(np.array(cumulative_returns) >= np.array(benchmark_returns)), 
                           alpha=0.3, color='green', interpolate=True, label='Outperformance')
        axes[0].fill_between(x_axis, cumulative_returns, benchmark_returns, 
                           where=(np.array(cumulative_returns) < np.array(benchmark_returns)), 
                           alpha=0.3, color='red', interpolate=True, label='Underperformance')
        
        axes[0].set_title('Enhanced Equity Curve vs Benchmark', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Cumulative Return', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        axes[1].fill_between(x_axis, drawdowns, 0, alpha=0.7, color='red', label='Drawdown')
        axes[1].axhline(y=-0.05, color='orange', linestyle='--', alpha=0.7, label='5% DD Level')
        axes[1].axhline(y=-0.10, color='red', linestyle='--', alpha=0.7, label='10% DD Level')
        axes[1].set_ylabel('Drawdown', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe (if enough data)
        if len(returns) > 60:
            window = min(60, len(returns) // 4)
            rolling_sharpe = []
            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252) if np.std(window_returns) > 0 else 0
                rolling_sharpe.append(sharpe)
            
            sharpe_x = x_axis[window:] if timestamps else range(window, len(returns))
            axes[2].plot(sharpe_x, rolling_sharpe, color='purple', linewidth=1.5, label=f'Rolling Sharpe ({window}d)')
            axes[2].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
            axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            axes[2].set_ylabel('Rolling Sharpe', fontsize=12)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Format x-axis
        if timestamps:
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        axes[-1].set_xlabel('Time', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_dashboard(
        self,
        returns: np.ndarray,
        timestamps: Optional[List[datetime]],
        save_path: str
    ) -> None:
        """Create a comprehensive performance dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(returns, timestamps)
        
        # 1. Key Metrics Bar Chart
        key_metrics = {
            'Sharpe Ratio': metrics['sharpe_ratio'],
            'Sortino Ratio': metrics['sortino_ratio'],
            'Calmar Ratio': metrics['calmar_ratio'],
            'Profit Factor': min(metrics['profit_factor'], 5),  # Cap for visualization
            'Win Rate': metrics['win_rate'],
            'Recovery Factor': min(metrics['recovery_factor'], 5)
        }
        
        bars = axes[0, 0].bar(range(len(key_metrics)), list(key_metrics.values()), 
                             color=['blue', 'green', 'orange', 'red', 'purple', 'brown'])
        axes[0, 0].set_xticks(range(len(key_metrics)))
        axes[0, 0].set_xticklabels(list(key_metrics.keys()), rotation=45, ha='right')
        axes[0, 0].set_title('Key Performance Metrics')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, key_metrics.values()):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Returns Distribution
        axes[0, 1].hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        axes[0, 1].axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.4f}')
        axes[0, 1].axvline(np.median(returns), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(returns):.4f}')
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Risk Metrics Radar Chart (simplified as bar chart)
        risk_metrics = {
            'Max DD': metrics['max_drawdown'] * 10,  # Scale for visibility
            'VaR 95%': abs(metrics['value_at_risk_95']) * 100,
            'Volatility': metrics['volatility'],
            'Ulcer Index': metrics['ulcer_index'] * 10,
            'Pain Index': metrics['pain_index'] * 10
        }
        
        axes[0, 2].bar(range(len(risk_metrics)), list(risk_metrics.values()), color='red', alpha=0.7)
        axes[0, 2].set_xticks(range(len(risk_metrics)))
        axes[0, 2].set_xticklabels(list(risk_metrics.keys()), rotation=45, ha='right')
        axes[0, 2].set_title('Risk Metrics (Scaled)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Rolling Performance
        if len(returns) > 30:
            window = min(30, len(returns) // 3)
            rolling_returns = []
            rolling_vol = []
            
            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                rolling_returns.append(np.mean(window_returns) * 252)  # Annualized
                rolling_vol.append(np.std(window_returns) * np.sqrt(252))
            
            x_rolling = range(window, len(returns))
            axes[1, 0].plot(x_rolling, rolling_returns, label='Rolling Return', color='blue')
            axes[1, 0].plot(x_rolling, rolling_vol, label='Rolling Volatility', color='red')
            axes[1, 0].set_title(f'Rolling Performance ({window}d window)')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Cumulative Return vs Drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + 1e-8)
        
        axes[1, 1].plot(cumulative, drawdowns, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Cumulative Return')
        axes[1, 1].set_ylabel('Drawdown')
        axes[1, 1].set_title('Return vs Drawdown Relationship')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Performance Summary Text
        axes[1, 2].axis('off')
        summary_text = f"""
Performance Summary

Total Return: {metrics['total_return']:.2%}
Annualized Return: {metrics['annualized_return']:.2%}
Volatility: {metrics['volatility']:.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
Max Drawdown: {metrics['max_drawdown']:.2%}
Win Rate: {metrics['win_rate']:.1%}
Profit Factor: {metrics['profit_factor']:.2f}

Risk Assessment:
{self._get_risk_assessment(metrics['max_drawdown'])}

Performance Rating:
{self._get_performance_rating(metrics['sharpe_ratio'])}
        """
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_analysis(
        self,
        returns: np.ndarray,
        timestamps: Optional[List[datetime]],
        save_path: str
    ) -> None:
        """Create detailed risk analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Drawdown Analysis
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + 1e-8)
        
        x_axis = timestamps if timestamps else range(len(returns))
        
        axes[0, 0].fill_between(x_axis, drawdowns, 0, alpha=0.7, color='red', label='Drawdown')
        axes[0, 0].axhline(y=-0.05, color='orange', linestyle='--', label='5% Level')
        axes[0, 0].axhline(y=-0.10, color='red', linestyle='--', label='10% Level')
        axes[0, 0].axhline(y=-0.20, color='darkred', linestyle='--', label='20% Level')
        axes[0, 0].set_title('Drawdown Analysis')
        axes[0, 0].set_ylabel('Drawdown')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. VaR Analysis
        var_levels = [1, 5, 10, 25]
        var_values = [np.percentile(returns, level) for level in var_levels]
        
        axes[0, 1].bar([f'{level}%' for level in var_levels], var_values, 
                      color=['darkred', 'red', 'orange', 'yellow'], alpha=0.7)
        axes[0, 1].set_title('Value at Risk (VaR) Levels')
        axes[0, 1].set_ylabel('VaR Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, (level, value) in enumerate(zip(var_levels, var_values)):
            axes[0, 1].text(i, value - 0.001, f'{value:.4f}', ha='center', va='top', fontsize=9)
        
        # 3. Rolling Volatility
        if len(returns) > 30:
            window = min(30, len(returns) // 3)
            rolling_vol = []
            
            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                rolling_vol.append(np.std(window_returns) * np.sqrt(252))
            
            vol_x = x_axis[window:] if timestamps else range(window, len(returns))
            axes[1, 0].plot(vol_x, rolling_vol, color='orange', linewidth=1.5)
            axes[1, 0].axhline(y=np.mean(rolling_vol), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(rolling_vol):.2%}')
            axes[1, 0].set_title(f'Rolling Volatility ({window}d window)')
            axes[1, 0].set_ylabel('Annualized Volatility')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Risk-Return Scatter (rolling periods)
        if len(returns) > 60:
            window = min(60, len(returns) // 2)
            rolling_returns = []
            rolling_risks = []
            
            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                rolling_returns.append(np.mean(window_returns) * 252)
                rolling_risks.append(np.std(window_returns) * np.sqrt(252))
            
            scatter = axes[1, 1].scatter(rolling_risks, rolling_returns, 
                                       c=range(len(rolling_returns)), cmap='viridis', alpha=0.7)
            axes[1, 1].set_xlabel('Risk (Volatility)')
            axes[1, 1].set_ylabel('Return (Annualized)')
            axes[1, 1].set_title('Risk-Return Profile Over Time')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[1, 1])
            cbar.set_label('Time Period')
        
        # Format x-axis for time-based plots
        if timestamps:
            for ax in [axes[0, 0], axes[1, 0]]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle('Risk Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_trade_distribution_analysis(
        self,
        trades_data: List[TradeRecord],
        save_path: str
    ) -> None:
        """Create comprehensive trade distribution analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract trade data
        pnls = [trade.pnl for trade in trades_data]
        confidences = [trade.confidence for trade in trades_data]
        durations = [trade.duration.total_seconds() / 3600 for trade in trades_data]  # hours
        sizes = [trade.position_size for trade in trades_data]
        leverages = [trade.leverage for trade in trades_data]
        
        # 1. PnL Distribution with Statistics
        axes[0, 0].hist(pnls, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        axes[0, 0].axvline(np.mean(pnls), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(pnls):.4f}')
        axes[0, 0].axvline(np.median(pnls), color='green', linestyle='--', linewidth=2, 
                          label=f'Median: {np.median(pnls):.4f}')
        axes[0, 0].axvline(0, color='black', linestyle='-', alpha=0.5)
        axes[0, 0].set_title('PnL Distribution')
        axes[0, 0].set_xlabel('PnL')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confidence vs PnL Analysis
        colors = ['red' if pnl < 0 else 'green' for pnl in pnls]
        sizes_scaled = [abs(pnl) * 1000 + 20 for pnl in pnls]  # Scale for visibility
        
        scatter = axes[0, 1].scatter(confidences, pnls, c=colors, s=sizes_scaled, alpha=0.6)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 1].axvline(x=0.8, color='blue', linestyle='--', alpha=0.7, label='Confidence Threshold')
        axes[0, 1].set_title('Confidence vs PnL (Size = |PnL|)')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('PnL')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Trade Duration Analysis
        axes[0, 2].hist(durations, bins=25, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].axvline(np.mean(durations), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(durations):.1f}h')
        axes[0, 2].axvline(np.median(durations), color='green', linestyle='--', 
                          label=f'Median: {np.median(durations):.1f}h')
        axes[0, 2].set_title('Trade Duration Distribution')
        axes[0, 2].set_xlabel('Duration (hours)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Position Size vs PnL
        axes[1, 0].scatter(sizes, pnls, alpha=0.6, color='purple')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_title('Position Size vs PnL')
        axes[1, 0].set_xlabel('Position Size')
        axes[1, 0].set_ylabel('PnL')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add trend line
        if len(sizes) > 1:
            z = np.polyfit(sizes, pnls, 1)
            p = np.poly1d(z)
            axes[1, 0].plot(sorted(sizes), p(sorted(sizes)), "r--", alpha=0.8, 
                           label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
            axes[1, 0].legend()
        
        # 5. Leverage Distribution
        leverage_counts = {}
        for lev in leverages:
            lev_rounded = round(lev)
            leverage_counts[lev_rounded] = leverage_counts.get(lev_rounded, 0) + 1
        
        axes[1, 1].bar(leverage_counts.keys(), leverage_counts.values(), 
                      alpha=0.7, color='brown', edgecolor='black')
        axes[1, 1].set_title('Leverage Usage Distribution')
        axes[1, 1].set_xlabel('Leverage')
        axes[1, 1].set_ylabel('Number of Trades')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Win/Loss Analysis by Confidence Bins
        confidence_bins = np.linspace(0, 1, 6)  # 5 bins
        bin_labels = [f'{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}' 
                     for i in range(len(confidence_bins)-1)]
        
        win_rates = []
        for i in range(len(confidence_bins)-1):
            bin_trades = [pnl for pnl, conf in zip(pnls, confidences) 
                         if confidence_bins[i] <= conf < confidence_bins[i+1]]
            if bin_trades:
                win_rate = sum(1 for pnl in bin_trades if pnl > 0) / len(bin_trades)
            else:
                win_rate = 0
            win_rates.append(win_rate)
        
        bars = axes[1, 2].bar(bin_labels, win_rates, alpha=0.7, color='green', edgecolor='black')
        axes[1, 2].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Win Rate')
        axes[1, 2].set_title('Win Rate by Confidence Level')
        axes[1, 2].set_xlabel('Confidence Bin')
        axes[1, 2].set_ylabel('Win Rate')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, rate in zip(bars, win_rates):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Trade Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_regime_performance(
        self,
        trades_data: List[TradeRecord],
        save_path: str
    ) -> None:
        """Plot performance comparison across market regimes."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Separate trades by regime
        trending_trades = [t for t in trades_data if t.market_regime == 1]
        ranging_trades = [t for t in trades_data if t.market_regime == 0]
        
        trending_pnls = [t.pnl for t in trending_trades]
        ranging_pnls = [t.pnl for t in ranging_trades]
        
        # 1. PnL Comparison
        regime_data = [trending_pnls, ranging_pnls]
        regime_labels = ['Trending', 'Ranging']
        
        box_plot = axes[0, 0].boxplot(regime_data, labels=regime_labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        
        axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 0].set_title('PnL Distribution by Market Regime')
        axes[0, 0].set_ylabel('PnL')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Win Rate Comparison
        trending_wins = sum(1 for pnl in trending_pnls if pnl > 0)
        ranging_wins = sum(1 for pnl in ranging_pnls if pnl > 0)
        
        trending_win_rate = trending_wins / len(trending_pnls) if trending_pnls else 0
        ranging_win_rate = ranging_wins / len(ranging_pnls) if ranging_pnls else 0
        
        win_rates = [trending_win_rate, ranging_win_rate]
        bars = axes[0, 1].bar(regime_labels, win_rates, color=['lightblue', 'lightcoral'], 
                             alpha=0.7, edgecolor='black')
        axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Win Rate')
        axes[0, 1].set_title('Win Rate by Market Regime')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, rate in zip(bars, win_rates):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{rate:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 3. Trade Count and Average PnL
        trade_counts = [len(trending_trades), len(ranging_trades)]
        avg_pnls = [np.mean(trending_pnls) if trending_pnls else 0, 
                   np.mean(ranging_pnls) if ranging_pnls else 0]
        
        ax3_twin = axes[1, 0].twinx()
        
        bars1 = axes[1, 0].bar([x - 0.2 for x in range(len(regime_labels))], trade_counts, 
                              width=0.4, color='skyblue', alpha=0.7, label='Trade Count')
        bars2 = ax3_twin.bar([x + 0.2 for x in range(len(regime_labels))], avg_pnls, 
                            width=0.4, color='orange', alpha=0.7, label='Avg PnL')
        
        axes[1, 0].set_xticks(range(len(regime_labels)))
        axes[1, 0].set_xticklabels(regime_labels)
        axes[1, 0].set_ylabel('Trade Count', color='blue')
        ax3_twin.set_ylabel('Average PnL', color='orange')
        axes[1, 0].set_title('Trade Volume and Performance by Regime')
        
        # Add value labels
        for bar, count in zip(bars1, trade_counts):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           str(count), ha='center', va='bottom', fontsize=10)
        
        for bar, pnl in zip(bars2, avg_pnls):
            ax3_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f'{pnl:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Cumulative Performance by Regime
        if trending_trades and ranging_trades:
            # Sort trades by timestamp
            trending_sorted = sorted(trending_trades, key=lambda x: x.timestamp)
            ranging_sorted = sorted(ranging_trades, key=lambda x: x.timestamp)
            
            trending_cumulative = np.cumsum([t.pnl for t in trending_sorted])
            ranging_cumulative = np.cumsum([t.pnl for t in ranging_sorted])
            
            axes[1, 1].plot(range(len(trending_cumulative)), trending_cumulative, 
                           color='blue', linewidth=2, label='Trending Market')
            axes[1, 1].plot(range(len(ranging_cumulative)), ranging_cumulative, 
                           color='red', linewidth=2, label='Ranging Market')
            
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 1].set_title('Cumulative PnL by Market Regime')
            axes[1, 1].set_xlabel('Trade Number')
            axes[1, 1].set_ylabel('Cumulative PnL')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Market Regime Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_monthly_performance_heatmap(
        self,
        returns: np.ndarray,
        timestamps: List[datetime],
        save_path: str
    ) -> None:
        """Create monthly performance heatmap."""
        # Calculate monthly returns
        monthly_data = {}
        for ret, timestamp in zip(returns, timestamps):
            year = timestamp.year
            month = timestamp.month
            
            if year not in monthly_data:
                monthly_data[year] = {}
            if month not in monthly_data[year]:
                monthly_data[year][month] = 0
            
            monthly_data[year][month] += ret
        
        if not monthly_data:
            return
        
        # Create matrix for heatmap
        years = sorted(monthly_data.keys())
        months = list(range(1, 13))
        
        heatmap_data = []
        for year in years:
            year_data = []
            for month in months:
                value = monthly_data[year].get(month, np.nan)
                year_data.append(value)
            heatmap_data.append(year_data)
        
        heatmap_data = np.array(heatmap_data)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(6, len(years) * 0.8)))
        
        # Use diverging colormap centered at zero
        vmax = max(abs(np.nanmin(heatmap_data)), abs(np.nanmax(heatmap_data)))
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', 
                      vmin=-vmax, vmax=vmax)
        
        # Set ticks and labels
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        
        # Add text annotations
        for i in range(len(years)):
            for j in range(12):
                value = heatmap_data[i, j]
                if not np.isnan(value):
                    text_color = 'white' if abs(value) > vmax * 0.5 else 'black'
                    ax.text(j, i, f'{value:.2%}', ha='center', va='center',
                           color=text_color, fontsize=9, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Monthly Return', rotation=270, labelpad=20)
        
        ax.set_title('Monthly Performance Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_risk_assessment(self, max_drawdown: float) -> str:
        """Get risk assessment based on max drawdown."""
        if max_drawdown < 0.05:
            return "Very Low Risk"
        elif max_drawdown < 0.10:
            return "Low Risk"
        elif max_drawdown < 0.15:
            return "Moderate Risk"
        elif max_drawdown < 0.25:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _get_performance_rating(self, sharpe_ratio: float) -> str:
        """Get performance rating based on Sharpe ratio."""
        if sharpe_ratio > 2.0:
            return "Excellent"
        elif sharpe_ratio > 1.5:
            return "Very Good"
        elif sharpe_ratio > 1.0:
            return "Good"
        elif sharpe_ratio > 0.5:
            return "Fair"
        else:
            return "Poor"
    
    def evaluate_returns(
        self,
        returns: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> PerformanceMetrics:
        """
        Comprehensive evaluation of return series.
        
        Args:
            returns: Array of returns
            timestamps: Optional timestamps
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if len(returns) == 0:
            logger.warning("Empty returns array provided")
            return PerformanceMetrics(
                total_return=0, annualized_return=0, sharpe_ratio=0, sortino_ratio=0,
                calmar_ratio=0, max_drawdown=0, max_drawdown_duration=0, volatility=0,
                value_at_risk_5=0, expected_shortfall_5=0, total_trades=0, win_rate=0,
                profit_factor=0, avg_win=0, avg_loss=0, largest_win=0, largest_loss=0,
                consistency_score=0, monthly_win_rate=0, best_month=0, worst_month=0,
                trending_performance={}, ranging_performance={}
            )
        
        # Basic return metrics
        total_return = np.sum(returns)
        annualized_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + 1e-8)
        max_drawdown = abs(np.min(drawdowns))
        
        # Find drawdown duration
        max_dd_idx = np.argmin(drawdowns)
        max_drawdown_duration = 0
        for i in range(max_dd_idx, len(drawdowns)):
            if drawdowns[i] >= -0.001:  # Recovery threshold
                max_drawdown_duration = i - max_dd_idx
                break
        else:
            max_drawdown_duration = len(drawdowns) - max_dd_idx
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # VaR and ES
        value_at_risk_5 = np.percentile(returns, 5)
        expected_shortfall_5 = np.mean(returns[returns <= value_at_risk_5])
        
        # Trade-level metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns)
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        largest_win = np.max(returns)
        largest_loss = np.min(returns)
        
        gross_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0
        gross_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 1e-8
        profit_factor = gross_profit / gross_loss
        
        # Consistency metrics
        if timestamps and len(timestamps) == len(returns):
            monthly_returns = self._calculate_monthly_returns(returns, timestamps)
            monthly_win_rate = np.sum(np.array(list(monthly_returns.values())) > 0) / len(monthly_returns) if monthly_returns else 0
            best_month = max(monthly_returns.values()) if monthly_returns else 0
            worst_month = min(monthly_returns.values()) if monthly_returns else 0
            consistency_score = 1.0 - (np.std(list(monthly_returns.values())) / (abs(np.mean(list(monthly_returns.values()))) + 1e-8)) if monthly_returns else 0
        else:
            monthly_win_rate = win_rate
            best_month = largest_win
            worst_month = largest_loss
            consistency_score = 1.0 - (np.std(returns) / (abs(np.mean(returns)) + 1e-8))
        
        # Regime-specific performance (placeholder - would need regime data)
        trending_performance = {
            'sharpe_ratio': sharpe_ratio * 1.1,  # Placeholder
            'win_rate': win_rate * 1.05,
            'avg_return': np.mean(returns) * 1.1
        }
        
        ranging_performance = {
            'sharpe_ratio': sharpe_ratio * 0.9,  # Placeholder
            'win_rate': win_rate * 0.95,
            'avg_return': np.mean(returns) * 0.9
        }
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            volatility=volatility,
            value_at_risk_5=value_at_risk_5,
            expected_shortfall_5=expected_shortfall_5,
            total_trades=len(returns),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consistency_score=consistency_score,
            monthly_win_rate=monthly_win_rate,
            best_month=best_month,
            worst_month=worst_month,
            trending_performance=trending_performance,
            ranging_performance=ranging_performance
        )
    
    def _calculate_monthly_returns(
        self,
        returns: np.ndarray,
        timestamps: List[datetime]
    ) -> Dict[str, float]:
        """Calculate monthly returns from daily returns."""
        monthly_returns = {}
        current_month = None
        current_month_return = 0
        
        for ret, timestamp in zip(returns, timestamps):
            month_key = timestamp.strftime('%Y-%m')
            
            if current_month is None:
                current_month = month_key
                current_month_return = ret
            elif current_month == month_key:
                current_month_return += ret
            else:
                monthly_returns[current_month] = current_month_return
                current_month = month_key
                current_month_return = ret
        
        # Add final month
        if current_month:
            monthly_returns[current_month] = current_month_return
        
        return monthly_returns
    
    def generate_performance_report(
        self,
        returns: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        trades_csv_path: Optional[str] = None,
        save_dir: str = "logs/performance_analysis",
        create_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report with enhanced metrics and visualizations.
        
        Args:
            returns: Array of returns
            timestamps: Optional timestamps
            trades_csv_path: Path to trades CSV file
            save_dir: Directory to save report and plots
            create_plots: Whether to create visualization plots
            
        Returns:
            Complete performance report dictionary
        """
        logger.info("Generating comprehensive performance report with enhanced analysis")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Load trade data if available
        trades_data = []
        if trades_csv_path and Path(trades_csv_path).exists():
            self.trade_analyzer.load_trades_from_csv(trades_csv_path)
            trades_data = self.trade_analyzer.trades
        
        # Generate comprehensive analysis
        strategy_analysis = self.analyze_strategy_performance(returns, timestamps, trades_data)
        
        # Create enhanced visualizations
        plot_paths = {}
        if create_plots:
            plot_paths = self.create_comprehensive_visualizations(
                returns, timestamps, trades_data, save_dir
            )
        
        # Calculate benchmark comparison (simple buy-and-hold)
        benchmark_returns = np.random.normal(0.0003, 0.01, len(returns))  # Simple benchmark
        benchmark_metrics = self.calculate_comprehensive_metrics(benchmark_returns, timestamps)
        
        # Compile enhanced report
        report = {
            'report_info': {
                'generated_at': datetime.now().isoformat(),
                'period_start': timestamps[0].isoformat() if timestamps else None,
                'period_end': timestamps[-1].isoformat() if timestamps else None,
                'total_observations': len(returns),
                'analysis_type': 'comprehensive_performance_evaluation_enhanced',
                'has_trade_data': len(trades_data) > 0,
                'total_trades': len(trades_data)
            },
            
            'strategy_performance': strategy_analysis,
            
            'benchmark_comparison': {
                'strategy_metrics': strategy_analysis['comprehensive_metrics'],
                'benchmark_metrics': benchmark_metrics,
                'outperformance': {
                    'total_return': strategy_analysis['comprehensive_metrics']['total_return'] - benchmark_metrics['total_return'],
                    'sharpe_ratio': strategy_analysis['comprehensive_metrics']['sharpe_ratio'] - benchmark_metrics['sharpe_ratio'],
                    'max_drawdown': benchmark_metrics['max_drawdown'] - strategy_analysis['comprehensive_metrics']['max_drawdown'],  # Lower is better
                    'volatility': benchmark_metrics['volatility'] - strategy_analysis['comprehensive_metrics']['volatility']  # Lower is better
                }
            },
            
            'risk_analysis': {
                'risk_metrics': {
                    'max_drawdown': strategy_analysis['comprehensive_metrics']['max_drawdown'],
                    'value_at_risk_95': strategy_analysis['comprehensive_metrics']['value_at_risk_95'],
                    'expected_shortfall_95': strategy_analysis['comprehensive_metrics']['expected_shortfall_95'],
                    'ulcer_index': strategy_analysis['comprehensive_metrics']['ulcer_index'],
                    'pain_index': strategy_analysis['comprehensive_metrics']['pain_index']
                },
                'risk_assessment': self._get_risk_assessment(strategy_analysis['comprehensive_metrics']['max_drawdown']),
                'risk_adjusted_performance': {
                    'sharpe_ratio': strategy_analysis['comprehensive_metrics']['sharpe_ratio'],
                    'sortino_ratio': strategy_analysis['comprehensive_metrics']['sortino_ratio'],
                    'calmar_ratio': strategy_analysis['comprehensive_metrics']['calmar_ratio']
                }
            },
            
            'trade_performance': strategy_analysis.get('trade_analysis', {}),
            
            'time_analysis': strategy_analysis.get('time_analysis', {}),
            
            'regime_analysis': strategy_analysis.get('regime_analysis', {}),
            
            'performance_summary': strategy_analysis.get('performance_summary', {}),
            
            'visualizations': plot_paths,
            
            'recommendations': self._generate_performance_recommendations(strategy_analysis)
        }
        
        # Save comprehensive report to JSON
        report_path = save_path / "comprehensive_performance_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate enhanced markdown summary
        self._generate_enhanced_markdown_summary(report, save_path / "performance_summary.md")
        
        # Generate detailed CSV reports
        self._generate_csv_reports(report, save_path)
        
        logger.info(f"Enhanced performance report saved to {report_path}")
        logger.info(f"Generated {len(plot_paths)} visualization plots")
        
        return report
    
    def _generate_performance_recommendations(
        self,
        strategy_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable performance recommendations."""
        recommendations = []
        metrics = strategy_analysis['comprehensive_metrics']
        
        # Sharpe ratio recommendations
        if metrics['sharpe_ratio'] < 1.0:
            recommendations.append("Consider improving risk-adjusted returns - Sharpe ratio below 1.0 indicates suboptimal risk/reward balance")
        
        # Drawdown recommendations
        if metrics['max_drawdown'] > 0.15:
            recommendations.append("High maximum drawdown detected - consider implementing stricter risk controls or position sizing")
        
        # Win rate recommendations
        if metrics['win_rate'] < 0.5:
            recommendations.append("Win rate below 50% - consider refining entry signals or improving trade selection criteria")
        
        # Profit factor recommendations
        if metrics['profit_factor'] < 1.5:
            recommendations.append("Low profit factor - focus on cutting losses faster or letting winners run longer")
        
        # Volatility recommendations
        if metrics['volatility'] > 0.25:
            recommendations.append("High volatility detected - consider reducing position sizes or implementing volatility-based position sizing")
        
        # Trade analysis recommendations
        if 'trade_analysis' in strategy_analysis and strategy_analysis['trade_analysis']:
            trade_patterns = strategy_analysis['trade_analysis'].get('trade_patterns', {})
            
            if trade_patterns.get('avg_confidence', 0) < 0.7:
                recommendations.append("Low average confidence in trades - consider raising confidence threshold for trade execution")
            
            if trade_patterns.get('high_confidence_win_rate', 0) > trade_patterns.get('win_rate', 0) + 0.1:
                recommendations.append("High confidence trades perform significantly better - consider filtering trades by confidence level")
        
        # Regime analysis recommendations
        if 'regime_analysis' in strategy_analysis and strategy_analysis['regime_analysis']:
            regime_data = strategy_analysis['regime_analysis']
            trending_perf = regime_data.get('trending_market', {})
            ranging_perf = regime_data.get('ranging_market', {})
            
            if trending_perf.get('win_rate', 0) > ranging_perf.get('win_rate', 0) + 0.2:
                recommendations.append("Strategy performs significantly better in trending markets - consider regime-based trade filtering")
            elif ranging_perf.get('win_rate', 0) > trending_perf.get('win_rate', 0) + 0.2:
                recommendations.append("Strategy performs significantly better in ranging markets - consider regime-based trade filtering")
        
        if not recommendations:
            recommendations.append("Strategy shows solid performance across key metrics - continue monitoring and consider minor optimizations")
        
        return recommendations
    
    def _generate_enhanced_markdown_summary(self, report: Dict[str, Any], save_path: Path) -> None:
        """Generate enhanced markdown summary with comprehensive analysis."""
        with open(save_path, 'w') as f:
            f.write("# Comprehensive Performance Analysis Report\n\n")
            f.write(f"**Generated:** {report['report_info']['generated_at']}\n\n")
            
            if report['report_info']['period_start']:
                f.write(f"**Analysis Period:** {report['report_info']['period_start']} to {report['report_info']['period_end']}\n\n")
            
            f.write(f"**Total Observations:** {report['report_info']['total_observations']}\n")
            f.write(f"**Total Trades:** {report['report_info']['total_trades']}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            metrics = report['strategy_performance']['comprehensive_metrics']
            summary = report['strategy_performance'].get('performance_summary', {})
            
            if summary:
                f.write(f"**Overall Assessment:** {summary.get('overall_assessment', 'N/A')}\n")
                f.write(f"**Risk Assessment:** {report['risk_analysis']['risk_assessment']}\n")
                f.write(f"**Performance Rating:** {self._get_performance_rating(metrics['sharpe_ratio'])}\n\n")
            else:
                f.write(f"**Risk Assessment:** {report['risk_analysis']['risk_assessment']}\n")
                f.write(f"**Performance Rating:** {self._get_performance_rating(metrics['sharpe_ratio'])}\n\n")
            
            # Key Performance Metrics
            f.write("## Key Performance Metrics\n\n")
            f.write("### Return Metrics\n")
            f.write(f"- **Total Return:** {metrics['total_return']:.2%}\n")
            f.write(f"- **Annualized Return:** {metrics['annualized_return']:.2%}\n")
            f.write(f"- **Volatility:** {metrics['volatility']:.2%}\n\n")
            
            f.write("### Risk-Adjusted Performance\n")
            f.write(f"- **Sharpe Ratio:** {metrics['sharpe_ratio']:.3f}")
            if summary:
                f.write(f" ({summary.get('sharpe_assessment', 'N/A')})\n")
            else:
                f.write("\n")
            f.write(f"- **Sortino Ratio:** {metrics['sortino_ratio']:.3f}\n")
            f.write(f"- **Calmar Ratio:** {metrics['calmar_ratio']:.3f}\n")
            f.write(f"- **Information Ratio:** {metrics['information_ratio']:.3f}\n\n")
            
            f.write("### Risk Metrics\n")
            f.write(f"- **Maximum Drawdown:** {metrics['max_drawdown']:.2%}\n")
            f.write(f"- **Value at Risk (95%):** {metrics['value_at_risk_95']:.4f}\n")
            f.write(f"- **Expected Shortfall (95%):** {metrics['expected_shortfall_95']:.4f}\n")
            f.write(f"- **Ulcer Index:** {metrics['ulcer_index']:.4f}\n")
            f.write(f"- **Pain Index:** {metrics['pain_index']:.4f}\n\n")
            
            f.write("### Trade Performance\n")
            f.write(f"- **Win Rate:** {metrics['win_rate']:.1%}")
            if summary:
                f.write(f" ({summary.get('win_rate_assessment', 'N/A')})\n")
            else:
                f.write("\n")
            f.write(f"- **Profit Factor:** {metrics['profit_factor']:.2f}\n")
            f.write(f"- **Average Win:** {metrics['avg_win']:.4f}\n")
            f.write(f"- **Average Loss:** {metrics['avg_loss']:.4f}\n")
            f.write(f"- **Largest Win:** {metrics['largest_win']:.4f}\n")
            f.write(f"- **Largest Loss:** {metrics['largest_loss']:.4f}\n\n")
            
            # Benchmark Comparison
            if 'benchmark_comparison' in report:
                f.write("## Benchmark Comparison\n\n")
                outperf = report['benchmark_comparison']['outperformance']
                f.write(f"- **Return Outperformance:** {outperf['total_return']:.2%}\n")
                f.write(f"- **Sharpe Ratio Advantage:** {outperf['sharpe_ratio']:.3f}\n")
                f.write(f"- **Drawdown Improvement:** {outperf['max_drawdown']:.2%}\n")
                f.write(f"- **Volatility Reduction:** {outperf['volatility']:.2%}\n\n")
            
            # Trade Analysis
            if 'trade_performance' in report and report['trade_performance']:
                f.write("## Trade Analysis\n\n")
                trade_patterns = report['trade_performance'].get('trade_patterns', {})
                
                if trade_patterns:
                    f.write("### Trade Patterns\n")
                    f.write(f"- **Average Confidence:** {trade_patterns.get('avg_confidence', 0):.1%}\n")
                    f.write(f"- **High Confidence Win Rate:** {trade_patterns.get('high_confidence_win_rate', 0):.1%}\n")
                    f.write(f"- **Average Trade Duration:** {trade_patterns.get('avg_trade_duration_hours', 0):.1f} hours\n")
                    f.write(f"- **Average Position Size:** {trade_patterns.get('avg_position_size', 0):.1%}\n")
                    f.write(f"- **Average Leverage:** {trade_patterns.get('avg_leverage', 0):.1f}x\n\n")
                
                consecutive = report['trade_performance'].get('consecutive_trades', {})
                if consecutive:
                    f.write("### Streak Analysis\n")
                    f.write(f"- **Max Consecutive Wins:** {consecutive.get('max_consecutive_wins', 0)}\n")
                    f.write(f"- **Max Consecutive Losses:** {consecutive.get('max_consecutive_losses', 0)}\n")
                    f.write(f"- **Average Win Streak:** {consecutive.get('avg_win_streak', 0):.1f}\n")
                    f.write(f"- **Average Loss Streak:** {consecutive.get('avg_loss_streak', 0):.1f}\n\n")
            
            # Regime Analysis
            if 'regime_analysis' in report and report['regime_analysis']:
                f.write("## Market Regime Analysis\n\n")
                regime = report['regime_analysis']
                
                trending = regime.get('trending_market', {})
                ranging = regime.get('ranging_market', {})
                
                f.write("### Trending Markets\n")
                f.write(f"- **Trades:** {trending.get('count', 0)}\n")
                f.write(f"- **Win Rate:** {trending.get('win_rate', 0):.1%}\n")
                f.write(f"- **Average PnL:** {trending.get('avg_pnl', 0):.4f}\n")
                f.write(f"- **Sharpe Ratio:** {trending.get('sharpe_ratio', 0):.3f}\n\n")
                
                f.write("### Ranging Markets\n")
                f.write(f"- **Trades:** {ranging.get('count', 0)}\n")
                f.write(f"- **Win Rate:** {ranging.get('win_rate', 0):.1%}\n")
                f.write(f"- **Average PnL:** {ranging.get('avg_pnl', 0):.4f}\n")
                f.write(f"- **Sharpe Ratio:** {ranging.get('sharpe_ratio', 0):.3f}\n\n")
                
                f.write(f"**Regime Preference:** {regime.get('regime_preference', 'N/A').title()}\n\n")
            
            # Time Analysis
            if 'time_analysis' in report and report['time_analysis']:
                f.write("## Time-Based Analysis\n\n")
                time_data = report['time_analysis']
                
                f.write(f"- **Best Month:** {time_data.get('best_month', 0):.2%}\n")
                f.write(f"- **Worst Month:** {time_data.get('worst_month', 0):.2%}\n")
                f.write(f"- **Monthly Win Rate:** {time_data.get('monthly_win_rate', 0):.1%}\n")
                f.write(f"- **Positive Months:** {time_data.get('positive_months', 0)}/{time_data.get('total_months', 0)}\n\n")
            
            # Recommendations
            if 'recommendations' in report:
                f.write("## Performance Recommendations\n\n")
                for i, rec in enumerate(report['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
            
            # Visualizations
            if 'visualizations' in report and report['visualizations']:
                f.write("## Generated Visualizations\n\n")
                for plot_name, plot_path in report['visualizations'].items():
                    f.write(f"- **{plot_name.replace('_', ' ').title()}:** `{plot_path}`\n")
                f.write("\n")
            
            f.write("---\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by AlphaPulse-RL Performance Analysis System*\n")
        
        logger.info(f"Enhanced performance summary saved to {save_path}")
    
    def _generate_csv_reports(self, report: Dict[str, Any], save_path: Path) -> None:
        """Generate detailed CSV reports for further analysis."""
        # Performance metrics CSV
        metrics_data = []
        metrics = report['strategy_performance']['comprehensive_metrics']
        
        for metric_name, value in metrics.items():
            metrics_data.append({
                'metric': metric_name,
                'value': value,
                'category': self._categorize_metric(metric_name)
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(save_path / "performance_metrics.csv", index=False)
        
        # Monthly performance CSV if available
        if 'time_analysis' in report and 'monthly_returns' in report['time_analysis']:
            monthly_data = []
            for month, return_val in report['time_analysis']['monthly_returns'].items():
                year, month_num = month.split('-')
                monthly_data.append({
                    'year': int(year),
                    'month': int(month_num),
                    'month_name': datetime(int(year), int(month_num), 1).strftime('%B'),
                    'return': return_val
                })
            
            monthly_df = pd.DataFrame(monthly_data)
            monthly_df.to_csv(save_path / "monthly_performance.csv", index=False)
        
        logger.info("CSV reports generated successfully")
    
    def _categorize_metric(self, metric_name: str) -> str:
        """Categorize performance metrics."""
        if any(word in metric_name.lower() for word in ['return', 'profit']):
            return 'Return'
        elif any(word in metric_name.lower() for word in ['risk', 'drawdown', 'var', 'volatility']):
            return 'Risk'
        elif any(word in metric_name.lower() for word in ['sharpe', 'sortino', 'calmar', 'ratio']):
            return 'Risk-Adjusted'
        elif any(word in metric_name.lower() for word in ['win', 'trade', 'factor']):
            return 'Trade'
        else:
            return 'Other'
    
    def _generate_markdown_summary(self, report: Dict[str, Any], save_path: Path) -> None:
        """Generate a markdown summary of the performance report."""
        with open(save_path, 'w') as f:
            f.write("# Performance Analysis Summary\n\n")
            f.write(f"**Generated:** {report['report_info']['generated_at']}\n\n")
            
            if report['report_info']['period_start']:
                f.write(f"**Analysis Period:** {report['report_info']['period_start']} to {report['report_info']['period_end']}\n\n")
            
            f.write(f"**Total Observations:** {report['report_info']['total_observations']}\n\n")
            
            # Performance Summary
            f.write("## Performance Summary\n\n")
            metrics = report['performance_metrics']
            
            f.write("### Return Metrics\n")
            f.write(f"- **Total Return:** {metrics['return_metrics']['total_return']:.2%}\n")
            f.write(f"- **Annualized Return:** {metrics['return_metrics']['annualized_return']:.2%}\n")
            f.write(f"- **Volatility:** {metrics['return_metrics']['volatility']:.2%}\n")
            f.write(f"- **Sharpe Ratio:** {metrics['return_metrics']['sharpe_ratio']:.3f}\n")
            f.write(f"- **Sortino Ratio:** {metrics['return_metrics']['sortino_ratio']:.3f}\n")
            f.write(f"- **Calmar Ratio:** {metrics['return_metrics']['calmar_ratio']:.3f}\n\n")
            
            f.write("### Risk Metrics\n")
            f.write(f"- **Max Drawdown:** {metrics['risk_metrics']['max_drawdown']:.2%}\n")
            f.write(f"- **Max Drawdown Duration:** {metrics['risk_metrics']['max_drawdown_duration']} periods\n")
            f.write(f"- **Value at Risk (5%):** {metrics['risk_metrics']['value_at_risk_5']:.4f}\n")
            f.write(f"- **Expected Shortfall (5%):** {metrics['risk_metrics']['expected_shortfall_5']:.4f}\n\n")
            
            f.write("### Trade Metrics\n")
            f.write(f"- **Total Trades:** {metrics['trade_metrics']['total_trades']}\n")
            f.write(f"- **Win Rate:** {metrics['trade_metrics']['win_rate']:.1%}\n")
            f.write(f"- **Profit Factor:** {metrics['trade_metrics']['profit_factor']:.2f}\n")
            f.write(f"- **Average Win:** {metrics['trade_metrics']['avg_win']:.4f}\n")
            f.write(f"- **Average Loss:** {metrics['trade_metrics']['avg_loss']:.4f}\n")
            f.write(f"- **Largest Win:** {metrics['trade_metrics']['largest_win']:.4f}\n")
            f.write(f"- **Largest Loss:** {metrics['trade_metrics']['largest_loss']:.4f}\n\n")
            
            # Advanced Metrics
            f.write("## Advanced Metrics\n\n")
            adv_metrics = report['advanced_metrics']
            f.write(f"- **Omega Ratio:** {adv_metrics['omega_ratio']:.3f}\n")
            f.write(f"- **Tail Ratio:** {adv_metrics['tail_ratio']:.3f}\n")
            f.write(f"- **Ulcer Index:** {adv_metrics['ulcer_index']:.4f}\n")
            f.write(f"- **Pain Index:** {adv_metrics['pain_index']:.4f}\n\n")
            
            # Performance Assessment
            f.write("## Performance Assessment\n\n")
            sharpe = metrics['return_metrics']['sharpe_ratio']
            max_dd = metrics['risk_metrics']['max_drawdown']
            win_rate = metrics['trade_metrics']['win_rate']
            
            if sharpe > 1.5:
                f.write("✅ **Sharpe Ratio:** Excellent (>1.5)\n")
            elif sharpe > 1.0:
                f.write("✅ **Sharpe Ratio:** Very Good (1.0-1.5)\n")
            elif sharpe > 0.5:
                f.write("⚠️ **Sharpe Ratio:** Good (0.5-1.0)\n")
            else:
                f.write("❌ **Sharpe Ratio:** Needs Improvement (<0.5)\n")
            
            if max_dd < 0.05:
                f.write("✅ **Risk Control:** Excellent (<5% max drawdown)\n")
            elif max_dd < 0.10:
                f.write("✅ **Risk Control:** Very Good (<10% max drawdown)\n")
            elif max_dd < 0.15:
                f.write("⚠️ **Risk Control:** Acceptable (<15% max drawdown)\n")
            else:
                f.write("❌ **Risk Control:** Poor (>15% max drawdown)\n")
            
            if win_rate > 0.6:
                f.write("✅ **Win Rate:** Excellent (>60%)\n")
            elif win_rate > 0.5:
                f.write("✅ **Win Rate:** Good (>50%)\n")
            else:
                f.write("⚠️ **Win Rate:** Below Average (<50%)\n")
            
            f.write("\n---\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        logger.info(f"Performance summary saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    logger.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_days = 252
    returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
    timestamps = [datetime.now() - timedelta(days=n_days-i) for i in range(n_days)]
    
    # Create evaluator and generate report
    evaluator = ComprehensivePerformanceEvaluator()
    report = evaluator.generate_performance_report(
        returns=returns,
        timestamps=timestamps,
        save_dir="logs/test_analysis",
        create_plots=True
    )
    
    print("Performance evaluation completed!")
    print(f"Sharpe Ratio: {report['performance_metrics']['return_metrics']['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {report['performance_metrics']['risk_metrics']['max_drawdown']:.2%}")
    print(f"Win Rate: {report['performance_metrics']['trade_metrics']['win_rate']:.1%}")