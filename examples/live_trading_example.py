"""
Example script demonstrating how to use the LiveTrader for AlphaPulse-RL trading system.

This example shows:
1. How to configure and initialize the LiveTrader
2. How to start and stop live trading
3. How to monitor trading status and metrics
4. How to handle emergency stops
"""

import asyncio
import logging
import yaml
from pathlib import Path
import signal
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from trading.live_trader import LiveTrader, create_live_trader


# Example configuration
EXAMPLE_CONFIG = {
    'trading': {
        'pairs': ['BTCUSDT', 'ETHUSDT'],
        'interval_seconds': 30,
        'confidence_threshold': 0.8,
        'max_positions': 2,
        'cooldown_minutes': 5
    },
    'portfolio': {
        'initial_balance': 1000.0,
        'data_dir': 'data/portfolio',
        'backup_dir': 'data/portfolio/backups',
        'max_backups': 10,
        'auto_save_interval_minutes': 5
    },
    'agent': {
        'model_path': 'models/ppo_agent.pth',
        'lr_actor': 3e-4,
        'lr_critic': 1e-3,
        'device': 'cpu'
    },
    'data_fetcher': {
        'api_base_url': 'https://api.weex.com',
        'timeout_seconds': 30,
        'rate_limit_requests_per_second': 10,
        'cache_duration_minutes': 60
    },
    'feature_engine': {
        'max_history_length': 200
    },
    'risk': {
        'config_path': 'config/trading_params.yaml'
    },
    'exchange': {
        'api_key': 'your_api_key_here',
        'secret_key': 'your_secret_key_here',
        'timeout_seconds': 30
    },
    'execution': {
        'max_retries': 3,
        'retry_delay': 1.0,
        'commission_rate': 0.001,
        'slippage_rate': 0.0005
    },
    'logging': {
        'level': 'INFO',
        'system_log_file': 'logs/alphapulse.log',
        'trade_history_file': 'logs/trades.csv',
        'ai_decisions_file': 'logs/ai_decisions.json',
        'portfolio_metrics_file': 'logs/portfolio_metrics.json',
        'system_health_file': 'logs/system_health.json',
        'max_json_records': 10000,
        'enable_real_time_monitoring': True
    }
}


class LiveTradingExample:
    """Example class demonstrating LiveTrader usage"""
    
    def __init__(self):
        self.trader: LiveTrader = None
        self.is_running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
        self.is_running = False
    
    async def setup_config(self):
        """Setup configuration files if they don't exist"""
        # Create config directory
        Path('config').mkdir(exist_ok=True)
        
        # Create main config file
        config_file = Path('config/config.yaml')
        if not config_file.exists():
            with open(config_file, 'w') as f:
                yaml.dump(EXAMPLE_CONFIG, f, indent=2)
            print(f"Created example config file: {config_file}")
        
        # Create trading parameters config
        trading_params_file = Path('config/trading_params.yaml')
        if not trading_params_file.exists():
            trading_params = {
                'risk': {
                    'max_leverage': 12.0,
                    'max_position_size_percent': 10.0,
                    'max_daily_loss_percent': 3.0,
                    'max_total_drawdown_percent': 12.0,
                    'volatility_threshold': 0.05
                }
            }
            with open(trading_params_file, 'w') as f:
                yaml.dump(trading_params, f, indent=2)
            print(f"Created trading parameters file: {trading_params_file}")
    
    async def initialize_trader(self):
        """Initialize the LiveTrader"""
        try:
            print("Initializing LiveTrader...")
            self.trader = await create_live_trader('config/config.yaml')
            print("LiveTrader initialized successfully")
            
            # Perform health check
            health_status = await self.trader.health_check()
            print(f"System health check: {health_status['overall_status']}")
            
            for component, status in health_status['components'].items():
                print(f"  {component}: {status}")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize LiveTrader: {e}")
            return False
    
    async def monitor_trading(self):
        """Monitor trading status and metrics"""
        while self.is_running and self.trader.is_trading:
            try:
                # Get trading status
                status = self.trader.get_trading_status()
                
                print("\n" + "="*50)
                print("TRADING STATUS")
                print("="*50)
                print(f"Trading Active: {status['is_trading']}")
                print(f"Emergency Mode: {status['emergency_mode']}")
                print(f"Active Positions: {status['active_positions']}")
                
                # Portfolio metrics
                portfolio = status['portfolio_metrics']
                print(f"\nPortfolio Equity: ${portfolio['total_equity']:.2f}")
                print(f"Daily P&L: ${portfolio['daily_pnl']:.2f} ({portfolio['daily_pnl_pct']:.2f}%)")
                print(f"Total P&L: ${portfolio['total_pnl']:.2f} ({portfolio['total_pnl_pct']:.2f}%)")
                print(f"Max Drawdown: {portfolio['max_drawdown_pct']:.2f}%")
                
                # Risk metrics
                risk = status['risk_metrics']
                print(f"\nRisk Score: {risk['risk_score']:.1f}/100")
                print(f"Current Drawdown: {risk['current_drawdown']:.2f}%")
                print(f"Margin Utilization: {portfolio['margin_utilization_pct']:.2f}%")
                
                # Real-time metrics (if available)
                if status.get('real_time_metrics', {}).get('monitoring_enabled'):
                    rt_metrics = status['real_time_metrics']['current_metrics']
                    if rt_metrics:
                        print(f"\nReal-time Metrics:")
                        print(f"  Equity Trend: {rt_metrics.get('equity_trend', 'N/A')}")
                        print(f"  Recent Trades: {rt_metrics.get('recent_trades_count', 0)}")
                        print(f"  Win Rate: {rt_metrics.get('win_rate', 0):.2%}")
                
                # Trade statistics
                trade_stats = status.get('trade_statistics_24h', {})
                if trade_stats:
                    print(f"\n24h Trade Statistics:")
                    print(f"  Total Decisions: {trade_stats.get('total_decisions', 0)}")
                    print(f"  Executed Trades: {trade_stats.get('executed_trades', 0)}")
                    print(f"  Execution Rate: {trade_stats.get('execution_rate', 0):.2%}")
                    print(f"  Avg Confidence: {trade_stats.get('avg_confidence', 0):.3f}")
                
                # Recent trades
                recent_trades = self.trader.get_recent_trades(5)
                if recent_trades:
                    print(f"\nRecent Trades ({len(recent_trades)}):")
                    for trade in recent_trades[-3:]:  # Show last 3
                        timestamp = trade['timestamp'][:19]  # Remove microseconds
                        pair = trade['pair']
                        decision = trade['decision_type']
                        confidence = trade['confidence']
                        print(f"  {timestamp} | {pair} | {decision} | Conf: {confidence:.3f}")
                
                print("="*50)
                
                # Wait before next update
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                print(f"Error monitoring trading: {e}")
                await asyncio.sleep(10)
    
    async def run_live_trading(self):
        """Run live trading with monitoring"""
        try:
            # Setup configuration
            await self.setup_config()
            
            # Initialize trader
            if not await self.initialize_trader():
                return
            
            print("\nStarting live trading...")
            print("Press Ctrl+C to stop gracefully")
            
            self.is_running = True
            
            # Start trading and monitoring concurrently
            trading_task = asyncio.create_task(self.trader.start_trading())
            monitoring_task = asyncio.create_task(self.monitor_trading())
            
            # Wait for either task to complete or signal to stop
            done, pending = await asyncio.wait(
                [trading_task, monitoring_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt")
        except Exception as e:
            print(f"Error in live trading: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.trader:
            print("\nStopping trading and cleaning up...")
            await self.trader.stop_trading()
            print("Cleanup completed")
    
    async def demo_emergency_stop(self):
        """Demonstrate emergency stop functionality"""
        if not self.trader:
            print("Trader not initialized")
            return
        
        print("Initiating emergency stop...")
        await self.trader.emergency_stop()
        print("Emergency stop completed")
    
    async def demo_health_check(self):
        """Demonstrate health check functionality"""
        if not self.trader:
            print("Trader not initialized")
            return
        
        print("Performing system health check...")
        health_status = await self.trader.health_check()
        
        print(f"Overall Status: {health_status['overall_status']}")
        print("Component Status:")
        for component, status in health_status['components'].items():
            print(f"  {component}: {status}")


async def main():
    """Main function to run the example"""
    example = LiveTradingExample()
    
    # You can choose different demo modes:
    
    # 1. Full live trading demo (default)
    await example.run_live_trading()
    
    # 2. Health check demo only
    # await example.setup_config()
    # if await example.initialize_trader():
    #     await example.demo_health_check()
    
    # 3. Emergency stop demo
    # await example.setup_config()
    # if await example.initialize_trader():
    #     await example.demo_emergency_stop()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("AlphaPulse-RL Live Trading Example")
    print("==================================")
    print()
    print("This example demonstrates the LiveTrader functionality.")
    print("Make sure to configure your API keys in config/config.yaml")
    print("before running with real trading.")
    print()
    
    # Run the example
    asyncio.run(main())