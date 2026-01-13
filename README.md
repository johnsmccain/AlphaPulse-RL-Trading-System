# AlphaPulse-RL Trading System

A risk-aware adaptive trading agent that uses Proximal Policy Optimization (PPO) reinforcement learning to trade Bitcoin and Ethereum on the WEEX exchange. The system dynamically switches between trend-following and mean-reversion strategies while maintaining strict risk controls to achieve consistent performance and high Sharpe ratios.

## Features

- **Adaptive Strategy**: Dynamically switches between trend-following and mean-reversion based on market regime detection
- **Risk Management**: Strict position size, leverage, and drawdown limits with emergency flattening
- **PPO Reinforcement Learning**: Uses state-of-the-art RL for trading decisions
- **Comprehensive Logging**: Full audit trail of all trading decisions and system actions
- **Modular Architecture**: Separates concerns for reliability, auditability, and maintainability

## System Architecture

```
/alpha-pulse-rl
├── data/                    # Data fetching and feature engineering
├── env/                     # OpenAI Gym trading environment
├── models/                  # PPO agent and training pipeline
├── trading/                 # Live trading and execution logic
├── risk/                    # Risk management system
├── logs/                    # Trade history and AI decision logs
├── config/                  # Configuration files
└── tests/                   # Unit and integration tests
```

## Quick Start

### Prerequisites

- Python 3.8+
- WEEX API credentials
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd alpha-pulse-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system:
   - Edit `config/config.yaml` for system settings
   - Edit `config/trading_params.yaml` for trading parameters
   - Set WEEX API credentials in environment variables

### Configuration

#### System Configuration (`config/config.yaml`)

Key settings:
- **Exchange**: WEEX API configuration and rate limits
- **Trading Pairs**: BTC/ETH pairs to trade
- **Logging**: Log levels and file settings
- **Performance**: Memory and caching settings

#### Trading Parameters (`config/trading_params.yaml`)

Key parameters:
- **Risk Limits**: Maximum leverage (12x), position size (10%), daily loss (3%), total drawdown (12%)
- **Strategy**: Confidence threshold (0.8), trading intervals
- **PPO Settings**: Learning rate, network architecture, training parameters

### Usage

#### Training the Model

```python
from models.train import train_ppo_agent
from config.logging_config import setup_logging

# Initialize logging
setup_logging()

# Train the PPO agent
agent = train_ppo_agent(
    total_timesteps=1000000,
    save_path="models/ppo_agent.zip"
)
```

#### Backtesting

```python
from models.evaluate import run_backtest

# Run backtest on historical data
results = run_backtest(
    model_path="models/ppo_agent.zip",
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

#### Live Trading

```python
from trading.live_trader import LiveTrader
from config.logging_config import setup_logging

# Initialize logging
setup_logging()

# Start live trading
trader = LiveTrader(
    model_path="models/ppo_agent.zip",
    config_path="config/config.yaml"
)
trader.start_trading()
```

## Risk Management

The system implements multiple layers of risk control:

### Hard Limits
- **Maximum Leverage**: 12x on all positions
- **Position Size**: Maximum 10% of total equity per trade
- **Daily Loss**: Maximum 3% daily loss limit
- **Total Drawdown**: Emergency position flattening at 12% drawdown

### Dynamic Controls
- **Volatility Filtering**: Blocks trades during extreme market volatility
- **Confidence Threshold**: Only executes trades with >80% model confidence
- **Market Regime Awareness**: Adapts strategy based on trending vs ranging markets

### Emergency Procedures
- **Automatic Flattening**: All positions closed if drawdown exceeds 12%
- **Trading Halt**: System stops trading on repeated failures or anomalies
- **Manual Override**: Operators can force position closure or system shutdown

## Logging and Monitoring

### Trade History (`logs/trades.csv`)
Complete record of all trades with:
- Timestamp, trading pair, action details
- Entry/exit prices and P&L
- Model confidence and market regime
- Portfolio balance tracking

### AI Decisions (`logs/ai_decisions.json`)
Detailed decision logs including:
- Market state vector and model predictions
- Risk validation results
- Execution outcomes and errors
- System performance metrics

### System Logs
Standard application logs with:
- INFO: Normal operations and trade executions
- WARNING: Risk limit approaches and data issues
- ERROR: System failures and emergency actions

## Performance Metrics

The system optimizes for risk-adjusted returns using:

### Reward Function
```
reward = pnl - 0.1 * volatility_penalty - 0.2 * drawdown_penalty - 0.01 * overtrading_penalty - transaction_costs
```

### Key Metrics
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses

## Development

### Project Structure

- **data/**: Market data fetching and feature engineering
- **env/**: OpenAI Gym trading environment for training
- **models/**: PPO agent implementation and training pipeline
- **trading/**: Live trading orchestration and execution
- **risk/**: Risk management and limit enforcement
- **config/**: System and trading parameter configuration
- **tests/**: Unit tests and integration tests

### Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

## Security Considerations

- **API Keys**: Store in environment variables, never in code
- **Rate Limiting**: Respects exchange limits to prevent bans
- **Input Validation**: All inputs sanitized and validated
- **Access Control**: Restricted file and system access

## Troubleshooting

### Common Issues

1. **API Connection Failures**
   - Check API credentials and network connectivity
   - Verify rate limits are not exceeded
   - Review exchange status and maintenance schedules

2. **Model Performance Issues**
   - Retrain model with recent market data
   - Adjust confidence threshold or risk parameters
   - Review feature engineering for data quality

3. **Risk Limit Violations**
   - Check portfolio balance and position sizes
   - Review drawdown calculations and limits
   - Verify risk manager configuration

### Support

For technical support or questions:
- Review system logs in `logs/` directory
- Check configuration files for parameter issues
- Consult the design document for architectural details

## License

[License information]

## Disclaimer

This trading system is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.