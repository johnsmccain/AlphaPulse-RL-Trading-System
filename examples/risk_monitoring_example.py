#!/usr/bin/env python3
"""
Example demonstrating the enhanced risk monitoring and alerting system.

This example shows how to use the RiskManager with integrated RiskMonitor
for comprehensive risk management in the AlphaPulse-RL trading system.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.risk_manager import RiskManager
from trading.portfolio import PortfolioState, Position


def simulate_trading_session():
    """Simulate a trading session with risk monitoring."""
    print("=== AlphaPulse-RL Risk Monitoring Example ===\n")
    
    # Initialize risk manager with monitoring enabled
    print("1. Initializing Risk Manager with monitoring...")
    risk_manager = RiskManager(enable_monitoring=True)
    print("   ✓ Risk monitoring system active")
    
    # Create initial portfolio
    print("\n2. Setting up initial portfolio...")
    portfolio = PortfolioState(
        balance=10000.0,
        daily_start_balance=10000.0,
        peak_balance=10000.0
    )
    print(f"   ✓ Initial balance: ${portfolio.balance:,.2f}")
    
    # Simulate market data updates
    print("\n3. Simulating market data updates...")
    pairs = ["BTCUSDT", "ETHUSDT"]
    prices = {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}
    
    # Add some price history for volatility calculations
    for i in range(50):
        for pair in pairs:
            # Simulate price movements with varying volatility
            if i < 30:
                # Normal volatility period
                price_change = np.random.normal(0, 0.01) * prices[pair]
            else:
                # High volatility period
                price_change = np.random.normal(0, 0.03) * prices[pair]
            
            prices[pair] += price_change
            risk_manager.update_market_data(pair, prices[pair])
    
    print(f"   ✓ Updated prices: BTC=${prices['BTCUSDT']:,.2f}, ETH=${prices['ETHUSDT']:,.2f}")
    
    # Check volatility regimes
    print("\n4. Checking volatility regimes...")
    for pair in pairs:
        volatility_info = risk_manager.check_volatility_regime(pair, prices[pair])
        print(f"   {pair}: {volatility_info['regime']} regime, "
              f"volatility={volatility_info['volatility']:.4f}, "
              f"safe_to_trade={volatility_info['safe_to_trade']}")
    
    # Simulate some trades
    print("\n5. Simulating trade validation...")
    
    # Test different trade scenarios
    trade_scenarios = [
        {
            'name': 'Conservative Long',
            'action': [0.3, 0.02, 2.0],  # Small long position, low leverage
            'pair': 'BTCUSDT'
        },
        {
            'name': 'Aggressive Long',
            'action': [0.8, 0.08, 8.0],  # Large position, high leverage
            'pair': 'BTCUSDT'
        },
        {
            'name': 'Maximum Leverage',
            'action': [0.5, 0.05, 12.0],  # Max allowed leverage
            'pair': 'ETHUSDT'
        },
        {
            'name': 'Excessive Leverage',
            'action': [0.5, 0.05, 15.0],  # Above max leverage (should fail)
            'pair': 'ETHUSDT'
        }
    ]
    
    for scenario in trade_scenarios:
        is_valid, reason = risk_manager.validate_trade(
            scenario['action'], 
            portfolio, 
            prices[scenario['pair']], 
            pair=scenario['pair']
        )
        status = "✓ APPROVED" if is_valid else "✗ REJECTED"
        print(f"   {scenario['name']}: {status} - {reason}")
    
    # Add a position to portfolio
    print("\n6. Adding position to portfolio...")
    btc_price = prices['BTCUSDT']
    position_value = 0.05 * portfolio.balance  # 5% of portfolio
    position_size = position_value / btc_price
    
    position = Position(
        pair="BTCUSDT",
        side="long",
        size=position_size,
        leverage=4.0,
        entry_price=btc_price,
        current_price=btc_price,
        unrealized_pnl=0.0,
        timestamp=datetime.now()
    )
    portfolio.add_position(position)
    print(f"   ✓ Added BTC position: {position_size:.6f} BTC at ${btc_price:,.2f}")
    
    # Simulate price movement and portfolio monitoring
    print("\n7. Simulating market movement and monitoring...")
    
    # Simulate adverse price movement
    new_btc_price = btc_price * 0.92  # 8% drop
    prices['BTCUSDT'] = new_btc_price
    portfolio.update_position_prices(prices)
    
    print(f"   Market moved: BTC dropped to ${new_btc_price:,.2f} (-8%)")
    print(f"   Portfolio equity: ${portfolio.get_total_equity():,.2f}")
    print(f"   Unrealized PnL: ${portfolio.positions['BTCUSDT'].unrealized_pnl:,.2f}")
    
    # Run comprehensive risk monitoring
    monitoring_result = risk_manager.monitor_portfolio_risk(portfolio, prices)
    
    print(f"\n8. Risk monitoring results:")
    risk_metrics = monitoring_result['risk_metrics']
    print(f"   Current drawdown: {risk_metrics.current_drawdown:.2f}%")
    print(f"   Daily PnL: {risk_metrics.daily_pnl_percent:.2f}%")
    print(f"   Risk score: {risk_metrics.risk_score:.1f}/100")
    print(f"   Position exposure: {risk_metrics.position_exposure_percent:.1f}%")
    
    # Display any alerts
    alerts = monitoring_result['alerts']
    if alerts:
        print(f"\n9. Risk alerts generated ({len(alerts)}):")
        for alert in alerts:
            print(f"   {alert.severity.upper()}: {alert.message}")
    else:
        print(f"\n9. No risk alerts generated")
    
    # Show risk assessment
    risk_report = monitoring_result['risk_report']
    risk_assessment = risk_report['risk_assessment']
    print(f"\n10. Risk assessment:")
    print(f"    Risk level: {risk_assessment['risk_level'].upper()}")
    if risk_assessment['recommendations']:
        print(f"    Recommendations:")
        for rec in risk_assessment['recommendations']:
            print(f"      - {rec}")
    
    print(f"\n=== Risk Monitoring Example Complete ===")


if __name__ == "__main__":
    try:
        simulate_trading_session()
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)