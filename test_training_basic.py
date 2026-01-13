#!/usr/bin/env python3
"""
Basic test script to verify training pipeline structure without PyTorch.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test basic imports that don't require PyTorch
        from models.model_utils import ModelConfig, ActionSpaceUtils
        print("✅ Model utilities imported successfully")
        
        from data.feature_engineering import FeatureVector
        print("✅ Feature engineering imported successfully")
        
        # Test configuration creation
        config = ModelConfig()
        print("✅ Model configuration created successfully")
        
        # Test action space utilities
        test_action = np.array([0.5, 0.05, 8.0])
        is_valid = ActionSpaceUtils.validate_action(test_action)
        print(f"✅ Action validation works: {is_valid}")
        
        interpretation = ActionSpaceUtils.interpret_action(test_action)
        print(f"✅ Action interpretation: {interpretation}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_data_structures():
    """Test data structures and utilities."""
    print("\nTesting data structures...")
    
    try:
        # Create dummy data
        dummy_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='5min'),
            'open': np.random.randn(100).cumsum() + 50000,
            'high': np.random.randn(100).cumsum() + 50100,
            'low': np.random.randn(100).cumsum() + 49900,
            'close': np.random.randn(100).cumsum() + 50000,
            'volume': np.random.randint(1000, 10000, 100),
            'pair': ['BTCUSDT'] * 100,
            'returns_5m': np.random.normal(0, 0.01, 100),
            'returns_15m': np.random.normal(0, 0.015, 100),
            'rsi_14': np.random.uniform(0.2, 0.8, 100),
            'macd_histogram': np.random.normal(0, 0.001, 100),
            'atr_percentage': np.random.uniform(0.01, 0.05, 100),
            'volume_zscore': np.random.normal(0, 1, 100),
            'orderbook_imbalance': np.random.normal(0, 0.1, 100),
            'funding_rate': np.random.normal(0.0001, 0.0005, 100),
            'volatility_regime': np.random.choice([0, 1], 100, p=[0.6, 0.4])
        })
        print("✅ Dummy data created successfully")
        
        # Test feature vector creation
        from data.feature_engineering import FeatureVector
        feature = FeatureVector(
            returns_5m=0.01,
            returns_15m=0.015,
            rsi_14=0.6,
            macd_histogram=0.001,
            atr_percentage=0.02,
            volume_zscore=0.5,
            orderbook_imbalance=0.1,
            funding_rate=0.0001,
            volatility_regime=1
        )
        feature_array = feature.to_array()
        print(f"✅ Feature vector created: shape {feature_array.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data structures: {e}")
        return False

def test_configuration():
    """Test configuration management."""
    print("\nTesting configuration...")
    
    try:
        from models.model_utils import ModelConfig
        
        # Test default configuration
        config = ModelConfig()
        lr_actor = config.get('agent.lr_actor')
        print(f"✅ Default config loaded: lr_actor = {lr_actor}")
        
        # Test setting values
        config.set('agent.lr_actor', 1e-4)
        new_lr = config.get('agent.lr_actor')
        print(f"✅ Config value updated: lr_actor = {new_lr}")
        
        # Test training config creation (without importing train module)
        training_config = {
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
        print("✅ Training configuration created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in configuration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics():
    """Test trading metrics calculations."""
    print("\nTesting trading metrics...")
    
    try:
        from models.evaluate import TradingMetrics
        
        # Create dummy returns
        returns = np.random.normal(0.001, 0.02, 100)
        
        metrics = TradingMetrics()
        
        # Test metric calculations
        sharpe = metrics.sharpe_ratio(returns)
        max_dd, start_idx, end_idx = metrics.max_drawdown(returns)
        win_rate = metrics.win_rate(returns)
        profit_factor = metrics.profit_factor(returns)
        
        print(f"✅ Sharpe ratio: {sharpe:.3f}")
        print(f"✅ Max drawdown: {max_dd:.3f}")
        print(f"✅ Win rate: {win_rate:.3f}")
        print(f"✅ Profit factor: {profit_factor:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in metrics: {e}")
        return False

def main():
    """Run all tests."""
    print("🔧 Testing AlphaPulse-RL Training Pipeline (Basic)")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_structures,
        test_configuration,
        test_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! Training pipeline structure is correct.")
        print("\nNote: Full training requires PyTorch installation.")
        print("To install PyTorch: pip install torch")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)