#!/usr/bin/env python3
"""
Quick validation test for the WeexTradingEnv implementation
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.weex_trading_env import WeexTradingEnv
from env.env_utils import create_test_environment, run_environment_validation

def main():
    print("=== WeexTradingEnv Validation Test ===\n")
    
    try:
        # Create test environment
        print("1. Creating test environment...")
        env = create_test_environment()
        print(f"   ✓ Environment created successfully")
        print(f"   ✓ Action space: {env.action_space}")
        print(f"   ✓ Observation space: {env.observation_space}")
        
        # Test basic functionality
        print("\n2. Testing basic functionality...")
        
        # Reset environment
        obs = env.reset()
        print(f"   ✓ Reset successful, observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"   ✓ Step {i+1}: reward={reward:.4f}, done={done}, equity=${info['portfolio']['equity']:.2f}")
            
            if done:
                print("   ✓ Episode terminated naturally")
                break
        
        # Test environment validation
        print("\n3. Running comprehensive validation...")
        validation_results = run_environment_validation(env)
        
        if validation_results['overall_success']:
            print("   ✓ All basic tests passed")
        else:
            print("   ⚠ Some tests failed")
        
        print(f"   ✓ Stress test: {validation_results['stress_tests']['episodes_completed']} episodes completed")
        print(f"   ✓ Average episode length: {validation_results['stress_tests']['average_episode_length']:.1f}")
        
        # Test environment utilities
        print("\n4. Testing environment utilities...")
        
        # Test wrappers
        from env.env_utils import create_wrapped_environment
        wrapped_env = create_wrapped_environment(env, normalize_obs=True, scale_rewards=True)
        print("   ✓ Environment wrappers created successfully")
        
        # Test wrapped environment
        obs = wrapped_env.reset()
        action = wrapped_env.action_space.sample()
        obs, reward, done, info = wrapped_env.step(action)
        print(f"   ✓ Wrapped environment step successful: reward={reward:.4f}")
        
        print("\n=== All Tests Passed Successfully! ===")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)