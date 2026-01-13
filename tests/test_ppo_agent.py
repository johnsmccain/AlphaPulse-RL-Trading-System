"""
Unit tests for PPO Agent prediction and confidence calculation.

This module tests the core functionality of the PPO agent including:
- Prediction accuracy and consistency
- Confidence calculation
- Model output validation
- Action bounds validation

Requirements tested: 1.4, 1.5
"""

import unittest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock, Mock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if torch is available and import appropriate PPO agent
try:
    import torch
    from models.ppo_agent import PPOAgent
    from models.model_utils import ActionSpaceUtils
    from models.performance_optimizer import PerformanceOptimizer
    TORCH_AVAILABLE = True
except ImportError:
    from models.ppo_agent_mock import PPOAgent
    TORCH_AVAILABLE = False
    # Create mock torch for testing
    torch = Mock()
    torch.FloatTensor = Mock()
    
    # Create mock utilities
    class MockTensor:
        def __init__(self, data):
            self.data = np.array(data)
        
        def __getitem__(self, key):
            return self.data[key]
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        @property
        def shape(self):
            return self.data.shape
        
        @property
        def ndim(self):
            return self.data.ndim
    
    class MockTorchModule:
        @staticmethod
        def tensor(data):
            return MockTensor(data)
        
        @staticmethod
        def FloatTensor(data):
            return MockTensor(data)
        
        @staticmethod
        def randn(*shape):
            return MockTensor(np.random.randn(*shape))
        
        class no_grad:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
    
    torch = MockTorchModule()
    
    class ActionSpaceUtils:
        @staticmethod
        def validate_action(action):
            if len(action) != 3:
                return False
            direction, size, leverage = action
            return (-1 <= direction <= 1 and 
                   0 <= size <= 0.1 and 
                   1 <= leverage <= 12)
        
        @staticmethod
        def clip_action(action):
            direction = np.clip(action[0], -1, 1)
            size = np.clip(action[1], 0, 0.1)
            leverage = np.clip(action[2], 1, 12)
            return np.array([direction, size, leverage])
        
        @staticmethod
        def interpret_action(action):
            direction, size, leverage = action
            return {
                'side': 'long' if direction > 0 else 'short',
                'direction': direction,
                'position_size': size,
                'leverage': leverage,
                'should_trade': abs(direction) > 0.1 and size > 0.001
            }
    
    class PerformanceOptimizer:
        def __init__(self):
            pass
        
        def optimize_prediction(self, agent, state):
            return agent.predict(state)
        
        def cache_prediction(self, state, action):
            pass
        
        def optimize_model_inference(self, model, name):
            return model
    
    # Create a mock OptimizedPPOAgent class
    class OptimizedPPOAgent(PPOAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.optimizer = PerformanceOptimizer()
    torch.tensor = Mock()
    torch.randn = Mock()
    torch.zeros = Mock()
    torch.device = Mock()

# Import model utilities (these should work without torch)
from models.model_utils import ActionSpaceUtils


class TestPPOAgentPrediction(unittest.TestCase):
    """Test PPO agent prediction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = PPOAgent(device='cpu')
        self.test_state = np.random.randn(9)
        self.batch_states = np.random.randn(5, 9)
        
    def test_predict_deterministic_consistency(self):
        """Test that deterministic predictions are consistent."""
        # Make multiple predictions with same state
        prediction1 = self.agent.predict(self.test_state, deterministic=True)
        prediction2 = self.agent.predict(self.test_state, deterministic=True)
        
        # Should be identical for deterministic predictions
        np.testing.assert_array_almost_equal(prediction1, prediction2, decimal=6)
        
    def test_predict_output_shape(self):
        """Test prediction output has correct shape."""
        prediction = self.agent.predict(self.test_state, deterministic=True)
        
        # Should return 3-dimensional action
        self.assertEqual(prediction.shape, (3,))
        
    def test_predict_action_bounds(self):
        """Test that predictions are within valid action bounds."""
        # Test multiple random states
        for _ in range(10):
            state = np.random.randn(9)
            prediction = self.agent.predict(state, deterministic=True)
            
            # Validate action bounds
            self.assertTrue(ActionSpaceUtils.validate_action(prediction),
                          f"Invalid action: {prediction}")
            
            # Check individual bounds
            direction, size, leverage = prediction
            self.assertGreaterEqual(direction, -1.0)
            self.assertLessEqual(direction, 1.0)
            self.assertGreaterEqual(size, 0.0)
            self.assertLessEqual(size, 0.1)
            self.assertGreaterEqual(leverage, 1.0)
            self.assertLessEqual(leverage, 12.0)
    
    def test_predict_stochastic_variation(self):
        """Test that stochastic predictions show variation."""
        predictions = []
        for _ in range(10):
            pred = self.agent.predict(self.test_state, deterministic=False)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Should have some variation in stochastic mode
        std_devs = np.std(predictions, axis=0)
        self.assertTrue(np.any(std_devs > 1e-6), 
                       "Stochastic predictions should show variation")
    
    def test_predict_batch_consistency(self):
        """Test batch prediction consistency with single predictions."""
        # Get single predictions
        single_predictions = []
        for state in self.batch_states:
            pred = self.agent.predict(state, deterministic=True)
            single_predictions.append(pred)
        
        # Get batch predictions using get_action_and_value
        state_tensor = torch.FloatTensor(self.batch_states)
        batch_actions, _, _ = self.agent.get_action_and_value(state_tensor)
        
        # Handle both torch tensors and numpy arrays
        if hasattr(batch_actions, 'detach'):
            batch_predictions = batch_actions.detach().numpy()
        else:
            batch_predictions = np.array(batch_actions)
        
        # Should be similar (allowing for small numerical differences)
        for i, (single, batch) in enumerate(zip(single_predictions, batch_predictions)):
            np.testing.assert_array_almost_equal(single, batch, decimal=3,
                                               err_msg=f"Mismatch at index {i}")


class TestPPOAgentConfidence(unittest.TestCase):
    """Test PPO agent confidence calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = PPOAgent(device='cpu')
        self.test_state = np.random.randn(9)
        
    def test_confidence_output_range(self):
        """Test confidence output is in valid range [0, 1]."""
        # Test multiple random states
        for _ in range(20):
            state = np.random.randn(9)
            confidence = self.agent.get_confidence(state)
            
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
    
    def test_confidence_consistency(self):
        """Test confidence calculation consistency."""
        # Multiple calls with same state should return same confidence
        confidence1 = self.agent.get_confidence(self.test_state)
        confidence2 = self.agent.get_confidence(self.test_state)
        
        self.assertAlmostEqual(confidence1, confidence2, places=6)
    
    def test_confidence_variation_across_states(self):
        """Test that confidence varies across different states."""
        confidences = []
        states = []
        
        # Generate diverse states
        for _ in range(10):
            state = np.random.randn(9) * np.random.uniform(0.1, 5.0)
            states.append(state)
            confidence = self.agent.get_confidence(state)
            confidences.append(confidence)
        
        # Should have some variation across different states
        confidence_std = np.std(confidences)
        self.assertGreater(confidence_std, 1e-6, 
                          "Confidence should vary across different states")
    
    def test_confidence_extreme_states(self):
        """Test confidence calculation with extreme state values."""
        # Test with very large values
        extreme_state_large = np.ones(9) * 100.0
        confidence_large = self.agent.get_confidence(extreme_state_large)
        self.assertGreaterEqual(confidence_large, 0.0)
        self.assertLessEqual(confidence_large, 1.0)
        
        # Test with very small values
        extreme_state_small = np.ones(9) * 0.001
        confidence_small = self.agent.get_confidence(extreme_state_small)
        self.assertGreaterEqual(confidence_small, 0.0)
        self.assertLessEqual(confidence_small, 1.0)
        
        # Test with zero state
        zero_state = np.zeros(9)
        confidence_zero = self.agent.get_confidence(zero_state)
        self.assertGreaterEqual(confidence_zero, 0.0)
        self.assertLessEqual(confidence_zero, 1.0)


class TestModelOutputValidation(unittest.TestCase):
    """Test model output validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = PPOAgent(device='cpu')
        
    def test_action_scaling_bounds(self):
        """Test that action scaling produces valid bounds."""
        # Test with extreme raw actions
        if TORCH_AVAILABLE:
            raw_actions = torch.tensor([[-1.0, -1.0, -1.0],  # Minimum
                                       [1.0, 1.0, 1.0],      # Maximum
                                       [0.0, 0.0, 0.0]])     # Middle
        else:
            raw_actions = np.array([[-1.0, -1.0, -1.0],  # Minimum
                                   [1.0, 1.0, 1.0],      # Maximum
                                   [0.0, 0.0, 0.0]])     # Middle
        
        scaled_actions = self.agent._scale_actions(raw_actions)
        
        for i, action in enumerate(scaled_actions):
            if hasattr(action, 'numpy'):
                action_np = action.numpy()
            else:
                action_np = np.array(action)
            self.assertTrue(ActionSpaceUtils.validate_action(action_np),
                          f"Invalid scaled action at index {i}: {action_np}")
    
    def test_network_output_bounds(self):
        """Test that network outputs are properly bounded."""
        # Test actor network output bounds
        test_states = torch.randn(10, 9)
        
        with torch.no_grad():
            raw_actions = self.agent.actor(test_states)
            
            # Actor should output values in [-1, 1] due to tanh activation
            self.assertTrue(torch.all(raw_actions >= -1.0))
            self.assertTrue(torch.all(raw_actions <= 1.0))
    
    def test_critic_output_validity(self):
        """Test that critic outputs are valid."""
        test_states = torch.randn(10, 9)
        
        with torch.no_grad():
            values = self.agent.critic(test_states)
            
            # Values should be finite
            self.assertTrue(torch.all(torch.isfinite(values)))
            # Values should have correct shape
            self.assertEqual(values.shape, (10, 1))
    
    def test_action_interpretation_validity(self):
        """Test that action interpretation produces valid results."""
        # Test multiple predictions
        for _ in range(20):
            state = np.random.randn(9)
            action = self.agent.predict(state, deterministic=True)
            
            # Interpret action
            interpretation = ActionSpaceUtils.interpret_action(action)
            
            # Check required keys
            required_keys = ['side', 'direction', 'position_size', 'leverage', 'should_trade']
            for key in required_keys:
                self.assertIn(key, interpretation)
            
            # Check value validity
            self.assertIn(interpretation['side'], ['hold', 'long', 'short'])
            self.assertIsInstance(interpretation['should_trade'], bool)
            self.assertGreaterEqual(interpretation['position_size'], 0.0)
            self.assertLessEqual(interpretation['position_size'], 0.1)
            self.assertGreaterEqual(interpretation['leverage'], 1.0)
            self.assertLessEqual(interpretation['leverage'], 12.0)


class TestOptimizedPPOAgent(unittest.TestCase):
    """Test optimized PPO agent functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock performance optimizer
        self.mock_optimizer = MagicMock(spec=PerformanceOptimizer)
        self.mock_optimizer.optimize_model_inference.side_effect = lambda x, name: x
        self.mock_optimizer.monitor = MagicMock()
        
        config = {
            'state_dim': 9,
            'action_dim': 3,
            'device': 'cpu',
            'prediction_cache_size': 10,
            'confidence_cache_size': 10
        }
        
        self.optimized_agent = OptimizedPPOAgent(config, self.mock_optimizer)
        self.test_state = np.random.randn(9)
    
    def test_optimized_prediction_consistency(self):
        """Test optimized prediction consistency with base agent."""
        # Create base agent for comparison
        base_agent = PPOAgent(device='cpu')
        
        # Copy weights to ensure same predictions
        self.optimized_agent.actor.load_state_dict(base_agent.actor.state_dict())
        self.optimized_agent.critic.load_state_dict(base_agent.critic.state_dict())
        
        # Compare predictions
        base_pred = base_agent.predict(self.test_state, deterministic=True)
        opt_pred = self.optimized_agent.predict_optimized(self.test_state, 
                                                         deterministic=True, 
                                                         use_cache=False)
        
        np.testing.assert_array_almost_equal(base_pred, opt_pred, decimal=5)
    
    def test_prediction_caching(self):
        """Test prediction caching functionality."""
        # First prediction (cache miss)
        pred1 = self.optimized_agent.predict_optimized(self.test_state, 
                                                      deterministic=True, 
                                                      use_cache=True)
        
        # Second prediction (cache hit)
        pred2 = self.optimized_agent.predict_optimized(self.test_state, 
                                                      deterministic=True, 
                                                      use_cache=True)
        
        # Should be identical due to caching
        np.testing.assert_array_equal(pred1, pred2)
        
        # Check cache statistics
        stats = self.optimized_agent.get_performance_stats()
        self.assertGreater(stats['cache_hits'], 0)
    
    def test_confidence_caching(self):
        """Test confidence caching functionality."""
        # First confidence calculation (cache miss)
        conf1 = self.optimized_agent.get_confidence_optimized(self.test_state, 
                                                             use_cache=True)
        
        # Second confidence calculation (cache hit)
        conf2 = self.optimized_agent.get_confidence_optimized(self.test_state, 
                                                             use_cache=True)
        
        # Should be identical due to caching
        self.assertEqual(conf1, conf2)
    
    def test_batch_prediction_validity(self):
        """Test batch prediction produces valid outputs."""
        states = [np.random.randn(9) for _ in range(5)]
        
        batch_predictions = self.optimized_agent.batch_predict(states, deterministic=True)
        
        # Check output validity
        self.assertEqual(len(batch_predictions), 5)
        
        for pred in batch_predictions:
            self.assertTrue(ActionSpaceUtils.validate_action(pred))


class TestModelValidationUtils(unittest.TestCase):
    """Test model validation utility functions."""
    
    def test_action_validation_edge_cases(self):
        """Test action validation with edge cases."""
        # Valid edge cases
        valid_actions = [
            np.array([-1.0, 0.0, 1.0]),    # Minimum direction, size, leverage
            np.array([1.0, 0.1, 12.0]),    # Maximum direction, size, leverage
            np.array([0.0, 0.05, 6.5])     # Middle values
        ]
        
        for action in valid_actions:
            self.assertTrue(ActionSpaceUtils.validate_action(action))
        
        # Invalid edge cases
        invalid_actions = [
            np.array([-1.1, 0.05, 6.0]),   # Direction too low
            np.array([1.1, 0.05, 6.0]),    # Direction too high
            np.array([0.0, -0.01, 6.0]),   # Size too low
            np.array([0.0, 0.11, 6.0]),    # Size too high
            np.array([0.0, 0.05, 0.9]),    # Leverage too low
            np.array([0.0, 0.05, 12.1]),   # Leverage too high
            np.array([0.0, 0.05])          # Wrong dimension
        ]
        
        for action in invalid_actions:
            self.assertFalse(ActionSpaceUtils.validate_action(action))
    
    def test_action_clipping(self):
        """Test action clipping functionality."""
        # Test clipping out-of-bounds actions
        invalid_action = np.array([2.0, 0.2, 15.0])
        clipped_action = ActionSpaceUtils.clip_action(invalid_action)
        
        # Should be valid after clipping
        self.assertTrue(ActionSpaceUtils.validate_action(clipped_action))
        
        # Check specific bounds
        self.assertEqual(clipped_action[0], 1.0)   # Direction clipped to max
        self.assertEqual(clipped_action[1], 0.1)   # Size clipped to max
        self.assertEqual(clipped_action[2], 12.0)  # Leverage clipped to max


if __name__ == '__main__':
    # Set up test environment
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    unittest.main(verbosity=2)