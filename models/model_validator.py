"""
Model validation utilities for PPO agent output validation.

This module provides comprehensive validation for PPO agent outputs including:
- Action bounds validation
- Confidence score validation
- Model output consistency checks
- Performance validation

Requirements: 1.4, 1.5
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


@dataclass
class ValidationResult:
    """Result of model validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]
    
    def add_error(self, message: str) -> None:
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)
    
    def add_metric(self, name: str, value: float) -> None:
        """Add validation metric."""
        self.metrics[name] = value


class PPOModelValidator:
    """Comprehensive validator for PPO model outputs."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Initialize model validator.
        
        Args:
            validation_level: Strictness level for validation
        """
        self.validation_level = validation_level
        
        # Action bounds
        self.action_bounds = {
            'position_direction': (-1.0, 1.0),
            'position_size': (0.0, 0.1),
            'leverage_multiplier': (1.0, 12.0)
        }
        
        # Confidence bounds
        self.confidence_bounds = (0.0, 1.0)
        
        # Tolerance levels based on validation strictness
        self.tolerances = {
            ValidationLevel.BASIC: {
                'action_tolerance': 1e-3,
                'confidence_tolerance': 1e-6,
                'consistency_tolerance': 1e-2
            },
            ValidationLevel.STANDARD: {
                'action_tolerance': 1e-4,
                'confidence_tolerance': 1e-7,
                'consistency_tolerance': 1e-3
            },
            ValidationLevel.STRICT: {
                'action_tolerance': 1e-6,
                'confidence_tolerance': 1e-8,
                'consistency_tolerance': 1e-4
            }
        }
    
    def validate_action(self, action: np.ndarray) -> ValidationResult:
        """
        Validate a single action output.
        
        Args:
            action: Action array [direction, size, leverage]
            
        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], metrics={})
        tolerance = self.tolerances[self.validation_level]['action_tolerance']
        
        # Check action dimension
        if len(action) != 3:
            result.add_error(f"Invalid action dimension: expected 3, got {len(action)}")
            return result
        
        direction, size, leverage = action
        
        # Validate position direction [-1, 1]
        dir_min, dir_max = self.action_bounds['position_direction']
        if direction < dir_min - tolerance or direction > dir_max + tolerance:
            result.add_error(f"Position direction {direction:.6f} out of bounds [{dir_min}, {dir_max}]")
        elif direction < dir_min or direction > dir_max:
            result.add_warning(f"Position direction {direction:.6f} slightly out of bounds")
        
        # Validate position size [0, 0.1]
        size_min, size_max = self.action_bounds['position_size']
        if size < size_min - tolerance or size > size_max + tolerance:
            result.add_error(f"Position size {size:.6f} out of bounds [{size_min}, {size_max}]")
        elif size < size_min or size > size_max:
            result.add_warning(f"Position size {size:.6f} slightly out of bounds")
        
        # Validate leverage [1, 12]
        lev_min, lev_max = self.action_bounds['leverage_multiplier']
        if leverage < lev_min - tolerance or leverage > lev_max + tolerance:
            result.add_error(f"Leverage {leverage:.6f} out of bounds [{lev_min}, {lev_max}]")
        elif leverage < lev_min or leverage > lev_max:
            result.add_warning(f"Leverage {leverage:.6f} slightly out of bounds")
        
        # Add metrics
        result.add_metric('direction_value', float(direction))
        result.add_metric('size_value', float(size))
        result.add_metric('leverage_value', float(leverage))
        result.add_metric('action_magnitude', float(np.linalg.norm(action)))
        
        return result
    
    def validate_confidence(self, confidence: float) -> ValidationResult:
        """
        Validate confidence score.
        
        Args:
            confidence: Confidence score
            
        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], metrics={})
        tolerance = self.tolerances[self.validation_level]['confidence_tolerance']
        
        conf_min, conf_max = self.confidence_bounds
        
        # Check if confidence is a valid number
        if not isinstance(confidence, (int, float)) or np.isnan(confidence) or np.isinf(confidence):
            result.add_error(f"Invalid confidence value: {confidence}")
            return result
        
        # Check bounds
        if confidence < conf_min - tolerance or confidence > conf_max + tolerance:
            result.add_error(f"Confidence {confidence:.8f} out of bounds [{conf_min}, {conf_max}]")
        elif confidence < conf_min or confidence > conf_max:
            result.add_warning(f"Confidence {confidence:.8f} slightly out of bounds")
        
        # Add metrics
        result.add_metric('confidence_value', float(confidence))
        
        return result
    
    def validate_batch_actions(self, actions: np.ndarray) -> ValidationResult:
        """
        Validate batch of actions.
        
        Args:
            actions: Batch of actions with shape (batch_size, 3)
            
        Returns:
            ValidationResult with batch validation status
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], metrics={})
        
        if len(actions.shape) != 2 or actions.shape[1] != 3:
            result.add_error(f"Invalid batch action shape: expected (N, 3), got {actions.shape}")
            return result
        
        batch_size = actions.shape[0]
        valid_actions = 0
        total_warnings = 0
        
        for i, action in enumerate(actions):
            action_result = self.validate_action(action)
            
            if action_result.is_valid:
                valid_actions += 1
            else:
                result.add_error(f"Action {i}: {'; '.join(action_result.errors)}")
            
            total_warnings += len(action_result.warnings)
            if action_result.warnings:
                result.add_warning(f"Action {i}: {'; '.join(action_result.warnings)}")
        
        # Add batch metrics
        result.add_metric('batch_size', float(batch_size))
        result.add_metric('valid_actions', float(valid_actions))
        result.add_metric('validation_rate', float(valid_actions / batch_size))
        result.add_metric('total_warnings', float(total_warnings))
        
        # Batch is valid if all actions are valid
        if valid_actions != batch_size:
            result.is_valid = False
        
        return result
    
    def validate_prediction_consistency(self, agent, state: np.ndarray, 
                                      num_predictions: int = 5) -> ValidationResult:
        """
        Validate prediction consistency for deterministic predictions.
        
        Args:
            agent: PPO agent to test
            state: Input state
            num_predictions: Number of predictions to compare
            
        Returns:
            ValidationResult with consistency validation
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], metrics={})
        tolerance = self.tolerances[self.validation_level]['consistency_tolerance']
        
        try:
            # Get multiple deterministic predictions
            predictions = []
            for _ in range(num_predictions):
                pred = agent.predict(state, deterministic=True)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Calculate consistency metrics
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            max_std = np.max(std_pred)
            
            # Check consistency
            if max_std > tolerance:
                result.add_error(f"Inconsistent predictions: max std = {max_std:.8f} > {tolerance}")
            
            # Add metrics
            result.add_metric('max_std_deviation', float(max_std))
            result.add_metric('mean_std_deviation', float(np.mean(std_pred)))
            result.add_metric('num_predictions', float(num_predictions))
            
        except Exception as e:
            result.add_error(f"Error during consistency validation: {str(e)}")
        
        return result
    
    def validate_confidence_consistency(self, agent, state: np.ndarray,
                                      num_calculations: int = 5) -> ValidationResult:
        """
        Validate confidence calculation consistency.
        
        Args:
            agent: PPO agent to test
            state: Input state
            num_calculations: Number of confidence calculations to compare
            
        Returns:
            ValidationResult with confidence consistency validation
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], metrics={})
        tolerance = self.tolerances[self.validation_level]['consistency_tolerance']
        
        try:
            # Get multiple confidence calculations
            confidences = []
            for _ in range(num_calculations):
                conf = agent.get_confidence(state)
                confidences.append(conf)
            
            confidences = np.array(confidences)
            
            # Calculate consistency metrics
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            
            # Check consistency
            if std_conf > tolerance:
                result.add_error(f"Inconsistent confidence: std = {std_conf:.8f} > {tolerance}")
            
            # Add metrics
            result.add_metric('confidence_std_deviation', float(std_conf))
            result.add_metric('confidence_mean', float(mean_conf))
            result.add_metric('num_calculations', float(num_calculations))
            
        except Exception as e:
            result.add_error(f"Error during confidence consistency validation: {str(e)}")
        
        return result
    
    def validate_model_outputs(self, agent, test_states: List[np.ndarray]) -> ValidationResult:
        """
        Comprehensive validation of model outputs.
        
        Args:
            agent: PPO agent to validate
            test_states: List of test states
            
        Returns:
            ValidationResult with comprehensive validation
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], metrics={})
        
        if not test_states:
            result.add_error("No test states provided")
            return result
        
        valid_predictions = 0
        valid_confidences = 0
        total_states = len(test_states)
        
        # Validate each state
        for i, state in enumerate(test_states):
            try:
                # Validate prediction
                prediction = agent.predict(state, deterministic=True)
                pred_result = self.validate_action(prediction)
                
                if pred_result.is_valid:
                    valid_predictions += 1
                else:
                    result.add_error(f"State {i} prediction: {'; '.join(pred_result.errors)}")
                
                if pred_result.warnings:
                    result.add_warning(f"State {i} prediction: {'; '.join(pred_result.warnings)}")
                
                # Validate confidence
                confidence = agent.get_confidence(state)
                conf_result = self.validate_confidence(confidence)
                
                if conf_result.is_valid:
                    valid_confidences += 1
                else:
                    result.add_error(f"State {i} confidence: {'; '.join(conf_result.errors)}")
                
                if conf_result.warnings:
                    result.add_warning(f"State {i} confidence: {'; '.join(conf_result.warnings)}")
                
            except Exception as e:
                result.add_error(f"Error validating state {i}: {str(e)}")
        
        # Add overall metrics
        result.add_metric('total_states', float(total_states))
        result.add_metric('valid_predictions', float(valid_predictions))
        result.add_metric('valid_confidences', float(valid_confidences))
        result.add_metric('prediction_validation_rate', float(valid_predictions / total_states))
        result.add_metric('confidence_validation_rate', float(valid_confidences / total_states))
        
        # Overall validation passes if all outputs are valid
        if valid_predictions != total_states or valid_confidences != total_states:
            result.is_valid = False
        
        return result
    
    def generate_validation_report(self, result: ValidationResult) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            result: ValidationResult to report on
            
        Returns:
            Formatted validation report
        """
        report = []
        report.append("=" * 60)
        report.append("PPO MODEL VALIDATION REPORT")
        report.append("=" * 60)
        
        # Overall status
        status = "✅ PASSED" if result.is_valid else "❌ FAILED"
        report.append(f"Overall Status: {status}")
        report.append("")
        
        # Errors
        if result.errors:
            report.append(f"Errors ({len(result.errors)}):")
            for i, error in enumerate(result.errors, 1):
                report.append(f"  {i}. {error}")
            report.append("")
        
        # Warnings
        if result.warnings:
            report.append(f"Warnings ({len(result.warnings)}):")
            for i, warning in enumerate(result.warnings, 1):
                report.append(f"  {i}. {warning}")
            report.append("")
        
        # Metrics
        if result.metrics:
            report.append("Validation Metrics:")
            for name, value in result.metrics.items():
                if isinstance(value, float):
                    report.append(f"  {name}: {value:.6f}")
                else:
                    report.append(f"  {name}: {value}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def create_test_states(num_states: int = 10, state_dim: int = 9) -> List[np.ndarray]:
    """
    Create diverse test states for validation.
    
    Args:
        num_states: Number of test states to create
        state_dim: Dimension of each state
        
    Returns:
        List of test states
    """
    test_states = []
    
    # Normal random states
    for _ in range(num_states // 3):
        state = np.random.randn(state_dim)
        test_states.append(state)
    
    # Extreme positive states
    for _ in range(num_states // 3):
        state = np.random.randn(state_dim) * 5.0 + 2.0
        test_states.append(state)
    
    # Extreme negative states
    for _ in range(num_states // 3):
        state = np.random.randn(state_dim) * 5.0 - 2.0
        test_states.append(state)
    
    # Edge cases
    remaining = num_states - len(test_states)
    for _ in range(remaining):
        if len(test_states) % 2 == 0:
            state = np.zeros(state_dim)  # Zero state
        else:
            state = np.ones(state_dim) * 0.001  # Near-zero state
        test_states.append(state)
    
    return test_states


if __name__ == "__main__":
    # Example usage
    from models.ppo_agent import PPOAgent
    
    # Create agent and validator
    agent = PPOAgent(device='cpu')
    validator = PPOModelValidator(ValidationLevel.STANDARD)
    
    # Create test states
    test_states = create_test_states(20)
    
    # Run comprehensive validation
    result = validator.validate_model_outputs(agent, test_states)
    
    # Print report
    report = validator.generate_validation_report(result)
    print(report)