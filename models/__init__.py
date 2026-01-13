"""
AlphaPulse-RL Models Package

This package contains the PPO agent implementation and training pipeline
for the AlphaPulse-RL trading system.
"""

# Import non-torch dependent modules first
from .model_utils import (
    ModelConfig,
    ModelCheckpoint,
    ActionSpaceUtils,
    StateNormalizer,
    setup_device,
    count_parameters,
    set_seed,
    get_model_summary
)

from .evaluate import PerformanceEvaluator, TradingMetrics

# Conditional imports for torch-dependent modules
try:
    from .ppo_agent import PPOAgent, ActorNetwork, CriticNetwork
    from .train import PPOTrainer, ExperienceBuffer, create_training_config
    TORCH_MODULES_AVAILABLE = True
except ImportError:
    TORCH_MODULES_AVAILABLE = False
    PPOAgent = None
    ActorNetwork = None
    CriticNetwork = None
    PPOTrainer = None
    ExperienceBuffer = None
    create_training_config = None

# Define __all__ based on what's available
__all__ = [
    'TradingMetrics',
    'PerformanceEvaluator',
    'ModelConfig',
    'ModelCheckpoint',
    'ActionSpaceUtils',
    'StateNormalizer',
    'setup_device',
    'count_parameters',
    'set_seed',
    'get_model_summary'
]

if TORCH_MODULES_AVAILABLE:
    __all__.extend([
        'PPOAgent',
        'ActorNetwork', 
        'CriticNetwork',
        'PPOTrainer',
        'ExperienceBuffer',
        'create_training_config'
    ])