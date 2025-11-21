"""
Refusal Directions: Mechanistic interpretability research on refusal in LLMs.

Based on the paper "Refusal in LLMs is mediated by a single direction" (NeurIPS 2024)
"""

from .data import RefusalDataset, Instruction, format_instruction
from .model import ModelWrapper
from .directions import RefusalDirectionExtractor, analyze_direction
from .interventions import (
    InterventionManager,
    create_ablation_model,
    create_addition_model,
    create_orthogonalized_model
)
from .evaluation import (
    RefusalEvaluator,
    SafetyEvaluator,
    CombinedEvaluator,
    compute_attack_success_rate
)

__version__ = "0.1.0"

__all__ = [
    # Data
    "RefusalDataset",
    "Instruction",
    "format_instruction",
    
    # Model
    "ModelWrapper",
    
    # Directions
    "RefusalDirectionExtractor",
    "analyze_direction",
    
    # Interventions
    "InterventionManager",
    "create_ablation_model",
    "create_addition_model",
    "create_orthogonalized_model",
    
    # Evaluation
    "RefusalEvaluator",
    "SafetyEvaluator",
    "CombinedEvaluator",
    "compute_attack_success_rate",
]
