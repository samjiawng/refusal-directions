"""
Intervention techniques: directional ablation, activation addition, weight orthogonalization.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable
from copy import deepcopy

from .model import ModelWrapper


class InterventionManager:
    """Manage interventions on model activations and weights."""
    
    def __init__(self, model: ModelWrapper):
        """
        Initialize intervention manager.
        
        Args:
            model: ModelWrapper instance
        """
        self.model = model
        self.original_weights = {}
        self.hooks = []
        self.intervention_active = False
        
    def clear_hooks(self):
        """Remove all intervention hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.intervention_active = False
    
    def directional_ablation(
        self,
        direction: torch.Tensor,
        layers: Optional[List[int]] = None,
        positions: str = "all"
    ):
        """
        Apply directional ablation - zero out component along direction.
        
        This modifies activations during forward pass by projecting out the direction:
        x' = x - (x · r̂)r̂
        
        Args:
            direction: Direction vector to ablate (will be normalized)
            layers: Layer indices to apply intervention. If None, applies to all layers.
            positions: Which token positions to intervene on ('all', 'last', 'post_instruction')
        """
        self.clear_hooks()
        
        # Normalize direction
        direction_unit = direction / (direction.norm() + 1e-8)
        direction_unit = direction_unit.to(self.model.device)
        
        if layers is None:
            layers = list(range(self.model.n_layers))
        
        def ablation_hook(module, input, output):
            """Hook function to ablate direction from activations."""
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Clone to avoid in-place modification issues
            hidden_states = hidden_states.clone()
            
            # Project out the direction: x' = x - (x · r̂)r̂
            # hidden_states shape: (batch, seq_len, hidden_dim)
            projection = torch.matmul(hidden_states, direction_unit)  # (batch, seq_len)
            projection = projection.unsqueeze(-1)  # (batch, seq_len, 1)
            hidden_states = hidden_states - projection * direction_unit
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        
        # Register hooks on specified layers
        for layer_idx in layers:
            layer = self.model.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(ablation_hook)
            self.hooks.append(hook)
        
        self.intervention_active = True
        print(f"Directional ablation active on {len(layers)} layers")
    
    def activation_addition(
        self,
        direction: torch.Tensor,
        layer_idx: int,
        coefficient: float = 1.0,
        positions: str = "all"
    ):
        """
        Add direction to activations at a specific layer.
        
        This modifies activations during forward pass:
        x' = x + α·r
        
        Args:
            direction: Direction vector to add
            layer_idx: Layer index to apply intervention
            coefficient: Scaling coefficient α
            positions: Which token positions to intervene on
        """
        self.clear_hooks()
        
        direction = direction.to(self.model.device)
        
        def addition_hook(module, input, output):
            """Hook function to add direction to activations."""
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            hidden_states = hidden_states.clone()
            
            # Add scaled direction: x' = x + α·r
            # Broadcasting will handle adding to all positions
            hidden_states = hidden_states + coefficient * direction
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        
        # Register hook on specified layer
        layer = self.model.model.model.layers[layer_idx]
        hook = layer.register_forward_hook(addition_hook)
        self.hooks.append(hook)
        
        self.intervention_active = True
        print(f"Activation addition active on layer {layer_idx} with coefficient {coefficient}")
    
    def weight_orthogonalization(
        self,
        direction: torch.Tensor,
        save_original: bool = True
    ):
        """
        Orthogonalize model weights with respect to direction.
        
        This is a permanent modification that prevents the model from writing
        to the direction in the residual stream:
        W' = W - r̂r̂ᵀW
        
        Args:
            direction: Direction to orthogonalize against (will be normalized)
            save_original: Whether to save original weights for restoration
        """
        # Normalize direction
        direction_unit = direction / (direction.norm() + 1e-8)
        direction_unit = direction_unit.to(self.model.device)
        
        # Outer product for projection matrix
        projection_matrix = torch.outer(direction_unit, direction_unit)  # (d_model, d_model)
        
        print("Orthogonalizing weights...")
        
        modified_params = []
        
        # Orthogonalize all matrices that write to residual stream
        for name, param in self.model.model.named_parameters():
            # Check if this parameter writes to residual stream
            should_modify = any([
                "embed" in name.lower(),
                "lm_head" in name.lower(),
                name.endswith("o_proj.weight"),  # Attention output projection
                name.endswith("down_proj.weight"),  # MLP output projection
                name.endswith("o_proj.bias"),
                name.endswith("down_proj.bias"),
            ])
            
            if should_modify and param.requires_grad:
                if save_original and name not in self.original_weights:
                    self.original_weights[name] = param.data.clone()
                
                with torch.no_grad():
                    if param.ndim == 2:  # Weight matrix
                        # For matrices writing to residual: W' = W - r̂r̂ᵀW
                        # Shape handling depends on whether this is (d_out, d_in) or (d_in, d_out)
                        if param.shape[0] == self.model.d_model:
                            # Shape: (d_model, d_in)
                            param.data = param.data - projection_matrix @ param.data
                        elif param.shape[1] == self.model.d_model:
                            # Shape: (d_out, d_model)
                            param.data = param.data - param.data @ projection_matrix.T
                    elif param.ndim == 1:  # Bias vector
                        if param.shape[0] == self.model.d_model:
                            # Shape: (d_model,)
                            param.data = param.data - (param.data @ direction_unit) * direction_unit
                
                modified_params.append(name)
        
        print(f"Orthogonalized {len(modified_params)} parameter tensors")
        print("Weight orthogonalization complete - this is a permanent modification")
    
    def restore_original_weights(self):
        """Restore original weights if they were saved."""
        if not self.original_weights:
            print("No original weights to restore")
            return
        
        print("Restoring original weights...")
        for name, original_weight in self.original_weights.items():
            param = dict(self.model.model.named_parameters())[name]
            param.data.copy_(original_weight)
        
        self.original_weights = {}
        print("Original weights restored")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up hooks."""
        self.clear_hooks()


def create_ablation_model(
    model: ModelWrapper,
    direction: torch.Tensor,
    layers: Optional[List[int]] = None
) -> InterventionManager:
    """
    Convenience function to create a model with directional ablation active.
    
    Args:
        model: ModelWrapper instance
        direction: Direction to ablate
        layers: Layers to apply ablation
        
    Returns:
        InterventionManager with ablation active
    """
    manager = InterventionManager(model)
    manager.directional_ablation(direction, layers)
    return manager


def create_addition_model(
    model: ModelWrapper,
    direction: torch.Tensor,
    layer_idx: int,
    coefficient: float = 1.0
) -> InterventionManager:
    """
    Convenience function to create a model with activation addition active.
    
    Args:
        model: ModelWrapper instance
        direction: Direction to add
        layer_idx: Layer to apply addition
        coefficient: Scaling coefficient
        
    Returns:
        InterventionManager with addition active
    """
    manager = InterventionManager(model)
    manager.activation_addition(direction, layer_idx, coefficient)
    return manager


def create_orthogonalized_model(
    model: ModelWrapper,
    direction: torch.Tensor,
    save_original: bool = True
) -> InterventionManager:
    """
    Convenience function to orthogonalize model weights.
    
    WARNING: This permanently modifies the model weights!
    
    Args:
        model: ModelWrapper instance
        direction: Direction to orthogonalize against
        save_original: Whether to save original weights
        
    Returns:
        InterventionManager instance
    """
    manager = InterventionManager(model)
    manager.weight_orthogonalization(direction, save_original)
    return manager


if __name__ == "__main__":
    # Test interventions
    print("Testing interventions...")
    
    from .model import ModelWrapper
    
    # Load model
    model = ModelWrapper(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        load_in_8bit=True
    )
    
    # Create a random direction for testing
    test_direction = torch.randn(model.d_model)
    
    print("\nTesting directional ablation...")
    manager = InterventionManager(model)
    manager.directional_ablation(test_direction, layers=[15, 16, 17])
    
    # Test generation with ablation
    test_prompt = ["The capital of France is"]
    completion = model.generate(test_prompt, max_new_tokens=10)
    print(f"With ablation: {completion[0]}")
    
    manager.clear_hooks()
    
    print("\nTesting activation addition...")
    manager.activation_addition(test_direction, layer_idx=16, coefficient=0.5)
    completion = model.generate(test_prompt, max_new_tokens=10)
    print(f"With addition: {completion[0]}")
    
    manager.clear_hooks()
    
    print("\nTests complete!")
