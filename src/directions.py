"""
Refusal direction extraction using difference-in-means.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pickle
from pathlib import Path

from .model import ModelWrapper
from .data import Instruction, format_instruction


class RefusalDirectionExtractor:
    """Extract refusal directions using difference-in-means."""
    
    def __init__(
        self,
        model: ModelWrapper,
        chat_template: str = "llama3"
    ):
        """
        Initialize extractor.
        
        Args:
            model: ModelWrapper instance
            chat_template: Chat template to use for formatting instructions
        """
        self.model = model
        self.chat_template = chat_template
        self.n_layers = model.n_layers
        self.d_model = model.d_model
        
        # Storage for computed directions
        self.mean_harmful = {}  # layer -> mean activation
        self.mean_harmless = {}  # layer -> mean activation
        self.directions = {}  # layer -> difference vector
        
    def compute_mean_activations(
        self,
        instructions: List[Instruction],
        batch_size: int = 8
    ) -> Dict[int, torch.Tensor]:
        """
        Compute mean activations across instructions for each layer.
        
        Args:
            instructions: List of instructions
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping layer index to mean activation vector
        """
        # Format instructions
        formatted = [format_instruction(str(inst), self.chat_template) 
                     for inst in instructions]
        
        # Collect activations in batches
        all_layer_acts = {i: [] for i in range(self.n_layers)}
        
        for i in tqdm(range(0, len(formatted), batch_size), desc="Computing activations"):
            batch = formatted[i:i + batch_size]
            
            # Get post-instruction activations
            acts = self.model.get_post_instruction_activations(batch)
            
            # Store by layer
            for layer_name, layer_acts in acts.items():
                layer_idx = int(layer_name.split("_")[1])
                all_layer_acts[layer_idx].append(layer_acts)
        
        # Compute means
        mean_acts = {}
        for layer_idx, acts_list in all_layer_acts.items():
            # Concatenate all batches and compute mean
            all_acts = torch.cat(acts_list, dim=0)  # (n_samples, d_model)
            mean_acts[layer_idx] = all_acts.mean(dim=0)  # (d_model,)
        
        return mean_acts
    
    def extract_directions(
        self,
        harmful_instructions: List[Instruction],
        harmless_instructions: List[Instruction],
        batch_size: int = 8
    ) -> Dict[int, torch.Tensor]:
        """
        Extract refusal directions for all layers using difference-in-means.
        
        Args:
            harmful_instructions: List of harmful instructions
            harmless_instructions: List of harmless instructions
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping layer index to direction vector
        """
        print("Computing mean activations for harmful instructions...")
        self.mean_harmful = self.compute_mean_activations(
            harmful_instructions, batch_size
        )
        
        print("Computing mean activations for harmless instructions...")
        self.mean_harmless = self.compute_mean_activations(
            harmless_instructions, batch_size
        )
        
        print("Computing difference-in-means directions...")
        self.directions = {}
        for layer_idx in range(self.n_layers):
            self.directions[layer_idx] = (
                self.mean_harmful[layer_idx] - self.mean_harmless[layer_idx]
            )
        
        return self.directions
    
    def select_best_direction(
        self,
        val_harmful: List[Instruction],
        val_harmless: List[Instruction],
        criterion: str = "ablation_success",
        top_k: int = 5
    ) -> Tuple[int, torch.Tensor]:
        """
        Select the best direction from all layers based on validation performance.
        
        Args:
            val_harmful: Validation harmful instructions
            val_harmless: Validation harmless instructions
            criterion: Selection criterion ('ablation_success', 'addition_success', 'combined')
            top_k: Number of top directions to consider
            
        Returns:
            Tuple of (best_layer_idx, best_direction_vector)
        """
        if not self.directions:
            raise ValueError("Must call extract_directions first!")
        
        print(f"Selecting best direction using {criterion} criterion...")
        
        # Evaluate each direction
        scores = {}
        for layer_idx, direction in tqdm(self.directions.items(), desc="Evaluating layers"):
            score = self._evaluate_direction(
                layer_idx, direction, val_harmful, val_harmless, criterion
            )
            scores[layer_idx] = score
        
        # Select best
        sorted_layers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {top_k} layers:")
        for i, (layer_idx, score) in enumerate(sorted_layers[:top_k]):
            print(f"  {i+1}. Layer {layer_idx}: score={score:.4f}")
        
        best_layer = sorted_layers[0][0]
        best_direction = self.directions[best_layer]
        
        print(f"\nSelected layer {best_layer} with score {scores[best_layer]:.4f}")
        
        return best_layer, best_direction
    
    def _evaluate_direction(
        self,
        layer_idx: int,
        direction: torch.Tensor,
        val_harmful: List[Instruction],
        val_harmless: List[Instruction],
        criterion: str
    ) -> float:
        """
        Evaluate a direction based on how well it separates harmful from harmless.
        
        This is a simplified evaluation - the full paper uses actual generation
        and refusal detection. Here we use activation projection as a proxy.
        """
        # Format instructions
        harmful_formatted = [format_instruction(str(inst), self.chat_template) 
                           for inst in val_harmful[:16]]  # Use subset for speed
        harmless_formatted = [format_instruction(str(inst), self.chat_template) 
                            for inst in val_harmless[:16]]
        
        # Get activations
        harmful_acts = self.model.get_post_instruction_activations(
            harmful_formatted, layers=[layer_idx]
        )[f"layer_{layer_idx}"]
        
        harmless_acts = self.model.get_post_instruction_activations(
            harmless_formatted, layers=[layer_idx]
        )[f"layer_{layer_idx}"]
        
        # Normalize direction
        direction_norm = direction / (direction.norm() + 1e-8)
        
        # Project activations onto direction
        harmful_proj = (harmful_acts @ direction_norm.to(harmful_acts.device)).cpu().numpy()
        harmless_proj = (harmless_acts @ direction_norm.to(harmless_acts.device)).cpu().numpy()
        
        # Score based on separation
        harmful_mean = harmful_proj.mean()
        harmless_mean = harmless_proj.mean()
        
        separation = harmful_mean - harmless_mean
        
        # Also consider variance (want low variance within each class)
        harmful_std = harmful_proj.std()
        harmless_std = harmless_proj.std()
        avg_std = (harmful_std + harmless_std) / 2
        
        # Cohen's d as score
        score = separation / (avg_std + 1e-8)
        
        return float(score)
    
    def get_unit_direction(self, layer_idx: int) -> torch.Tensor:
        """Get unit-norm direction for a layer."""
        direction = self.directions[layer_idx]
        return direction / (direction.norm() + 1e-8)
    
    def save(self, path: str):
        """Save extracted directions to disk."""
        save_dict = {
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "chat_template": self.chat_template,
            "mean_harmful": {k: v.cpu() for k, v in self.mean_harmful.items()},
            "mean_harmless": {k: v.cpu() for k, v in self.mean_harmless.items()},
            "directions": {k: v.cpu() for k, v in self.directions.items()},
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Saved directions to {path}")
    
    @classmethod
    def load(cls, path: str, model: ModelWrapper) -> 'RefusalDirectionExtractor':
        """Load extracted directions from disk."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        extractor = cls(model, chat_template=save_dict["chat_template"])
        extractor.mean_harmful = save_dict["mean_harmful"]
        extractor.mean_harmless = save_dict["mean_harmless"]
        extractor.directions = save_dict["directions"]
        
        print(f"Loaded directions from {path}")
        return extractor


def analyze_direction(
    direction: torch.Tensor,
    top_k: int = 10
) -> Dict:
    """
    Analyze properties of a direction vector.
    
    Args:
        direction: Direction vector
        top_k: Number of top dimensions to report
        
    Returns:
        Dictionary with analysis results
    """
    direction_np = direction.cpu().numpy()
    
    # Magnitude
    magnitude = np.linalg.norm(direction_np)
    
    # Top dimensions
    abs_values = np.abs(direction_np)
    top_indices = np.argsort(abs_values)[-top_k:][::-1]
    top_values = direction_np[top_indices]
    
    # Sparsity (what fraction of dimensions are near zero)
    threshold = 0.01 * abs_values.max()
    sparsity = (abs_values < threshold).sum() / len(abs_values)
    
    return {
        "magnitude": float(magnitude),
        "top_dimensions": list(zip(top_indices.tolist(), top_values.tolist())),
        "sparsity": float(sparsity),
        "max_value": float(abs_values.max()),
        "mean_abs_value": float(abs_values.mean())
    }


if __name__ == "__main__":
    # Test direction extraction
    print("Testing RefusalDirectionExtractor...")
    
    from .data import RefusalDataset
    
    # Load data
    dataset = RefusalDataset()
    harmful_train = dataset.get_harmful_train(n=32)  # Small for testing
    harmless_train = dataset.get_harmless_train(n=32)
    harmful_val = dataset.get_harmful_val(n=16)
    harmless_val = dataset.get_harmless_val(n=16)
    
    # Load model
    model = ModelWrapper(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        load_in_8bit=True
    )
    
    # Extract directions
    extractor = RefusalDirectionExtractor(model)
    directions = extractor.extract_directions(
        harmful_train, harmless_train, batch_size=8
    )
    
    print(f"\nExtracted directions for {len(directions)} layers")
    
    # Select best direction
    best_layer, best_direction = extractor.select_best_direction(
        harmful_val, harmless_val
    )
    
    # Analyze
    analysis = analyze_direction(best_direction)
    print(f"\nDirection analysis for layer {best_layer}:")
    print(f"  Magnitude: {analysis['magnitude']:.4f}")
    print(f"  Sparsity: {analysis['sparsity']:.4f}")
    print(f"  Max value: {analysis['max_value']:.4f}")
