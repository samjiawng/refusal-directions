"""
Model loading and activation collection utilities.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class ModelWrapper:
    """Wrapper for HuggingFace models with activation collection."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False
    ):
        """
        Initialize model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            torch_dtype: Data type for model weights
            load_in_8bit: Whether to use 8-bit quantization
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto" if device == "cuda" else None,
        }
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        self.n_layers = len(self.model.model.layers)
        self.d_model = self.model.config.hidden_size
        
        print(f"Model loaded: {self.n_layers} layers, {self.d_model} hidden dimensions")
        
        # Storage for activations during forward pass
        self.activations = {}
        self.hooks = []
        
    def get_layer_names(self) -> List[str]:
        """Get names of all transformer layers."""
        return [f"model.layers.{i}" for i in range(self.n_layers)]
    
    def tokenize(self, texts: List[str], add_special_tokens: bool = True) -> Dict:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=add_special_tokens
        ).to(self.device)
    
    def _get_activation_hook(self, name: str):
        """Create hook function to capture activations."""
        def hook(module, input, output):
            # For decoder layers, output is a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Store on CPU to save GPU memory
            self.activations[name] = hidden_states.detach().cpu()
        
        return hook
    
    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        """
        Register forward hooks to capture activations.
        
        Args:
            layer_indices: List of layer indices to hook. If None, hooks all layers.
        """
        self.clear_hooks()
        
        if layer_indices is None:
            layer_indices = list(range(self.n_layers))
        
        for idx in layer_indices:
            layer = self.model.model.layers[idx]
            hook = layer.register_forward_hook(
                self._get_activation_hook(f"layer_{idx}")
            )
            self.hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def get_activations(
        self,
        texts: List[str],
        layers: Optional[List[int]] = None,
        positions: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get activations for a batch of texts at specified layers and positions.
        
        Args:
            texts: List of input texts
            layers: Layer indices to collect from. If None, collects from all layers.
            positions: Token positions to collect. If None, collects all positions.
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        self.register_hooks(layers)
        
        # Tokenize and run forward pass
        inputs = self.tokenize(texts)
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Extract activations at specified positions
        result = {}
        for layer_name, acts in self.activations.items():
            if positions is not None:
                # acts shape: (batch, seq_len, hidden_dim)
                # Select specific positions
                acts = acts[:, positions, :]
            result[layer_name] = acts
        
        self.clear_hooks()
        return result
    
    def get_post_instruction_activations(
        self,
        texts: List[str],
        layers: Optional[List[int]] = None,
        post_token_start: int = -5  # Default: last 5 tokens
    ) -> Dict[str, torch.Tensor]:
        """
        Get activations from the post-instruction region.
        
        Args:
            texts: List of formatted instruction texts
            layers: Layers to collect from
            post_token_start: Start position of post-instruction tokens (negative indexing)
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        self.register_hooks(layers)
        
        inputs = self.tokenize(texts)
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Extract post-instruction activations
        result = {}
        for layer_name, acts in self.activations.items():
            # Take last few tokens (post-instruction region)
            # Shape: (batch, seq_len, hidden_dim)
            post_acts = acts[:, post_token_start:, :]
            # Average over post-instruction tokens
            post_acts_mean = post_acts.mean(dim=1)  # (batch, hidden_dim)
            result[layer_name] = post_acts_mean
        
        self.clear_hooks()
        return result
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Generate completions for prompts.
        
        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)
            do_sample: Whether to use sampling
            
        Returns:
            List of generated text strings
        """
        inputs = self.tokenize(prompts)
        
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            **kwargs
        }
        
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        
        outputs = self.model.generate(
            **inputs,
            **generate_kwargs
        )
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        
        completions = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        return completions
    
    def get_residual_stream_shape(self) -> Tuple[int, int]:
        """Get the shape of residual stream activations."""
        return (self.n_layers, self.d_model)


if __name__ == "__main__":
    # Test model loading and activation collection
    print("Testing ModelWrapper...")
    
    # Use a smaller model for testing
    model = ModelWrapper(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        load_in_8bit=True  # Use 8-bit for testing to save memory
    )
    
    # Test tokenization
    test_text = ["Hello, how are you?", "What is the capital of France?"]
    tokens = model.tokenize(test_text)
    print(f"\nTokenized {len(test_text)} texts")
    print(f"Input shape: {tokens['input_ids'].shape}")
    
    # Test activation collection
    print("\nCollecting activations from layers 0, 15, 31...")
    acts = model.get_activations(test_text, layers=[0, 15, 31])
    for layer_name, activation in acts.items():
        print(f"{layer_name}: {activation.shape}")
    
    # Test generation
    print("\nTesting generation...")
    prompts = ["The capital of France is"]
    completions = model.generate(prompts, max_new_tokens=10)
    print(f"Prompt: {prompts[0]}")
    print(f"Completion: {completions[0]}")
