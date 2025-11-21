"""
Data loading and management for refusal directions experiments.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random


@dataclass
class Instruction:
    """A single instruction with optional category label."""
    instruction: str
    category: str = None
    
    def __str__(self):
        return self.instruction


class RefusalDataset:
    """Dataset manager for harmful and harmless instructions."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.harmful = self._load_instructions("harmful_instructions.json")
        self.harmless = self._load_instructions("harmless_instructions.json")
        
    def _load_instructions(self, filename: str) -> Dict[str, List[Instruction]]:
        """Load instructions from JSON file."""
        filepath = self.data_dir / filename
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return {
            split: [Instruction(**item) for item in items]
            for split, items in data.items()
        }
    
    def get_harmful_train(self, n: int = None) -> List[Instruction]:
        """Get training harmful instructions."""
        data = self.harmful["train"]
        return data[:n] if n else data
    
    def get_harmful_val(self, n: int = None) -> List[Instruction]:
        """Get validation harmful instructions."""
        data = self.harmful["validation"]
        return data[:n] if n else data
    
    def get_harmless_train(self, n: int = None) -> List[Instruction]:
        """Get training harmless instructions."""
        data = self.harmless["train"]
        return data[:n] if n else data
    
    def get_harmless_val(self, n: int = None) -> List[Instruction]:
        """Get validation harmless instructions."""
        data = self.harmless["validation"]
        return data[:n] if n else data
    
    def get_contrastive_pairs(self, split: str = "train", n: int = 128) -> Tuple[List[Instruction], List[Instruction]]:
        """Get paired harmful and harmless instructions for contrast."""
        if split == "train":
            harmful = self.get_harmful_train(n)
            harmless = self.get_harmless_train(n)
        else:
            harmful = self.get_harmful_val(n)
            harmless = self.get_harmless_val(n)
        
        return harmful, harmless
    
    def get_by_category(self, category: str, split: str = "train") -> List[Instruction]:
        """Get harmful instructions by category."""
        data = self.harmful[split]
        return [inst for inst in data if inst.category == category]
    
    def get_all_categories(self) -> List[str]:
        """Get list of all harm categories."""
        all_insts = self.harmful["train"] + self.harmful["validation"]
        return sorted(set(inst.category for inst in all_insts if inst.category))


def format_instruction(instruction: str, chat_template: str = "llama3") -> str:
    """
    Format instruction with appropriate chat template.
    
    Args:
        instruction: The instruction text
        chat_template: Template type ('llama3', 'llama2', 'gemma', etc.)
    
    Returns:
        Formatted instruction string
    """
    templates = {
        "llama3": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "llama2": f"[INST] {instruction} [/INST]",
        "gemma": f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
        "qwen": f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
        "yi": f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    }
    
    return templates.get(chat_template, templates["llama3"])


def get_post_instruction_tokens(chat_template: str = "llama3") -> List[str]:
    """
    Get the post-instruction tokens for a given chat template.
    These are tokens that appear after the user instruction and before the assistant response.
    
    Args:
        chat_template: Template type
        
    Returns:
        List of token strings that comprise the post-instruction region
    """
    # These will need to be adjusted based on actual tokenization
    # This is a simplified version - you may need to tokenize and inspect
    post_tokens = {
        "llama3": ["<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n\n"],
        "llama2": ["[/INST]", " "],
        "gemma": ["<end_of_turn>", "\n", "<start_of_turn>", "model", "\n"],
        "qwen": ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"],
        "yi": ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"]
    }
    
    return post_tokens.get(chat_template, post_tokens["llama3"])


if __name__ == "__main__":
    # Test the dataset loading
    dataset = RefusalDataset()
    
    print(f"Loaded {len(dataset.get_harmful_train())} harmful train instructions")
    print(f"Loaded {len(dataset.get_harmful_val())} harmful val instructions")
    print(f"Loaded {len(dataset.get_harmless_train())} harmless train instructions")
    print(f"Loaded {len(dataset.get_harmless_val())} harmless val instructions")
    
    print("\nCategories:", dataset.get_all_categories())
    
    print("\nExample harmful instruction:")
    print(dataset.get_harmful_train()[0])
    
    print("\nExample harmless instruction:")
    print(dataset.get_harmless_train()[0])
    
    print("\nFormatted with Llama 3 template:")
    print(format_instruction(str(dataset.get_harmful_train()[0]), "llama3"))
