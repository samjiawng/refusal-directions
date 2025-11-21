#!/usr/bin/env python3
"""
Extract refusal directions from Llama 3.1 using difference-in-means.

Usage:
    python experiments/extract_directions.py --model meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    RefusalDataset,
    ModelWrapper,
    RefusalDirectionExtractor,
    analyze_direction
)


def main():
    parser = argparse.ArgumentParser(description="Extract refusal directions")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing instruction datasets"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/directions.pkl",
        help="Output path for extracted directions"
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=128,
        help="Number of training examples per dataset"
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=32,
        help="Number of validation examples per dataset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="llama3",
        choices=["llama3", "llama2", "gemma", "qwen", "yi"],
        help="Chat template to use"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("REFUSAL DIRECTION EXTRACTION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Training examples: {args.n_train} per dataset")
    print(f"Validation examples: {args.n_val} per dataset")
    print(f"Chat template: {args.chat_template}")
    print("="*60 + "\n")
    
    # Load dataset
    print("Loading datasets...")
    dataset = RefusalDataset(data_dir=args.data_dir)
    
    harmful_train = dataset.get_harmful_train(args.n_train)
    harmless_train = dataset.get_harmless_train(args.n_train)
    harmful_val = dataset.get_harmful_val(args.n_val)
    harmless_val = dataset.get_harmless_val(args.n_val)
    
    print(f"Loaded {len(harmful_train)} harmful and {len(harmless_train)} harmless training examples")
    print(f"Loaded {len(harmful_val)} harmful and {len(harmless_val)} harmless validation examples")
    
    # Load model
    print("\nLoading model...")
    model = ModelWrapper(
        model_name=args.model,
        load_in_8bit=args.load_in_8bit
    )
    
    # Extract directions
    print("\nExtracting directions...")
    extractor = RefusalDirectionExtractor(model, chat_template=args.chat_template)
    
    directions = extractor.extract_directions(
        harmful_train,
        harmless_train,
        batch_size=args.batch_size
    )
    
    print(f"\nExtracted directions for {len(directions)} layers")
    
    # Select best direction
    print("\nSelecting best direction...")
    best_layer, best_direction = extractor.select_best_direction(
        harmful_val,
        harmless_val,
        top_k=10
    )
    
    # Analyze best direction
    print("\nAnalyzing best direction...")
    analysis = analyze_direction(best_direction, top_k=20)
    
    print(f"\nDirection Properties:")
    print(f"  Magnitude: {analysis['magnitude']:.4f}")
    print(f"  Sparsity: {analysis['sparsity']:.2%}")
    print(f"  Max absolute value: {analysis['max_value']:.4f}")
    print(f"  Mean absolute value: {analysis['mean_abs_value']:.4f}")
    
    print(f"\nTop 10 dimensions:")
    for i, (dim, val) in enumerate(analysis['top_dimensions'][:10], 1):
        print(f"  {i}. Dimension {dim}: {val:.4f}")
    
    # Save directions
    print(f"\nSaving directions to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    extractor.save(args.output)
    
    # Also save best direction separately
    best_path = Path(args.output).parent / "best_direction.pt"
    import torch
    torch.save({
        "layer": best_layer,
        "direction": best_direction,
        "analysis": analysis
    }, best_path)
    print(f"Saved best direction to {best_path}")
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Best layer: {best_layer}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
