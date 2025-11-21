#!/usr/bin/env python3
"""
Test activation addition for inducing refusal on harmless instructions.

This script loads a pre-extracted refusal direction and tests its effectiveness
at inducing refusal when added to the model's activations.

Usage:
    python experiments/test_addition.py --directions results/best_direction.pt
"""

import argparse
import sys
from pathlib import Path
import torch
import json
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    RefusalDataset,
    ModelWrapper,
    InterventionManager,
    CombinedEvaluator,
    format_instruction
)


def main():
    parser = argparse.ArgumentParser(description="Test activation addition")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--directions",
        type=str,
        required=True,
        help="Path to saved directions (.pt file)"
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
        default="results/addition_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=100,
        help="Number of test examples to evaluate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--coefficient",
        type=float,
        default=1.0,
        help="Scaling coefficient for activation addition"
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
        help="Chat template to use"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ACTIVATION ADDITION TEST")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Directions: {args.directions}")
    print(f"Test examples: {args.n_test}")
    print(f"Coefficient: {args.coefficient}")
    print("="*60 + "\n")
    
    # Load directions
    print("Loading directions...")
    directions_data = torch.load(args.directions)
    best_layer = directions_data["layer"]
    direction = directions_data["direction"]
    
    print(f"Loaded direction from layer {best_layer}")
    print(f"Direction shape: {direction.shape}")
    
    # Load dataset - use HARMLESS instructions
    print("\nLoading dataset...")
    dataset = RefusalDataset(data_dir=args.data_dir)
    harmless_test = dataset.get_harmless_val(args.n_test)
    
    print(f"Loaded {len(harmless_test)} harmless test examples")
    
    # Load model
    print("\nLoading model...")
    model = ModelWrapper(
        model_name=args.model,
        load_in_8bit=args.load_in_8bit
    )
    
    # Initialize evaluator (skip safety classifier since these are harmless)
    print("\nInitializing evaluator...")
    evaluator = CombinedEvaluator(use_safety_classifier=False)
    
    # Format prompts
    prompts_raw = [str(inst) for inst in harmless_test]
    prompts_formatted = [format_instruction(p, args.chat_template) for p in prompts_raw]
    
    # Baseline: Generate without intervention
    print("\n" + "="*60)
    print("BASELINE (No Intervention)")
    print("="*60)
    
    baseline_completions = []
    for i in tqdm(range(0, len(prompts_formatted), args.batch_size), desc="Generating"):
        batch = prompts_formatted[i:i + args.batch_size]
        completions = model.generate(
            batch,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0
        )
        baseline_completions.extend(completions)
    
    print("\nEvaluating baseline...")
    baseline_results = evaluator.evaluate(
        prompts_raw,
        baseline_completions,
        compute_safety=False
    )
    
    print("\nBaseline Results:")
    evaluator.print_results(baseline_results)
    
    # With intervention: Activation addition
    print("\n" + "="*60)
    print("WITH ACTIVATION ADDITION")
    print("="*60)
    
    # Apply intervention
    intervention = InterventionManager(model)
    intervention.activation_addition(
        direction,
        layer_idx=best_layer,
        coefficient=args.coefficient
    )
    
    addition_completions = []
    for i in tqdm(range(0, len(prompts_formatted), args.batch_size), desc="Generating"):
        batch = prompts_formatted[i:i + args.batch_size]
        completions = model.generate(
            batch,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0
        )
        addition_completions.extend(completions)
    
    # Clear intervention
    intervention.clear_hooks()
    
    print("\nEvaluating with activation addition...")
    addition_results = evaluator.evaluate(
        prompts_raw,
        addition_completions,
        compute_safety=False
    )
    
    print("\nActivation Addition Results:")
    evaluator.print_results(addition_results)
    
    # Compute induced refusal rate
    baseline_refusal_rate = baseline_results["refusal_rate"]
    addition_refusal_rate = addition_results["refusal_rate"]
    induced_refusal_increase = addition_refusal_rate - baseline_refusal_rate
    
    print(f"\nInduced Refusal Increase: {induced_refusal_increase:.2%}")
    print(f"(from {baseline_refusal_rate:.2%} to {addition_refusal_rate:.2%})")
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "config": {
            "model": args.model,
            "directions": args.directions,
            "n_test": args.n_test,
            "coefficient": args.coefficient,
            "best_layer": int(best_layer),
        },
        "baseline": {
            k: v for k, v in baseline_results.items()
            if k != "refusal_scores"
        },
        "addition": {
            k: v for k, v in addition_results.items()
            if k != "refusal_scores"
        },
        "induced_refusal_increase": float(induced_refusal_increase)
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Baseline refusal rate: {baseline_refusal_rate:.2%}")
    print(f"Addition refusal rate: {addition_refusal_rate:.2%}")
    print(f"Induced refusal increase: {induced_refusal_increase:.2%}")
    
    # Show some examples
    print("\n" + "="*60)
    print("EXAMPLE COMPLETIONS")
    print("="*60)
    
    for i in range(min(5, len(harmless_test))):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {prompts_raw[i][:100]}...")
        print(f"\nBaseline: {baseline_completions[i][:200]}...")
        print(f"\nWith addition: {addition_completions[i][:200]}...")
        print("-"*60)


if __name__ == "__main__":
    main()
