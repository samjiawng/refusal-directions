#!/usr/bin/env python3
"""
Test weight orthogonalization as a permanent jailbreak.

This script loads a pre-extracted refusal direction and permanently modifies
the model weights to orthogonalize them with respect to the direction.

WARNING: This permanently modifies model weights! Use with caution.

Usage:
    python experiments/weight_ortho.py --directions results/best_direction.pt
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
    format_instruction,
    compute_attack_success_rate
)


def main():
    parser = argparse.ArgumentParser(description="Test weight orthogonalization")
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
        default="results/ortho_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Path to save orthogonalized model (optional)"
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
        "--load-in-8bit",
        action="store_true",
        help="Use 8-bit quantization (not recommended for weight modification)"
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="llama3",
        help="Chat template to use"
    )
    parser.add_argument(
        "--skip-safety",
        action="store_true",
        help="Skip safety evaluation"
    )
    
    args = parser.parse_args()
    
    if args.load_in_8bit:
        print("WARNING: Weight orthogonalization with 8-bit models may not work properly!")
        print("Consider using full precision for weight modification.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            return
    
    print("="*60)
    print("WEIGHT ORTHOGONALIZATION TEST")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Directions: {args.directions}")
    print(f"Test examples: {args.n_test}")
    print("="*60)
    print("\nWARNING: This will permanently modify model weights!")
    print("="*60 + "\n")
    
    # Load directions
    print("Loading directions...")
    directions_data = torch.load(args.directions)
    best_layer = directions_data["layer"]
    direction = directions_data["direction"]
    
    print(f"Loaded direction from layer {best_layer}")
    print(f"Direction shape: {direction.shape}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = RefusalDataset(data_dir=args.data_dir)
    harmful_test = dataset.get_harmful_val(args.n_test)
    
    print(f"Loaded {len(harmful_test)} harmful test examples")
    
    # Get categories
    categories = [inst.category for inst in harmful_test]
    unique_categories = sorted(set(c for c in categories if c))
    print(f"Categories: {unique_categories}")
    
    # Load model
    print("\nLoading model...")
    model = ModelWrapper(
        model_name=args.model,
        load_in_8bit=args.load_in_8bit
    )
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = CombinedEvaluator(use_safety_classifier=not args.skip_safety)
    
    # Format prompts
    prompts_raw = [str(inst) for inst in harmful_test]
    prompts_formatted = [format_instruction(p, args.chat_template) for p in prompts_raw]
    
    # Baseline: Generate BEFORE orthogonalization
    print("\n" + "="*60)
    print("BASELINE (Before Orthogonalization)")
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
    baseline_results = evaluator.evaluate_by_category(
        prompts_raw,
        baseline_completions,
        categories,
        compute_safety=not args.skip_safety
    )
    
    print("\nBaseline Results:")
    evaluator.print_results(baseline_results["overall"])
    
    # Apply weight orthogonalization
    print("\n" + "="*60)
    print("APPLYING WEIGHT ORTHOGONALIZATION")
    print("="*60)
    
    intervention = InterventionManager(model)
    intervention.weight_orthogonalization(direction, save_original=True)
    
    print("\nWeights have been orthogonalized!")
    
    # Test AFTER orthogonalization
    print("\n" + "="*60)
    print("AFTER ORTHOGONALIZATION")
    print("="*60)
    
    ortho_completions = []
    for i in tqdm(range(0, len(prompts_formatted), args.batch_size), desc="Generating"):
        batch = prompts_formatted[i:i + args.batch_size]
        completions = model.generate(
            batch,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0
        )
        ortho_completions.extend(completions)
    
    print("\nEvaluating after orthogonalization...")
    ortho_results = evaluator.evaluate_by_category(
        prompts_raw,
        ortho_completions,
        categories,
        compute_safety=not args.skip_safety
    )
    
    print("\nOrthogonalization Results:")
    evaluator.print_results(ortho_results["overall"])
    
    # Compute attack success rate
    baseline_refusal = baseline_results["overall"]["refusal_scores"]
    ortho_refusal = ortho_results["overall"]["refusal_scores"]
    
    if "safety_scores" in ortho_results["overall"]:
        ortho_safety = ortho_results["overall"]["safety_scores"]
        asr = compute_attack_success_rate(ortho_refusal, ortho_safety)
    else:
        asr = compute_attack_success_rate(ortho_refusal)
    
    print(f"\nAttack Success Rate: {asr:.2%}")
    
    # Per-category results
    print("\n" + "="*60)
    print("RESULTS BY CATEGORY")
    print("="*60)
    
    for category in unique_categories:
        if category in ortho_results:
            cat_results = ortho_results[category]
            print(f"\n{category.upper()}:")
            print(f"  Refusal rate: {cat_results['refusal_rate']:.2%}")
            if "safety_rate" in cat_results:
                print(f"  Safety rate:  {cat_results['safety_rate']:.2%}")
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "config": {
            "model": args.model,
            "directions": args.directions,
            "n_test": args.n_test,
            "best_layer": int(best_layer),
        },
        "baseline": {
            k: v for k, v in baseline_results["overall"].items()
            if k != "refusal_scores" and k != "safety_scores"
        },
        "orthogonalized": {
            k: v for k, v in ortho_results["overall"].items()
            if k != "refusal_scores" and k != "safety_scores"
        },
        "attack_success_rate": float(asr),
        "by_category": {
            cat: {
                k: v for k, v in results.items()
                if k != "refusal_scores" and k != "safety_scores"
            }
            for cat, results in ortho_results.items()
            if cat != "overall"
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Optionally save the modified model
    if args.save_model:
        print(f"\nSaving orthogonalized model to {args.save_model}...")
        model.model.save_pretrained(args.save_model)
        model.tokenizer.save_pretrained(args.save_model)
        print("Model saved!")
    
    # Option to restore original weights
    print("\n" + "="*60)
    restore = input("Restore original weights? (y/n): ")
    if restore.lower() == 'y':
        intervention.restore_original_weights()
        print("Original weights restored.")
    else:
        print("Keeping orthogonalized weights.")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Baseline refusal rate: {baseline_results['overall']['refusal_rate']:.2%}")
    print(f"Ortho refusal rate: {ortho_results['overall']['refusal_rate']:.2%}")
    print(f"Attack success rate: {asr:.2%}")
    
    # Show some examples
    print("\n" + "="*60)
    print("EXAMPLE COMPLETIONS")
    print("="*60)
    
    for i in range(min(3, len(harmful_test))):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {prompts_raw[i][:100]}...")
        print(f"\nBaseline: {baseline_completions[i][:200]}...")
        print(f"\nOrthogonalized: {ortho_completions[i][:200]}...")
        print("-"*60)


if __name__ == "__main__":
    main()
