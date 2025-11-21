#!/usr/bin/env python3
"""
Test directional ablation as a jailbreak technique.

This script loads a pre-extracted refusal direction and tests its effectiveness
at bypassing refusal when ablated from the model.

Usage:
    python experiments/test_ablation.py --directions results/best_direction.pt
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
    parser = argparse.ArgumentParser(description="Test directional ablation")
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
        default="results/ablation_results.json",
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
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to apply ablation (comma-separated or 'all')"
    )
    parser.add_argument(
        "--skip-safety",
        action="store_true",
        help="Skip safety evaluation (faster but less informative)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("DIRECTIONAL ABLATION JAILBREAK TEST")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Directions: {args.directions}")
    print(f"Test examples: {args.n_test}")
    print("="*60 + "\n")
    
    # Load directions
    print("Loading directions...")
    directions_data = torch.load(args.directions)
    best_layer = directions_data["layer"]
    direction = directions_data["direction"]
    
    print(f"Loaded direction from layer {best_layer}")
    print(f"Direction shape: {direction.shape}")
    
    # Parse layers
    if args.layers == "all":
        layers = None
        print("Applying ablation to all layers")
    else:
        layers = [int(x) for x in args.layers.split(",")]
        print(f"Applying ablation to layers: {layers}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = RefusalDataset(data_dir=args.data_dir)
    harmful_test = dataset.get_harmful_val(args.n_test)
    
    print(f"Loaded {len(harmful_test)} harmful test examples")
    
    # Get categories for analysis
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
    baseline_results = evaluator.evaluate_by_category(
        prompts_raw,
        baseline_completions,
        categories,
        compute_safety=not args.skip_safety
    )
    
    print("\nBaseline Results:")
    evaluator.print_results(baseline_results["overall"])
    
    # With intervention: Directional ablation
    print("\n" + "="*60)
    print("WITH DIRECTIONAL ABLATION")
    print("="*60)
    
    # Apply intervention
    intervention = InterventionManager(model)
    intervention.directional_ablation(direction, layers=layers)
    
    ablation_completions = []
    for i in tqdm(range(0, len(prompts_formatted), args.batch_size), desc="Generating"):
        batch = prompts_formatted[i:i + args.batch_size]
        completions = model.generate(
            batch,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0
        )
        ablation_completions.extend(completions)
    
    # Clear intervention
    intervention.clear_hooks()
    
    print("\nEvaluating with ablation...")
    ablation_results = evaluator.evaluate_by_category(
        prompts_raw,
        ablation_completions,
        categories,
        compute_safety=not args.skip_safety
    )
    
    print("\nAblation Results:")
    evaluator.print_results(ablation_results["overall"])
    
    # Compute attack success rate
    baseline_refusal = baseline_results["overall"]["refusal_scores"]
    ablation_refusal = ablation_results["overall"]["refusal_scores"]
    
    if "safety_scores" in ablation_results["overall"]:
        ablation_safety = ablation_results["overall"]["safety_scores"]
        asr = compute_attack_success_rate(ablation_refusal, ablation_safety)
    else:
        asr = compute_attack_success_rate(ablation_refusal)
    
    print(f"\nAttack Success Rate: {asr:.2%}")
    
    # Per-category results
    print("\n" + "="*60)
    print("RESULTS BY CATEGORY")
    print("="*60)
    
    for category in unique_categories:
        if category in ablation_results:
            cat_results = ablation_results[category]
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
            "layers": layers if layers else "all",
            "best_layer": int(best_layer),
        },
        "baseline": {
            k: v for k, v in baseline_results["overall"].items()
            if k != "refusal_scores" and k != "safety_scores"
        },
        "ablation": {
            k: v for k, v in ablation_results["overall"].items()
            if k != "refusal_scores" and k != "safety_scores"
        },
        "attack_success_rate": float(asr),
        "by_category": {
            cat: {
                k: v for k, v in results.items()
                if k != "refusal_scores" and k != "safety_scores"
            }
            for cat, results in ablation_results.items()
            if cat != "overall"
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Baseline refusal rate: {baseline_results['overall']['refusal_rate']:.2%}")
    print(f"Ablation refusal rate: {ablation_results['overall']['refusal_rate']:.2%}")
    print(f"Attack success rate: {asr:.2%}")
    
    # Show some examples
    print("\n" + "="*60)
    print("EXAMPLE COMPLETIONS")
    print("="*60)
    
    for i in range(min(3, len(harmful_test))):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {prompts_raw[i][:100]}...")
        print(f"\nBaseline: {baseline_completions[i][:200]}...")
        print(f"\nWith ablation: {ablation_completions[i][:200]}...")
        print("-"*60)


if __name__ == "__main__":
    main()
