#!/bin/bash
# Complete pipeline for refusal directions experiments
# Usage: ./run_pipeline.sh [model_name] [n_test]

set -e  # Exit on error

# Default arguments
MODEL=${1:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
N_TEST=${2:-100}

echo "=========================================="
echo "REFUSAL DIRECTIONS EXPERIMENT PIPELINE"
echo "=========================================="
echo "Model: $MODEL"
echo "Test samples: $N_TEST"
echo "=========================================="
echo ""

# Step 1: Extract directions
echo "Step 1/4: Extracting refusal directions..."
python experiments/extract_directions.py \
    --model "$MODEL" \
    --n-train 128 \
    --n-val 32 \
    --load-in-8bit \
    --output results/directions.pkl

echo ""
echo "✓ Directions extracted"
echo ""

# Step 2: Test ablation
echo "Step 2/4: Testing directional ablation..."
python experiments/test_ablation.py \
    --model "$MODEL" \
    --directions results/best_direction.pt \
    --n-test $N_TEST \
    --load-in-8bit \
    --skip-safety \
    --output results/ablation_results.json

echo ""
echo "✓ Ablation test complete"
echo ""

# Step 3: Test addition
echo "Step 3/4: Testing activation addition..."
python experiments/test_addition.py \
    --model "$MODEL" \
    --directions results/best_direction.pt \
    --n-test $N_TEST \
    --load-in-8bit \
    --output results/addition_results.json

echo ""
echo "✓ Addition test complete"
echo ""

# Step 4: Test orthogonalization (optional - commented out by default)
# Uncomment to run weight orthogonalization test
# echo "Step 4/4: Testing weight orthogonalization..."
# python experiments/weight_ortho.py \
#     --model "$MODEL" \
#     --directions results/best_direction.pt \
#     --n-test $N_TEST \
#     --output results/ortho_results.json

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/best_direction.pt"
echo "  - results/ablation_results.json"
echo "  - results/addition_results.json"
echo ""
echo "View results in notebooks/analysis.ipynb"
echo ""
