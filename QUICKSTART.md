# Quick Start Guide

Get up and running with refusal directions in 5 minutes!

## Prerequisites

1. **Python 3.8+** with GPU support
2. **Hugging Face account** with Llama 3.1 access
3. **~16GB GPU memory** (for 8B model with 8-bit quantization)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face (you'll need access to Llama 3.1)
huggingface-cli login
```

## Option 1: Run Complete Pipeline (Recommended)

The easiest way to run all experiments:

```bash
# Run full pipeline with default settings
./run_pipeline.sh

# Or specify model and test size
./run_pipeline.sh meta-llama/Meta-Llama-3.1-8B-Instruct 50
```

This will:
1. Extract refusal directions (Layer 15 usually selected)
2. Test ablation on 100 harmful instructions
3. Test addition on 100 harmless instructions
4. Save all results to `results/`

**Time estimate**: ~30-45 minutes on a single GPU

## Option 2: Run Step-by-Step

### Step 1: Extract Directions

```bash
python experiments/extract_directions.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --load-in-8bit
```

**Output**: `results/best_direction.pt`

**Time**: ~10 minutes

### Step 2: Test Ablation (Jailbreak)

```bash
python experiments/test_ablation.py \
    --directions results/best_direction.pt \
    --n-test 100 \
    --load-in-8bit \
    --skip-safety  # Optional: speeds up testing
```

**Output**: `results/ablation_results.json`

**Time**: ~15 minutes

### Step 3: Test Addition (Induce Refusal)

```bash
python experiments/test_addition.py \
    --directions results/best_direction.pt \
    --n-test 100 \
    --load-in-8bit
```

**Output**: `results/addition_results.json`

**Time**: ~10 minutes

### Step 4: Analyze Results

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Expected Results

You should see:

### Directional Ablation
- **Baseline refusal rate**: 95-100% (model refuses harmful requests)
- **After ablation**: 5-25% (jailbreak successful!)
- **Attack success rate**: 60-85%

### Activation Addition
- **Baseline refusal rate**: 0-5% (model complies with harmless requests)
- **After addition**: 75-95% (induced refusal!)

## Quick Test (5 minutes)

Want to just verify it works? Run with minimal samples:

```bash
# Extract with 32 samples instead of 128
python experiments/extract_directions.py \
    --n-train 32 \
    --n-val 16 \
    --load-in-8bit

# Test with 20 samples instead of 100
python experiments/test_ablation.py \
    --directions results/best_direction.pt \
    --n-test 20 \
    --skip-safety \
    --load-in-8bit
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Use smaller batch sizes
python experiments/extract_directions.py --batch-size 4 --load-in-8bit
```

### Model Access Denied

You need to:
1. Go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Click "Request Access"
3. Wait for approval (usually instant)
4. Run `huggingface-cli login`

### Slow Generation

This is normal! Generation with 8B model:
- ~1-2 minutes per 100 examples
- Can reduce with `--n-test 20`

### LlamaGuard Not Loading

Skip safety evaluation (still useful):
```bash
python experiments/test_ablation.py --skip-safety
```

## What's Next?

1. **Visualize results**: Open `notebooks/analysis.ipynb`
2. **Try different layers**: `--layers "10,15,20"`
3. **Adjust coefficients**: `--coefficient 0.5`
4. **Test orthogonalization**: Run `experiments/weight_ortho.py`
5. **Try other models**: Change `--model` to any Llama 3.1 variant

## Project Structure

```
refusal-directions/
├── data/              # 128+32 instructions per category
├── src/               # Core library code
├── experiments/       # Runnable scripts
├── notebooks/         # Analysis notebooks
└── results/          # Your experiment outputs (created automatically)
```

## Key Files

- `results/best_direction.pt` - Extracted refusal direction
- `results/ablation_results.json` - Jailbreak test results
- `results/addition_results.json` - Refusal induction results
- `notebooks/analysis.ipynb` - Visualization and analysis

## Need Help?

1. Check the full [README.md](README.md) for detailed documentation
2. Review the paper: https://arxiv.org/abs/2406.11717
3. Check console output for helpful error messages

## Common Workflows

### Just want to see it work?
```bash
./run_pipeline.sh meta-llama/Meta-Llama-3.1-8B-Instruct 20
```

### Research mode (full evaluation)?
```bash
./run_pipeline.sh meta-llama/Meta-Llama-3.1-8B-Instruct 100
```

### Analyze specific layers?
```bash
python experiments/test_ablation.py \
    --directions results/best_direction.pt \
    --layers "8,12,16,20,24"
```

### Test on harmless instructions?
```bash
python experiments/test_addition.py \
    --directions results/best_direction.pt \
    --coefficient 1.5  # Stronger addition
```

Happy researching! 🔬
