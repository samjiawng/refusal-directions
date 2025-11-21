# VS Code Setup Instructions

Quick guide to get this project running in VS Code.

## Initial Setup

### 1. Open in VS Code

```bash
cd path/to/refusal-directions
code .
```

### 2. Set Up Python Environment

#### Option A: Using Conda (Recommended)

```bash
# Create environment
conda create -n refusal python=3.10

# Activate
conda activate refusal

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using venv

```bash
# Create environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure VS Code Python Interpreter

1. Press `Cmd/Ctrl + Shift + P`
2. Type "Python: Select Interpreter"
3. Choose your `refusal` conda environment or `venv`

### 4. Set Up Hugging Face

```bash
# Login to Hugging Face
huggingface-cli login

# Paste your token when prompted
```

Get your token from: https://huggingface.co/settings/tokens

## VS Code Extensions (Recommended)

Install these for better development experience:

1. **Python** (Microsoft) - Python language support
2. **Pylance** (Microsoft) - Fast Python IntelliSense
3. **Jupyter** (Microsoft) - For running notebooks
4. **GitLens** (optional) - Git visualization

## Running Experiments

### Using VS Code Terminal

```bash
# Run full pipeline
./run_pipeline.sh

# Or run individual scripts
python experiments/extract_directions.py --load-in-8bit
```

### Using VS Code Run Configuration

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Extract Directions",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/experiments/extract_directions.py",
            "args": [
                "--model", "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "--load-in-8bit",
                "--n-train", "128"
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "Test Ablation",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/experiments/test_ablation.py",
            "args": [
                "--directions", "results/best_direction.pt",
                "--n-test", "100",
                "--load-in-8bit",
                "--skip-safety"
            ],
            "console": "integratedTerminal"
        }
    ]
}
```

Then press `F5` or use the Run menu.

## Running Jupyter Notebooks

1. Open `notebooks/analysis.ipynb`
2. VS Code will prompt to install Jupyter extension
3. Select your Python interpreter
4. Click "Run All" or run cells individually

## Project Structure in VS Code

```
refusal-directions/          # Root workspace folder
├── .vscode/                 # VS Code configuration (create this)
│   └── launch.json         # Run configurations
├── data/                   # Datasets
│   ├── harmful_instructions.json
│   └── harmless_instructions.json
├── src/                    # Library code
│   ├── __init__.py
│   ├── data.py
│   ├── directions.py
│   ├── evaluation.py
│   ├── interventions.py
│   └── model.py
├── experiments/            # Runnable scripts
│   ├── extract_directions.py
│   ├── test_ablation.py
│   ├── test_addition.py
│   └── weight_ortho.py
├── notebooks/             # Analysis notebooks
│   └── analysis.ipynb
├── results/              # Generated outputs (created on first run)
├── requirements.txt      # Dependencies
├── README.md            # Full documentation
├── QUICKSTART.md       # Quick start guide
└── run_pipeline.sh    # Automated pipeline
```

## Useful VS Code Shortcuts

- `Cmd/Ctrl + ~` - Toggle terminal
- `Cmd/Ctrl + Shift + P` - Command palette
- `Cmd/Ctrl + P` - Quick file open
- `F5` - Start debugging
- `Shift + Enter` - Run Jupyter cell

## Debugging

### Set Breakpoints

1. Click left of line number to set breakpoint
2. Press `F5` to start debugging
3. Use Debug Console to inspect variables

### Useful Debug Configurations

Add to `.vscode/launch.json`:

```json
{
    "name": "Debug Current File",
    "type": "python",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal"
}
```

## Common Issues

### Import Errors

Make sure you've:
1. Selected correct Python interpreter
2. Installed all requirements: `pip install -r requirements.txt`
3. Are in the project root directory

### GPU Not Detected

Check CUDA setup:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Should print `True`. If not, reinstall PyTorch with CUDA support.

### Module Not Found

Add workspace to Python path in `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/src"
    ]
}
```

## Workflow Tips

### Quick Testing

1. Use small sample sizes during development:
   ```bash
   python experiments/extract_directions.py --n-train 32 --n-val 16
   ```

2. Skip safety evaluation for faster iteration:
   ```bash
   python experiments/test_ablation.py --skip-safety
   ```

### Organizing Results

Create experiment folders:
```bash
mkdir -p results/exp1 results/exp2
python experiments/test_ablation.py --output results/exp1/ablation.json
```

### Tracking Changes

Use Git in VS Code:
1. `Cmd/Ctrl + Shift + G` - Open Source Control
2. Stage changes
3. Commit with meaningful messages

## Performance Monitoring

Monitor GPU usage:
```bash
# In separate terminal
watch -n 1 nvidia-smi
```

Or use VS Code terminal split view:
1. `Cmd/Ctrl + Shift + 5` - Split terminal
2. Run experiment in one, monitor in other

## Next Steps

1. Read [QUICKSTART.md](QUICKSTART.md) for usage examples
2. Read [README.md](README.md) for detailed documentation
3. Open `notebooks/analysis.ipynb` to explore results
4. Run `./run_pipeline.sh` to start experimenting!

Happy coding! 🚀
