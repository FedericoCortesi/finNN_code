# finNN_code: Financial Neural Network Training Pipeline

A neural network training pipeline for financial time series prediction using walk-forward cross-validation. This repository implements a framework for training and evaluating machine learning models on financial data with proper temporal validation and hyperparameter optimization.

## 🚀 Quick Start — Run Experiments

**To run experiments and generate forecasts:**

Open and run the notebook at:
```
src/benchmarks/forecast.ipynb
```

This Jupyter notebook contains the complete experimental pipeline and will generate predictions on financial time series data.

### Setup

1. **Create a Python environment (Python >= 3.12):**

```bash
python -m venv .venv
source .venv/bin/activate
```

2. **Install the package with dependencies:**

```bash
pip install -e .
```

3. **Run the forecast notebook:**
   - Open `src/benchmarks/forecast.ipynb` in Jupyter or VS Code
   - Run all cells to execute the full pipeline

## 🏗️ Project Structure

```
finNN_code/
├── src/
│   ├── benchmarks/
│   │   └── forecast.ipynb           # ⭐ Main notebook — run experiments here
│   ├── config/                      # Configuration files and types
│   │   ├── config_types.py          # Pydantic models for config validation
│   │   ├── default.yaml             # Production experiment settings
│   │   ├── debug.yaml               # Quick debug configuration
│   │   └── search_debug.yaml        # Hyperparameter search settings
│   ├── data/                        # Data files and analysis
│   │   ├── data_analysis.ipynb      # Exploratory data analysis
│   │   ├── get_data.ipynb           # Data fetching and preprocessing
│   │   ├── permnos_info.csv         # Stock permno metadata
│   │   ├── permons_list.txt         # List of stock permnos
│   │   └── sp500_daily_data.parquet # S&P 500 daily OHLCV data
│   ├── hyperparams_search/          # Hyperparameter optimization (Optuna)
│   │   └── search_utils.py          # Search space & trial management
│   ├── models/                      # Neural network architectures
│   │   └── mlp.py                   # Multi-layer perceptron model
│   ├── pipeline/                    # Data processing & walk-forward logic
│   │   ├── __init__.py
│   │   ├── data_loader.py           # DataLoader wrappers
│   │   ├── preprocessing.py         # Feature engineering & normalization
│   │   └── walkforward.py           # Walk-forward cross-validation splits
│   ├── training_routine/            # Trainer and metric computations
│   │   ├── __init__.py
│   │   ├── trainer.py               # Main training loop
│   │   ├── metrics.py               # Loss functions & evaluation metrics
│   │   └── callbacks.py             # Early stopping, checkpointing
│   ├── price_prediction/            # Experiment outputs & results
│   │   └── experiments/             # Timestamped experiment folders
│   ├── utils/                       # Logging, GPU checks, helpers
│   │   ├── __init__.py
│   │   ├── gpu_test.py              # GPU availability checker
│   │   ├── logger.py                # Logging configuration
│   │   └── helpers.py               # Utility functions
│   ├── volatility/                  # Volatility analysis
│   │   └── volatility_analysis.py   # Volatility metrics & features
│   ├── run_experiments.py           # CLI entry point for experiments
│   └── __init__.py
├── logs/                            # SLURM job logs
├── train_job.slurm                  # Example SLURM submission script
├── pyproject.toml                   # Project metadata & dependencies
├── ssh_guide.md                     # Remote machine setup guide
└── README.md                        # This file
```

---

## 📁 Detailed File Documentation

### **Configuration** (`src/config/`)

**`config_types.py`**
- Pydantic data classes defining the schema for all configuration parameters
- Ensures type safety and validation when loading YAML configs
- Usage: Automatically loaded by the configuration manager

**`default.yaml`**
- Production-ready experiment configuration
- Contains full dataset, larger batch sizes, extended training
- Use this for final experiments and benchmarks

**`debug.yaml`**
- Lightweight configuration for quick iteration
- Smaller dataset, fewer epochs, smaller batch sizes
- Use this during development to test code changes quickly

**`search_debug.yaml`**
- Configuration for Optuna hyperparameter search
- Defines search space, number of trials, and optimization objectives
- Use this to find optimal hyperparameters

### **Data** (`src/data/`)

**`get_data.ipynb`**
- Fetches raw financial data from external sources
- Cleans, validates, and preprocesses OHLCV (Open, High, Low, Close, Volume) data
- Outputs: `sp500_daily_data.parquet` with standardized format

**`data_analysis.ipynb`**
- Exploratory data analysis and visualization
- Computes statistics, correlations, and temporal patterns
- Generates plots and summary reports

**`sp500_daily_data.parquet`**
- Main dataset: S&P 500 daily OHLCV price data
- Format: Parquet (columnar, efficient compression)
- Used by all training pipelines

**`permnos_info.csv`**
- Metadata for stock tickers (permno = permanent number ID)
- Columns: permno, ticker, company_name, sector, etc.
- Used for filtering and tracking stocks

### **Models** (`src/models/`)

**`mlp.py`**
- Multi-layer perceptron (feedforward neural network)
- Configurable hidden layers, activation functions, dropout
- Usage example:
```python
from src.models.mlp import MLP
model = MLP(input_dim=50, hidden_dims=[128, 64], output_dim=1)
```

### **Pipeline** (`src/pipeline/`)

**`preprocessing.py`**
- Feature engineering (technical indicators: moving averages, RSI, etc.)
- Normalization/scaling using StandardScaler or MinMaxScaler
- Missing value handling and outlier detection

**`walkforward.py`**
- Implements walk-forward (rolling) cross-validation splits
- Prevents look-ahead bias by ensuring train < validation < test
- Returns temporal split indices for each fold

**`data_loader.py`**
- PyTorch DataLoader wrappers
- Handles batching, shuffling, and data feeding during training
- Supports different sampling strategies

Usage example:
```python
from src.pipeline.walkforward import WalkForwardSplitter
splitter = WalkForwardSplitter(n_splits=5, train_size=0.6, val_size=0.2)
for train_idx, val_idx, test_idx in splitter.split(data):
    train_data = data[train_idx]
    val_data = data[val_idx]
    test_data = data[test_idx]
```

### **Training Routine** (`src/training_routine/`)

**`trainer.py`**
- Main training loop: forward pass, loss computation, backward pass, optimization
- Handles device placement (CPU/GPU), mixed precision training
- Implements epoch-based training with validation checks

**`metrics.py`**
- Loss functions: MSE, MAE, Huber loss, etc.
- Evaluation metrics: RMSE, MAPE, Sharpe ratio, etc.
- Computes financial-specific metrics (directional accuracy, profit/loss)

**`callbacks.py`**
- Early stopping: stops training if validation loss plateaus
- Model checkpointing: saves best model state
- Learning rate scheduling

Usage example:
```python
from src.training_routine.trainer import Trainer
trainer = Trainer(model, optimizer, loss_fn, device='cuda')
trainer.train(train_loader, val_loader, epochs=100)
best_model = trainer.best_model
```

### **Hyperparameter Search** (`src/hyperparams_search/`)

**`search_utils.py`**
- Defines Optuna search space (learning rate, hidden dims, dropout, etc.)
- Objective function for optimization
- Trial management and result aggregation

Usage:
```bash
python src/run_experiments.py --config src/config/search_debug.yaml
```

### **Utilities** (`src/utils/`)

**`gpu_test.py`**
- Detects CUDA availability and prints GPU info
- Useful for debugging device placement issues

**`logger.py`**
- Configures logging for console and file output
- Sets log levels and formatters

**`helpers.py`**
- Common utility functions (path handling, file I/O, etc.)

### **Volatility** (`src/volatility/`)

**`volatility_analysis.py`**
- Computes rolling volatility metrics
- Generates volatility-based features for the model
- Useful for risk-adjusted analysis

---

## ⚙️ Configuration System

Experiment settings are defined in YAML files under `src/config/`:

- **`default.yaml`** — Production experiment configuration
- **`debug.yaml`** — Quick debug runs with smaller datasets
- **`search_debug.yaml`** — Hyperparameter search configuration

Each config defines:
```yaml
experiment:
  name: "experiment_name"
  description: "what this experiment does"

data:
  source: "sp500_daily_data.parquet"
  lookback_window: 50  # days of history
  target_horizon: 1    # days ahead to predict

model:
  type: "mlp"
  input_dim: 50
  hidden_dims: [128, 64]
  output_dim: 1
  dropout: 0.2

trainer:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"

walkforward:
  n_splits: 5
  train_size: 0.6
  val_size: 0.2
  test_size: 0.2
```

### Walk-Forward Cross-Validation

The pipeline uses rolling (walk-forward) windows to avoid look-ahead bias:
- Trains on past data
- Validates on intermediate periods
- Tests on future intervals
- Repeats across multiple overlapping time windows

---

## 📊 Running Experiments

### **Option 1: Jupyter Notebook (Recommended for Exploration)**

```bash
jupyter notebook src/benchmarks/forecast.ipynb
# Or in VS Code: Open notebook and run cells
```

Advantages:
- Interactive debugging
- Visualize results inline
- Easy hyperparameter tweaking

### **Option 2: Command Line (Recommended for Production)**

```bash
# Quick debug run
python src/run_experiments.py --config src/config/debug.yaml

# Full experiment
python src/run_experiments.py --config src/config/default.yaml

# Hyperparameter search
python src/run_experiments.py --config src/config/search_debug.yaml
```

Output structure:
```
src/price_prediction/experiments/
└── YYYYMMDD_HHMMSS_experiment_name/
    ├── config_snapshot.json          # Exact config used
    ├── results.csv                   # Aggregated metrics across all folds
    ├── trial_0/
    │   ├── fold_0/
    │   │   ├── model_best.pth        # Best model checkpoint
    │   │   ├── training_log.json     # Loss/metric history
    │   │   └── predictions.csv       # Predictions on test set
    │   └── fold_1/
    │       └── ...
    └── trial_1/
        └── ...
```

---

## 🖥️ GPU Support

### Verify GPU Availability
```bash
python src/utils/gpu_test.py
```

Expected output:
```
GPU available: True
Device: NVIDIA A100
CUDA Version: 12.0
```

The trainer automatically uses GPU if available, falls back to CPU otherwise.

---

## 🚀 Running on HPC Clusters (SLURM)

### Basic SLURM Submission

```bash
sbatch train_job.slurm
```

### Monitor Job

```bash
squeue -u $USER              # List your jobs
scancel <job_id>             # Cancel a job
tail -f logs/slurm_<job_id>.err  # Watch logs
```

### Customize `train_job.slurm`

```bash
# Edit resource requests
#SBATCH --gpus=1              # Number of GPUs
#SBATCH --time=04:00:00       # Max runtime (HH:MM:SS)
#SBATCH --mem=32G             # Memory
#SBATCH --cpus-per-task=8     # CPU cores
```

---

## 🔧 Development & Debugging

### Quick Iteration Workflow

1. **Make code changes** to `src/models/`, `src/pipeline/`, etc.
2. **Use `debug.yaml`** for fast testing (smaller dataset, fewer epochs)
3. **Run from command line**:
   ```bash
   python src/run_experiments.py --config src/config/debug.yaml
   ```
4. **Check logs**:
   ```bash
   tail -f logs/training_<timestamp>.log
   ```

### Interactive Development

1. **Start Jupyter**:
   ```bash
   jupyter notebook src/
   ```
2. **Open `src/debug.ipynb`** for iterative development
3. **Import and test modules**:
   ```python
   from src.models.mlp import MLP
   from src.pipeline.preprocessing import FeatureEngineer
   model = MLP(input_dim=50, hidden_dims=[64], output_dim=1)
   ```

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -e .` to install package in editable mode |
| CUDA out of memory | Reduce `batch_size` in config or use `debug.yaml` |
| GPU not detected | Run `python src/utils/gpu_test.py` to diagnose |
| Data file not found | Ensure `sp500_daily_data.parquet` exists in `src/data/` |

---

## 📦 Dependencies

Key packages (see `pyproject.toml`):
- **PyTorch** — Neural networks & training
- **TensorFlow** — Alternative deep learning framework
- **scikit-learn** — Machine learning utilities, preprocessing
- **Optuna** — Hyperparameter optimization
- **pandas, numpy** — Data processing & numerical computing
- **statsmodels** — Statistical modeling
- **tqdm** — Progress bars
- **PyYAML** — Configuration file parsing
- **matplotlib, seaborn** — Visualization

---

## 📈 Typical Workflow

### 1. Data Preparation
```bash
# Fetch and explore data
jupyter notebook src/data/get_data.ipynb
jupyter notebook src/data/data_analysis.ipynb
```

### 2. Model Development
```bash
# Test model in debug mode
python src/run_experiments.py --config src/config/debug.yaml
```

### 3. Hyperparameter Tuning
```bash
# Run Optuna search
python src/run_experiments.py --config src/config/search_debug.yaml
# Update best hyperparameters in default.yaml
```

### 4. Final Evaluation
```bash
# Run full experiment
python src/run_experiments.py --config src/config/default.yaml
```

### 5. Analysis & Reporting
```bash
# Results saved to src/price_prediction/experiments/
# Load and visualize results in Jupyter
jupyter notebook src/benchmarks/forecast.ipynb
```

---

## 📝 Adding Your Own Models

1. **Create a new model file** in `src/models/`:
   ```python
   # src/models/lstm.py
   import torch.nn as nn
   
   class LSTM(nn.Module):
       def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
           super().__init__()
           self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
           self.fc = nn.Linear(hidden_dim, output_dim)
       
       def forward(self, x):
           out, _ = self.lstm(x)
           return self.fc(out[:, -1, :])
   ```

2. **Register in config** (`src/config/default.yaml`):
   ```yaml
   model:
     type: "lstm"
     input_dim: 50
     hidden_dim: 64
     num_layers: 2
     output_dim: 1
   ```

3. **Update config_types.py** to include your model config

---

## 🤝 Contributing

1. Create a new branch: `git checkout -b feature/my-feature`
2. Make changes following the project structure
3. Test with `debug.yaml`: `python src/run_experiments.py --config src/config/debug.yaml`
4. Commit and push: `git push origin feature/my-feature`

---

**Happy experimenting!** 🚀

For setup on remote machines, see `ssh_guide.md`.
