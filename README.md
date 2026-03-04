# Financial Neural Network Training Pipeline

A neural network training pipeline for financial time series prediction using walk-forward cross-validation. This framework trains and evaluates neural networks (MLP, CNN, LSTM, Transformer) on S&P 500 data with proper temporal validation and Optuna-based hyperparameter optimization.

## 🚀 Quick Start

### Setup
```bash
# Create environment (Python >= 3.12)
python -m venv .venv
source .venv/bin/activate

# Install package
pip install -e .
```

### Run Experiments
```bash
# Interactive notebook (recommended for exploration)
jupyter notebook src/benchmarks/forecast.ipynb

# Or command line (recommended for production)
python src/run_experiments.py --config src/config/default.yaml
```

---

## 📁 Project Structure

```
finNN_code/
├── src/
│   ├── benchmarks/              # Evaluation & result visualization
│   │   └── forecast.ipynb       # ⭐ Main experiment notebook
│   ├── config/                  # Configuration & experiment definitions
│   │   ├── config_types.py      # Pydantic schema validation
│   │   ├── yaml_configs/        # Pre-configured experiments
│   │   ├── seed_runs/           # Multi-seed configurations
│   │   └── snr_runs/            # Signal-to-noise ratio experiments
│   ├── data/                    # Datasets & data processing
│   │   ├── sp500_daily_data.parquet
│   │   ├── get_data.ipynb
│   │   └── data_analysis.ipynb
│   ├── financial_tests/         # Portfolio analysis & empirical tests
│   ├── hyperparams_search/      # Optuna-based hyperparameter optimization
│   ├── models/                  # Neural network architectures
│   │   ├── mlp.py               # Multi-layer perceptron
│   │   ├── lstm.py              # LSTM with stacked layers
│   │   ├── simplecnn.py         # 1D Convolutional network
│   │   └── transformer.py       # Transformer with attention
│   ├── pipeline/                # Data processing & walk-forward CV
│   │   ├── walkforward.py       # Temporal cross-validation splits
│   │   ├── preprocessing.py     # Feature engineering & scaling
│   │   └── pipeline_utils.py    # Utilities (scaling, noise injection)
│   ├── training_routine/        # Training loop & metrics
│   │   ├── trainer.py           # GPU-optimized training
│   │   ├── metrics.py           # Loss & evaluation metrics
│   │   └── training_utils.py    # Early stopping, checkpointing
│   ├── utils/                   # Logging, paths, GPU management
│   │   ├── logging_utils.py     # Experiment tracking & results
│   │   ├── custom_formatter.py  # Colored console logging
│   │   ├── paths.py             # Centralized path definitions
│   │   ├── gpu_test.py          # GPU diagnostics
│   │   └── random_setup.py      # Reproducibility settings
│   ├── price_prediction/        # Price prediction experiments
│   ├── volatility/              # Volatility prediction experiments
│   └── run_experiments.py       # CLI entry point
├── logs/                        # SLURM job logs
├── train_job.slurm              # HPC submission script
├── pyproject.toml               # Dependencies
├── ssh_guide.md                 # Remote setup guide
└── README.md                    # This file
```

**📖 For detailed documentation, read README files in each `src/*/` directory:**
- **`src/benchmarks/README.md`** – Benchmarking, evaluation, and result visualization
- **`src/config/README.md`** – Configuration system, YAML schemas, and experiment setup
- **`src/financial_tests/README.md`** – Portfolio construction and empirical analysis
- **`src/hyperparams_search/README.md`** – Optuna hyperparameter optimization and search strategies
- **`src/models/README.md`** – Neural network architectures (MLP, LSTM, CNN, Transformer)
- **`src/pipeline/README.md`** – Walk-forward validation, data preprocessing, and scaling
- **`src/training_routine/README.md`** – Training loop, optimizers, and GPU optimization
- **`src/utils/README.md`** – Logging, experiment tracking, paths, and utilities

---

## 🏗️ Architecture Overview

### Four Neural Network Models

| Model | Type | Best For |
|-------|------|----------|
| **MLP** | Feedforward | Fast baseline, parameter search |
| **CNN** | 1D Convolution | Local temporal patterns, efficiency |
| **LSTM** | Recurrent | Long-term dependencies, sequences |
| **Transformer** | Attention | Full context, parallel computation |

**See `src/models/README.md` for architecture specifications and hyperparameters.**

### Three Optimizers

- **Adam** – Adaptive learning rates (default, fast)
- **SGD** – Classical with momentum option
- **Muon** – Second-order approximation (often best performance)

### Walk-Forward Cross-Validation

Temporal cross-validation preventing look-ahead bias:
```
Fold 0: Train (t=0-753) → Val (t=754-1004) → Test (t=1005-1255)
Fold 1: Train (t=252-1004) → Val (t=1005-1255) → Test (t=1256-1506)
Fold 2: Train (t=504-1255) → Val (t=1256-1506) → Test (t=1507-1757)
...
```

**See `src/pipeline/README.md` for detailed walk-forward mechanics.**

---

## ⚙️ Configuration System

Experiments defined in YAML under `src/config/yaml_configs/`:

```yaml
model:
  name: lstm
  hparams:
    lstm_hidden_sizes: [256, 128]
    lstm_dropout: 0.1
    bidirectional: false

trainer:
  hparams:
    epochs: 100
    batch_size: 512
    optimizer_type: muon
    lr: 0.001
    weight_decay: 0.0001

walkforward:
  lags: 20              # Feature window length
  ratio_train: 3        # Relative duration (train:val:test)
  ratio_val: 1
  ratio_test: 1
  step: 251             # Trading days per period
  scale: true
  scale_type: standard
```

**Configuration Options:**
- `yaml_configs/` – Pre-configured architecture + optimizer combinations
- `seed_runs/` – 14 random seeds per config for robustness testing
- `snr_runs/` – Signal-to-noise ratio experiments

**See `src/config/README.md` for full schema and configuration details.**

---

## 📊 Running Experiments

### Interactive Notebook
```bash
jupyter notebook src/benchmarks/forecast.ipynb
```
- Load data, train models, generate predictions, visualize results
- Best for exploration and debugging

### Command Line
```bash
# Debug run (fast, small dataset, 1 fold)
python src/run_experiments.py --config src/config/debug.yaml

# Full experiment (complete dataset, all folds)
python src/run_experiments.py --config src/config/default.yaml

# Hyperparameter search (Optuna optimization)
python src/run_experiments.py --config src/config/search_debug.yaml
```

### Output Structure
```
src/volatility/experiments/
└── exp_035_lstm_100_muon/           # Experiment ID + name
    ├── trial_search_best/           # Best trial from search
    │   ├── results.csv              # Metrics per fold
    │   ├── config_snapshot.json     # Config + environment
    │   ├── fold_000/
    │   │   ├── model_best.pt        # Best checkpoint
    │   │   ├── training_history.json
    │   │   └── predictions.csv
    │   └── fold_001/, fold_002/, ...
    └── trial_20240304_143022/       # Named trials
        └── ...
```

**See `src/training_routine/README.md` for training details and metrics.**

---

## 🔬 Analysis & Evaluation

### Benchmarking
Compare models against OLS and LASSO baselines:
```bash
jupyter notebook src/benchmarks/compare_runs.ipynb
```

### Portfolio Analysis
Construct volatility-managed portfolios and compute Sharpe ratios, returns, turnover:
```bash
python src/financial_tests/empiricals.py
```

**See `src/benchmarks/README.md` and `src/financial_tests/README.md` for details.**

---

## 🖥️ GPU & HPC

### Verify GPU
```bash
python -c "from utils.gpu_test import gpu_test; gpu_test()"
```

### SLURM Submission
```bash
sbatch train_job.slurm
squeue -u $USER
tail -f logs/slurm_*.err
```

**See `src/utils/README.md` for GPU diagnostics and troubleshooting.**

---

## 🔧 Development

### Quick Iteration
1. Edit code in `src/models/`, `src/pipeline/`, etc.
2. Use `debug.yaml` for fast testing (1 fold, fewer epochs)
3. Run: `python src/run_experiments.py --config src/config/debug.yaml`

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | `pip install -e .` |
| CUDA out of memory | Reduce `batch_size` in config |
| GPU not found | `python -c "from utils.gpu_test import gpu_test; gpu_test()"` |
| Data missing | Ensure `sp500_daily_data.parquet` exists |

**See `src/utils/README.md` for logging, reproducibility, and paths.**

---

## 📚 Key Concepts

### Walk-Forward Validation
Prevents information leakage by using only past data for training. Each fold's test set is always in the future relative to training. Essential for time series.

**Learn more:** `src/pipeline/README.md`

### Hyperparameter Search
Optuna-based optimization with MedianPruner for early stopping. Samples from search specifications in config, evaluates on validation set, automatically saves best trial.

**Learn more:** `src/hyperparams_search/README.md`

### Experiment Logging
Automatic folder structure with config snapshots, results CSV, and metadata JSONL. All metrics logged for easy aggregation and analysis.

**Learn more:** `src/utils/README.md`

### Training Features
- GPU-optimized with mixed precision (bfloat16)
- Early stopping with checkpoint restoration
- Transfer learning via model initialization
- Per-layer hyperparameter search

**Learn more:** `src/training_routine/README.md`

---

## 📦 Dependencies

Key packages in `pyproject.toml`:
- **PyTorch** – Neural networks & GPU training
- **Optuna** – Hyperparameter optimization
- **scikit-learn** – Preprocessing, baseline models
- **pandas, numpy** – Data manipulation
- **statsmodels** – Statistical tests & models
- **PyYAML** – Configuration parsing
- **matplotlib, seaborn** – Visualization

---

## 📈 Typical Workflow

### 1. Data Preparation
```bash
jupyter notebook src/data/get_data.ipynb
jupyter notebook src/data/data_analysis.ipynb
```

### 2. Model Development
```bash
python src/run_experiments.py --config src/config/debug.yaml
```

### 3. Hyperparameter Tuning
```bash
python src/run_experiments.py --config src/config/search_debug.yaml
```

### 4. Final Training
```bash
python src/run_experiments.py --config src/config/default.yaml
```

### 5. Analysis & Comparison
```bash
jupyter notebook src/benchmarks/compare_runs.ipynb
python src/financial_tests/empiricals.py
```

---

## 📝 Adding Models

1. Create in `src/models/my_model.py`
2. Use `@register_model("model_name")` decorator
3. Update `config_types.py` with model config
4. Add to YAML configs

---

## 🤝 Contributing

1. Branch: `git checkout -b feature/my-feature`
2. Code in appropriate `src/*/` directory
3. Test: `python src/run_experiments.py --config src/config/debug.yaml`
4. Commit & push

---

**For detailed documentation on any component, read the README in its `src/*/` directory.** 🚀

For remote machine setup, see `ssh_guide.md`.
