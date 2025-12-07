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
│   │   ├── config_types.py
│   │   ├── default.yaml
│   │   ├── debug.yaml
│   │   └── search_debug.yaml
│   ├── data/                        # Data files and analysis
│   │   ├── data_analysis.ipynb
│   │   ├── get_data.ipynb
│   │   ├── permnos_info.csv
│   │   ├── permons_list.txt
│   │   └── sp500_daily_data.parquet
│   ├── hyperparams_search/          # Hyperparameter optimization (Optuna)
│   │   └── search_utils.py
│   ├── models/                      # Neural network architectures
│   │   └── mlp.py
│   ├── pipeline/                    # Data processing & walk-forward logic
│   ├── price_prediction/            # Experiment outputs & benchmarks
│   ├── training_routine/            # Trainer and metric computations
│   ├── utils/                       # Logging, GPU checks, helpers
│   └── volatility/                  # Volatility analysis
├── logs/                            # SLURM job logs
├── train_job.slurm                  # Example SLURM submission script
├── pyproject.toml                   # Project metadata & dependencies
├── ssh_guide.md                     # Remote machine setup guide
└── README.md                        # This file
```

## ⚙️ Configuration

Experiment settings are defined in YAML files under `src/config/`:

- **`default.yaml`** — Production experiment configuration
- **`debug.yaml`** — Quick debug runs with smaller datasets
- **`search_debug.yaml`** — Hyperparameter search configuration

Each config defines:
- `experiment` — experiment name and metadata
- `data` — data source and preprocessing
- `model` — neural network architecture and hyperparameters
- `trainer` — training loop settings (epochs, batch size, learning rate)
- `walkforward` — walk-forward cross-validation parameters

### Walk-Forward Cross-Validation

The pipeline uses rolling (walk-forward) windows to avoid look-ahead bias:
- Trains on past data
- Validates on intermediate periods
- Tests on future intervals

## 📊 Running Experiments from Command Line

If you prefer command-line execution instead of the notebook:

```bash
# Quick debug run
python src/run_experiments.py --config src/config/debug.yaml

# Full experiment
python src/run_experiments.py --config src/config/default.yaml

# Hyperparameter search
python src/run_experiments.py --config src/config/search_debug.yaml
```

Experiment results are saved to `src/price_prediction/experiments/` with timestamped folders containing:
- `config_snapshot.json` — Exact config used
- `results.csv` — Aggregated metrics
- `trial_*/fold_*/model_best.pth` — Trained model checkpoints
- `trial_*/fold_*/training_log.json` — Loss/metric logs per epoch

## 🖥️ GPU Support

To verify GPU availability:
```bash
python src/utils/gpu_test.py
```

## 🚀 Running on HPC Clusters (SLURM)

Use the provided `train_job.slurm` script as a starting point:

```bash
sbatch train_job.slurm
squeue -u $USER
tail -f logs/slurm_<job_id>.err
```

Customize resources and module loads as needed for your cluster.

## 🔧 Development & Debugging

- Interactive notebooks for exploration: `src/debug.ipynb`, `src/data/data_analysis.ipynb`
- Use `debug.yaml` for faster iteration with smaller configs
- Check GPU: `python src/utils/gpu_test.py`
- Inspect logs: `tail -f logs/slurm_<job_id>.err`

## 📦 Dependencies

Key packages (see `pyproject.toml`):
- **PyTorch** — Neural networks
- **TensorFlow** — Deep learning
- **scikit-learn** — Machine learning utilities
- **Optuna** — Hyperparameter optimization
- **pandas, numpy** — Data processing
- **statsmodels** — Statistical modeling
- **tqdm** — Progress bars
- **PyYAML** — Configuration files

---

**Happy experimenting!** 🚀

For setup on remote machines, see `ssh_guide.md`.
