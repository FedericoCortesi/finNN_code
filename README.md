# finNN_code: Financial Neural Network Training Pipeline

A neural network training pipeline for financial time series prediction using walk-forward cross-validation. This repository implements a framework for training and evaluating machine learning models on financial data with proper temporal validation and hyperparameter optimization.

## 🏗️ Project Structure

```
finNN_code/
├── src/                              # Main source code
│   ├── config/                       # Configuration files and types
│   │   ├── __init__.py
│   │   ├── config_types.py          # Configuration dataclasses
│   │   ├── default.yaml             # Default experiment configuration
│   │   ├── debug.yaml               # Debug configuration
│   │   └── search_debug.yaml        # Hyperparameter search debug config
│   ├── data/                        # Data files and analysis
│   │   ├── __init__.py
│   │   ├── data_analysis.ipynb      # Exploratory data analysis
│   │   ├── get_data.ipynb           # Data acquisition notebook
│   │   ├── permnos_info.csv         # S&P 500 company metadata
│   │   ├── permons_list.txt         # List of company identifiers
│   │   └── sp500_daily_data.parquet # S&P 500 daily price data
│   ├── hyperparams_search/         # Hyperparameter optimization
│   │   ├── __init__.py
│   │   └── search_utils.py          # Optuna-based hyperparameter search
│   ├── models/                      # Neural network architectures
│   │   ├── __init__.py
│   │   └── mlp.py                   # Multi-layer perceptron implementation
│   ├── pipeline/                    # Data processing and validation
# finNN_code — Financial Neural Network Training Pipeline

This repository is a compact but feature-complete training framework for financial time-series prediction. It focuses on correct temporal validation (walk-forward cross-validation), reproducible experiments, and easy comparisons against simple benchmark models.

The README below explains the repository layout, how to run experiments, and the benchmark folder so you can get started quickly.

**Short audience note:** this is aimed at a programming-savvy reader who may not be a machine learning expert. Commands and file paths are shown as copy-paste examples.

**Repository Structure (high level)**

```
`/` - project root
├── `src/`                         # Package sources (package-dir for setuptools)
│   ├── `config/`                  # YAML configs & typed config classes
│   ├── `data/`                    # Data, notebooks, small fixtures
│   ├── `hyperparams_search/`      # Optuna helper code
│   ├── `models/`                  # Neural net model implementations (e.g. `mlp.py`)
│   ├── `pipeline/`                # Preprocessing & walk-forward logic
│   ├── `price_prediction/`        # Experiments, benchmarks, visuals
│   ├── `training_routine/`        # Trainer and metric computations
│   └── `utils/`                   # Logging, paths, GPU checks, helpers
├── `logs/`                         # SLURM job logs (stdout/stderr)
├── `train_job.sh`                  # Example SLURM submission script
├── `pyproject.toml`                # Project metadata & minimal deps
├── `ssh_guide.md`                  # Tips for running on remote machines
└── `README.md`                     # This file
```

**What to expect in `src/price_prediction/benchmarks/`**
- **Purpose:** simple baselines (e.g. linear regressions) used to compare against neural models.
- **Files:** small notebooks and helper scripts such as `regressions.ipynb` that run straightforward models over the same walk-forward splits as the main experiments.
- **How to use:** run the notebook or call the helper functions to produce baseline metrics and plots that live alongside experiment outputs for direct comparison.

**Quick Start — run an experiment locally**

1. Create a Python environment (Python >= 3.12 is the project requirement):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the package in editable mode and dependencies. The project uses a minimal `pyproject.toml` (see `requires-python` and base deps). If you prefer `pip`:

```bash
pip install -e .
pip install -r requirements.txt  # if you keep one locally; otherwise install extras manually
```

3. Confirm GPU (optional):

```bash
python src/utils/gpu_test.py
```

4. Run an experiment using a YAML config in `src/config/` (examples: `default.yaml`, `debug.yaml`, `search_debug.yaml`):

```bash
python src/run_experiments.py --config src/config/default.yaml
```

- Use `debug.yaml` for quick/development runs:

```bash
python src/run_experiments.py --config src/config/debug.yaml
```

- To run a hyperparameter search (Optuna-backed):

```bash
python src/run_experiments.py --config src/config/search_debug.yaml
```

**How experiments are organized on disk**

When an experiment runs it creates a directory under `src/price_prediction/experiments/` with a timestamped name. The typical layout:

```
`src/price_prediction/experiments/exp_<id>_<timestamp>_<name>/`
├── `config_snapshot.json`       # Exact config used to run the experiment
├── `results.csv`                # Aggregated per-fold metrics
├── `trial_000/`                 # If hyperparameter search: each trial
│   └── `fold_000/`              # Per-fold subfolders
│       ├── `model_best.pth`     # Best checkpoint for that fold
│       └── `training_log.json`  # Loss/metrics per epoch
└── ...
```

This layout makes it straightforward to compare experiments, re-run with the same config, or load models for inspection.

**Benchmarks folder — quick guide**

- Location: `src/price_prediction/benchmarks/`
- Goal: run simple, interpretable baselines (linear regression, simple moving averages, etc.) over the same walk-forward splits.
- Typical workflow:
  - Reuse the preprocessing and walk-forward splits from `src/pipeline/` so baselines are directly comparable.
  - Open `src/price_prediction/benchmarks/regressions.ipynb` and run the cells to reproduce baseline metrics and plots.

**Walk-Forward Cross-Validation (brief)**

- The pipeline uses rolling (walk-forward) windows to avoid look-ahead bias. Each fold trains on past data and tests on future intervals.
- Common config keys (in `src/config/*.yaml`): `step`, `ratio_train`, `ratio_val`, `ratio_test`, `lags`, and `max_folds`.

Example (conceptual):

```yaml
walkforward:
  step: 251
  ratio_train: 3
  ratio_val: 1
  ratio_test: 1
  lags: 20
```

The code calculates absolute window sizes from these ratios and the available time span in your data.

**Running on a cluster (SLURM)**

- Use the provided `train_job.sh` script as a starting point:

```bash
sbatch train_job.sh
squeue -u $USER
tail -f logs/slurm_<job_id>.out
```

- `train_job.sh` contains environment setup and example `python src/run_experiments.py` commands. Customize resources and modules as needed.

**Configuration and customization**

- Configs are YAML files under `src/config/`. They contain sections like `experiment`, `data`, `model`, and `trainer`.
- To add a new model: implement it in `src/models/`, register it in the model factory (where models are created), and add corresponding `model.hparams` to your YAML.

**Development: notebooks and debugging**

- Interactive notebooks are in `src/` and `src/data/` (e.g. `src/debug.ipynb`, `src/data/data_analysis.ipynb`). Use these for exploratory work and visual checks.
- For quick iteration use `src/config/debug.yaml` and run smaller experiments locally.

**Minimal troubleshooting notes**

- GPU not visible: run `python src/utils/gpu_test.py` and check `nvidia-smi`.
- SLURM failures: inspect `logs/slurm_<job_id>.err` and ensure environment modules match the cluster configuration.
- NaNs in metrics: check data completeness, walk-forward window sizes, and `lags` in the config.

--

If you want, I can also:
- add a small `requirements.txt` derived from `pyproject.toml`,
- add a one-line example `make` or `invoke` task to run a debug experiment,
- or run a smoke test locally (if you want me to execute commands here).

Happy experimenting! 🚀
