# Hyperparameters Search Directory

This directory contains utilities and tools for automated hyperparameter optimization using Optuna. It provides functionality for sampling hyperparameters, managing Optuna trials, and integrating with the training pipeline.

## 📁 Contents

### Core Utilities

- **`search_utils.py`** – Hyperparameter search utilities
  - Configuration parsing and conversion between dataclass and dict formats
  - Optuna trial objective function for model training
  - Hyperparameter sampling from search specifications
  - Trial callback management and early stopping
  - Integration with the training routine and experiment logger

- **`__init__.py`** – Package initialization

## 🔧 Key Functions

### Configuration Parsing

**`to_dict(cfg)`**
- Converts Pydantic dataclass configs to dictionaries
- Handles both dataclass and dict-like objects
- Creates deep copies to avoid mutation

**`_spec_type(spec)`**
- Identifies hyperparameter specification type
- Supports: float, int, categorical
- Works with both dict and dataclass formats

### Hyperparameter Sampling

**`_suggest_from_spec(trial, name, spec)`**
- Samples hyperparameters from Optuna trials
- Supports different parameter types:
  - **Float**: Continuous parameters with optional log scale
  - **Int**: Integer parameters with bounds
  - **Categorical**: Discrete choices from a list
- Uses unique parameter names for proper Optuna tracking

**`sample_hparams_into_cfg(base_cfg, trial)`**
- Main function for sampling hyperparameters
- Reads `model.search` and `trainer.search` from config
- Creates new configuration with sampled values
- Handles variable layer counts for architecture search
- Features:
  - Global model choices (activation, dropout) sampled once
  - Per-layer width sampling with layer-specific parameter names
  - Fallback to existing values if not searched
  - Preserves non-searched configuration elements

**Returns**: Dictionary with sampled hyperparameters merged into config

### Optimization Callbacks

**`_make_report_cb(trial, mode="min", patience=10)`**
- Creates callback for trial reporting during training
- Integrates with Optuna's MedianPruner for early stopping
- Monitors validation metrics across epochs
- Prunes unpromising trials early to save compute

### Objective Function

**`optuna_objective(trial, config, fold_data, n_fold, input_shp, output_shp)`**
- Main Optuna objective function
- Workflow:
  1. Sample hyperparameters from trial
  2. Create new config with sampled values
  3. Initialize experiment logger and trainer
  4. Build fresh model for trial
  5. Set up reporting callback with early stopping
  6. Train and evaluate on fold data
  7. Return optimization metric (val_loss, val_metric, etc.)
- Returns: Scalar metric value for Optuna to optimize

## 📋 Configuration Search Specification Format

Hyperparameter search ranges are defined in YAML config under `model.search` and `trainer.search`:

```yaml
model:
  search:
    activation:
      type: cat
      choices: [relu, gelu, tanh]
    dropout_rate:
      type: float
      low: 0.0
      high: 0.5
    n_layers:
      type: int
      low: 1
      high: 4
    width:
      type: int
      low: 64
      high: 512

trainer:
  search:
    lr:
      type: float
      low: 0.0001
      high: 0.1
      log: true
    batch_size:
      type: int
      low: 32
      high: 512
```

## 🚀 Typical Usage

### 1. Define Search Space in Config
```yaml
model:
  hparams:
    hidden_sizes: [128, 64]
    dropout_rate: 0.1
  search:
    hidden_sizes:
      type: cat
      choices: [[128, 64], [128, 128]]
    dropout_rate:
      type: float
      low: 0.0
      high: 0.5

trainer:
  hparams:
  ...
  search:
    lr:
      type: float
      low: 0.00001
      high: 0.1
      log: true
```

### 2. Create Optuna Study
```python
from hyperparams_search.search_utils import optuna_objective

study = optuna.create_study(
    direction='minimize',  # or 'maximize'
    pruner=optuna.pruners.MedianPruner()
)

study.optimize(
    lambda trial: optuna_objective(
        trial,
        config=cfg,
        fold_data=fold_data,
        n_fold=0,
        input_shp=(T, D),
        output_shp=(1,)
    ),
    n_trials=100
)
```

### 3. Access Best Trial
```python
best_trial = study.best_trial
best_params = best_trial.params
print(f"Best value: {best_trial.value}")
print(f"Best params: {best_params}")
```

## 🎯 Optimization Strategy

- **Sampler**: Optuna's default TPE (Tree-structured Parzen Estimator)
- **Pruner**: MedianPruner for early stopping of bad trials
- **Patience**: Configurable early stopping patience via `optuna_patience` in trainer config
- **Direction**: Min/max defined by `experiment.mode` in config
- **Metric**: Monitored via `experiment.monitor` (e.g., `val_loss`)

## 🔗 Integration Points

- **Config System**: Works with `AppConfig` dataclasses and YAML configs
- **Training**: Integrates with `Trainer` class for model training
- **Models**: Compatible with all model architectures (MLP, CNN, LSTM, Transformer)
- **Logging**: Uses `ExperimentLogger` for trial tracking
- **Fold Validation**: Designed for walk-forward cross-validation scenarios

## ⚠️ Important Notes

- Each trial gets a **fresh model and trainer** to avoid state carryover
- Layer-specific width parameters enable learning architecture patterns across depths
- Early stopping via pruning saves significant computation
- Trial naming convention: `{fold_idx:03d}_{trial_number:03d}`
- Configuration is validated before sampling to catch errors early
- Sampled hyperparameters are logged per trial for reproducibility

## 📊 Performance Considerations

- **Variable Layer Counts**: Per-layer width parameters scale logarithmically with trials
- **Search Space Size**: Larger search spaces require more trials for convergence
- **Early Stopping**: Patience setting balances accuracy vs. compute cost
- **Fold-Level Optimization**: Each fold can have its own optimized hyperparameters

## 🔄 Workflow Example

```
Config with search specs
    ↓
Optuna Study created
    ↓
For each trial:
  - Sample hyperparameters
  - Train model on fold
  - Report metrics per epoch
  - Prune if bad performance
  ↓
Best trial selected
    ↓
Use best hyperparameters for full training
```
