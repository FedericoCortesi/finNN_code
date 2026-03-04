# Training Routine Directory

This directory contains the training loop, optimization, early stopping, and evaluation metrics for neural network models. It provides GPU-optimized training with flexible loss functions, multiple optimizers, and comprehensive metric computation.

## 📁 Contents

### Core Modules

- **`trainer.py`** – Main training class and orchestrator
  - Model compilation and initialization
  - Training loop with batching and GPU optimization
  - Evaluation and metrics computation
  - Checkpointing and early stopping
  - Support for hyperparameter search (Optuna integration)

- **`metrics.py`** – Loss functions and evaluation metrics
  - Standard metrics: MSE, MAE, Directional Accuracy
  - Custom QLIKE loss for volatility forecasts
  - Efficient PyTorch implementations

- **`training_utils.py`** – Utility functions
  - Early stopping logic
  - CPU/GPU state management for checkpoints
  - Optimizer state handling

- **`__init__.py`** – Package initialization (auto-created)

---

## 🏃 Trainer Class

### Initialization

```python
from training_routine.trainer import Trainer
from utils.logging_utils import ExperimentLogger

logger = ExperimentLogger(cfg)
trainer = Trainer(cfg, logger)
```

**Requirements:**
- `cfg`: `AppConfig` object with model/trainer/experiment settings
- `logger`: `ExperimentLogger` for tracking experiment results
- **GPU Required**: Trainer enforces CUDA availability

### Main Methods

#### `compile(model: nn.Module)`
Prepares model for training with optimizer and loss function.

**Features:**
- Loads initialization checkpoint if specified in config
- Moves model to GPU
- Creates optimizer (Adam, SGD, Muon)
- Sets up loss function (MSE, MAE, QLIKE)
- Optional torch.compile for inference optimization

**Initialization from Checkpoint:**
```yaml
trainer:
  hparams:
    initialization: exp_035_mlp_100_muon/trial_search_best/fold_000
```

Loads model weights from a previous experiment for transfer learning.

#### `fit_eval_fold(model, fold_data, fold=0, trial=None, merge_train_val=False, report_cb=None)`
Complete training and evaluation on a single fold.

**Input:**
- `fold_data`: Tuple of (Xtr, ytr, Xv, yv, Xte, yte, ...) from WFCVGenerator
- `fold`: Fold index for logging
- `trial`: Optuna trial (optional, for hyperparameter search)
- `merge_train_val`: Whether to merge train+val for final test evaluation
- `report_cb`: Callback for epoch-wise reporting (e.g., Optuna pruning)

**Output:**
- `(train_metrics, val_metrics, test_metrics)` - Dictionaries of computed metrics

**Workflow:**
1. Load/prepare data as tensors
2. Create training loop
3. For each epoch:
   - Train on batches
   - Evaluate on validation (if `merge_train_val=False`)
   - Check early stopping criteria
   - Report to callback (Optuna)
4. Load best checkpoint
5. Evaluate on test set
6. Return metrics

---

## 🎯 Optimizers

### Supported Optimizers

| Optimizer | Config | Parameters | Use Case |
|-----------|--------|------------|----------|
| **Adam** | `adam` | `lr`, `weight_decay` | Default, fast convergence |
| **SGD** | `sgd` | `lr`, `weight_decay`, `momentum` | Classic, sometimes better generalization |
| **Muon** | `muon` | `lr`, `weight_decay` | Second-order approximation, often best |

### Optimizer Setup

Weight decay applied selectively:
- **With decay**: 2D weight matrices (Linear, Conv layers)
- **Without decay**: Biases, 1D normalization parameters

```python
# From trainer.compile()
self.optimizer = optim.Adam(
    [{"params": decay, "weight_decay": weight_decay},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=lr,
    fused=True  # Faster on modern GPUs
)
```

---

## 💥 Loss Functions

### Available Losses

#### MSE (Mean Squared Error)
```python
loss = nn.MSELoss()
# L = (1/n) * Σ(ŷ - y)²
```
- Default for regression
- Penalizes large errors quadratically
- Config: `loss: mse` or `mean_squared_error`

#### MAE (Mean Absolute Error)
```python
loss = nn.L1Loss()
# L = (1/n) * Σ|ŷ - y|
```
- Robust to outliers
- Linear penalty
- Config: `loss: mae` or `mean_absolute_error` or `l1`

#### QLIKE (Quasi-Likelihood)
```python
loss = QLikeLoss()
# L = log(σ²_pred) + σ²_true / σ²_pred
```
- Custom loss for volatility forecasting
- Ensures strictly positive predictions
- Asymmetric penalty structure
- Config: `loss: qlike`

### Loss Configuration

```yaml
trainer:
  hparams:
    loss: mse
```

---

## 📊 Evaluation Metrics

All metrics computed with `@torch.no_grad()` for efficiency:

### MSE (Mean Squared Error)
```
MSE = (1/n) * Σ(ŷ - y)²
```
Lower is better. Scale-dependent.

### MAE (Mean Absolute Error)
```
MAE = (1/n) * Σ|ŷ - y|
```
More interpretable than MSE. Same units as target.

### Directional Accuracy
```
DA = (1/n) * Σ[sign(ŷ) == sign(y)]
```
Percentage of correct sign predictions. Useful for directional forecasts (0-100%).

### Undershooting
```
US = (1/n) * Σ(y > ŷ)
```
Percentage of predictions below true value. Useful for understanding prediction bias.

### Metric Batching

For memory efficiency, metrics computed in batches:

```python
@torch.inference_mode()
def _evaluate_fold(self, X, y, eval_bs=8192):
    for i in range(0, len(X), eval_bs):
        xb = X[i:i+eval_bs]
        yb = y[i:i+eval_bs]
        pred = self.model(xb)
        # Accumulate metrics
```

---

## 🔄 Training Loop Details

### Epoch Structure

```
For each epoch:
  │
  ├─ Shuffle training data
  ├─ For each batch:
  │   ├─ Forward pass
  │   ├─ Compute loss
  │   ├─ Backward pass
  │   └─ Optimizer step
  │
  ├─ Evaluate on validation (every val_every epochs)
  ├─ Check early stopping
  ├─ Save checkpoint if improved
  └─ Report to callback (Optuna)
```

### Configuration Parameters

```yaml
trainer:
  hparams:
    epochs: 100                    # Max epochs
    batch_size: 512                # Training batch size
    lr: 0.001                      # Learning rate
    weight_decay: 0.0001           # L2 regularization
    optimizer_type: adam           # Optimizer choice
    momentum: 0.9                  # SGD momentum (if SGD)
    torch_patience: 20             # Early stopping patience
    min_delta: 1e-10               # Improvement threshold
    val_every: 5                   # Validation frequency (epochs)

experiment:
  monitor: val_loss                # Metric to track for checkpointing
  mode: min                        # min/max (minimize or maximize)
  store_test_loss: false           # Save test metrics
  n_steps: null                    # Optional: stop after N steps
```

### Batch Processing

Two-phase batching for large datasets:

```python
# Training: shuffle and batch
perm = torch.randperm(N)
for i in range(0, N, batch_size):
    idx = perm[i:i+batch_size]
    xb = Xtr_tensor[idx]
    yb = ytr_tensor[idx]
    # Forward + backward

# Evaluation: sequential batches for consistency
for i in range(0, len(X), eval_bs):
    xb = X[i:i+eval_bs]
    yb = y[i:i+eval_bs]
    # Inference only
```

---

## ✓ Early Stopping

### Mechanism

```python
from training_routine.training_utils import early_stopping_step

new_best, stalled, improved, should_stop = early_stopping_step(
    epoch=epoch,
    val_loss=val_loss,
    best_val=best_val,
    stalled=stalled,
    patience=20,
    min_delta=1e-10,
    mode="min",
    model=model,
    optimizer=optimizer
)
```

**Logic:**
- Track best validation metric across epochs
- Increment "stalled" counter when no improvement
- Stop if `stalled >= patience`
- `min_delta`: Minimum improvement to count as progress
- Save checkpoint on improvement

**Configuration:**

```yaml
trainer:
  hparams:
    torch_patience: 20             # Epochs to wait before stopping
    min_delta: 1e-10               # Minimum improvement threshold
```

---

## 💾 Checkpointing

### Checkpoint Content

```python
best_state = {
    "epoch": 15,
    "model_state": {...},          # Model weights
    "optimizer_state": {...},      # Optimizer momentum, etc.
}
```

### State Management

```python
def _state_dict_cpu(model):
    """Move model state to CPU to save GPU memory"""
    return {k: v.detach().to("cpu") for k, v in model.state_dict().items()}

def _optimizer_state_cpu(optimizer):
    """Move optimizer tensors to CPU"""
    # Recursively move optimizer state tensors
```

Checkpoints saved to disk during training and loaded on best epoch.

---

## ⚡ GPU Optimizations

### Automatic Mixed Precision (AMP)

```python
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    pred = model(xb)
    loss = loss_fn(pred, yb)
```

- Uses bfloat16 for activations (faster, less memory)
- Keeps float32 for accumulation (more accurate)
- Selectively disabled for LSTM in eval

### torch.compile

```python
if name == "transformer":
    self.model = torch.compile(self.model)
elif name == "simplecnn":
    self.model = torch.compile(self.model, backend="eager", fullgraph=False)
```

- Compiles model to optimize for target hardware
- Eager backend for models with control flow
- Optional fullgraph=True for maximum speedup

### Batch Processing

- Variable batch sizes for train (shuffle) and eval (consistency)
- Evaluation batch size > training for better memory utilization
- Sequential batching prevents duplicate computation

---

## 🔗 Hyperparameter Search Integration

### Optuna Reporting

```python
report_cb = _make_report_cb(trial, mode="min", patience=10)
trainer.fit_eval_fold(model, fold_data, report_cb=report_cb)
```

Callback reports validation metric each epoch:
- Enables MedianPruner to stop bad trials early
- Saves significant computation for large search spaces

### Trial Configuration

```yaml
trainer:
  search:
    lr:
      type: float
      low: 0.0001
      high: 0.1
      log: true
    weight_decay:
      type: float
      low: 0.0
      high: 0.001
```

---

## 📈 Metric Output Format

### Training Metrics Dictionary

```python
train_metrics = {
    "loss": 0.045,
    "mae": 0.152,
    "mse": 0.045,
    "diracc": 52.3,              # Directional accuracy %
    "undershooting": 48.5         # Undershooting %
}
```

Same structure for validation and test metrics.

---

## 🔧 Typical Usage

### Single Fold Training

```python
from models import create_model

# Create model
model = create_model(cfg.model, input_shape, output_shape)

# Train on fold
trainer = Trainer(cfg, logger)
tr_metrics, val_metrics, te_metrics = trainer.fit_eval_fold(
    model,
    fold_data,
    fold=0,
    merge_train_val=False
)

print(f"Validation Loss: {val_metrics['loss']:.4f}")
print(f"Test Directional Accuracy: {te_metrics['diracc']:.1f}%")
```

### Hyperparameter Search

```python
def objective(trial):
    # Sample hparams
    sampled_cfg = sample_hparams_into_cfg(cfg, trial)
    
    # Train on fold
    model = create_model(sampled_cfg.model, input_shape, output_shape)
    trainer = Trainer(sampled_cfg, logger)
    
    _, val_metrics, _ = trainer.fit_eval_fold(
        model,
        fold_data,
        report_cb=report_callback,
        merge_train_val=False
    )
    
    return val_metrics[cfg.experiment.monitor]

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

---

## ⚠️ Important Notes

### GPU Requirements
- CUDA/GPU mandatory (will error if not available)
- bfloat16 optimizations require modern GPUs (Ampere or newer)

### Device Management
- All tensors automatically moved to GPU
- Checkpoints saved to CPU before disk
- Model reloaded to GPU for next fold

### Reproducibility
- Random seed in data shuffling affects training
- Each trial starts fresh (no state carryover)
- Deterministic metrics computation

### Memory Efficiency
- Batch-wise evaluation prevents OOM on large datasets
- Checkpoint states moved to CPU automatically
- AMP reduces memory during forward/backward

---

## 🔗 Integration

- **Models**: Works with MLP, LSTM, CNN, Transformer
- **Config**: Loaded from `config.config_types`
- **Pipeline**: Consumes fold data from `pipeline.walkforward`
- **Search**: Integrated with `hyperparams_search.search_utils`
- **Logging**: Tracks metrics via `utils.logging_utils`

---

## 📚 References

- **Early Stopping**: Prechelt (1998) - "Early Stopping - But When?"
- **Weight Decay**: Loshchilov & Hutter (2017) - "Decoupled Weight Decay Regularization"
- **Muon Optimizer**: Ivgi et al. (2024) - "Muon: Second-order Optimization for SGD"
