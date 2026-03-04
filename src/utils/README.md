# Utils Directory

This directory contains utility functions and helpers for logging, GPU management, path resolution, and miscellaneous operations used throughout the project.

## 📁 Contents

### Core Modules

- **`logging_utils.py`** – Experiment tracking and results logging
  - `ExperimentLogger` class for managing experiment trials and results
  - Folder structure creation and file management
  - CSV/JSON results logging
  - Configuration snapshots

- **`custom_formatter.py`** – Console logging with color formatting
  - `CustomFormatter` for pretty-printed logs
  - `setup_logger()` function for consistent logger initialization
  - Color-coded log levels

- **`paths.py`** – Project path definitions
  - Absolute paths to key directories
  - Constants for data, configs, experiments
  - Centralized path management

- **`gpu_test.py`** – GPU availability and setup diagnostics
  - `gpu_test()` function to verify CUDA/GPU availability
  - Detailed GPU information logging
  - Startup validation

- **`inference_utils.py`** – Inference and formatting helpers
  - `format_legend_name()` for consistent result naming

- **`random_setup.py`** – Reproducibility utilities
  - `set_global_seed()` for deterministic training

- **`__init__.py`** – Package initialization

---

## 🎯 Custom Formatter & Logging

### CustomFormatter Class

Provides colored console output for different log levels:

| Level | Color | Usage |
|-------|-------|-------|
| DEBUG | Blue | Verbose debugging info |
| INFO | Grey | General information |
| WARNING | Yellow | Warning messages |
| ERROR | Red | Error messages |
| CRITICAL | Bold Red | Critical failures |

**Format**: `timestamp - logger_name - level - message (file:line_number)`

### setup_logger Function

```python
from utils.custom_formatter import setup_logger

logger = setup_logger(name="MyModule", level="INFO")
logger.debug("Detailed info")
logger.info("Important message")
logger.warning("Potential issue")
logger.error("Error occurred")
```

**Parameters:**
- `name`: Logger identifier (appears in log output)
- `level`: Log level ("DEBUG", "INFO", "WARNING")

**Returns:** Configured `logging.Logger` instance

**Features:**
- Clears existing handlers (prevents duplicate logs)
- Single StreamHandler for console output
- Custom formatter applied automatically
- Level validation

---

## 📂 Experiment Logging

### ExperimentLogger Class

Manages folder structure and result tracking for experiments.

**Folder Structure:**
```
{exp_type}/experiments/
├── exp_001_mlp_baseline/
│   ├── trial_search_best/
│   │   ├── results.csv
│   │   ├── metadata.jsonl
│   │   ├── config_snapshot.json
│   │   ├── fold_000/
│   │   │   ├── model_best.pt
│   │   │   └── training_history.json
│   │   ├── fold_001/
│   │   └── ...
│   └── trial_20240304_143022/
│       └── ...
└── exp_002_lstm_attention/
    └── ...
```

### Initialization

```python
from utils.logging_utils import ExperimentLogger

logger = ExperimentLogger(cfg)
```

**Parameters:**
- `cfg`: `AppConfig` object containing experiment configuration
- `console_logger`: Optional logger for messages (created if None)

**Automatic Setup:**
- Creates experiment type directory (price_prediction/volatility)
- Reuses existing experiment with same name
- Allocates new experiment ID if new experiment

### Trial Management

#### Begin Trial

```python
trial_dir = logger.begin_trial(name="search_best")
```

**Creates:**
- Trial subdirectory: `trial_{name}`
- `results.csv` - Results header with metric columns
- `metadata.jsonl` - Metadata logging file
- `config_snapshot.json` - Full config + environment info

**Parameters:**
- `name`: Optional identifier. If `None`, uses timestamp (YYYYMMDD_HHMMSS)

**Returns:** Trial directory path

#### Get Path

```python
path = logger.path(f"fold_000/model_best.pt")
```

Constructs full path within trial directory for saving files.

#### Append Result

```python
logger.append_result(
    trial=0,
    fold=0,
    tr_loss=0.045, val_loss=0.048, test_loss=0.050,
    tr_mae=0.152, val_mae=0.155, test_mae=0.158,
    tr_directional_accuracy_pct=52.3, val_directional_accuracy_pct=51.8, test_directional_accuracy_pct=51.5,
    tr_undershooting_pct=48.2, val_undershooting_pct=48.5, test_undershooting_pct=48.7,
    seconds=1234.5,
    model_path="fold_000/model_best.pt"
)
```

Appends row to `results.csv` with all metrics.

#### Log Metadata

```python
logger.log({"learning_rate": 0.001, "batch_size": 512, "epoch": 10})
```

Appends JSON-formatted metadata to `metadata.jsonl` for detailed logging.

### Results CSV Format

```
trial,fold,tr_loss,val_loss,test_loss,tr_mae,val_mae,test_mae,tr_directional_accuracy_pct,val_directional_accuracy_pct,test_directional_accuracy_pct,tr_undershooting_pct,val_undershooting_pct,test_undershooting_pct,seconds,model_path
```

One row per fold per trial, enabling easy aggregation and analysis.

### Config Snapshot

Saved as JSON in `config_snapshot.json`:

```json
{
  "cfg": {
    "model": {...},
    "trainer": {...},
    "experiment": {...},
    "walkforward": {...}
  },
  "env": {
    "host": "gpu-server-01",
    "platform": "Linux-5.10.0",
    "time": "20240304_143022"
  }
}
```

Captures full configuration and environment for reproducibility.

---

## 🗂️ Path Resolution

### Path Constants

**File**: `paths.py`

```python
from utils.paths import (
    REPO_ROOT,
    SRC_DIR,
    CONFIG_DIR,
    DATA_DIR,
    PRICE_EXPERIMENTS_DIR,
    VOL_EXPERIMENTS_DIR,
    SP500_PATH,
    PERMNOS_PATH,
    INFO_PATH
)
```

**Key Paths:**
- `REPO_ROOT` - Repository root directory
- `SRC_DIR` - Source code directory (`src/`)
- `CONFIG_DIR` - Configuration files (`src/config/`)
- `DATA_DIR` - Data directory (`src/data/`)
- `PRICE_EXPERIMENTS_DIR` - Price prediction experiments
- `VOL_EXPERIMENTS_DIR` - Volatility experiments
- `SP500_PATH` - S&P 500 daily data parquet file
- `PERMNOS_PATH` - Stock permno list
- `INFO_PATH` - Stock information CSV

**Usage:**
```python
from utils.paths import DATA_DIR

data = pd.read_parquet(DATA_DIR / "sp500_daily_data.parquet")
```

**Benefits:**
- Centralized path management
- Works regardless of CWD
- Cross-platform compatibility (uses `pathlib.Path`)
- Easy to update paths in one place

---

## 🔧 GPU Management

### gpu_test Function

```python
from utils.gpu_test import gpu_test

gpu_test()
```

**Checks:**
1. PyTorch and CUDA versions
2. Available GPU devices and names
3. CUDA build status
4. Active GPU device
5. CUDA driver/runtime version
6. libcuda.so.1 loadability

**Output Example:**
```
DEBUG - GPU-Test - PyTorch version: 2.0.1
DEBUG - GPU-Test - CUDA runtime version: 11.8
DEBUG - GPU-Test - Visible GPUs: ['NVIDIA A100 80GB', 'NVIDIA A100 80GB']
DEBUG - GPU-Test - Built with CUDA?: True
DEBUG - GPU-Test - Active device ID: 0
DEBUG - GPU-Test - Active device name: NVIDIA A100 80GB
DEBUG - GPU-Test - libcuda.so.1 is loadable
```

**Error Handling:**
- Exits if CUDA not available
- Exits if libcuda library missing
- Returns status code 0 on success, 1 on failure

**Usage in Scripts:**
```bash
python -c "from utils.gpu_test import gpu_test; gpu_test()"
```

---

## 🎲 Reproducibility

### set_global_seed Function

```python
from utils.random_setup import set_global_seed

set_global_seed(42)
```

**Sets:**
1. Python `random.seed()`
2. NumPy `np.random.seed()`
3. PyTorch `torch.manual_seed()` (CPU)
4. PyTorch CUDA seeds (GPU)
5. Deterministic algorithms flag
6. CuDNN settings

**Ensures:**
- Reproducible random number generation
- Deterministic GPU operations (slightly slower)
- Consistent results across runs
- No performance surprises from algorithm selection

**Important Notes:**
- Call early in script before any random operations
- Some GPU operations may still have minor non-determinism
- Trades some performance for reproducibility

**Environment Variable:**
Sets `CUBLAS_WORKSPACE_CONFIG=":4096:8"` for CUDA compatibility.

---

## 🎨 Formatting Helpers

### format_legend_name Function

```python
from utils.inference_utils import format_legend_name

name = format_legend_name("exp_035_mlp_100_muon_lr")
# Returns: "MLP 100 MUON"
```

**Purpose:**
- Converts experiment name to readable legend format
- Extracts architecture, hidden dim, optimizer
- Standardizes capitalization

**Format:**
- Input: `exp_{id}_{arch}_{hidden}_{optimizer}_{optional}`
- Output: `{ARCH} {HIDDEN} {OPTIMIZER}`

**Examples:**
- `exp_035_mlp_100_muon` → `MLP 100 MUON`
- `exp_179_transformer_100_adam_lr` → `TRANSFORMER 100 ADAM`
- `exp_169_cnn_100_muon_icml_3` → `CNN 100 MUON`

---

## 📋 Typical Usage Workflow

### Initialization

```python
from utils.custom_formatter import setup_logger
from utils.logging_utils import ExperimentLogger
from utils.random_setup import set_global_seed

# Setup
logger_console = setup_logger("Main", "INFO")
set_global_seed(42)

# Create experiment logger
logger_exp = ExperimentLogger(cfg)
```

### Training

```python
# Begin trial
trial_dir = logger_exp.begin_trial("search_best")

for fold_idx, fold_data in enumerate(wf.folds()):
    trainer = Trainer(cfg, logger_exp)
    model = create_model(cfg.model, input_shape, output_shape)
    
    tr_metrics, val_metrics, te_metrics = trainer.fit_eval_fold(
        model, fold_data, fold=fold_idx
    )
    
    # Log results
    logger_exp.append_result(
        trial=0,
        fold=fold_idx,
        tr_loss=tr_metrics["loss"],
        val_loss=val_metrics["loss"],
        test_loss=te_metrics["loss"],
        # ... other metrics
        seconds=trainer.elapsed,
        model_path=f"fold_{fold_idx:03d}/model_best.pt"
    )
```

### Analysis

```python
# Read results
results_df = pd.read_csv(logger_exp.results_csv)
print(results_df.groupby("fold").mean())
```

---

## 🔗 Integration

- **Trainer**: Uses logging for checkpoints and results
- **Pipeline**: Uses paths for data loading
- **Models**: Use GPU during training
- **Scripts**: Use reproducibility settings globally
- **Notebooks**: Use loggers and path constants

---

## ⚠️ Important Notes

### Logging Best Practices
- Use `setup_logger()` for all loggers (consistent formatting)
- Set level to "INFO" in production, "DEBUG" in development
- Avoid global loggers in libraries (creates conflicts)

### Path Management
- Always use `paths.py` constants for cross-platform compatibility
- Don't hardcode paths with `/` or `\` (use `pathlib.Path`)
- Update `paths.py` if directory structure changes

### GPU Testing
- Run `gpu_test()` at startup to catch GPU issues early
- Useful in cluster environments with variable GPU availability
- Helps diagnose CUDA configuration problems

### Reproducibility
- Call `set_global_seed()` before any data loading/model creation
- Note that reproducibility has performance cost (~5-10% overhead)
- Some operations may still have unavoidable non-determinism

### Experiment Logging
- Each trial automatically gets unique directory
- Results CSV accumulates across trials
- Config snapshots enable experiment recreation
- Metadata JSONL allows custom logging

---

## 📚 References

- **Logging**: Python `logging` module documentation
- **Pathlib**: Python 3 pathlib for cross-platform paths
- **PyTorch Reproducibility**: PyTorch reproducibility documentation
- **CUDA Setup**: NVIDIA CUDA toolkit documentation
