# Configuration Directory

This directory contains all configuration files and utilities for the finNN pipeline. It defines experiment parameters, model architectures, training hyperparameters, and data processing settings.

## 📁 Contents

### Core Configuration Files

- **`config_types.py`** – Pydantic configuration schema
  - Defines all configuration data classes with type validation
  - Includes: `ModelConfig`, `TrainerConfig`, `WFConfig` (walk-forward), `AppConfig`
  - Search specifications for hyperparameter optimization
  - Validates that all parameters meet constraints during loading

- **`example.yaml`** – Example configuration file
  - Shows full structure with all available options
  - Includes model architectures (MLP, CNN, LSTM, Transformer)
  - Trainer hyperparameters, optimizer settings, loss functions
  - Walk-forward cross-validation parameters

### Configuration Subdirectories

#### `yaml_configs/` – Experiment configurations
Pre-configured YAML files for different experimental setups:
- **Debug configs**: `cnn_debug.yaml`, `debug_prc.yaml`, `debug_vol.yaml`
- **Search configs**: `cnn_search.yaml`, `search_debug.yaml`, `lstm_search_1.yaml`, `lstm_search_2.yaml`
- **Architecture-specific**: `vol_cnn.yaml`, `vol_lstm.yaml`, `vol_mlp.yaml`, `vol_transformer.yaml`
- **Optimizer variants**: Configs with Muon, Adam, and SGD optimizers
- **Price vs. Volatility**: `prc_cnn.yaml`, `prc_lstm.yaml`, `prc_mlp.yaml` for price prediction
- **Special purpose**: `lstm_best.yaml`, `lstm_intervention.yaml`, `mlp_intervention.yaml`, `mlp_snr.yaml`

#### `seed_runs/` – Multi-seed experiments
Configurations for running models with different random seeds:
- Multiple configurations per architecture × optimizer combination
- Naming: `{arch}_{hidden_dim}_{optimizer}_seeds_{seed}.yaml`
- Covers: CNN, LSTM, MLP, Transformer
- Optimizers: Adam, Muon, SGD
- Seed range: 0-13 (14 different seeds per setup)
- **Script**: `make_seeds_yaml.py` – Generates seed-based configs from templates

#### `snr_runs/` – Signal-to-noise ratio experiments
Configurations for SNR (Signal-to-Noise Ratio) analysis:
- Named like: `{arch}_{hidden_dim}_{optimizer}.yaml`
- Used to evaluate model robustness under noise
- Includes optimizer variants (Adam, Muon, SGD)
- **Script**: `make_noise_yaml.py` – Generates noise-based configs

### Utility Scripts

- **`interventions.py`** – Configuration generator for intervention studies
  - Creates configs that combine different model initializations
  - Generates combinations of base models with intervention targets
  - Outputs new YAML files with intervention-specific settings
  - Automatically disables hyperparameter search for interventions

## 🔧 Configuration Structure

### Main Configuration Sections

```yaml
model:                          # Model architecture
  name: lstm                    # Model type
  hparams:                      # Model hyperparameters
    lstm_hidden_sizes: [256]
    dropout_rate: 0.1
  search: {}                    # Search space (for Optuna)

trainer:                        # Training configuration
  hparams:                      # Trainer hyperparameters
    epochs: 20
    batch_size: 512
    optimizer: muon
    lr: 0.001
  search: {}                    # Hyperparameter search space

wf:                            # Walk-forward cross-validation
  target_col: ret              # Prediction target
  lags: 20                     # Feature lookback window
  ratio_train: 3               # Train ratio
  ratio_val: 1                 # Validation ratio
  ratio_test: 1                # Test ratio
  step: 251                    # Trading days per period
  scale_type: standard         # Scaling method

experiment:                    # Experiment metadata
  name: experiment_name
  hyperparams_search: false    # Enable Optuna search
  seed: 42
```

## 📊 Model Architecture Support

The configuration system supports four neural network architectures:

1. **MLP** – Multi-layer perceptron
2. **CNN** – Convolutional neural network
3. **LSTM** – Long short-term memory RNN
4. **Transformer** – Attention-based architecture

Each can be configured with custom:
- Hidden dimensions and layer counts
- Activation functions
- Dropout and layer normalization
- Optimizer (Adam, SGD, Muon)
- Learning rate and regularization

## 🚀 Usage

### Loading a Configuration
```python
from config.config_types import AppConfig
import json

cfg_json = json.loads(Path("config.yaml").read_text())
cfg = AppConfig.from_dict(cfg_json)
```

### Running with a Specific Configuration
```bash
python run_experiments.py --config_path yaml_configs/vol_lstm.yaml
```

### Generating New Seed-Based Configs
```bash
python seed_runs/make_seeds_yaml.py
```

### Generating Noise-Based Configs for SNR Studies
```bash
python snr_runs/make_noise_yaml.py
```

### Creating Intervention Configs
```bash
python interventions.py
```

## 📝 Notes

- All YAML configurations are validated against the schema in `config_types.py`
- Search specifications define hyperparameter ranges for Optuna-based optimization
- Walk-forward parameters define temporal cross-validation strategy
- Seed-based and SNR-based configs allow reproducible multi-trial experiments
- Intervention configs enable transfer learning and initialization studies
