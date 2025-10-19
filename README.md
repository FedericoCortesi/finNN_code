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
│   │   ├── __init__.py
│   │   ├── preprocessing.py         # Data preprocessing utilities
│   │   ├── walkforward.py          # Walk-forward cross-validation engine
│   │   └── wf_config.py            # Walk-forward configuration dataclass
│   ├── price_prediction/           # Price prediction experiments
│   │   ├── __init__.py
│   │   ├── visuals.ipynb           # Results visualization
│   │   ├── benchmarks/             # Baseline model implementations
│   │   │   ├── __init__.py
│   │   │   └── regressions.ipynb   # Linear regression benchmarks
│   │   ├── experiments/            # Experiment results and saved models
│   │   │   ├── exp_001_20251011_162012_mlp/
│   │   │   ├── exp_002_20251012_165113_pippo/
│   │   │   └── ...                 # Other experiment directories
│   │   └── legacy/                 # Legacy training code
│   │       ├── __init__.py
│   │       ├── debug.py
│   │       ├── training_models.py
│   │       └── training_pipeline.ipynb
│   ├── training_routine/           # Training infrastructure
│   │   ├── trainer.py              # Model training orchestrator
│   │   └── metrics.py              # Training metrics computation
│   ├── utils/                      # Utility functions
│   │   ├── __init__.py
│   │   ├── custom_formatter.py     # Custom logging formatters
│   │   ├── gpu_test.py             # GPU availability testing
│   │   ├── logging_utils.py        # Experiment logging utilities
│   │   └── paths.py                # Path management
│   ├── debug.ipynb                 # Main debugging notebook
│   └── run_experiments.py          # Main experiment runner script
├── logs/                           # SLURM job logs
│   ├── slurm_*.out                 # SLURM stdout logs
│   └── slurm_*.err                 # SLURM stderr logs
├── train_job.sh                    # SLURM job submission script
├── pyproject.toml                  # Project dependencies and metadata
├── ssh_guide.md                    # SSH connection guide
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.12
- PyTorch (for GPU training)
- CUDA-capable GPU (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd finNN_code
   ```

2. **Install the package:**
   ```bash
   pip install -e .
   ```

3. **Verify GPU setup:**
   ```bash
   python src/utils/gpu_test.py
   ```

### Running Your First Experiment

1. **Basic experiment with default settings:**
   ```bash
   python src/run_experiments.py --config default.yaml
   ```

2. **Debug experiment (smaller dataset, fewer epochs):**
   ```bash
   python src/run_experiments.py --config debug.yaml
   ```

3. **Custom experiment name:**
   ```bash
   python src/run_experiments.py --config default.yaml --exp-name my_first_experiment
   ```

4. **Hyperparameter search:**
   ```bash
   python src/run_experiments.py --config search_debug.yaml
   ```

## 📊 Walk-Forward Cross-Validation

The core of this framework is the walk-forward cross-validation system, which ensures proper temporal validation for time series data:

### How It Works

1. **Data Splitting**: The time series is divided into overlapping windows
2. **Temporal Ordering**: Training always occurs on past data, validation on recent past, testing on future
3. **Rolling Windows**: The validation window moves forward in time with each fold
4. **No Look-Ahead Bias**: Strict temporal ordering prevents data leakage

### Configuration Parameters

```yaml
walkforward:
  max_folds: 20              # Maximum number of folds (null = all possible)
  step: 251                  # Days to advance between folds (1 year ≈ 251 trading days)
  ratio_train: 3             # Training period ratio
  ratio_val: 1               # Validation period ratio  
  ratio_test: 1              # Test period ratio
  lags: 20                   # Number of lagged features to use
  T: null                    # Total periods (auto-calculated from data if null)
```

**How it Works:**
The system automatically calculates `T_train`, `T_val`, and `T_test` based on the total available time periods and the specified ratios. Each fold advances by `step` periods while maintaining the same window sizes.

## ⚙️ Configuration System

Experiments are configured using YAML files in `src/config/`. The configuration system supports:

### Core Configuration Sections

```yaml
# Experiment metadata
experiment:
  name: "mlp"                        # Experiment identifier
  random_state: 1234                 # Random seed for reproducibility
  monitor: "val_loss"                # Metric to monitor for early stopping
  mode: "min"                        # Optimization direction (min/max)
  hyperparams_search: false          # Enable hyperparameter optimization

# Data configuration
data:
  df_path: null                      # Path to data file (auto-loads if null)
  target_col: "y"                    # Target variable column name
  feature_cols: ["feature_0", "feature_1", ...]  # Feature column names
  standardize: true                  # Apply standardization
  per_asset_norm: true               # Normalize per asset independently

# Model architecture
model:
  name: "mlp"                        # Model type (mlp, cnn1d, etc.)
  hparams:
    hidden_sizes: [32, 32]           # Hidden layer dimensions
    dropout_rate: 0.2                # Dropout rate
    activation: "relu"               # Activation function
    l2_reg: 0.001                    # L2 regularization strength
    output_activation: null          # Output layer activation

# Training parameters
trainer:
  params:
    epochs: 50                       # Maximum training epochs
    batch_size: 128                  # Training batch size
    lr: 1.0e-3                       # Learning rate
    loss: "mse"                      # Loss function
    metrics: ["mae", "mse"]          # Evaluation metrics
```

## 🧠 Model Architecture

### Available Models

Currently implemented:
- **MLP**: Multi-layer perceptron with configurable layers, dropout, and regularization

### Model Features

- **Configurable Architecture**: Variable number of hidden layers and neurons
- **Regularization**: Dropout and L2 regularization support
- **Flexible Activations**: ReLU, tanh, sigmoid, and other standard activations
- **Time Series Adaptation**: Designed for lagged feature inputs from financial data

### Adding New Models

1. Add model implementation to `src/models/`
2. Update the `create_model` function to handle your new model type
3. Configure model parameters in YAML files

## 🔬 Experiment Management

### Running Experiments

The experiment system automatically:

- **Creates unique experiment directories** with timestamps
- **Saves model checkpoints** for each fold
- **Logs comprehensive metrics** (RMSE, directional accuracy, etc.)
- **Tracks hyperparameters** and configuration
- **Enables experiment reproduction** with saved configs

### Results Structure

```
src/price_prediction/experiments/exp_XXX_YYYYMMDD_HHMMSS_name/
├── trial_000/                       # Trial directory for hyperparameter search
│   ├── fold_000/                    # Fold-specific results
│   │   ├── model_best.pth           # Best model checkpoint
│   │   └── training_log.json        # Training metrics and losses
│   └── fold_001/
│       └── ...
├── config_snapshot.json             # Experiment configuration
└── results.csv                      # Aggregated results across folds
```

### Results Analysis

Each experiment automatically saves:
- **Model Checkpoints**: Best model weights per fold
- **Training Logs**: Loss curves, metrics, and training progress
- **Configuration**: Complete experiment setup for reproducibility
- **Results CSV**: Performance metrics aggregated across folds

## 🖥️ High-Performance Computing

### SLURM Integration

For cluster environments, use the provided SLURM script:

```bash
# Submit job to SLURM
sbatch train_job.sh

# Monitor job status
squeue -u $USER

# Check logs
tail -f logs/slurm_<job_id>.out
tail -f logs/slurm_<job_id>.err
```

The `train_job.sh` script:
- Requests H200 GPU resources (configurable)
- Loads miniforge/conda environment
- Sets up proper CUDA paths
- Runs experiments with logging
- Saves all outputs to `logs/` directory

### GPU Utilization

The framework uses PyTorch for GPU training:
- Automatic GPU detection via `gpu_test.py`
- Efficient data loading with multiple workers
- Memory-optimized batch processing
- Support for mixed precision training

## 📈 Performance Monitoring

### Logging System

The framework includes comprehensive logging via `ExperimentLogger`:

- **Structured Logging**: Automatic directory creation with timestamps
- **Training Metrics**: Loss curves, validation performance tracking
- **Model Checkpointing**: Automatic saving of best models per fold
- **Configuration Snapshots**: Complete reproducibility information

### Hyperparameter Optimization

Built-in Optuna integration for automated hyperparameter search:

```yaml
experiment:
  hyperparams_search: true           # Enable search
  n_trials: 50                       # Number of trials
  mode: "min"                        # Optimization direction

# Define search spaces in configuration
search_spaces:
  model.hparams.hidden_sizes: [[32], [64], [32, 32], [64, 32]]
  trainer.params.lr: [1e-4, 1e-3, 1e-2]
  trainer.params.batch_size: [128, 256, 512]
```

## 🔧 Advanced Usage

### Custom Data Sources

The framework can work with custom financial datasets:

1. **Data Format**: Long-format DataFrame with:
   - `permno`: Asset identifier  
   - `t`: Time index (integer)
   - Feature columns (as specified in config)
   - Target column (`y` by default)

2. **Loading Custom Data:**
   ```bash
   python src/run_experiments.py --config default.yaml --data path/to/your/data.parquet
   ```

### Debugging and Development

1. **Debug Configuration**: Use smaller datasets and fewer epochs
   ```bash
   python src/run_experiments.py --config debug.yaml
   ```

2. **Jupyter Notebooks**: Interactive analysis available:
   - `src/debug.ipynb`: Main debugging notebook
   - `src/data/data_analysis.ipynb`: Data exploration
   - `src/price_prediction/visuals.ipynb`: Results visualization

### Extending the Framework

1. **Add New Models**: Extend `src/models/` with new architectures
2. **Custom Metrics**: Modify `src/training_routine/metrics.py`
3. **New Preprocessing**: Extend `src/pipeline/preprocessing.py`

## 🧪 Development and Testing

### Interactive Development

```bash
# Main debugging notebook
jupyter notebook src/debug.ipynb

# Data exploration and analysis  
jupyter notebook src/data/data_analysis.ipynb
jupyter notebook src/data/get_data.ipynb

# Results visualization
jupyter notebook src/price_prediction/visuals.ipynb

# Benchmark comparisons
jupyter notebook src/price_prediction/benchmarks/regressions.ipynb
```

### Code Structure

- **Configuration-Driven**: YAML-based experiment setup with typed configurations
- **Modular Pipeline**: Separate components for data, models, training, and evaluation  
- **Walk-Forward Validation**: Proper temporal validation for financial time series
- **Experiment Tracking**: Automatic logging and reproducibility

## 📚 Key Concepts

### Walk-Forward Validation vs Traditional CV

Traditional cross-validation randomly splits data, which can cause **look-ahead bias** in time series. Walk-forward validation:

- ✅ Respects temporal ordering
- ✅ Simulates real-world deployment
- ✅ Provides realistic performance estimates
- ✅ Prevents data leakage

### Financial Time Series Considerations

- **Stationarity**: Markets change over time
- **Regime Changes**: Model performance varies across market conditions  
- **Transaction Costs**: Real-world implementation considerations
- **Risk Management**: Drawdown and volatility controls

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   ```bash
   python src/utils/gpu_test.py
   # Should show CUDA availability and GPU information
   ```

2. **NaN Metrics in Results**:
   - Check for empty test sets in walk-forward validation
   - Verify data has sufficient non-NaN values for each window
   - Review `T`, `step`, and `lags` parameters in walk-forward config

3. **High CPU Usage During Training**:
   - Increase `batch_size` for better GPU utilization
   - Use multiple workers in data loading
   - Check if data preprocessing is done on CPU vs GPU

4. **SLURM Job Failures**:
   - Check logs: `cat logs/slurm_<job_id>.err`
   - Verify conda environment and module loading
   - Ensure sufficient time and memory allocation

5. **Configuration Errors**:
   - Validate YAML syntax
   - Check that all required fields are present
   - Use `debug.yaml` for testing configuration changes

### Performance Tips

- **Start Small**: Use `debug.yaml` for initial testing
- **Monitor Resources**: Check GPU utilization with `nvidia-smi`
- **Batch Size**: Larger batches (512-2048) often improve GPU efficiency
- **Data Loading**: Use sufficient workers to avoid CPU bottlenecks

### Getting Help

- Check experiment logs for detailed error messages
- Use the debugging notebooks for interactive troubleshooting
- Review configuration examples in `src/config/`

---

**Happy Training! 🚀**