# finNN_code: Financial Neural Network Training Pipeline

A comprehensive neural network training pipeline for financial time series prediction using walk-forward cross-validation. This repository implements a robust framework for training and evaluating machine learning models on financial data with proper temporal validation.

## 🏗️ Project Structure

```
finNN_code/
├── src/                              # Main source code
│   ├── config/                       # Configuration files
│   │   ├── default.yaml             # Default experiment configuration
│   │   └── debug.yaml               # Debug configuration
│   ├── data/                        # Data handling and analysis
│   │   ├── data_analysis.ipynb      # Exploratory data analysis
│   │   ├── get_data.ipynb           # Data acquisition notebook
│   │   └── permnos_info.csv         # Asset metadata
│   ├── models/                      # Neural network architectures
│   │   └── mlp.py                   # Multi-layer perceptron implementation
│   ├── pipeline/                    # Data processing and validation
│   │   ├── preprocessing.py         # Data preprocessing utilities
│   │   ├── walkforward.py          # Walk-forward cross-validation engine
│   │   ├── wf_config.py            # Walk-forward configuration
│   │   └── pipeline_test.ipynb     # Pipeline testing notebook
│   ├── price_prediction/           # Main experiment runner
│   │   ├── run_experiments.py      # Main experiment script
│   │   ├── benchmarks/             # Baseline model implementations
│   │   └── experiments/            # Experiment results and saved models
│   ├── training_routine/           # Training infrastructure
│   │   ├── trainer.py              # Model training orchestrator
│   │   └── callbacks.py            # Custom Keras callbacks
│   └── utils/                      # Utility functions
│       ├── logging_utils.py        # Experiment logging
│       ├── gpu_test.py             # GPU availability testing
│       ├── custom_formatter.py     # Custom logging formatters
│       └── paths.py                # Path management
├── train_job.sh                    # SLURM job submission script
├── pyproject.toml                  # Project dependencies and metadata
└── logs/                           # Training logs and SLURM outputs
```

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.9
- TensorFlow ≥ 2.15
- GPU support (recommended for training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/FedericoCortesi/finNN_code.git
   cd finNN_code
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Verify GPU setup (optional but recommended):**
   ```bash
   python src/utils/gpu_test.py
   ```

### Running Your First Experiment

1. **Basic experiment with default settings:**
   ```bash
   python src/price_prediction/run_experiments.py
   ```

2. **Custom configuration:**
   ```bash
   python src/price_prediction/run_experiments.py --config custom_config.yaml
   ```

3. **Override experiment name:**
   ```bash
   python src/price_prediction/run_experiments.py --exp-name my_experiment
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
  max_folds: 20              # Number of validation folds (null = all possible)
  step: 251                  # Days to advance between folds (1 year ≈ 251 trading days)
  ratio_train: 3             # Training period ratio
  ratio_val: 1               # Validation period ratio  
  ratio_test: 1              # Test period ratio
  lags: 20                   # Number of lagged features to use
```

**Example Timeline:**
- Fold 1: Train[1-753], Val[754-1004], Test[1005-1255]
- Fold 2: Train[252-1004], Val[1005-1255], Test[1256-1506]
- Fold 3: Train[503-1255], Val[1256-1506], Test[1507-1757]

## ⚙️ Configuration System

Experiments are configured using YAML files in `src/config/`. The configuration system supports:

### Core Configuration Sections

```yaml
# Experiment metadata
experiment:
  name: "mlp"                        # Experiment identifier
  seed: 1234                         # Random seed for reproducibility
  output_root: "experiments"         # Results directory
  monitor: "val_loss"                # Metric to monitor for early stopping
  mode: "min"                        # Optimization direction (min/max)

# Data configuration
data:
  target_col: "y"                    # Target variable column name
  feature_cols: [...]                # List of feature column names
  standardize: true                  # Apply standardization
  per_asset_norm: true               # Normalize per asset independently

# Model architecture
model:
  name: "mlp"                        # Model type
  hparams:
    hidden_sizes: [32, 32]           # Hidden layer dimensions
    dropout_rate: 0.2                # Dropout rate
    activation: "relu"               # Activation function
    l2_reg: 0.001                    # L2 regularization strength

# Training parameters
trainer:
  epochs: 50                         # Maximum training epochs
  batch_size: 128                    # Training batch size
  lr: 1.0e-3                         # Learning rate
  loss: "mse"                        # Loss function
  metrics: ["mae", "mse"]            # Evaluation metrics
```

## 🧠 Model Architecture

### Multi-Layer Perceptron (MLP)

The framework currently supports MLP models with the following features:

- **Configurable Architecture**: Variable number of hidden layers and neurons
- **Regularization**: Dropout and L2 regularization
- **Flexible Activations**: Support for standard activation functions
- **Time Series Adaptation**: Designed for lagged feature inputs

### Adding New Models

To add a new model architecture:

1. Create a new file in `src/models/`
2. Implement a `build_model(hparams, input_shape)` function
3. Update the configuration to reference your new model

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
experiments/exp_001_20251011_162012_mlp/
├── model_fold_0.keras               # Saved model for fold 0
├── model_fold_1.keras               # Saved model for fold 1
├── ...
├── model_fold_N.keras               # Saved model for fold N
├── model_results.csv                # Comprehensive results summary
└── experiment_config.yaml           # Saved experiment configuration
```

### Results Analysis

Each experiment produces a results CSV with the following columns:
- `fold`: Fold number
- `best_hyperparameters`: Optimal hyperparameters found
- `in_sample_rmse`: Training set RMSE
- `out_of_sample_rmse`: Test set RMSE  
- `in_sample_dir_acc`: Training directional accuracy
- `out_of_sample_dir_acc`: Test directional accuracy
- `model_path`: Path to saved model

## 🖥️ High-Performance Computing

### SLURM Integration

For cluster environments, use the provided SLURM script:

```bash
sbatch train_job.sh
```

The script automatically:
- Requests GPU resources
- Sets up the conda environment
- Runs the experiment
- Saves logs to the `logs/` directory

### GPU Utilization

The framework is optimized for GPU training:
- Automatic GPU detection and configuration
- TensorFlow GPU optimization
- Memory management for large datasets

## 📈 Performance Monitoring

### Logging System

The framework includes comprehensive logging:

- **Experiment Logger**: Tracks experiment metadata and results
- **Training Logger**: Monitors training progress and metrics
- **Console Logger**: Provides real-time feedback

### Metrics Tracked

- **Regression Metrics**: MSE, MAE, RMSE
- **Financial Metrics**: Directional accuracy, Sharpe ratio
- **Training Metrics**: Loss curves, validation performance
- **Computational Metrics**: Training time, memory usage

## 🔧 Advanced Usage

### Custom Data Sources

To use your own data:

1. **Format Requirements**: Long-format DataFrame with columns:
   - `permno`: Asset identifier
   - `date` or `t`: Time identifier  
   - Feature columns (as specified in config)
   - Target column

2. **Pass data path:**
   ```bash
   python src/price_prediction/run_experiments.py --data path/to/your/data.parquet
   ```

### Hyperparameter Optimization

The framework supports hyperparameter search:

```yaml
trainer:
  search: true                       # Enable hyperparameter search
```

### Debugging

Use the debug configuration for faster iteration:

```bash
python src/price_prediction/run_experiments.py --config debug.yaml
```

## 🧪 Development and Testing

### Running Tests

```bash
# Test the walk-forward pipeline
jupyter notebook src/pipeline/pipeline_test.ipynb

# Test data preprocessing
jupyter notebook src/data/data_analysis.ipynb
```

### Code Structure Guidelines

- **Modular Design**: Each component has a specific responsibility
- **Configuration-Driven**: All parameters externalized to YAML
- **Logging Integration**: Comprehensive logging throughout
- **Type Hints**: Full type annotation for better development experience

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
   ```

2. **Memory Issues**:
   - Reduce `batch_size` in configuration
   - Reduce `max_folds` for shorter experiments

3. **SLURM Job Failures**:
   - Check logs in `logs/slurm_*.err`
   - Verify module and conda environment setup

4. **Data Loading Errors**:
   - Verify data format matches expected schema
   - Check column names in configuration

### Getting Help

- 📧 **Issues**: Open an issue on GitHub
- 📖 **Documentation**: Check the inline code documentation
- 🔍 **Logs**: Always check the experiment logs for detailed error information

---

**Happy Training! 🚀**