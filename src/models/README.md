# Models Directory

This directory contains neural network model architectures for financial time series forecasting. All models are registered in a factory pattern and can be instantiated dynamically from configuration.

## 📁 Contents

### Core Modules

- **`__init__.py`** – Model factory and registry
  - Decorator-based model registration system
  - `create_model()` factory function for instantiation
  - Lazy-loading of model modules
  - Returns properly configured `nn.Module` instances

- **`mlp.py`** – Multi-Layer Perceptron (MLP)
- **`lstm.py`** – Long Short-Term Memory (LSTM) 
- **`simplecnn.py`** – 1D Convolutional Neural Network (CNN)
- **`transformer.py`** – Transformer with positional encoding

## 🏗️ Architecture Overview

All models follow a consistent interface:

```python
model = create_model(
    model_cfg=ModelConfig(name="lstm", hparams={...}),
    input_shape=(T, D),      # (sequence_length, features)
    output_shape=(1,)        # regression output dimension
)
```


## 📊 MLP Regressor

**File**: `mlp.py`

Fully connected feedforward network for flattened sequential data.

### Architecture
```
Input(T,) → Dense(h₁) → Activation → Dropout → ... → Dense(h_n) → Output(1)
```

### Key Features
- Customizable hidden layer sizes via `mlp_hidden_sizes`
- Per-layer activation functions (ReLU, GELU, Tanh, Sigmoid, LeakyReLU)
- Optional dropout between layers
- Output activation support (usually Linear for regression)

### Hyperparameters
```yaml
model:
  name: mlp
  hparams:
    mlp_hidden_sizes: [256, 128, 64]        # Hidden layer dimensions
    mlp_activation: [relu, relu, relu]      # Per-layer activations
    dropout_rate: 0.2                        # Dropout probability
    output_activation: null                  # Output layer activation
```

### Input/Output
- **Input**: Flattened sequence (T,) - e.g., last 20 days of returns
- **Output**: Single value - predicted volatility/return

---

## 🔄 LSTM Regressor

**File**: `lstm.py`

Stacked LSTM architecture with configurable hidden dimensions per layer.

### Architecture
```
Input(T, D) → LSTM₁(h₁) → Dropout → LSTM₂(h₂) → ... → Readout → MLP Head → Output(1)
```

### Key Features
- **Per-layer LSTM dimensions** - Each LSTM layer can have different hidden size
- **Bidirectional option** - Optional bidirectional processing
- **Multiple readout modes**:
  - `last`: Use final timestep hidden state
  - `mean`: Average pooling across timesteps
  - `max`: Max pooling across timesteps
- **Layer normalization** option with `use_ln`
- **MLP head** for final transformation
- **Orthogonal weight initialization** for forget gates

### Hyperparameters
```yaml
model:
  name: lstm
  hparams:
    lstm_hidden_sizes: [256, 128]          # Per-layer hidden sizes
    lstm_dropout: 0.1                       # Inter-layer dropout
    bidirectional: false                    # Bidirectional processing
    use_ln: true                            # Layer normalization
    readout: last                           # Readout mode
    mlp_hidden_sizes: [64]                  # MLP head layers
    mlp_activation: [relu]                  # MLP activations
    dropout_rate: 0.1                       # MLP dropout
```

### Input/Output
- **Input**: Sequence (T, D) - e.g., (20, num_features)
- **Output**: Single value - prediction

### Forward Pass
1. Each LSTM layer processes input sequentially
2. Inter-layer dropout applied between LSTM stacks
3. Readout selects relevant features (last, mean, or max)
4. MLP head transforms to output dimension

---

## 🔷 1D CNN Regressor

**File**: `simplecnn.py`

Convolutional architecture for temporal feature extraction.

### Architecture
```
Input(1, T) → [Conv1d → BatchNorm → Activation]* → AdaptivePool → MLP Head → Output(1)
```

### Key Features
- **Stacked Conv1d layers** with configurable channels
- **Batch normalization** option per layer
- **Multiple pooling types**:
  - `adaptive_avg`: Adaptive average pooling
  - `adaptive_max`: Adaptive max pooling
- **Configurable kernel size and padding**
- **Fully connected head** for final regression

### Hyperparameters
```yaml
model:
  name: simplecnn
  hparams:
    conv_channels: [64, 128, 256]          # Output channels per conv layer
    conv_activation: [relu, relu, relu]    # Per-layer activations
    use_bn: true                            # Batch normalization
    kernel_size: 3                          # Convolution kernel size
    padding: 1                              # Zero padding
    pool: adaptive_max                      # Pooling type
    pool_k: 4                               # Pooling output size
    mlp_hidden_sizes: [64]                  # FC layers after pooling
    mlp_activation: [relu]                  # FC activations
    dropout_rate: 0.1                       # Dropout rate
```

### Input/Output
- **Input**: Image-like (1, T) - Time series as single channel
- **Output**: Single value - prediction

### Design Notes
- Processes temporal sequences as spatial convolutions
- Good for learning local temporal patterns
- Pooling reduces dimensionality before MLP head

---

## 🎯 Transformer Regressor

**File**: `transformer.py`

Multi-head attention-based architecture with positional encoding.

### Architecture
```
Input(T, D) → Projection → PositionalEncoding → TransformerEncoder → Readout → MLP Head → Output(1)
```

### Key Features
- **Positional encoding** for temporal position awareness
- **Multi-head self-attention** with configurable heads and layers
- **Feed-forward sub-layers** with configurable dimensions
- **Multiple projection modes**:
  - `linear`: Simple linear projection
  - `conv`: Convolutional stem (3 layers with 3×3 kernels)
- **Multiple readout modes** (last, mean, max)
- **Optional layer normalization**

### Hyperparameters
```yaml
model:
  name: transformer
  hparams:
    d_model: 128                            # Model embedding dimension
    nhead: 8                                # Number of attention heads
    num_layers: 2                           # Transformer encoder layers
    dim_feedforward: 512                    # Feed-forward hidden size
    transformer_dropout: 0.1                # Attention dropout
    pe_dropout: 0.05                        # Positional encoding dropout
    activation: gelu                        # Feed-forward activation
    projection: linear                      # Input projection type
    readout: mean                           # Readout strategy
    use_ln: false                           # Layer normalization
    mlp_hidden_sizes: [64]                  # MLP head layers
    mlp_activation: [relu]                  # MLP activations
    dropout_rate: 0.0                       # MLP dropout
```

### Input/Output
- **Input**: Sequence (T, D) - e.g., (20, num_features)
- **Output**: Single value - prediction

### Forward Pass
1. Input projected to `d_model` dimension
2. Positional encoding adds temporal position information
3. Transformer encoder processes full sequence with attention
4. Readout extracts features from encoded sequence
5. MLP head produces final output

### Attention Mechanism
- Multi-head self-attention enables learning different context types
- Parallel processing of all timesteps (unlike RNNs)
- Can capture long-range dependencies efficiently

---

## 🔧 Model Factory

### Registration System

Models use a decorator-based registry:

```python
@register_model("mlp")
class MLPRegressor(nn.Module):
    def __init__(self, hparams, input_shape, output_shape):
        ...
```

### Creating Models

```python
from models import create_model

model = create_model(
    model_cfg=ModelConfig(name="lstm", hparams={...}),
    input_shape=(20, 5),   # (sequence_length, features)
    output_shape=(1,)
)

# Model is ready for training
output = model(input_tensor)
```

### Supported Names
- `"mlp"` → MLPRegressor
- `"lstm"` → LSTMRegressor  
- `"simplecnn"` → CNN1D
- `"transformer"` → TransformerRegressor

---

## 📋 Common Activation Functions

Supported across all models:

- **ReLU**: Rectified Linear Unit (default for hidden layers)
- **GELU**: Gaussian Error Linear Unit (Transformer default)
- **Tanh**: Hyperbolic tangent
- **Sigmoid**: Sigmoid activation
- **LeakyReLU**: Leaky variant of ReLU
- **Linear**: Identity (no activation)

---

## 🎛️ Hyperparameter Search Configuration

All models support Optuna-based search via the search block:

```yaml
model:
  search:
    n_layers:
      type: int
      low: 1
      high: 4
    width:
      type: int
      low: 64
      high: 512
    dropout_rate:
      type: float
      low: 0.0
      high: 0.5
```

---

## 📝 Implementation Notes

### Initialization
- **LSTM**: Orthogonal initialization for recurrent weights, forget gate bias = 1.0
- **CNN**: Standard PyTorch defaults
- **Transformer**: Standard PyTorch defaults
- **MLP**: Standard PyTorch defaults

### Batch Dimensions
All models expect batched inputs with shape `(batch_size, ...)`:
- MLP: `(B, T)` for flattened sequences
- LSTM/Transformer: `(B, T, D)` for sequential data  
- CNN: `(B, 1, T)` for 1D convolutions

### Device Handling
Models are device-agnostic. Move to device before forward pass:
```python
model = model.to(device)
output = model(input.to(device))
```

---

## 🔗 Integration

- **Training**: Integrated with `training_routine.Trainer`
- **Config**: Loaded from YAML via `config.config_types`
- **Hyperparameter Search**: Compatible with `hyperparams_search.search_utils`
- **Inference**: Used in `benchmarks.benchmark.py` for predictions

---

## ⚠️ Important Notes

- All models are registered as decorators at import time
- Factory uses lazy-loading to avoid circular imports
- Output layer is always linear (suitable for regression)
- Batch-first convention throughout (easier for time series)
- Input shapes must match expectations (use preprocessing if needed)
