# Pipeline Directory

This directory contains data processing, feature engineering, and walk-forward cross-validation functionality for time series financial prediction. It handles loading, preprocessing, windowing, and scaling of data.

## 📁 Contents

### Core Modules

- **`walkforward.py`** – Walk-Forward Cross-Validation (WFCV) generator
  - Time-aware temporal cross-validation splits
  - Feature window generation with configurable lag
  - Data scaling and normalization
  - Merge train/val for final evaluation

- **`preprocessing.py`** – Data loading and cleaning
  - CRSP data import and parsing
  - Stock split and share adjustments
  - Missing value handling (NaN, extreme values)
  - Target variable computation (returns, volatility)

- **`pipeline_utils.py`** – Utility functions
  - Multiple scaling strategies (StandardScaler, RobustScaler, transforms)
  - Variance injection for robustness testing
  - Data type conversions and list parsing

- **`__init__.py`** – Package initialization

### Notebooks

- **`pipeline_test.ipynb`** – Testing and validation notebook
  - Example usage of pipeline components
  - Data verification and diagnostics

---

## 🔄 Walk-Forward Cross-Validation

### Overview

Walk-forward validation is a time-aware cross-validation strategy for time series that prevents look-ahead bias. The dataset is split into sequential train/validation/test windows that walk forward in time.

### WFCVGenerator Class

**Purpose**: Generates temporal cross-validation folds with automatic feature windowing.

```python
from pipeline.walkforward import WFCVGenerator
from config.config_types import WFConfig

config = WFConfig(
    target_col="ret",           # Prediction target
    lags=20,                    # Number of past observations as features
    ratio_train=3,              # Train duration ratio
    ratio_val=1,                # Val duration ratio
    ratio_test=1,               # Test duration ratio
    step=251                    # Trading days per "year"
)

wf = WFCVGenerator(config)
for fold_data in wf.folds():
    Xtr, ytr, Xv, yv, Xte, yte, ... = fold_data
    # Use fold data for training
```

### Key Configuration Parameters

```yaml
walkforward:
  target_col: ret              # Column to predict
  lookback: 0                  # Additional past observations to include
  lags: 20                     # Feature window length (lookback window)
  ratio_train: 3               # Multiple of step size for train
  ratio_val: 1                 # Multiple of step size for validation
  ratio_test: 1                # Multiple of step size for test
  step: 251                    # Trading days per period (annual)
  max_folds: null              # Optional cap on number of folds
  min_folds: null              # Optional minimum number of folds
  scale: true                  # Apply scaling to data
  scale_type: standard         # Type of scaling (see below)
  annualize: false             # Annualize return statistics
  portfolios: 0                # Portfolio construction (0 = no)
  noise: []                    # Gaussian noise variance levels
  clip: 0                      # Winsorization parameter (quantile)
```

### Output Tuple Format

For each fold, `wf.folds()` yields:

```python
(
    Xtr,                # Train features (N_train, T, D) or (N_train, D)
    ytr,                # Train target (N_train,)
    Xv,                 # Validation features
    yv,                 # Validation target
    Xte,                # Test features
    yte,                # Test target
    Xtr_val,            # Train + Val features (merged)
    ytr_val,            # Train + Val target
    Xte_merged,         # Test features (from merged training)
    yte_merged,         # Test target
    id_tr,              # Stock IDs for train
    id_v,               # Stock IDs for validation
    id_te,              # Stock IDs for test
    window_train,       # Time indices for train window
    window_val,         # Time indices for val window
    window_test,        # Time indices for test window
    X_scaler,           # Fitted scaler for train
    y_scaler,           # Fitted target scaler
    X_scaler_merged,    # Fitted scaler for train+val
    y_scaler_merged     # Fitted target scaler for train+val
)
```

---

## 🔧 Data Preprocessing

### Preprocessing Pipeline

The `preprocessing.py` module provides:

1. **Data Import** - Load CRSP parquet data
2. **Split Adjustment** - Adjust for stock splits and share issuance
3. **Data Cleaning** - Handle missing values and outliers
4. **Feature Engineering** - Compute returns, volatility, and other metrics

### Split & Share Adjustments

CRSP data includes cumulative adjustment factors:

- **Price adjustment**: $adjusted = raw / cfacpr$ (cumulative price factor)
- **Volume adjustment**: $adjusted = raw \times cfacshr$ (cumulative share factor)

Adjustments applied within each stock (permno) forward-filled to handle missing factors.

### NaN Handling

- Forward-fill returns within each stock
- Remove stocks with < 1000 trading days
- Winsorize extreme values (default: none, configurable via `clip`)
- Interpolate missing prices in valid ranges

### Feature Computation

```python
def compute_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns from prices"""
    return np.diff(np.log(prices))

def compute_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling volatility (standard deviation)"""
    return pd.Series(returns).rolling(window).std().values
```

---

## 📊 Data Scaling Strategies

### Available Scalers

Implemented in `pipeline_utils.py`:

| Type | Formula | Use Case |
|------|---------|----------|
| `standard` | $(x - \mu) / \sigma$ | Default, assumes normality |
| `robust` | $(x - q50) / IQR$ | For outliers, uses quantiles |
| `asinh` | $\arcsinh(x)$ | Heavy-tailed distributions |
| `log` | $\log(x)$ | Positive values, multiplicative |
| `log1p` | $\log(1 + x)$ | Positive/negative, symmetric |
| `sqrt` | $\sqrt{1 + x}$ | Variance stabilization |
| `asinhstandard` | StandardScaler($\arcsinh(x)$) | Combined approach |
| `log1pstandard` | StandardScaler($\log(1+x)$) | Combined approach |

### Usage

```python
from pipeline.pipeline_utils import scale_split

Xtr_s, ytr_s, Xv_s, yv_s, Xte_s, yte_s, *merged = scale_split(
    Xtr, ytr, Xv, yv, Xte, yte,
    scale_type="standard",
    merge=True
)
```

**Key Properties**:
- Scaler fitted on **training data only** (no leakage)
- Validation and test transformed with training scaler
- Optional merging of train+val for final evaluation
- Type preservation (converts to float64)

---

## 📈 Feature Windowing

### Lag-Based Feature Construction

For each stock-date combination:

1. **Select lagged observations**: Last `lags` days of returns/features
2. **Create feature vector**: $(r_{t-lags}, ..., r_{t-1})$
3. **Target**: $y_t = r_t$ (or volatility, etc.)

### Window Example

```
Dates:      t-4   t-3   t-2   t-1   t
Returns:    -0.02  0.01  0.03  -0.01 0.02 ← Predicting this
                  |←─────────────────|
                  Feature window (lags=4)
```

### Efficient Windowing

The `WFCVGenerator` uses efficient numpy-based windowing:

1. **Pivot to wide format**: (permnos × dates)
2. **Apply rolling windows**: (permnos × lags+1)
3. **Extract and reshape**: Ready for model training

This approach avoids per-row DataFrame operations for speed.

---

## 🎯 Feature Engineering Options

### Variance Injection

Add synthetic noise for robustness testing:

```python
from pipeline.pipeline_utils import apply_variance_injection

X_noisy = apply_variance_injection(X, noise_levels=[0.01, 0.05, 0.10])
```

Enables evaluation under different signal-to-noise ratios (SNR).

### Portfolio Construction

Optional: Create portfolio-level aggregates instead of stock-level:

```yaml
walkforward:
  portfolios: 10  # Create 10 portfolios (e.g., deciles by volatility)
```

---

## 🔄 Typical Usage Workflow

### 1. Initialize Walk-Forward Generator

```python
from pipeline.walkforward import WFCVGenerator
from config.config_types import WFConfig

config = WFConfig(
    lags=20,
    ratio_train=3,
    ratio_val=1,
    ratio_test=1,
    step=251,
    scale=True
)

wf = WFCVGenerator(config)
```

### 2. Iterate Over Folds

```python
for fold_idx, fold_data in enumerate(wf.folds()):
    Xtr, ytr, Xv, yv, Xte, yte, *rest = fold_data
    
    # Train model on (Xtr, ytr)
    # Evaluate on (Xv, yv)
    # Test on (Xte, yte)
```

### 3. Get Predictions

```python
# Model predictions on each fold
y_pred_test = model.predict(Xte)

# Evaluation on validation
y_pred_val = model.predict(Xv)
```

---

## 📋 Important Notes

### Data Requirements

- **Date column**: Must be numeric, sorted, continuous
- **Stock IDs**: Unique identifier per security (permno)
- **Target values**: Must be numeric and finite
- **Minimum length**: At least 2×(T_train + T_val + T_test) observations

### Scaling Strategy

- **Always fit on train**: Prevent information leakage
- **Transform val/test**: Use training scaler for consistency
- **Optional merge**: Train+Val together for final test evaluation
- **Preserve scalers**: Saved in fold output for inference

### Memory Efficiency

- Uses numpy arrays (efficient memory layout)
- Pivots to wide format once per fold
- Slicing operations are O(n) without copying

### Time Consistency

- Walk-forward ensures **no look-ahead bias**
- Later folds use later data for training
- Each test fold is in the future relative to training
- Proper temporal cross-validation for time series

---

## 🔗 Integration

- **Training**: Used by `training_routine.Trainer` for fitting models
- **Config**: Loaded from `config.config_types.WFConfig`
- **Models**: All architectures support WF-generated data
- **Benchmarking**: `benchmarks.benchmark.py` uses WF for evaluation

---

## ⚠️ Common Issues

### Issue: "Missing required columns"
**Solution**: Ensure dataframe has `time_col`, `target_col`, and `id_col`

### Issue: "Duplicate (permno, t) rows"
**Solution**: Deduplicate before creating WFCVGenerator

### Issue: "Window size too large"
**Solution**: Reduce `lags` or increase data length

### Issue: "All NaN after scaling"
**Solution**: Check that input data contains finite values; review preprocessing

---

## 📚 References

- **Walk-Forward Analysis**: Pardo (2008) - proper temporal backtesting strategy
- **Time Series Validation**: Robert Nau's guide on forecasting validation
- **CRSP Data**: Center for Research in Security Prices documentation
