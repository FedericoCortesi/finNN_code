# Financial Tests Directory

This directory contains financial analysis and empirical testing of volatility forecasts and portfolio strategies. It includes notebooks for data preparation and scripts for statistical and portfolio analysis.

## 📁 Contents

### Core Notebooks

- **`create_df.ipynb`** – Data preparation and feature engineering
  - Loads S&P 500 daily OHLCV data
  - Applies walk-forward cross-validation splits
  - Generates neural network predictions (CNN, LSTM, MLP, Transformer)
  - Computes baseline predictions (OLS, LASSO)
  - Creates merged datasets for analysis
  - Handles data scaling and normalization
  - Outputs processed data with predictions for downstream analysis

### Analysis Scripts

- **`empiricals.py`** – Financial empirical analysis and portfolio construction
  - Implements volatility-managed portfolio strategies
  - Constructs equal-weight portfolios by quintiles
  - Computes performance metrics:
    - Annualized returns and volatility
    - Sharpe ratio
    - Maximum drawdown
    - Portfolio turnover
  - Analyzes volatility-sorted portfolios
  - Generates publication-ready tables and figures
  - Compares performance across different model architectures and optimizers

### Data Files

**Forecast Data:**
- `sp500_vol_forecasts_2000_2024_v2.parquet` – Main volatility forecast dataset (2000-2024)
- `sp500_vol_forecasts_2000_2024_v2_extended.parquet` – Extended version with additional features
- `sp500_vol_forecasts_2000_2024_v2_extended_notransform.parquet` – Without standardization
- `df_preds_vol.parquet` – Processed predictions dataframe

**Required columns in forecast data:**
- `date` – Trading date
- `permno` – Stock identifier
- `ret` – Stock return
- `y` – Target volatility value
- `pred_<arch>_<L>_<optimizer>` – Model predictions (e.g., `pred_cnn_100_muon`, `pred_lstm_100_adam`)
- `ols` – OLS baseline predictions
- `lasso_0.05` – LASSO baseline predictions (with alpha=0.05)

### Output Directory

**`paper_figs_appendix/`** – Generated analysis outputs
- **Figures (PDF):**
  - `fig_frontier_Q1Q5_two_panel.pdf` – Efficient frontier for volatility-sorted portfolios
  - `fig_turnover_panels_*.pdf` – Portfolio turnover analysis by optimizer
  
- **Tables (CSV & LaTeX):**
  - `table_appendix_vol_sorted_summary.csv` – Summary statistics for volatility-sorted portfolios
  - `table_appendix_vol_sorted_summary.tex` – Publication-ready LaTeX table

## 🚀 Workflow

### Step 1: Prepare Data
Run `create_df.ipynb` to:
1. Load configuration for desired model(s)
2. Apply walk-forward cross-validation splits
3. Generate predictions using trained neural networks
4. Compute baseline predictions (OLS, LASSO)
5. Merge all predictions into analysis-ready dataframe

### Step 2: Analyze Results
Run `empiricals.py` to:
1. Load prediction dataframe
2. Construct volatility-sorted quintile portfolios
3. Compute portfolio performance metrics
4. Generate turnover and transaction cost analysis
5. Create figures and tables for paper/presentation

## 📊 Key Metrics & Functions

### Portfolio Construction
- **Quintile Assignment**: Splits volatility forecast into 5 groups (Q1=lowest vol, Q5=highest vol)
- **Equal-Weight Portfolios**: Each stock in quintile weighted equally
- **Rebalancing**: Daily or custom frequency

### Performance Metrics
- **Annualized Return**: $r_{ann} = \mu(r) \times 252$
- **Annualized Volatility**: $\sigma_{ann} = \sigma(r) \times \sqrt{252}$
- **Sharpe Ratio**: $S = \frac{r_{ann}}{\sigma_{ann}}$
- **Maximum Drawdown**: Peak-to-trough decline
- **Portfolio Turnover**: Average daily portfolio rebalancing cost

### Risk Models
- **Volatility-Managed Strategy**: Weights inversely proportional to predicted volatility
- **Baseline Strategies**: 
  - Equal-weight across all stocks
  - OLS volatility forecast baseline
  - LASSO volatility forecast baseline

## 🔧 Configuration

Models evaluated in the analysis:
- **Architectures**: CNN, LSTM, MLP, Transformer
- **Optimizers**: Muon, Adam, SGD
- **Hidden Dimensions**: 100-hidden units (default)

Rolling window parameters:
- `ROLL_TURN`: 126 days (~6 months) for rolling analysis
- `ROLL_TURN_LONG`: 252 days (~1 year) for annual analysis
- `ANN`: 252 (annualization factor for daily returns)

## 📝 Notes

- All analysis uses daily trading data from S&P 500
- Walk-forward validation ensures no look-ahead bias
- Predictions are standardized across train/val/test splits
- Portfolio construction assumes equal-weight rebalancing daily
- Turnover calculated accounting for stock entry/exit
- Results saved with timestamp for reproducibility
- LaTeX tables can be directly included in academic papers

## 🔗 Dependencies

- `pandas`, `polars` – Data manipulation
- `numpy` – Numerical computations
- `torch` – Neural network inference
- `matplotlib` – Visualization
- `pyarrow` – Parquet file I/O
- Configuration and pipeline utilities from main package

## ⚠️ Important

Ensure that the parquet files contain predictions from all required models before running `empiricals.py`. Model names must match the expected format: `pred_{arch}_{hidden_dim}_{optimizer}`
