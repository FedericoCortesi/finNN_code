# Benchmarks Directory

This directory contains benchmarking, evaluation, and visualization tools for the finNN neural network pipeline. It includes scripts and notebooks for model comparison, performance analysis, and result visualization.

## 📁 Contents

### Core Scripts

- **`benchmark.py`** – Main benchmarking module
  - Model evaluation against baseline methods (OLS, LASSO)
  - Directional accuracy and undershooting metrics
  - Batch prediction utilities for efficient inference
  - Walk-forward validation framework integration
  - Supports single and ensemble model evaluation

- **`results_tables.py`** – Result aggregation and reporting
  - Generates summary tables from benchmark runs
  - Organizes results by model architecture and configuration

### Jupyter Notebooks

- **`forecast.ipynb`** ⭐ – **Main entry point for experiments**
  - Complete experimental pipeline from data loading to prediction
  - Recommended starting point for running the full workflow

- **`compare_runs.ipynb`** – Model comparison analysis
  - Side-by-side performance comparison across multiple runs
  - Statistical evaluation and ranking

- **`estimate_lambda_max.ipynb`** – EoSS estimation
  - Determines sharpness of the solution found by a given model

- **`visuals.ipynb`** – Visualization and plotting 
  - Older compared to `forecast.ipynb`
  - Result visualizations and comparative plots
  - Performance charts and diagnostics

### Output Directories

- **`tables/`** – Generated result tables (CSV format)
  - Contains timestamped results CSV files from benchmark runs
  - Example: `results_ensemble_02012026_121731.csv`

- **`images/`** – Generated visualizations
  - Plots and figures from `visuals.ipynb`
  - SNR (signal-to-noise ratio) analysis subdirectory
  - Model interaction analysis subdirectory


## 📝 Notes

- Predictions and metrics are normalized by target variance when `USE_NMSE=True`
- Training and validation sets can be merged for final evaluation with `MERGE_TRAIN_VAL=True`
- Results are timestamped for easy tracking of multiple experiment runs
- GPU acceleration is available for neural network inference when CUDA is present
