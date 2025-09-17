# Deep Learning for Financial Time Series: Towards a Common Benchmark

> *"I can calculate the motion of heavenly bodies, but not the madness of people."*  
> — Isaac Newton  

This repository contains the initial steps of a research project on **deep learning applications in finance**, with a particular focus on **price prediction** and **volatility forecasting** of liquid financial instruments.  

Despite a wide body of literature, results in this field remain **mixed** and **difficult to compare**. The central aim of this project is to **establish a common ground** for evaluating models, similar to how **ImageNet** standardized progress in computer vision.

---

## 📌 Motivation

The academic and practitioner communities have made significant progress in applying neural networks to financial data, but there are recurring problems:  

- **Lack of benchmark datasets**: Each paper tends to use different assets, time spans, and preprocessing pipelines.  
- **Lack of common evaluation metrics**: Unlike accuracy in classification tasks, finance lacks consensus on what metrics best capture predictive success.  
- **Heterogeneity of models**: LSTMs, RNNs, GNNs, and even reinforcement learning approaches are all applied, but without a unified baseline for comparison.  

Our goal is to **define the problem first**, then evaluate models under comparable conditions.

---

## 🧩 Research Questions

1. **What are the most relevant tasks?**  
   We start with:
   - **Price prediction**: At what horizon (e.g., 1-day ahead, 5-min ahead)? For what instruments?  
   - **Volatility prediction**: A critical input for risk management, options pricing, and trading strategies.  

2. **What makes a good benchmark?**  
   - **Dataset**: Needs to be liquid, broad-based, and representative.  
   - **Metric**: Should capture predictive accuracy in a way that is meaningful in finance (e.g., MSE, RMSE, directional accuracy, Sharpe ratio improvements).  

---

## 📚 Literature Review

We anchor our work in recent surveys and influential papers:  

- Bao et al. (2025): *Data-driven stock forecasting models based on neural networks*.  
- Ge et al. (2022): *Neural network-based approaches to volatility forecasting*.  
- Kumar et al. (2021): Exploration of heuristic training methods (e.g., particle swarm optimization).  
- Hudson River Trading (2022): Practitioner insights into how deep learning is (and isn’t) used in trading.  

These sources suggest **price and volatility forecasting dominate** the literature, while portfolio optimization tasks remain fragmented.

---

## 📊 Proposed Benchmark Dataset

We propose the **Russell 2000 index** and its liquid ETFs as a starting point:  

- **Diversity**: Less concentrated than the S&P 500, avoiding dominance by the "Magnificent 7."  
- **Liquidity**: Multiple ETFs (e.g., IWM) provide easy replication and historical coverage.  
- **Representativeness**: Captures a broad cross-section of U.S. industries, including small and mid-cap stocks often neglected in benchmarks.  
- **Extensibility**: Can be paired with sectoral or cross-asset data to test generalization.  

---

## ⚙️ Methodological Outline

1. **Data Pipeline**
   - Data sourcing (Bloomberg, WRDS, or open data APIs like Yahoo Finance).  
   - Preprocessing: normalization, log-returns, volatility proxies.  
   - Train/test splits with walk-forward validation.  

2. **Models to Compare**
   - Baselines: ARIMA, GARCH, random forests.  
   - Deep Learning: LSTM, GRU, Transformers, GNNs.  
   - Hybrid: NN + volatility models, or NN + RL for sequential decision-making.  

3. **Loss Functions**
   - Regression-based (MSE, RMSE, MAE).  
   - Finance-aware (quantile loss, volatility-scaled loss, directional accuracy).  

---

## ✅ Goals of This Repository

- Define **clear tasks** in financial prediction.  
- Provide **clean datasets and preprocessing scripts** for replication.  
- Establish **baseline results** across classical and deep learning models.  
- Contribute toward a **shared benchmark** for financial deep learning research.  

---

## 📂 Repository Structure


