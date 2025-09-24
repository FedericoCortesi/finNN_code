# Introduction 

We will study 3 tasks meaningful to finance, trying to establish a common ground to benchmark neural networks in financial applications:
- Daily close level prediction from OHLCV data 
- Classification (Up/Down) in HFT from LOB data and/or high frequency bid/ask data.
- Volatility forecasting at different time frames from price data

## Price prediction
- Standard problem in financial economics (EMH) and financial engineering/econometrics

We propose the following dataset:
- An S&P500 ETF: we choose the Vanguard S&P 500 ETF (VOO) for its high liquidity and general agreement of the S&P500 being a good proxy for the market.
- 