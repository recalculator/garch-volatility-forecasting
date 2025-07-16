GARCH Volatility Forecasting
This project implements volatility forecasting on stock price data using GARCH(1,1) models with Python.

Overview
- Download historical stock data (e.g., AAPL) using yfinance
- Calculate daily returns
- Fit a GARCH(1,1) model using the arch package
- Forecast future volatility and compare it to realized volatility
- Visualize results with matplotlib

Why This Matters
Volatility forecasting is crucial in quantitative finance and risk management. GARCH models help estimate changing market risk over time.

How to Run
1. Create and activate a Python virtual environment
2. Install dependencies using: pip install -r requirements.txt
3. Run the main script using: python main.py
This will download the data, fit the model, and generate plots comparing forecasted and realized volatility.

Requirements
- Python 3.9+ recommended
- Packages listed in requirements.txt

Results
The output plot shows how the GARCH model’s forecasted volatility tracks the realized market volatility over time.