import yfinance as yf
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

# Download adjusted data and flatten the column structure
ticker = 'AAPL'
data = yf.download(ticker, start="2018-01-01", end="2024-12-31", group_by="column", auto_adjust=True)

# Calculate daily returns
returns = 100 * data['Close'].pct_change().dropna()

# Fit a GARCH(1,1) model
model = arch_model(returns, vol='Garch', p=1, q=1)
res = model.fit(disp="off")

# Forecast future volatility (next 5 days)
forecast = res.forecast(horizon=5)
volatility = forecast.variance[-1:] ** 0.5

# Plot recent returns
plt.figure(figsize=(10, 4))
plt.plot(returns[-100:], label='Returns')
plt.title(f"{ticker} Daily Returns")
plt.legend()
plt.show()

# Print forecast
print("Forecasted volatility for next 5 days:")
print(volatility)

# === STEP 1: Realized Volatility ===
# Rolling 5-day standard deviation as realized volatility
realized_vol = returns.rolling(window=5).std()

# Align both series to the same index (drop NaNs)
realized_vol = realized_vol.dropna()
garch_vol = res.conditional_volatility

# Match index (last N values only for clean plot)
plot_start = -200
realized_vol = realized_vol[plot_start:]
garch_vol = garch_vol[plot_start:]

# === STEP 2: Plot ===
plt.figure(figsize=(12, 5))
plt.plot(realized_vol, label="Realized Volatility (5-day rolling std)", color="blue")
plt.plot(garch_vol, label="GARCH Forecasted Volatility", color="orange")
plt.title(f"{ticker} - GARCH Forecasted vs Realized Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility (%)")
plt.legend()
plt.tight_layout()
plt.show()