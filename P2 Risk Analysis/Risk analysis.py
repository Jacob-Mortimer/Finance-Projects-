# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 17:36:11 2025

@author: Jacon
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest


tickers = ["AAPL","MSFT","AMZN","JPM","XOM","TSLA"]
start = "2022-01-01"
end = "2024-12-31"

data = yf.download(tickers, start=start, end=end)
df = data['Close']          # just the closing prices
df = df.dropna()            # clean missing values

returns = np.log(df / df.shift(1)).dropna()
trading_days = 252

# annualised mean and vol
mean_daily = returns.mean()
cov_daily = returns.cov()

mu_annual = mean_daily * trading_days
cov_annual = cov_daily * trading_days
vol_annual = np.sqrt(np.diag(cov_annual))

# cumulative returns
cumulative = (1 + returns).cumprod()

plt.figure(figsize=(10,6))
for c in cumulative.columns:
    plt.plot(cumulative.index, cumulative[c], label=c)

plt.title("Cumulative Returns (2022–2024)")
plt.legend()
plt.show()

# Sharpe ratio (assuming r_f = 0 for now)
sharpe_ratios = mu_annual / vol_annual
print("Annualised Sharpe Ratios:\n", sharpe_ratios)

# correlation of daily returns
corr = returns.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Matrix of Daily Returns (2022–2024)")
plt.show()

VaR_hist = returns.quantile(0.05, axis=0)

print("Historical 95% Daily VaR:")
print(VaR_hist)

n_sims = 100000

VaR_mc = {}

for ticker in returns.columns:
    mu = mean_daily[ticker]
    sigma = returns[ticker].std()

    # Simulate n_sims returns from Normal(mu, sigma)
    sims = np.random.normal(mu, sigma, n_sims)

    # Monte Carlo VaR = 5th percentile of simulated returns
    VaR_mc[ticker] = np.percentile(sims, 100*0.05)

VaR_mc = pd.Series(VaR_mc)
print("Monte Carlo 95% Daily VaR:")
print(VaR_mc)

tickers = returns.columns.tolist()
n = len(tickers)

# 1) Choose portfolio weights (example: equal weight)
w = np.array([1.0/n]*n)

# 2) Historical portfolio returns & Historical VaR (daily)
port_hist = returns.dot(w)                     # historical portfolio daily returns
VaR_hist = np.percentile(port_hist, 100 * 0.05)  # e.g. 5th percentile (negative number)

# 3) Monte Carlo: simulate correlated daily returns
# Use Cholesky of cov_daily
cov = cov_daily.values if isinstance(cov_daily, pd.DataFrame) else cov_daily
mu = mean_daily.values
L = np.linalg.cholesky(cov)

Z = np.random.normal(size=(n_sims, n))
sim_returns = Z.dot(L.T) + mu  # each row = simulated daily returns for assets
sim_port = sim_returns.dot(w)  # simulated portfolio returns

VaR_mc = np.percentile(sim_port, 100 * 0.05)

# 4) Expected Shortfall (Average loss in worst alpha fraction)
ES_hist = port_hist[port_hist <= VaR_hist].mean()
ES_mc = sim_port[sim_port <= VaR_mc].mean()

# 5) Backtest: count exceedances of historical dataset
exceedances = (port_hist <= VaR_mc).sum()     # how many historical days fell below simulated VaR
n_obs = len(port_hist)
obs_rate = exceedances / n_obs

# Binomial test (Kupiec-style idea): null = exceedance probability = 0.05
bt = binomtest(k=exceedances, n=n_obs, p=0.05)
p_value = bt.pvalue

# === 6) Print results ===
print("Portfolio weights:", dict(zip(tickers, np.round(w,3))))
print(f"N historical days = {n_obs}")
print(f"Historical VaR (daily, {100*(1-0.05):.0f}% conf) = {VaR_hist:.4%}")
print(f"Monte Carlo VaR (daily, {100*(1-0.05):.0f}% conf)  = {VaR_mc:.4%}")
print(f"Historical ES (avg loss | loss <= VaR_hist) = {ES_hist:.4%}")
print(f"Monte Carlo ES = {ES_mc:.4%}")
print(f"Observed exceedances vs expected: {exceedances} observed, expected ~ {0.05*n_obs:.1f}")
print(f"Observed exceedance rate = {obs_rate:.3%}")
print(f"Binomial test p-value (H0: exceedance rate = {0.05}) = {p_value:.3f}")

# Quick interpretation helper
if p_value < 0.05:
    print("=> Backtest FAILS at 5% level: model exceedance rate statistically different from 0.05.")
else:
    print("=> Backtest PASSES: exceedance rate consistent with 0.05 (no statistical rejection).")