# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 12:40:45 2025

@author: Jacon
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest
from scipy.optimize import minimize

# risk-free rate (annual).
r_f = 0.0

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

n = len(tickers)
w_eq = np.array([1.0/n]*n)          # equal weights
mu = mu_annual.values               # vector of annual returns
cov = cov_annual.values             # annual covariance matrix

# helper functions: mu_p = w^T.mu sigma_p = sqrt(w^T.covmat(w))
def port_return(w, mu_vec):
    """Annual expected portfolio return given weights and mu vector"""
    return float(np.dot(w, mu_vec))

def port_vol(w, cov_mat):
    """Annual portfolio volatility (std)"""
    return float(np.sqrt(np.dot(w.T, np.dot(cov_mat, w))))

ret_eq = port_return(w_eq, mu)
vol_eq = port_vol(w_eq, cov)
sharpe_sh = (ret_eq - r_f) / vol_eq
print("Equal weight portfolio")
print("Weights:", dict(zip(tickers, np.round(w_eq,3))))
print(f"Expected return (annual): {ret_eq:.2%}")
print(f"Volatility (annual): {vol_eq:.2%}")
print(f" Sharpe: {sharpe_sh:.3f}")


def neg_sharpe(w, mu_vec, cov_mat, rf=r_f):
    ret = port_return(w, mu_vec)     # w^T * mu  (expected return)
    vol = port_vol(w, cov_mat)       # sqrt(w^T Σ w)  (volatility)
    if vol == 0:                     # avoid divide by zero
        return 1e6
    return - (ret - rf) / vol        # negative Sharpe

cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}) #weights must sum to 1 (fully invest)
bounds = tuple((0,1) for _ in range(n)) #no short selling (positive weight)
w0 = w_eq.copy() #start at equal

res = minimize(neg_sharpe, w0, args=(mu, cov),
               method='SLSQP', bounds=bounds, constraints=cons) # Sequential least squares programming (finds portfolio with lowest neg sharpe => highest sharpe)

w_sharpe = res.x
ret_sh = port_return(w_sharpe, mu)
vol_sh = port_vol(w_sharpe, cov)
sharpe_sh = (ret_sh - r_f) / vol_sh

print("Max-Sharpe portfolio (no shorting):")
print("Weights:", dict(zip(tickers, np.round(w_sharpe,3))))
print(f"Return: {ret_sh:.2%}, Vol: {vol_sh:.2%}, Sharpe: {sharpe_sh:.3f}")

r_min = min(mu)
r_max = max(mu)
target_returns = np.linspace(r_min, r_max, 40)

frontier_vols = []
frontier_weights = []

for r_target in target_returns:
    cons = (
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    {'type': 'eq', 'fun': lambda w, r_target=r_target: port_return(w, mu) - r_target}
    )
    res = minimize(lambda w: port_vol(w, cov), w0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        frontier_vols.append(np.nan)
        frontier_weights.append(None)
    else:
        frontier_vols.append(res.fun)
        frontier_weights.append(res.x)
        
plt.plot(frontier_vols, target_returns, label='Efficient frontier', lw=2)
plt.scatter(vol_eq, ret_eq, c='gray', marker='o', label='Equal weight')
plt.scatter(vol_sh, ret_sh, c='red', marker='*', s=150, label='Max Sharpe')
