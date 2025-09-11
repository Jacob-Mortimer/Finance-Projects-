# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 12:38:57 2025

@author: Jacon
"""
#Monte Carlo option pricing ST​=S0​⋅e^((r−0.5​σ^2)T+Zσ(t)^0.5)

import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

S0 = 100    # stock price today
K = 105     # strike price
T = 1.0     # time to maturity (1 year)
r = 0.05    # risk-free rate
sigma = 0.2 # volatility
n_sims = 10_000 # number of Monte Carlo simulations

# STEP 1: simulate stock price at maturity
Z = np.random.standard_normal(n_sims)  # random N(0,1)
ST = S0 * np.exp((r - 0.5 * sigma**2)*T + sigma*np.sqrt(T)*Z)

print(ST[:10])  # print first 10 simulated prices


payoffs = np.maximum(ST - K, 0.0)         # phi_i
N = len(payoffs)

# Monte Carlo estimator (discounted mean)
price_mc = math.exp(-r*T) * payoffs.mean()

# sample std of payoffs (ddof=1)
s_phi = payoffs.std(ddof=1)

# standard error of the estimator
se_mc = math.exp(-r*T) * s_phi / math.sqrt(N)

# 95% CI
z = 1.96
ci_lower = price_mc - z * se_mc
ci_upper = price_mc + z * se_mc

print(f"MC price      = {price_mc:.6f}")
print(f"SE (MC)       = {se_mc:.6f}")
print(f"95% CI        = [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"N simulations = {N}")

#Compare with BS

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

price_bs = black_scholes_call(S0, K, T, r, sigma)

abs_err = abs(price_mc - price_bs)
rel_err = abs_err / price_bs

print(f"Black–Scholes = {price_bs:.6f}")
print(f"Absolute err  = {abs_err:.6f}")
print(f"Relative err  = {rel_err:.4%}")

Ns = [1000, 5000, 10000, 20000, 50000, 100000]
estimates = []
ses = []

for n in Ns:
    Z = np.random.standard_normal(n)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoffs = np.maximum(ST - K, 0.0)
    est = np.exp(-r*T) * payoffs.mean()
    se = np.exp(-r*T) * payoffs.std(ddof=1) / (n**0.5)
    estimates.append(est)
    ses.append(se)

plt.errorbar(Ns, estimates, yerr=ses, fmt='o-')
plt.xscale('log')
plt.xlabel('N (log scale)')
plt.ylabel('Estimated Call Price')
plt.title('MC Convergence and SE')
plt.show()

