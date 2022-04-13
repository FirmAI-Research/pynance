from pathlib import Path

import numpy as np
from numpy.linalg import inv
from numpy.random import dirichlet
import pandas as pd

from sympy import symbols, solve, log, diff
from scipy.optimize import minimize_scalar, newton, minimize
from scipy.integrate import quad
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns

import stooq as sq

share, odds, probability = symbols('share odds probability')
Value = probability * log(1 + odds * share) + (1 - probability) * log(1 - share)
solve(diff(Value, share), share)

f, p = symbols('f p')
y = p * log(1 + f) + (1 - p) * log(1 - f)
solve(diff(y, f), f)


prices = sq.get_stooq_prices_and_tickers()
sp500 = prices.close


annual_returns = sp500.resample('A',level = 'date').last().pct_change().dropna().to_frame('sp500')
return_params = annual_returns.sp500.rolling(25).agg(['mean', 'std']).dropna()
return_ci = (return_params[['mean']]
                .assign(lower=return_params['mean'].sub(return_params['std'].mul(2)))
                .assign(upper=return_params['mean'].add(return_params['std'].mul(2))))
return_ci.plot(lw=2, figsize=(14, 8))
plt.tight_layout()
sns.despine();
plt.show()


def norm_integral(f, mean, std):
    val, er = quad(lambda s: np.log(1 + f * s) * norm.pdf(s, mean, std), 
                               mean - 3 * std, 
                               mean + 3 * std)
    return -val

def norm_dev_integral(f, mean, std):
    val, er = quad(lambda s: (s / (1 + f * s)) * norm.pdf(s, mean, std), m-3*std, mean+3*std)
    return val


def get_kelly_share(data):
    solution = minimize_scalar(norm_integral, 
                        args=(data['mean'], data['std']), 
                        bounds=[0, 2], 
                        method='bounded') 
    return solution.x

annual_returns['f'] = return_params.apply(get_kelly_share, axis=1)
return_params.plot(subplots=True, lw=2, figsize=(14, 8));
plt.show()
annual_returns.tail()
#(annual_returns[['sp500']]
# .assign(kelly=annual_returns.sp500.mul(annual_returns.f.shift()))
# .dropna()
# .loc['1900':]
# .add(1)
# .cumprod()
# .sub(1)
# .plot(lw=2));
annual_returns.f.describe()
return_ci.head()


m = .058
s = .216
# Option 1: minimize the expectation integral
sol = minimize_scalar(norm_integral, args=(m, s), bounds=[0., 2.], method='bounded')
print('Optimal Kelly fraction: {:.4f}'.format(sol.x))

# Option 2: take the derivative of the expectation and make it null
x0 = newton(norm_dev_integral, .1, args=(m, s))
print('Optimal Kelly fraction: {:.4f}'.format(x0))


import stooq as sq
prices = sq.get_stooq_prices_and_tickers()
prices.info()
monthly_returns = prices.loc['1988':'2017'].resample('M', level='date').last().pct_change().dropna(how='all').dropna(axis=1)
stocks = monthly_returns.columns
monthly_returns.info()
cov = monthly_returns.cov()
precision_matrix = pd.DataFrame(inv(cov), index=stocks, columns=stocks)
kelly_allocation = monthly_returns.mean().dot(precision_matrix)
kelly_allocation.describe()
kelly_allocation.sum()
kelly_allocation[kelly_allocation.abs()>5].sort_values(ascending=False).plot.barh(figsize=(8, 10))
plt.yticks(fontsize=12)
sns.despine()
plt.tight_layout();


ax = monthly_returns.loc['2010':].mul(kelly_allocation.div(kelly_allocation.sum())).sum(1).to_frame('Kelly').add(1).cumprod().sub(1).plot(figsize=(14,4));
sp500.filter(monthly_returns.loc['2010':].index).pct_change().add(1).cumprod().sub(1).to_frame('SP500').plot(ax=ax, legend=True)
plt.tight_layout()
sns.despine();
plt.show()