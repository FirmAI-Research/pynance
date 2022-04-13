import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import datetime as dt
import sys, os 
from datetime import date
import matplotlib.pyplot as plt 
import scipy 

cwd = os.getcwd()
p = cwd + '/io/'


initial_investment = 1000000
conf_level1 = 0.05 # At the 95% confidence interval; losses should not exceed x

# Create our portfolio of equities
tickers = ['AAPL','FB', 'C', 'DIS']
weights = np.array([0.25,0.25,0.25,0.25])

def build_portfolio_returns(tickers):
    import inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    sys.path.append(parentdir)
    from Performance import Returns as priceRet      # import Performance Module to retrieve portfolio daily price returns as pd.DataFrame()
    returns = priceRet.price_returns(symbol = tickers, date_start = '2020-04-21', time_sample='D', cumulative= True, plot = True)
    return returns
price_returns = build_portfolio_returns(tickers)

returns = price_returns.pct_change()
print(returns.tail(3))

cov_matrix = returns.cov()
print(cov_matrix.tail(3))

avg_rets = returns.mean()
port_mean = avg_rets.dot(weights)

port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
mean_investment = (1+port_mean) * initial_investment
stdev_investment = initial_investment * port_stdev


from scipy.stats import norm
cutoff1 = norm.ppf(conf_level1, mean_investment, stdev_investment)

var_1d1 = initial_investment - cutoff1
print(var_1d1)


# Calculate n Day VaR
var_array = []
num_days = int(252)
for x in range(1, num_days+1):    
    var_array.append(np.round(var_1d1 * np.sqrt(x),2))
    print(str(x) + " day VaR @ 95% confidence: " + str(np.round(var_1d1 * np.sqrt(x),2)))

# Build plot
plt.xlabel("Day #")
plt.ylabel("Max portfolio loss (USD)")
plt.title("Max portfolio loss (VaR) over 15-day period")
plt.plot(var_array, "r")
plt.show()
plt.savefig(f'{p}Portfolio_VaR.png')


# VaR assumes that the price returns of our assets are normaly distributed

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
# Repeat for each equity in portfolio
for c in price_returns.columns:
    returns[c].hist(bins=40, histtype="stepfilled",alpha=0.5)
    x = np.linspace(port_mean - 3*port_stdev, port_mean+3*port_stdev,100)
    plt.plot(x, scipy.stats.norm.pdf(x, port_mean, port_stdev), "r")
    plt.title(f"{c} returns (binned) vs. normal distribution")
    plt.savefig(f'{p}histo_{c}.png')

    plt.show()