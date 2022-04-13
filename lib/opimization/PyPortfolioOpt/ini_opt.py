# build initial df using pandas_datareader
# date range
#qunadl, tiingo, pandas_datareader

import yahoo_fin as yf 

import pandas_datareader.data as web
import pandas as pd
from datetime import datetime
import numpy as np
date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='H')


timestamp_date_rng = pd.to_datetime(date_rng, infer_datetime_format=True)

# Import the yfinance. If you get module not found error the run !pip install yfinance from your Jupyter notebook
import yfinance as yf
import matplotlib.pyplot as plt

def multi_stock():
	tickers_list = ['MLM','ITW','GOOGL','AAPL','MSFT','JNJ','AMT','SPG','VZ','JPM','C','GSK','ACGL','RE','ATO']
	data = yf.download(tickers_list,'2015-1-1')['Adj Close']
	print(data.head())

	def plot():
		((data.pct_change()+1).cumprod()).plot(figsize=(10, 7))
		plt.legend()
		plt.title("Returns", fontsize=16)
		plt.ylabel('Cumulative Returns', fontsize=14)
		plt.xlabel('Year', fontsize=14)
		plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
		plt.show()
	#plot()
	return data
df = multi_stock()


from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

mu = mean_historical_return(df) # pandas series of estimated expected returns for each asset
S = CovarianceShrinkage(df).ledoit_wolf() #estimated covariance matrix 
print(mu)
print(S)


# define a loop to iterate through all the different objective functions
from pypfopt import objective_functions as objfunc



print()
print('max sharpe')
from pypfopt.efficient_frontier import EfficientFrontier
ef = EfficientFrontier(mu, S)
ef.add_objective(objfunc.L2_reg, gamma=0.1) # incentivize optimizer to choose non zero  weights
weights = ef.max_sharpe()
x = ef.portfolio_performance(verbose=True)
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("weights.txt")  # saves to file
print(cleaned_weights)

x = ef.portfolio_performance(verbose=True)
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices = get_latest_prices(df)
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=20000)
allocation, leftover = da.lp_portfolio()
print(allocation)
print(leftover)



print()
print('min_volatility')
ef = EfficientFrontier(mu, S)
ef.add_objective(objfunc.L2_reg, gamma=0.1) # incentivize optimizer to choose non zero  weights
w = ef.min_volatility()
cw = ef.clean_weights()
x = ef.portfolio_performance(verbose=True)
print(cw)
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices = get_latest_prices(df)
da = DiscreteAllocation(cw, latest_prices, total_portfolio_value=20000)
allocation, leftover = da.lp_portfolio()
print(allocation)
print(leftover)





