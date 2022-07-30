import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import sys, os 
from datetime import date
import yfinance as yf
from functools import reduce
from scipy.stats import norm

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy

from lib.calendar import Calendar
cal = Calendar()

cwd = os.getcwd()
p = cwd + '/io/'

class VaR():

    def __init__(self, symbols, weights, start_date, initial_capital=1000000, conf_level = 0.05 ) -> None:
        
        self.symbols = symbols

        self.weights = weights
        
        self.start_date = start_date

        self.initial_capital = initial_capital

        self.conf_level = conf_level # At the 95% confidence interval; losses should not exceed x


    def get_returns(self):

        frames = []

        for ix, symbol in enumerate(self.symbols):
    
            returns = pd.DataFrame(yf.download(symbol, self.start_date, cal.today(), progress=False)['Adj Close'].pct_change().dropna())

            returns.rename(columns={'Adj Close': symbol}, inplace=True)

            frames.append(returns) 
        
        df = reduce(lambda x, y: pd.merge(x, y, on='Date', how = 'outer'), frames) # Each underlying holding may have a different date range available, so outer join and fillna returns as 0

        df.fillna(0, inplace=True)
        
        df['portf_rtn'] = pd.DataFrame(df.dot(self.weights)) # calculate weighted returns via matrix multiplication

        self.portfolio_returns = df

        return df



    def calculate_value_at_risk(self):
        
        cov_matrix = self.portfolio_returns.iloc[:, :-1].cov() # drop the last column to exclude the weighted portfolio return calculation 'portf_rtn'

        port_stdev = np.sqrt(np.array(self.weights).T.dot(cov_matrix).dot(np.array(self.weights)))

        mean_investment = (1+self.portfolio_returns) * self.initial_capital

        stdev_investment =  self.initial_capital * port_stdev

        cutoff = norm.ppf(self.conf_level, mean_investment, stdev_investment)

        var_1d1 = self.initial_capital - cutoff
        
        # Calculate n Day VaR
        var_array = []
        for x in range(1, 252+1):    
            var_array.append(np.round(var_1d1[x] * np.sqrt(x),2))
            # print(str(x) + " day VaR @ 95% confidence: " + str(np.round(var_1d1[x] * np.sqrt(x),2)))

        plt.xlabel("Day #")
        plt.ylabel("Max portfolio loss (USD)")
        plt.title("Max portfolio loss (VaR) over 252-day period")
        plt.plot(var_array, "r")
        plt.show()

        # for c in self.portfolio_returns.iloc[:, :-1].columns:
        #     self.portfolio_returns[c].hist(bins=40, histtype="stepfilled",alpha=0.5)
        #     x = np.linspace(self.portfolio_returns.iloc[:, :-1] - 3*port_stdev, self.portfolio_returns.iloc[:, :-1]+3*port_stdev,100)
        #     plt.plot(x, scipy.stats.norm.pdf(x, self.portfolio_returns.iloc[:, :-1], port_stdev), "r")
        #     plt.title(f"{c} returns (binned) vs. normal distribution")
        #     plt.show()



