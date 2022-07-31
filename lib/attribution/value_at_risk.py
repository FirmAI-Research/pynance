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
    '''
    
    Resoures:
        https://colab.research.google.com/gist/emarsden/f969dc6091162b3e5294731ec3c37b86/stock-market.ipynb#scrollTo=RtE_f5TFJk0o
    '''

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

    def plot_daily_percent_change(self):
        plt.rcParams["figure.figsize"] = (20,7)
        self.portfolio_returns['portf_rtn'].plot()
        plt.title("Daily Percent Change", weight="bold")
        plt.show()


    def plot_daily_percent_change_hist(self, normal = True):
        if normal == False:
            self.portfolio_returns['portf_rtn'].hist(bins=50, density=True, histtype="stepfilled", alpha=0.5)
            plt.title("Histogram of daily returns", weight="bold")
            plt.show()
        elif normal == True:

            tdf, tmean, tsigma = scipy.stats.t.fit(self.portfolio_returns['portf_rtn'])

            support = np.linspace(self.portfolio_returns['portf_rtn'].min(), self.portfolio_returns['portf_rtn'].max(), 100)
            self.portfolio_returns['portf_rtn'].hist(bins=40, density=True, histtype="stepfilled", alpha=0.5);
            plt.plot(support, scipy.stats.t.pdf(support, loc=tmean, scale=tsigma, df=tdf), "r-")
            plt.title("Daily change", weight="bold")
            plt.show()

        std = self.portfolio_returns['portf_rtn'].std()
        mean = self.portfolio_returns['portf_rtn'].mean()
        sigma = self.portfolio_returns['portf_rtn'].std()

        return std, mean, sigma


    def normal_distribution(self):
        scipy.stats.probplot(self.portfolio_returns['portf_rtn'], dist=scipy.stats.norm, plot=plt.figure().add_subplot(111))
        plt.title("Normal probability plot of daily returns", weight="bold")
        plt.show()
        var90 = self.portfolio_returns['portf_rtn'].quantile(0.1)
        var95 = self.portfolio_returns['portf_rtn'].quantile(0.05)
        var99 = self.portfolio_returns['portf_rtn'].quantile(0.01)

        var_values = pd.DataFrame([[var90, var95, var99]], columns = ['var90', 'var95', 'var99'])

        return var_values


    
    def monte_carlo_var(self):
        days = 300   # time horizon
        dt = 1/float(days)
        sigma = 0.04 # volatility
        mu = 0.05  # drift (average growth rate)

        def random_walk(startprice):
            price = np.zeros(days)
            shock = np.zeros(days)
            price[0] = startprice
            for i in range(1, days):
                shock[i] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
                price[i] = max(0, price[i-1] + shock[i] * price[i-1])
            return price

        plt.figure(figsize=(9,4))    
        for run in range(30):
            plt.plot(random_walk(10.0))
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.show()


        runs = 10_000
        simulations = np.zeros(runs)
        for run in range(runs):
            simulations[run] = random_walk(10.0)[days-1]
        q = np.percentile(simulations, 1)
        plt.hist(simulations, density=True, bins=30, histtype="stepfilled", alpha=0.5)
        plt.figtext(0.6, 0.8, "Start price: 10€")
        plt.figtext(0.6, 0.7, "Mean final price: {:.3}€".format(simulations.mean()))
        plt.figtext(0.6, 0.6, "VaR(0.99): {:.3}€".format(10 - q))
        plt.figtext(0.15, 0.6, "q(0.99): {:.3}€".format(q))
        plt.axvline(x=q, linewidth=4, color="r")
        plt.title("Final price distribution after {} days".format(days), weight="bold")
        plt.show()


    def calculate_value_at_risk_over_time(self):
        
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
        plt.ylabel("Max portfolio loss (%)")
        plt.title("Max portfolio loss (VaR) over 252-day period")
        plt.plot(var_array, "r")
        plt.show()

        # for c in self.portfolio_returns.iloc[:, :-1].columns:
        #     self.portfolio_returns[c].hist(bins=40, histtype="stepfilled",alpha=0.5)
        #     x = np.linspace(self.portfolio_returns.iloc[:, :-1] - 3*port_stdev, self.portfolio_returns.iloc[:, :-1]+3*port_stdev,100)
        #     plt.plot(x, scipy.stats.norm.pdf(x, self.portfolio_returns.iloc[:, :-1], port_stdev), "r")
        #     plt.title(f"{c} returns (binned) vs. normal distribution")
        #     plt.show()



