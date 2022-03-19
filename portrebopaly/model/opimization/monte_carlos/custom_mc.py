import pandas as pd
import numpy as np
import math
import datetime
from pandas_datareader import data, wb

import warnings
warnings.filterwarnings("ignore")
import xlsxwriter
import sys, os
poutdir = os.getcwd() + '/output/'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


import pandas as pd
import numpy as np
import datetime
import scipy.optimize as sco
from scipy import stats

import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

tickers = ['HYG', 'SPY', 'FDHY', 'GOVT', 'REET','LQD','IUSG','IWP','MDY','MBB','QQQ','USRT','FUTY','JKL','IWM','URTH','IAU','USMV', 'REZ','IDU','FMIL','MUB']

def build_df(tickers):
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2020, 9, 1)
    df = pd.DataFrame([data.DataReader(ticker, 'yahoo', start, end)['Adj Close'] for ticker in tickers]).T
    df.columns = tickers
    return df
 

print('| Historical DataFrame |')
df = build_df(tickers)
print(df.tail())
print(df.columns)

print('| Returns DataFrame |')
df_returns = df.pct_change()




def calculate_sortino_ratio(df_returns, tickerlist, num_iterations, rf, ntickers):
    print('Expected Returns From: ' + '1-1-2010 : 9-1-2020')
    sortino_dict, frames = {}, []

    for ticker in tickerlist:
        label, label2, label3 = str(ticker) + '_dRetBool' , str(ticker) + '_dRet', str(ticker) + '_expRet'

        ###  ###########*** Calculate Downside Deviation for Sortino Ratio  *** ###########
        # 1. Given the daily returns, for each security in portfolio, determine which days had downside returns
        df_returns.loc[df_returns[ticker] < 0, label] = True  # square the downside returns
        # 2. Square the value of the downside return days
        df_returns[label2] = [item for item in range(len(df_returns))]
        for i in range(len(df_returns.index)):
            if df_returns[label].iloc[i] == True:
                df_returns[label2].iloc[i] = df_returns[ticker].iloc[i] ** 2
            else:
                df_returns[label2].iloc[i] = 0
        # 3. Sum squared values from step 2
        downsum = df_returns[label2].sum()
        # 3b. Determine number of downside observations
        nobsv = len(df_returns.loc[df_returns[label] == True])
        # 4. Divide by number of observations
        deviate = downsum / nobsv
        # 5. Take square root of value from step 4
        sqdd = np.sqrt(deviate)  # squuared downside deviation
        # 6. Annualize the Squared downside deviation
        # asqdd = sqdd * math.pow(12, 1/2)
        asqdd = sqdd * np.sqrt(252)
        ### Expected (Average) Returns ###
        expected_return = df_returns[
                              ticker].sum() / 100  # ((df_returns[ticker].iloc[-1] - df_returns[ticker].iloc[1]) / df_returns[ticker].iloc[1]) #/ 100
        sortino = (expected_return - rf) / asqdd
        sortino_dict[ticker] = sortino # Create dictionary of Ticker Symbol and Sortino Value

    # Convert dict to dataframe
    sortino_df = pd.DataFrame.from_dict(sortino_dict, orient ='index').transpose() # sortino_df /= sortino_df.iloc[0].sum()
 
    # # Select random weights and normalize to set the sum to 1
    weights = np.array(np.random.random(len(sortino_df.iloc[0])))
    weights /= np.sum(weights)
    print(weights)
    weighted_sortino_df = np.sum(sortino_df * weights) * 100
    weighted_sortino_df /= weighted_sortino_df.sum()

    weighted_sortino_df['SUM'] = weighted_sortino_df.sum(axis=0) #Add SUM column of weighted portfolio allocation given to each symbol based on sortino ratio

    return weighted_sortino_df, sortino_df



def calculate_sharpe(df_returns):
    mean_returns = df_returns.mean()
    cov = df_returns.cov()
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return mean_returns, sharpe_ratio





num_iterations = 3
ntickers = len(tickers)
rf = 0.006
frames =[]
for i in range(num_iterations):
    print('Simulated Portfolio - Iteration # ' + str(i))
    simulated_portfolio_weighted_sortino, sortino_df = calculate_sortino_ratio(df_returns, tickers,num_iterations, rf, ntickers)
    frames.append(simulated_portfolio_weighted_sortino)



print('| Asset Sortino Ratio |' )
print(sortino_df)
print(' | Simulation Results Frame | ')

simulation_results_frame = pd.DataFrame(frames)
print(simulation_results_frame.tail())

totalreturns = (((df.iloc[-1] - df.iloc[0]) / df.iloc[0]) * 100)
totalreturndf = pd.DataFrame(totalreturns).T
print(totalreturndf)

simulation_results_frame['Score'] = ['' for item in simulation_results_frame.index]
for i in range(len(simulation_results_frame)):
    xx = simulation_results_frame.iloc[i] * totalreturndf.iloc[0]
    out = xx.sum()
    simulation_results_frame['Score'].iloc[i] = out

print(simulation_results_frame.columns)
print(simulation_results_frame.tail())











