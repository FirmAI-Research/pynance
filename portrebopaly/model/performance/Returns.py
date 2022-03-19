import yfinance as yf
from datetime import date

today = date.today()

import matplotlib.pyplot as plt 

import pandas as pd
import numpy as np 


def price_returns(symbol = [], date_start = None, date_end = today, time_sample = "D", cumulative = False, plot = False):
    """
    calculate percent returns or compound returns for time series of prices
    _________
    :param symbol: individual or array of tickers to retrieve historical prices for
    :type list: 
    :param date_start: date to start retrieval of data
    :type pd.datetime: 
    :param date_end: date to end retrieval of data
    :type pd.datetime:  
    ----------
    :return: timeseries of historical prices
    :rtype: pd.dataframe
    _________
    """
    print(f"{date_start}:{date_end}")

    if len(symbol) < 2:
        if (cumulative == False):
            returns = yf.download(symbol,date_start,date_end)['Adj Close'].resample(time_sample).last().pct_change()
        else:
            df = yf.download(symbol,date_start,date_end).resample(time_sample).last()
            returns = (1 + df['Close'].pct_change()).cumprod() - 1
        if plot == True:

            def price_plot():
                plt.plot(returns)
                plt.legend()
                plt.show()
            price_plot()

            def change_plot():
                pass
                # sdf['Daily Return'] = sdf['Adj Close'].pct_change()
                # sdf['Daily Return'].tail()
                # sdf['Daily Return'].plot(figsize=(14,5),legend=True,linestyle='--',marker='o', color = 'green')  
            change_plot() 
        
            def distro_plot():
                #sns.distplot(sdf['Daily Return'].dropna(),bins=100,color='green')
                pass
            distro_plot()
    else:
        df = yf.download(symbol,date_start)['Adj Close']
        return df



        #todo: add multiplot and support arr instead of symbol
        # to do: datasist box plots? & timeplots?


    return returns 

#returns = price_returns(symbol = ['MSFT'], date_start = '2020-04-21', time_sample='D', cumulative= True, plot = True)




def portfolio_returns(arr = [], weights = [], date_start = '2021-01-01', date_end = today, time_sample='D', cumulative= False, plot = False):
    """
    column bind time series of historical stock returns for an array of individual securities
    _________
    :param symbol: individual ticker to retrieve historical prices for
    :type list: 
    :param date_start: date to start retrieval of data
    :type pd.datetime: 
    :param date_end: date to end retrieval of data
    :type pd.datetime:  
    ----------
    :return: timeseries of historical prices
    :rtype: pd.dataframe
    _________
    """
    if cumulative == False:
        ''' series of daily price returns (i.e. pct changes over time) '''
        df_returns = yf.download(arr,date_start)['Adj Close'].pct_change()[1:]
        weighted_returns = (weights * df_returns)
        df_returns['weighted_returns'] = weighted_returns.sum(axis=1)


    if plot == True:
        def histogram():
            fig = plt.figure()
            ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
            ax1.hist(port_ret, bins = 60)
            ax1.set_xlabel('Portfolio returns')
            ax1.set_ylabel("Freq")
            ax1.set_title("Portfolio Returns calculated manually")
            plt.show()
        
        histogram()

        def multiplot():
            ''' assumes cumulative retunrs compounding over time '''
            ((df_returns['weighted_returns'].pct_change()+1).cumprod()).plot(figsize=(10, 7))
            plt.legend()
            plt.title("Returns", fontsize=16)
            plt.ylabel('Cumulative Returns', fontsize=14)
            plt.xlabel('Year', fontsize=14)
            plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
            plt.show()
        multiplot()

    return df_returns['weighted_returns']

#returns = portfolio_returns(arr = ['MSFT','AAPL'], weights = [0.5,0.5], date_start = '2020-01-01', date_end = today, time_sample='D', cumulative= True, plot = True)
