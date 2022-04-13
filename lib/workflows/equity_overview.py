import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import pandas_datareader as web
import yfinance as yfin

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

from datetime import datetime

from datasist import timeseries as dts

import ts_utils as util
import statsmodels.tsa.api as tsa


import sys, os
def multi_plot(df):
    ((df.pct_change()+1).cumprod()).plot(figsize=(10, 7))

    plt.legend()
    plt.title("Returns", fontsize=16)
    plt.ylabel('Cumulative Returns', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)

    plt.show()

def describe_data(df, SINGLE_TICKER):
    print(SINGLE_TICKER)
    print(df.head())
    print(df.describe())
    print(df.info())

def solo_analysis(df):
    df['Adj Close'].plot(legend=True,figsize=(12,5), color='green')
    plt.show()
    df['Volume'].plot(legend=True,figsize=(12,5), color='green')
    plt.show()


def returns_distro(df):
    nlag = [50,100,200]
    for ma in nlag:
        column_name = "%s_MA" %(str(ma))
        df[column_name] = df['Adj Close'].rolling(window=ma,center=False).mean()
    df.tail()
    df[['Adj Close','50_MA','100_MA','200_MA']].plot(subplots=False,figsize=(12,5))

    plt.show()

    df['Daily Return'] = df['Adj Close'].pct_change()
    df['Daily Return'].tail()
    df['Daily Return'].plot(figsize=(14,5),legend=True,linestyle='--',marker='o', color = 'green')
    plt.show()

    sns.distplot(df['Daily Return'].dropna(),bins=100,color='green')
    plt.show()

def correlate(df, TICKERS):
    df.tail()
    df_chg = df.pct_change()
    df_chg.tail()

    sns.jointplot(TICKERS[0],TICKERS[1],df,kind='scatter')
    plt.show()

    sns.pairplot(df.dropna())
    plt.show()

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, n =9, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

    plt.scatter(df.mean(),df.std(),s=25)
    plt.xlabel('Expected Return')
    plt.ylabel('Risk')
    plt.show()

    #For adding annotatios in the scatterplot
    for label,x,y in zip(df.columns,df.mean(),df.std()):
        plt.annotate(
        label,
        xy=(x,y),xytext=(-10,10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',)
        
def dts_time_series_plots(sdf, cdf,TICKERS):
    # solo
    timedf = sdf.reset_index()
    dts.timeplot(data=timedf, num_cols=['Adj Close', 'Volume'], time_col='Date', subplots=True, marker='.', 
                    figsize=(15,10), y_label='Daily Totals',save_fig=False, alpha=0.5, linestyle='None')

    # dts.time_boxplot(data=timedf, features=['High', 'Low'], x='Volume', subplots=True,)

    # combod df
    timedf = cdf.reset_index()
    print(timedf.head())
    dts.timeplot(data=timedf, num_cols=TICKERS, time_col='Date', subplots=True, marker='.', 
                    figsize=(15,10), y_label='Daily Totals',save_fig=False, alpha=0.5, linestyle='None')
                  

def practical_time_series(sdf, cdf):

    sdf = util.interpolate_na(data =sdf, col_name='Adj Close')
    # util.smoothing(data=sdf, col_name='Adj Close') # err
    util.check_stationarity(data =sdf, col_name='Adj Close')
    util.self_correlation(data =sdf, col_name='Adj Close')
    util.autoregression(data =sdf, col_name='Adj Close')
    util.arima(data =sdf, col_name='Adj Close')



def init():

    COMPETITORS =['AMZN','MSFT','GOOGL']
    SINGLE_TICKER = COMPETITORS[0]  

    def get_cbind_onloop(tickers_list):
        df = yfin.download(tickers_list,'2010-1-1')['Adj Close']
        print(df.tail(2))
        return df

    cdf = get_cbind_onloop(COMPETITORS)
    sdf = web.DataReader(SINGLE_TICKER, 'yahoo', start='2014', end=datetime(2017, 5, 24))

    
    # pass into decomposition volatility script
    import inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = (os.path.dirname(currentdir))
    sys.path.append(parentdir)
    from Attribution.Volatility import decomposition as decomp
    components = tsa.seasonal_decompose(sdf['Adj Close'], model='additive', freq = 30)
    decomp.additive_components(components)
    #differencing() #requires comparison; pass broader index
    decomp.autocorrelations(sdf['Adj Close'], the_title=SINGLE_TICKER)
    
    ''' init '''

    multi_plot(cdf)
    describe_data(cdf, SINGLE_TICKER)
    solo_analysis(sdf)
    return_df  = returns_distro(sdf)
    correlate(cdf, COMPETITORS)
    dts_time_series_plots(sdf, cdf, COMPETITORS)
    practical_time_series(sdf,cdf)





if __name__ == '__main__':
    init()