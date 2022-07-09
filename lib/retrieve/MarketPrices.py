import numpy as np
import pandas as pd 
import quandl

import yfinance as yf

from yahoo_fin.stock_info import *

import pandas_datareader as web


def historical_prices_cbind(arr= [],  date_start = '', date_end =None):
    """
    column bind time series of historical stock returns for an array of individual securities
    _________
    :param arr: array of individual tickers to retrieve historical prices for
    :type list: 

    :param plot: line chart of all columns in df   
    :type bool: 

    :param date_start: date to start retrieval of data
    :type pd.datetime: 

    :param date_end: date to end retrieval of data
    :type pd.datetime:  
    ----------
    :return: timeseries of historical prices
    :rtype: pd.dataframe
    _________
    """
    if date_end==None:
        date_end = pd.datetime.now().date()
        print(date_end)

    df = yf.download(arr,date_start)['Adj Close']
    print(df.head())

    return df




def historical_prices_solo(symbol = None, source = 'yahoo', date_start = '', date_end =None):
    """
    retrieve a historical time series of individual price returns for a single company
    _________
    :param symbol: the ticker symbol for a companies equity
    :type str: 

    :param source: the source to use to retrieve the data: yahoo, tiingo, quandl
    :type str:

    :param date_start: date to start retrieval of data
    :type pd.datetime: 

    :param date_end: date to end retrieval of data
    :type pd.datetime:  
    ----------
    :return: timeseries of historical prices
    :rtype: pd.dataframe
    _________
    """
    if date_end==None:
        date_end = pd.datetime.now().date()

    start = '2014'
    df = web.DataReader(symbol, 'yahoo', start=start, end=date_end)

    def get_quandl(self):
        symbol = symbol
        quandl = web.DataReader(symbol, 'quandl', '2015-01-01')
        quandl.info()

    def get_tiingo(self):
        df = web.get_data_tiingo(symbol, api_key=os.getenv('TIINGO_API_KEY'))
        return df
    
    return df




def get_yahoofin_quote_table(arr):
    """
    build an array of dates between two input dates
    _________
    :param symbol: start_date --> end_date
    :type pd.datetime: 
    ----------
    :return: timeseries of historical prices
    :rtype: pd.dataframe
    _________
    """
    frames = []
    for ticker in arr:
        print(str(ticker), sep='', end='  |  ', flush=True)
        try:
            quotedata = get_quote_table(ticker , dict_result = True)  # yahoo_fin
            print(quotedata)
            df = pd.DataFrame(list(quotedata.items()),columns = ['names',ticker])
            df = df[ticker]
            frames.append(df)
        except Exception as e:
            pass
    df = pd.concat(frames, axis = 1).transpose()
    df.columns = ['1yrTargetEst','52weekRange','Ask','AvgVol','Beta','Bid','DayRange','EPS',\
    'EarningsDate','ExDiv','DivYield','MktCap','Open','PE','Close','Quote','Volume']
    print(df.head())
    return df





def get_book_data(symbol):
    """
    retrieve a historical time series of individual price returns for a single company
    _________
    :param symbol: the ticker symbol for a companies equity
    :type str: 
    ----------
    :return:
    :rtype: pd.dataframe
    _________
    """
    book = web.get_iex_book(symbol)
    orders = pd.concat([pd.DataFrame(book[side]).assign(side=side) for side in ['bids', 'asks']])
    print(orders)
    list(book.keys())
    for key in book.keys():
        try:
            print(f'\n{key}')
            print(pd.DataFrame(book[key]))
        except:
            print(book[key])
    #pd.DataFrame(book['trades']).head()

get_book_data('AAPL')






def write_csv_onloop(arr= [],  date_start = '', date_end =None, ):
    """
    retrieve a historical time series of individual price returns for a single company
    _________
    :param symbol: the ticker symbol for a companies equity
    :type str: 

    :param source: the source to use to retrieve the data: yahoo, tiingo, quandl
    :type str:

    :param date_start: date to start retrieval of data
    :type pd.datetime: 

    :param date_end: date to end retrieval of data
    :type pd.datetime:  
    ----------
    :return: timeseries of historical prices
    :rtype: pd.dataframe
    _________
    """
    if date_end==None:
        date_end = pd.datetime.now().date()

    tickers = arr
    data_source = 'yahoo'
    for _ticker in tickers:
        df = web.DataReader(_ticker, data_source, date_start, date_end)
        p = f'{cwd}/output/individuals/{_ticker}.csv'
        df.to_csv(p)





def dt_tools(datestr):
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute


def get_daterange():
    """
    build an array of dates between two input dates
    _________
    :param symbol: start_date --> end_date
    :type pd.datetime: 
    ----------
    :return: timeseries of historical prices
    :rtype: pd.dataframe
    _________
    """
    date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='H')
    timestamp_date_rng = pd.to_datetime(date_rng, infer_datetime_format=True)
    return timestamp_date_rng




def plot_returns():
    import matplotlib.pyplot as plt 
    ((df.pct_change()+1).cumprod()).plot(figsize=(10, 7))
    plt.legend()
    plt.title(f"Returns for {arr}", fontsize=16)
    plt.ylabel('Cumulative Returns', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.show()


