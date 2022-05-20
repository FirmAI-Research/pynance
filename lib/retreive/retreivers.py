
###########################################################################################

tickers = ['GOOG','AMZN','MSFT','AAPL', 'BLK','JPM','IAU']
import pandas_datareader as data
import pandas as pd
data_source = 'yahoo'
start_date = '2000-01-01'
end_date = '2022-04-30'
frames = []
for ticker in tickers:
    stockdata = data.DataReader(ticker, data_source, start_date, end_date)['Close']
    stockdata =  pd.DataFrame(stockdata).rename(columns={'Close':ticker})
    frames.append(stockdata)
df = pd.concat(frames, axis=1)
df.to_csv('/Users/michaelsands/data/stock_prices.csv')

import datetime
start = datetime.datetime (2000, 1, 1)
end = datetime.datetime (2022, 4, 30)
seriesid = ['PAYEMS', 'GDP', 'UNRATE', 'CPILFESL', 'T10YIE', 'DGS10', 'DGS30']
frames = []
for ticker in seriesid:
    stockdata = data.DataReader(ticker, 'fred', start_date, end_date)
    frames.append(stockdata)
df = pd.concat(frames, axis=1)
df.to_csv('/Users/michaelsands/data/fred_prices.csv')



def historical_prices_cbind(arr= [],  date_start = '', date_end =None):
    if date_end==None:
        date_end = pd.datetime.now().date()
        print(date_end)
    df = yf.download(arr,date_start)['Adj Close']
    print(df.head())
    return df
