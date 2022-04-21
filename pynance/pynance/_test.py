import sys,os
lib_dirp = os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )) # NOTE: relative path from manage.py to lib/
sys.path.append(lib_dirp) 

""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ @Test: learn recursive neural net                                                                                                                   │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """


# import lib.learn.rnn.rnn as rnn
# rnn.init()

# import lib.learn.rnn.rnn as rnn


""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ @Test: Time series analysis class in lib/                                                                                              │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
# import pandas as pd
# import numpy as np
# import seaborn as sns
# from lib.time_series.timeseries import TimeSeries
# fp = r"C:\dev\pynance\_tmp\ff\ff_factors.csv"
# ts = TimeSeries(data = fp)
# ts.decomposition()

""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ @Test: regression : find relevant features from fundamental data on future share prices                                                                                      │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
import pandas as pd
import numpy as np
import seaborn as sns
from lib.learn.regression.regression import Regression
# import pandas_datareader as wb
# start = '2000-01-01'
# end = '2021-12-31'
# ticker = 'AAPL'
# aapl = wb.DataReader(ticker, 'yahoo', start, end)
# aapl.to_csv(r"aapl_prices.csv")
# x = pd.read_csv('/Users/michaelsands/code/pynance/_tmp/SHARADAR-SF1_AAPL.csv')
# y = pd.read_csv('/Users/michaelsands/code/pynance/_tmp/aapl_prices.csv')
# y.Date = pd.to_datetime(y.Date)
# y.set_index('Date', inplace=True)
# y = y.resample('Q').last()
# print(y)

# x.calendardate = pd.to_datetime(x.calendardate)
# x.set_index('calendardate', inplace=True)
# x = x.loc[x['dimension'] == 'MRQ']
# print(x)

# df = x.merge(y, left_index=True, right_index=True)
df = pd.read_csv('/Users/michaelsands/code/pynance/_tmp/aapl_fundamental_features.csv')
df['NextMonthsClose'] = df['Adj Close'].shift(1)
df.dropna(subset=['NextMonthsClose'], inplace=True)
df.drop(columns=['datekey', 'reportperiod', 'lastupdated','Unnamed: 0', 'ticker', 'dimension','Adj Close'], inplace=True)
df.dropna(axis=1, inplace=True)
print(df)
print(df.columns.tolist())
df = df[['NextMonthsClose', 'workingcapital', 'netinc', 'eps', 'ebitda', 'fcf', 'rnd' ]]
reg = Regression(data=df, dep_var='NextMonthsClose')
reg.cast_numeric_cols()
reg.split(test_size=0.3)
reg.scale()
reg.train_model()
print(reg.df)
print(reg.model_summary)
print(reg.model.pvalues)

# NOTE: running test_model before oos_predict is not working
# reg.test_model()
reg.oos_predict()