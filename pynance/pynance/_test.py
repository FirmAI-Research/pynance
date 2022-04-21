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
from lib.nasdaq import Nasdaq, Fundamentals
import nasdaqdatalink
import pandas_datareader as wb

def create_fundamental_regression_dataset():
  ticker = 'AMZN'
  # fundamentals
  fun = Fundamentals()
  fun.authenticate()
  dfx = nasdaqdatalink.get_table(fun.name, dimension="MRQ", ticker=ticker, paginate=True) 
  dfx.calendardate = pd.to_datetime(dfx.calendardate)
  dfx.set_index('calendardate', inplace=True)
  # market prices
  start = '2000-01-01'
  end = '2022-03-31'
  ticker = ticker
  dfy = wb.DataReader(ticker, 'yahoo', start, end)
  dfy = dfy.resample('Q').last()
  # mere
  df = dfx.merge(dfy, left_index=True, right_index=True)
  df.to_csv(r"C:\dev\pynance\_tmp\fundamentals_regression.csv")
create_fundamental_regression_dataset()


''' Regress next quarter's closing price by previous quarter end fundamental data '''
# pre-process
df = pd.read_csv(r"C:\dev\pynance\_tmp\fundamentals_regression.csv")
df['NextQtrClose'] = df['Adj Close'].shift(1)
df = df[['NextQtrClose', 'reportperiod','workingcapital', 'netinc', 'eps', 'ebitda', 'fcf', 'rnd' ]]

datekey_index = df.reportperiod
df.drop('reportperiod', axis=1, inplace=True) # drop datekey to exclude from regression and concat later on

latest_fundamental_data = df.iloc[0]

df.dropna(subset=['NextQtrClose'], inplace=True) # drop the most recent data which is NaN due to shift
print(df)

# model and predict
reg = Regression(data=df, dep_var='NextQtrClose')
reg.cast_numeric_cols()
reg.split(test_size=0.40)
reg.scale()
reg.train_model()
# print(reg.df)
# print(reg.model_summary)
# print(reg.model.pvalues)
# reg.test_model() # NOTE: running test_model before oos_predict is not working; run one or the other at a given time

print(latest_fundamental_data)

reg.oos_predict(X_new = latest_fundamental_data)