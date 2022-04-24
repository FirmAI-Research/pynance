import sys,os
import pandas as pd
import seaborn as sns
import numpy as np
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
# import pandas as pd
# import numpy as np
# import seaborn as sns
# from lib.learn.regression.regression import Regression
# from lib.nasdaq import Nasdaq, Fundamentals
# import nasdaqdatalink
# import pandas_datareader as wb

# def create_fundamental_regression_dataset():
#   ticker = 'AMZN'
#   # fundamentals
#   fun = Fundamentals()
#   fun.authenticate()
#   dfx = nasdaqdatalink.get_table(fun.name, dimension="MRQ", ticker=ticker, paginate=True) 
#   dfx.calendardate = pd.to_datetime(dfx.calendardate)
#   dfx.set_index('calendardate', inplace=True)
#   # market prices
#   start = '2000-01-01'
#   end = '2022-03-31'
#   ticker = ticker
#   dfy = wb.DataReader(ticker, 'yahoo', start, end)
#   dfy = dfy.resample('Q').last()
#   # mere
#   df = dfx.merge(dfy, left_index=True, right_index=True)
#   df.to_csv(r"C:\dev\pynance\_tmp\fundamentals_regression.csv")
# create_fundamental_regression_dataset()


# ''' Regress next quarter's closing price by previous quarter end fundamental data '''
# # pre-process
# df = pd.read_csv(r"C:\dev\pynance\_tmp\fundamentals_regression.csv")
# df['NextQtrClose'] = df['Adj Close'].shift(1)
# df = df[['NextQtrClose', 'reportperiod','workingcapital', 'netinc', 'eps', 'ebitda', 'fcf', 'rnd' ]]

# datekey_index = df.reportperiod
# df.drop('reportperiod', axis=1, inplace=True) # drop datekey to exclude from regression and concat later on

# latest_fundamental_data = df.iloc[0]

# df.dropna(subset=['NextQtrClose'], inplace=True) # drop the most recent data which is NaN due to shift
# print(df)

# # model and predict
# reg = Regression(data=df, dep_var='NextQtrClose')
# reg.cast_numeric_cols()
# reg.split(test_size=0.40)
# reg.scale()
# reg.train_model()
# # print(reg.df)
# # print(reg.model_summary)
# # print(reg.model.pvalues)
# # reg.test_model() # NOTE: running test_model before oos_predict is not working; run one or the other at a given time

# print(latest_fundamental_data)

# reg.oos_predict(X_new = latest_fundamental_data)


""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ @Test: regression : find relevant features from fundamental data on future share prices                                                                                      │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
# def housing_regression():
#   root = r'C:\dev\pynance\_tmp\housing\\'
#   housing_price_index = pd.read_csv(root + '/monthly-hpi.csv')
#   unemployment = pd.read_csv(root + '/unemployment-macro.csv')
#   federal_funds_rate = pd.read_csv(root + '/fed_funds.csv')
#   shiller = pd.read_csv(root + '/shiller.csv')
#   gross_domestic_product = pd.read_csv(root + '/gdp.csv')
#   df = (shiller.merge(housing_price_index, on='date')
#                       .merge(unemployment, on='date')
#                       .merge(federal_funds_rate, on='date')
#                       .merge(gross_domestic_product, on='date'))
#   df.drop(['date'], axis=1, inplace=True)
#   df = df.iloc[:, :7]
#   print(df.head())
#   print(df.tail())
#   from lib.learn.regression.regression import Regression
#   reg = Regression(data = df, dep_var = 'housing_price_index') # indep_var='total_unemployed'
#   reg.split(test_size=0.4)
#   reg.train_model()
#   # reg.reg_plots()
#   # reg.test_model()
#   # reg.oos_predict(most_recent=True ) # NOTE first value should be constant of 1; X_new = [1, 1282,220,3.39,16.5,8500,2900],
# housing_regression()



""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ @Test: feature engineering                                                                                   │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
from lib.learn.featureengine import FeatureEngine

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
              'machine-learning-databases/wine/wine.data',
              header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                'Alcalinity of ash', 'Magnesium', 'Total phenols',
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                'Color intensity', 'Hue',
                'OD280/OD315 of diluted wines', 'Proline']
print(df_wine.head())

fe = FeatureEngine(df_wine)
fe.eig()
fe.explained_variance()