# Sector Percentile Ranks; Financial Statements; DCF
import sys, os
import nasdaqdatalink
import pandas as pd 

from calendar_dates import Calendar

import nasdaq_data_link as nasdaq
from nasdaq_data_link import Sharadar
from numeric import custom_formatting

import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:,.4f}'.format

cal = Calendar()

""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Fundamentals View                                                                                                  │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
from lib.equity.fundamentals import Fundamentals, Columns, DCF, Ranks

# ticker = ['JNJ','PG','CVS']
ticker = ['JNJ']

# fun = Fundamentals( ticker = ticker,)
# print(fun)
# fun.get( columns = Columns.INCOME.value, limit = 5 ).style_terminal(fun.df, text = 'Income:')
# change = fun.percent_change()
# print(change)
# print(fun.describe(change))
# print(fun.delta())
# peers = fun.get_peers()


# fun = Fundamentals( ticker = peers,)
# print(fun)
# fun.get( columns = Columns.INCOME.value, limit = 5 ).style_terminal(fun.df, text = 'Income:')
# change = fun.percent_change()
# print(change)
# print(fun.describe(change))
# print(fun.delta())


# fun.get( columns = Columns.CASHFLOW.value, limit = 5 ).style_terminal(fun.df, text = 'Cash Flows:')
# change = fun.percent_change()
# print(change)
# print(fun.describe(change))
# print(fun.delta())


# fun = Fundamentals( ticker = peers,)
# print(fun)
# fun.get( columns = Columns.CASHFLOW.value, limit = 5 ).style_terminal(fun.df, text = 'Cash Flows:')
# change = fun.percent_change()
# print(change)
# print(fun.describe(change))
# print(fun.delta())


# fun.get( columns = Columns.PEERS.value, limit = 5 ).style_terminal(fun.df, text = 'Peers:')
# change = fun.percent_change()
# print(change)
# print(fun.describe(change))
# print(fun.delta())


# fun = Fundamentals( ticker = peers,)

# fun.get( columns = Columns.PEERS.value, limit = 5 ).style_terminal(fun.df, text = 'Peers:')
# change = fun.percent_change()
# print(change)
# print(fun.describe(change))
# print(fun.delta())


# fun.get( columns = Columns.EXP.value, limit = 5 ).style_terminal(fun.df, text = 'Expected:')

# r = Ranks(ticker = 'JNJ')
# print(r.get_ranks())


dcf = DCF(ticker=['JNJ'])
dcf.forecast_as_percent_of_revenue(type = 'INCOME')
dcf.forecast_as_percent_of_revenue(type = 'BALANCE')
dcf.forecast_as_percent_of_revenue(type = 'CF')

with custom_formatting():
  print(dcf.inc_forecast)
  print(dcf.bal_forecast)
  print(dcf.cf_forecast)

dcf.forecast_cf_from_opperations()
print(dcf.cf_from_opp)
dcf.discount()
dcf.terminal_value()
print(dcf.estimate_price_per_share())











