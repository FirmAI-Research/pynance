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

# ticker = ['JNJ','PG']
ticker = ['JNJ']

# fun = Fundamentals( ticker = ticker,)
# print(fun)
# fun.get( columns = Columns.INCOME.value, limit = 5 ).style_terminal(fun.df, text = 'Income:')
# change = fun.percent_change()
# print(change)
# print(change.describe())
# print(fun.delta())

# fun.get( columns = Columns.CASHFLOW.value, limit = 5 ).style_terminal(fun.df, text = 'Cash Flows:')
# change = fun.percent_change()
# print(change)
# print(change.describe())
# print(fun.delta())

# fun.get( columns = Columns.PEERS.value, limit = 5 ).style_terminal(fun.df, text = 'Peers:')
# change = fun.percent_change()
# print(change)
# print(change.describe())
# print(fun.delta())

# fun.get( columns = Columns.EXP.value, limit = 5 ).style_terminal(fun.df, text = 'Expected:')


# fun.get( columns = Columns.DCF.value, limit = 5 ).style_terminal(fun.df, text = 'Dcf:')
# dcf = DCF(fun)
# dcf.industry_cagr()


r = Ranks(ticker = 'JNJ')
print(r.get_ranks())














