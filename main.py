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
from lib.equity.fundamentals import Fundamentals, Columns, RanksETL

# ticker = ['JNJ','PG']
# fun = Fundamentals( ticker = ticker,)
# print(fun)
# fun.get( columns = Columns.INCOME.value, limit = 4 ).style_terminal(fun.df, text = 'Income:')
# fun.growth().style_terminal([fun.growth_pct, fun.growth_measures], text = ['Growth %:', 'Growth Measures:'])

# fun.get( columns = Columns.CASHFLOW.value, limit = 4 ).style_terminal(fun.df, text = 'Cash Flows:')
# fun.growth().style_terminal([fun.growth_pct, fun.growth_measures], text = ['Growth %:', 'Growth Measures:'])

# fun.get( columns = Columns.PEERS.value, limit = 4 ).style_terminal(fun.df, text = 'Peers:')
# fun.growth().style_terminal([fun.growth_pct, fun.growth_measures], text = ['Growth %:', 'Growth Measures:'])



r = RanksETL()
r.join_fundamentals_and_profiles()





















# import sys, os
# import nasdaqdatalink
# import pandas as pd

# proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(proj_root)
# sys.path.append(proj_root)

# from calendar_dates import Calendar
# cal = Calendar()

# import nasdaq_data_link as nasdaq
# from nasdaq_data_link import Sharadar

# from etl.fundamentals_etl import FundamentalsETL

# pd.options.display.float_format = '{:,.2f}'.format


# """ 
#   ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
#   │                                  Retreive data from NasdaqDataLink                                                 │
#   └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
# """
# def get_nasdaq_data():

#   query = nasdaq.Nasdaq()
#   df = nasdaqdatalink.get_table(Sharadar.METRICS.value, ticker = ['AAPL','AMZN'], date = cal.previous_quarter_end())
#   print(df)

# # get_nasdaq_data()






# """ 
#   ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
#   │                                      Equity Fundamentals Ranks                                                     |
#   └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
#  """
#  # use fundamentals.Ranks  object to calculate ranks and store in database
# def etl_equity_fundamental_ranks():
#   fun = FundamentalsETL()
#   fun.custom_calculations()

# etl_equity_fundamental_ranks()