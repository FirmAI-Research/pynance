# Sector Percentile Ranks; Financial Statements; DCF
import sys, os
import nasdaqdatalink
import pandas as pd 


from calendar_dates import Calendar
cal = Calendar()

import nasdaq_data_link as nasdaq
from nasdaq_data_link import Sharadar

""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Call DCF                                                                 │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
from lib.equity.fundamentals import Fundamentals
from lib.equity.fundamentals import DiscountedCashFlow

fun = Fundamentals( ticker = 'XOM' )
dcf = DiscountedCashFlow(fun)


























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