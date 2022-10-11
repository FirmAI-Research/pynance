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
from etl.fundamentals_etl import FundamentalsETL
from lib.equity.fundamentals import DiscountedCashFlow

fun = FundamentalsETL()
df = fun.df[fun.dcf_columns].set_index('calendardate').drop('ticker', axis=1)
dcf = DiscountedCashFlow(df)
