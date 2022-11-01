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
from lib.equity.attribution import Attribution
from lib.fixed_income.rates import Treasuries

ust = Treasuries(years = ['2021', '2022'])
print(ust.df)









