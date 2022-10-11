# Sector Percentile Ranks; Financial Statements; DCF
import sys, os
import nasdaqdatalink
import pandas as pd 

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(proj_root)
sys.path.append(proj_root)

from calendar_dates import Calendar
cal = Calendar()

import nasdaq_data_link as nasdaq
from nasdaq_data_link import Sharadar

from etl.fundamentals_etl import FundamentalsETL


class DiscountedCashFlow():


    def __init__(self, df):
        '''
        :param: df : time series of equity fundamentals for an individual company
        '''
        self.df = df
        self.forecast = pd.DataFrame()

    def build_assumptions_table(self):
        pass

    def wacc(self):
        pass

    def forecast_historical_average(self, col):
        pass

    def forecast_opperating_margin():
        pass
