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

from scipy.stats.mstats import gmean


class DiscountedCashFlow():


    def __init__(self, df):
        '''
        :param: df : time series of equity fundamentals for an individual company
        '''
        n_historical_periods = 5

        self.df = df.iloc[-n_historical_periods:]
        
        self.build_forecasting_table()
     

    def build_forecasting_table(self):

        self.forecast = pd.DataFrame()
        
        self.cagr(col = 'revenue')
        self.arithmetic_average(col = 'revenue')
        self.cagr(col = 'oppmargin')
        self.arithmetic_average(col = 'oppmargin')
        self.cagr(col = 'opinc')
        self.arithmetic_average(col = 'opinc')   

        print(self.forecast)


    def build_assumptions_table(self):
        pass


    def wacc(self):
        pass


    def arithmetic_average(self, col):
        label = col + '_aagr'
        self.forecast[label] = self.df[col].pct_change().dropna().mean()


    def cagr(self, col):
        label = col + '_cagr'

        series = self.df[col]
        start_value, end_value = series[0],  series[-1]
        num_periods = len(series)
        
        self.forecast[label]  = [(end_value / start_value) ** (1 / (num_periods - 1)) - 1 for x in self.df[col]]