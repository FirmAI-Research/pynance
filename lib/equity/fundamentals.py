# Fundamentals; Sector Percentile Ranks;  DCF; Financial Statements;
import sys, os
import nasdaqdatalink
import pandas as pd 
from scipy.stats.mstats import gmean

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(proj_root)
sys.path.append(proj_root)

from calendar_dates import Calendar
cal = Calendar()

import nasdaq_data_link as nasdaq
from nasdaq_data_link import Sharadar



class Fundamentals:
    ''' A class which accepts a Ticker as a parameter and returns equity fundamental's.
    '''
    # 1. query fundamental data for all companies as of the previous quarter end date
    # 2. calculate desired 'custom' metrics on fundamentals
    # 3. populate sql table for all companies, with custom metrics, other metrics to be used for ranks, and sector/industry. Table name = EquityFundamentalMetrics

    id_columns = [
        'ticker', 'calendardate'

    ]

    dcf_columns = id_columns + [
        'revenue', 'ebitda', 'ebit', 'netinc',  'opinc', 'taxexp', 'ebt'
        
    ]

    peer_compare_columns = id_columns + [ 
        'bvps', 'currentratio', 'de', 'dps', 'divyield', 'eps', 'evebit', 'evebitda', 'fcfps', 'grossmargin', 'netmargin', 'pb', 'pe', 'price', 'roa', 'roe', 'roic', 'ros', 
        'roc', 'fcfmargin', 'p/cf', 'oppmargin', 'interestcoverage', 'payoutratio', 'tax rate', 'retention ratio', 'expected netinc growth', 'expected roe growth', 
        'equity reinvestment rate','expected ebit growth'
    ]


    def __init__(self, ticker = None):

        nasdaq.Nasdaq()
        
        self.df = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ", ticker = ticker,  paginate=True) # All MRQ periods; One Ticker
        # self.df = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ",   paginate=True) # All MRQ periods; All Tickers

        self.custom_metric_calcs()

        self.df = self.df[::-1]


    def custom_metric_calcs(self):
        ''' additional metrics calculated based on raw fundamental data
        
        damodaran pg. 178
        '''

        df = self.df

        df['roc'] = df['netinc'] / (df['equity'] + df['debt'])
        df['fcfmargin'] = df['fcf'] / df['revenue']
        df['p/cf'] = df['marketcap'] / df['fcf']
        df['oppmargin'] = df['opinc'] / df['revenue']
        df['interestcoverage'] = df['ebit'] / df['intexp']
        df['payoutratio'] = df['dps'] / df['eps']
        df['tax rate'] = df['taxexp'] / df['ebt']
        df['retention ratio'] = (df['retearn']  / df['netinc']) / 100
        df['expected netinc growth'] = df['retention ratio'] * df['roe']
        df['expected roe growth'] = df['ebit'] *  df['tax rate'] / (df['equity'] + df['debt'])
        df['equity reinvestment rate'] =   df['expected netinc growth'] /  df['roe']  
        df['expected ebit growth'] = df['equity reinvestment rate'] * df['roc']
        df['sales to capital ratio'] = '' # reinvestment rate

        self.df = df




class DiscountedCashFlow():

    n_historical_periods = 5

    n_future_periods = 4
    

    def __init__(self, fun_obj):
        ''' Constructs a dcf using fundamental data passed in a ~Fundamentals~ object. Raw data is stored in Fundamentals().df

        :param: fun_obj : An object of type ~Fundamentals~
        '''
        df = fun_obj.df[fun_obj.dcf_columns].set_index('calendardate').drop('ticker', axis=1)

        self.fundamentals = df.iloc[-self.n_historical_periods:]

        print(self.fundamentals)
        
        self.build_forecasting_table()

        print(self.forecast)



    def build_forecasting_table(self):

        self.forecast = pd.DataFrame()
        
        self.cagr(col = 'revenue')
        self.arithmetic_average(col = 'revenue')
        
        self.cagr(col = 'ebitda')
        self.arithmetic_average(col = 'ebitda')        
        
        self.cagr(col = 'ebit')
        self.arithmetic_average(col = 'ebit')
        
        self.cagr(col = 'netinc')
        self.arithmetic_average(col = 'netinc')   
        
        self.cagr(col = 'opinc')
        self.arithmetic_average(col = 'opinc')   


    def arithmetic_average(self, col):
        label = col + '_aagr'
        self.forecast[label] = self.fundamentals[col].pct_change().dropna().mean()


    def cagr(self, col):
        label = col + '_cagr'

        start_value, end_value = self.fundamentals[col][0],  self.fundamentals[col][-1]
        num_periods = len(self.fundamentals[col])
        
        self.forecast[label]  = [(end_value / start_value) ** (1 / (num_periods - 1)) - 1 for x in self.fundamentals[col]]




    def build_assumptions_table(self):
        pass


    def wacc(self):
        pass




class RanksETL:


    def __init__(self):
        pass

    def query(self):
        pass




