# Fundamentals; Sector Percentile Ranks;  DCF; Financial Statements;
import sys, os
import nasdaqdatalink
import pandas as pd 
import numpy as np
from scipy.stats.mstats import gmean

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(proj_root)
sys.path.append(proj_root)

from calendar_dates import Calendar
cal = Calendar()

import nasdaq_data_link as nasdaq
from nasdaq_data_link import Sharadar

from numeric import custom_formatting


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

        self.ticker = ticker

        nasdaq.Nasdaq()
        
        self.df = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ", ticker = ticker,  paginate=True) # All MRQ periods; One Ticker
        # self.df = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ",   paginate=True) # All MRQ periods; All Tickers

        self.custom_metric_calcs()

        self.df = self.df[::-1]

        print(f'Fundamental data loaded for: {ticker}')


    def __str__(self):
        return f'Fundamentals:Object:{self.ticker}'


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

        with custom_formatting():
            print(self.fundamentals)
        
        self.view_forecasting_table()

        print(self.forecasting)

        self.build_forecasting_table()

        # print(self.assumptions)



    def view_forecasting_table(self):
        ''' View factors to grow metrics by in future periods; return multiIndex dataframe as view.
        
        '''

        def arithmetic(col):
            return self.fundamentals[col].pct_change().dropna().mean()

        def geometric(col):
            a = np.log(self.fundamentals[col].pct_change().dropna())
            return np.exp(a.mean()) 

        def stdev(col):
            return self.fundamentals[col].pct_change().dropna().std()


        self.fields = ['revenue', 'ebitda', 'ebit', 'netinc', 'opinc']

        arrays = [
            [field for field in self.fields for _ in range(3)],
            ['arithmetic', 'geometric', 'stdev'] * len(self.fields)
        ]

        tuples = list(zip(*arrays))

        index = pd.MultiIndex.from_tuples(tuples, names=["field", "calc"])

        values = []
        for item in self.fields:
            values.extend([arithmetic(col = item), geometric(col = item), stdev(col = item)])
        
        multi_idx = pd.Series(values, index=index).to_frame().transpose()

        self.forecasting = multi_idx



    def build_forecasting_table(self):
        ''' Build table of same shape as self.fundamentals containing the factors by which a companies fundamentals will be scaled. 
        '''
        # bull case; bear case; base case
        self.assumptions = pd.DataFrame(columns = self.fields, index = self.fundamentals.index)
        
        x = self.forecasting.loc[0, ("revenue", "geometric")]
        print(x)

        print(100 * ((1+x)**(np.arange(self.n_future_periods))))



    def build_assumptions_table(self):
        pass



    def wacc(self):
        pass




class RanksETL:


    def __init__(self):
        pass

    def query(self):
        pass




