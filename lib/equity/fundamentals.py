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

    growth_columns = id_columns + [
        'revenue', 'ebitda', 'ebit', 'netinc',  'opinc', 'taxexp', 'ebt'
        
    ]

    cf_columns =  id_columns + [
        'fcf', 'ncf', 'ncfbus', 'ncfcommon', 'ncfdebt', 'ncfdiv', 'ncff', 'ncfi', 'ncfinv', 'ncfo', 'ncfx'
    ]

    peer_columns = id_columns + [ 
        'de', 'divyield', 'eps', 'evebitda', 'fcfps', 'grossmargin', 'netmargin', 'pb', 'pe', 'price', 'roa', 'roe', 'roic', 'ros', 
        'roc', 'fcfmargin', 'p/cf', 'oppmargin', 'intcov',
    ]

    def __init__(self, ticker = None):

        if isinstance(ticker, list):
            
            self.ticker = ticker

        elif isinstance(ticker, str):

            if ticker == '*':
                print('Not slicing by an individual ticker')

            self.ticker = ticker

        
        nasdaq.Nasdaq()
        
        self.df = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ", ticker = self.ticker,  paginate=True) # All MRQ periods; One Ticker
        # self.df = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ",   paginate=True) # All MRQ periods; All Tickers

        self.custom_metric_calcs()

        self.df = self.df[::-1]

        print(f'Fundamental data loaded for: {ticker}')


    def __str__(self):
        return f'Fundamentals:Object:{self.ticker}'


    def custom_metric_calcs(self):
        ''' additional metrics calculated based on raw fundamental data
        
        damodaran pg. 178; Jansend 79-85
        '''

        df = self.df

        df['roc'] = df['netinc'] / (df['equity'] + df['debt'])
        df['fcfmargin'] = df['fcf'] / df['revenue']
        df['p/cf'] = df['marketcap'] / df['fcf']
        df['oppmargin'] = df['opinc'] / df['revenue']
        df['intcov'] = df['ebit'] / df['intexp'] # interestcoverage
        df['paoutratio'] = df['dps'] / df['eps'] # payoutratio
        df['taxrate'] = df['taxexp'] / df['ebt']
        df['retentionratio'] = (df['retearn']  / df['netinc']) / 100 #retentionratio
        df['expnetincgrow'] = df['retentionratio'] * df['roe'] # expected netinc growth
        df['exproegrow'] = df['ebit'] *  df['taxrate'] / (df['equity'] + df['debt']) # expected roe growth
        df['eqreinvestrate'] =   df['expnetincgrow'] /  df['roe']  # equity reinvestment rate
        df['expebitgrow'] = df['eqreinvestrate'] * df['roc'] # expebitgrow
        # df['sales to capital ratio'] = '' # reinvestment rate

        self.df = df



class Compare():
    '''
    '''
    n_historical_periods = 5

    def __init__(self, tickers):

        fun = Fundamentals(ticker = tickers)
        
        df = fun.df[fun.peer_columns].set_index('calendardate')

        self.df = df.pivot(columns = ['ticker']).iloc[-self.n_historical_periods:, :]




class Measures():

    n_historical_periods = 5


    def __init__(self, fun_obj, cols):
        '''Fundamental data passed in a ~Fundamentals~ object. Raw data is stored in Fundamentals().df

        :param: fun_obj : An object of type ~Fundamentals~
        '''
        self.fun_obj = fun_obj

        data = fun_obj.df[cols].set_index('calendardate').drop('ticker', axis=1)

        self.data = data.iloc[-self.n_historical_periods:]

        self.growth_measures()

        # with custom_formatting():
        #     print('Fundamental Data:')
        #     print(self.data)
        #     print('Measures:')
        #     print(self.measures)
        print(self.data.pct_change())


    def growth_measures(self):
        ''' Measure growth of fundamentals over time; return multiIndex dataframe as view.
        '''

        def arithmetic(col):
            return self.data[col].pct_change().dropna().mean()

        def geometric(col):
            a = np.log(self.data[col].pct_change().dropna())
            return np.exp(a.mean()) 

        def stdev(col):
            return self.data[col].pct_change().dropna().std()


        self.fields = self.data.columns

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

        self.measures = multi_idx.T

        self.measures.columns = self.fun_obj.ticker



class RanksETL:


    def __init__(self):
        pass

    def query(self):
        pass




