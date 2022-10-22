# Fundamentals; Sector Percentile Ranks;  DCF; Financial Statements;
from enum import Enum
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

class Columns(Enum):
    id_columns = [   'ticker', 'calendardate' ]

    CASHFLOW = id_columns +  ['ncfo','ncfi', 'ncff', 'ncfinv', 'fcf', 'ncf', 'ncfbus', 'ncfcommon', 'ncfdebt', 'ncfdiv',  'ncfx']
    INCOME = id_columns +  ['revenue', 'cogs','gp', 'opex','opinc','ebt','netinc','eps','depamor','ebitda']
    PEERS = id_columns +  [  'de', 'divyield', 'eps', 'evebitda', 'fcfps', 'grossmargin', 'netmargin', 'pb', 'pe', 'price', 'roa', 'roe', 'roic', 'ros', 'roc', 'fcfmargin', 'p/cf', 'oppmargin', 'intcov']

units = {
    'ncfo':'usd',
}


class Fundamentals:
    ''' Retreive equity fundamental's.
    '''

    def __str__(self):
        return f'Fundamentals:Object:{self.ticker}'


    def __init__(self, ticker = None, columns = None, limit = 5):

        self.ticker = ticker
        self.limit = limit
        self.columns = columns

        nasdaq.Nasdaq()

        if ticker is not None:
            
            self.df = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ", ticker = self.ticker,  paginate=True) # All MRQ periods; One Ticker
        
        else:
    
            self.df = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ",   paginate=True) # All MRQ periods; All Tickers

        self.calculate()

        if columns is not None:
        
            self.df = self.df[self.columns]


        if isinstance(ticker, list) and len(ticker) == 1:

            self.df = self.df[::-1].iloc[-limit:, :].set_index('calendardate').drop('ticker', axis=1) # reverse rows and limit n if an individual ticker is supplied 

        else:  # compare fundamentals of multiple companies side by side in multi indexed dataframe

            self.df = self.df[self.columns].set_index('calendardate').pivot(columns = ['ticker'])[::-1].iloc[-limit:, :]


    def calculate(self):
        ''' additional metrics calculated based on raw fundamental data
        damodaran pg. 178; Jansen 79-85
        '''
        df = self.df
        
        df['cogs'] = df['revenue'] - df['gp']
        
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



    def growth(self):
        ''' Measure growth of fundamentals over time; return multiIndex dataframe as view.
        '''
        def arithmetic(df, col):
            return df[col].pct_change().dropna().mean()

        def geometric(df, col):
            a = np.log(df[col].pct_change().dropna())
            return np.exp(a.mean()) 

        def stdev(df, col):
            return df[col].pct_change().dropna().std()

        self.fields = [c for c in self.columns if c not in  ['ticker', 'calendardate']]
        measure_names = ['arith', 'geo', 'stdev']
        n_measures = len(measure_names)

        # case for creating multiIndex frame with growth measures for a single ticker
        if len(self.ticker) == 1:
            arrays = [
                [field for field in self.fields for _ in range(n_measures)],
                measure_names * len(self.fields)
            ]

            tuples = list(zip(*arrays))

            index = pd.MultiIndex.from_tuples(tuples, names=["field", "calc"])

            values = []
            for item in self.fields:
                values.extend([arithmetic(self.df, col = item), geometric(self.df, col = item), stdev(self.df, col = item)])
        
        # # case for creating multiIndex frame with growth measures for multiple tickers
        else:
            arrays = [
                [ticker for ticker in self.ticker * (len(self.fields) * len(self.ticker) * n_measures)],
                [field for field in self.fields for _ in range(n_measures * len(self.ticker))],
                measure_names * (len(self.fields) * len(self.ticker))
            ]
            tuples = list(zip(*arrays))

            index = pd.MultiIndex.from_tuples(tuples, names=["ticker", "field", "calc"])

            values = []

            for item in self.fields:
                for ticker in self.ticker:
                    df = self.df.iloc[:, self.df.columns.get_level_values(1)==ticker]
                    # print(df[item])
                    values.extend([arithmetic(df, col = item).values[0], geometric(df, col = item).values[0], stdev(df, col = item).values[0]])
   
   
        multi_idx = pd.Series(values, index=index).to_frame().transpose()

        measures = multi_idx.T

        # measures.columns = self.ticker

        measures = measures.T
        return measures, self.df.pct_change().dropna(how = 'all', axis=0)








class RanksETL:


    def __init__(self):
        pass

    def query(self):
        pass




