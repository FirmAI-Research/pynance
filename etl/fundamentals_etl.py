import sys, os
import nasdaqdatalink
import pandas as pd 

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(proj_root)
sys.path.append(proj_root)

from calendar_dates import Calendar
cal = Calendar()

from numeric import *

import nasdaq_data_link as nasdaq
from nasdaq_data_link import Sharadar


class FundamentalsETL:
    # 1. query fundamental data for all companies as of the previous quarter end date
    # 2. calculate desired 'custom' metrics on fundamentals
    # 3. populate sql table for all companies, with custom metrics, other metrics to be used for ranks, and sector/industry. Table name = EquityFundamentalMetrics

    id_columns = [
        'ticker', 'calendardate'

    ]

    dcf_columns = id_columns + [
        'revenue', 'oppmargin', 'opinc', 'ebit', 'taxexp', 'ebt'
        
    ]

    peer_compare_columns = id_columns + [ 
        'bvps', 'currentratio', 'de', 'dps', 'divyield', 'eps', 'evebit', 'evebitda', 'fcfps', 'grossmargin', 'netmargin', 'pb', 'pe', 'price', 'roa', 'roe', 'roic', 'ros', 
        'roc', 'fcfmargin', 'p/cf', 'oppmargin', 'interestcoverage', 'payoutratio', 'tax rate', 'retention ratio', 'expected netinc growth', 'expected roe growth', 
        'equity reinvestment rate','expected ebit growth'
    ]


    def __init__(self):

        nasdaq.Nasdaq()
        
        self.df = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ", ticker = 'GOOGL',  paginate=True) # All MRQ periods; One Ticker

        # self.df = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ",   paginate=True) # All MRQ periods; All Tickers

        self.custom_calculations()

        self.df = self.df[::-1]


    def custom_calculations(self):
        ''' additional metrics calculated based on raw fundamental data '''

        df = self.df

        df['roc'] = df['netinc'] / (df['equity'] + df['debt'])
        df['fcfmargin'] = df['fcf'] / df['revenue']
        df['p/cf'] = df['marketcap'] / df['fcf']
        df['oppmargin'] = df['opinc'] / df['revenue']
        df['interestcoverage'] = df['ebit'] / df['intexp']
        df['payoutratio'] = df['dps'] / df['eps']
        df['tax rate'] = df['taxexp'] / df['ebt']
        # damodaran pg. 178
        df['retention ratio'] = (df['retearn']  / df['netinc']) / 100
        df['expected netinc growth'] = df['retention ratio'] * df['roe']
        df['expected roe growth'] = df['ebit'] *  df['tax rate'] / (df['equity'] + df['debt'])
        df['equity reinvestment rate'] =   df['expected netinc growth'] /  df['roe']  
        df['expected ebit growth'] = df['equity reinvestment rate'] * df['roc']
        df['sales to capital ratio'] = '' # reinvestment rate

        self.df = df




class RanksETL:


    def __init__(self):
        pass

    def query(self):
        pass




