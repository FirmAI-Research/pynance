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


import nasdaq_data_link as nasdaq
from nasdaq_data_link import Sharadar
from numeric import custom_formatting
from calendar_dates import Calendar
cal = Calendar()


class Columns(Enum):
    ID = ['ticker', 'calendardate' ]
    
    CASHFLOW = ID +  ['ncfo','ncfi', 'ncff', 'ncfinv', 'fcf', 'ncf', 'ncfbus', 'ncfcommon', 'ncfdebt', 'ncfdiv',  'ncfx']
    
    INCOME = ID +  ['revenue', 'cogs','gp', 'opex','opinc','ebt','netinc','eps','depamor','ebitda']
    
    BALANCE = ID + []
    
    PEERS = ID +  ['de', 'divyield', 'eps', 'evebitda', 'fcfps', 'grossmargin', 'netmargin', 'fcfmargin', 'p/cf', 'oppmargin', 'pb', 'pe', 'roa', 'roe', 'roic', 'ros', 'roc', 'intcov']
    
    EXP = ID + ['retentionratio', 'roe', 'retearn','expnetincgrow', 'exproegrow', 'eqreinvestrate', 'expebitgrow','expgrowthrate']
    
    DCF = ID + ['revenue','cogs','gp', 'rnd','sgna','ebit', 'taxrate', 'depamor','ebitda','capex']
    
    RANKS = list(set(ID + CASHFLOW + INCOME + BALANCE + PEERS + EXP + DCF + ['name', 'industry', 'sector', 'famaindustry', 'famasector', 'scalemarketcap','scalerevenue']))


class Fundamentals:
    ''' Retreive equity fundamental's.

    # https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1?qopts.export=true&api_key=API_KEY
    '''

    def __str__(self):
        return f'Fundamentals:Object:{self.ticker}'


    def __init__(self, ticker = None):

        self.ticker = ticker


        nasdaq.Nasdaq()

        if ticker is not None:
            
            self.data = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ", ticker = self.ticker,  paginate=True) # All MRQ periods; One Ticker
        
        else:
    
            self.data = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ",   paginate=True) # All MRQ periods; All Tickers

        self.calculate()


    def get(self, columns = None, limit = 8):

        self.limit = limit
        self.columns = columns
        self.df = self.data

        if columns is not None:
        
            self.df = self.df[self.columns]

        if isinstance(self.ticker, list) and len(self.ticker) == 1:

            self.df = self.df[::-1].iloc[-limit:, :].set_index('calendardate').drop('ticker', axis=1) 

        else:  
            if limit is not None:
                self.df = self.df[self.columns].set_index('calendardate').pivot(columns = ['ticker'])[::-1].iloc[-limit:, :]
            
            else:
                self.df = self.df[self.columns].set_index('calendardate').pivot(columns = ['ticker'])[::-1]
        
        return self


    def calculate(self):
        ''' additional metrics calculated based on raw fundamental data
        damodaran pg. 178; Jansen 79-85
        '''
        df = self.data
        
        df['cogs'] = df['revenue'] - df['gp']
        
        df['roc'] = df['netinc'] / (df['equity'] + df['debt'])

        df['roe'] = df['netinc'] / (df['equity']) # TODO Use TTM net income
        
        df['fcfmargin'] = df['fcf'] / df['revenue']
        
        df['p/cf'] = df['marketcap'] / df['fcf']
        
        df['oppmargin'] = df['opinc'] / df['revenue']
        
        df['intcov'] = df['ebit'] / df['intexp'] # interestcoverage
        
        df['paoutratio'] = df['dps'] / df['eps'] # payoutratio
        
        df['taxrate'] = df['taxexp'] / df['ebt']

        df['ebit1-t'] = ''
        
        df['retentionratio'] = (df['retearn']  / df['netinc']) / 100 #retentionratio
        
        df['expnetincgrow'] = df['retentionratio'] * df['roe'] # expected netinc growth
        
        df['exproegrow'] = df['ebit'] *  df['taxrate'] / (df['equity'] + df['debt']) # expected roe growth
        
        df['eqreinvestrate'] =   df['expnetincgrow'] /  df['roe']  # equity reinvestment rate
        
        df['expebitgrow'] = df['eqreinvestrate'] * df['roc'] # expebitgrow

        df['netmargin'] = (df['revenue'] - df['cogs']) / df['revenue']

        df['expgrowthrate'] = df['netmargin'] * (df['revenue']/df['equity']) * df ['retentionratio'] #expected growth rate; p.275
        # df['sales to capital ratio'] = '' # reinvestment rate
        self.data = df


    def percent_change(self):
        return self.df.pct_change().dropna(how = 'all', axis=0) 


    def delta(self):
        ''' Change from previous quarter; Quarter over Quarter change.'''
        sub = self.df.iloc[[-5, -2, -1], :].dropna(how='any', axis=1)
        
        sub = sub.iloc[-1].squeeze() - sub.iloc[:-1]
        
        sub.index.name = 'Change Since'
        
        return sub


    def estimates(self):
        ''' 
        https://www.barchart.com/stocks/quotes/SQ/earnings-estimates
        https://www.nasdaq.com/market-activity/stocks/amzn/analyst-research
        '''
        pass


    def full_export(self, curl = False):
        ''' Helper function for RanksETL
        '''
        import requests, zipfile, io

        dirp = "C:\data\FundamentalsZip"
        
        if curl:
            nas = nasdaq.Nasdaq()
            
            request_url = f'https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1?qopts.export=true&api_key={nas.api_key}'
            res = requests.get(request_url)
            
            df  = pd.DataFrame.from_dict(res.json())
            link = df.datatable_bulk_download.iloc[1].get('link')
            print(link)
            
            r = requests.get(link)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(dirp)
        
        fp = os.listdir(dirp)[0] # Only one file is returned in the unziped folder
        
        self.data = pd.read_csv(os.path.join(dirp, fp))

        self.calculate()

        return self.data


    def plot_clustered_bar_peer_fundamentals():
        ''' Clustered bar chart for multiple peers to the ticker and various metrics
        '''
        pass



    def style_terminal(self, df, text:list = None):
        if isinstance(text, list):
            for i in range(len(text)):
                print(text[i])
                with custom_formatting():
                    print(df[i])
        else:
            print(text)
            with custom_formatting():
                print(df)           


    def style_jupyter(self):
        pass

    # NOTE: deprecated
    def growth(self):
        ''' Measure growth of fundamentals over time; return multiIndex dataframe as view.
        '''
        def arithmetic(df, col):
            return df[col].pct_change().dropna().mean()

        def median(df, col):
            return df[col].pct_change().dropna().median()

        def stdev(df, col):
            return df[col].pct_change().dropna().std()

        self.fields = [c for c in self.columns if c not in  ['ticker', 'calendardate']]
        measure_names = ['arith', 'median', 'stdev']
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
                values.extend([arithmetic(self.df, col = item), median(self.df, col = item), stdev(self.df, col = item)])
        
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
                    values.extend([arithmetic(df, col = item).values[0], median(df, col = item).values[0], stdev(df, col = item).values[0]])
   
        multi_idx = pd.Series(values, index=index).to_frame().transpose()

        measures = multi_idx.T

        self.growth_measures = measures.T

        if len(self.ticker) > 1:
            self.growth_measures.columns=self.growth_measures.sort_index(axis=1,level=[1,2, 0],ascending=[True,True, True]).columns

        self.growth_pct = self.df.pct_change().dropna(how = 'all', axis=0) 
        return self




class Ranks:
    ''' ETL Proccess to calculate and load ranks to sqlite is stored in pynance/db/
    '''

    def __init__(self, ticker=None, industry=None):

        if ticker is None and industry is None:
            raise ValueError ('Must provide a ticker or an Industry')

        if ticker is not None and industry is None:
            tick = nasdaq.Tickers()
            tick.full_export(curl=False)
            self.industry = tick.get_industry(ticker)
        
        else:    
            self.industry = industry

        self.ticker = ticker

        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///C:\data\industry_fundamentals.db', echo=False)
        self.cnxn = engine.connect()


    def get_ranks(self):
        ''' Returns ranks for all periods and one ticker '''
        df =  pd.read_sql(f"select * from CompFunRanks where ticker == '{self.ticker}'", self.cnxn)
        return df.pivot(index = ['calendardate'], columns = ['variable'], values= ['value'])


    def get_box_plot_values(self):
        ''' Raw fundamental values for all companies in an industry peer group'''
        return pd.read_sql(f"select * from CompFunBase where industry == '{self.industry}'", self.cnxn)


    def plot_dual_axis_value_and_rank(self):
        ''' Plots a time series of fundamental values for an individual metric and company on one axis; With the % rank vs peer group on the second axis
        '''
        pass

    
    def plot_plot(self):
        ''' Box plot of fundamental values for a metric across all companies in a peer group. Star the individual companies value on the plot. 
        '''
        pass



class DCF:

    def __init__(self, funobj, ):
        self.funobj = funobj

        data = funobj.df

        print(self.industry_cagr())


    def industry_cagr(self):
        ''' Damodaran Industry average's using compdata
        '''

        from compdata import comp_data
        print(comp_data.industry_name_list)
        software = comp_data.Industry('Software (System & Application)')
        betas = software.get_betas()
        cagrL5Y = pd.DataFrame(data = betas)
        return cagrL5Y

    def industry_gross_profit_margin(self):
        margins = software.get_margins()
        sf1 = pd.DataFrame(data = margins)
        sf1

        #rnd


