# Fundamentals; Sector Percentile Ranks;  DCF; Financial Statements;
from enum import Enum
import sys, os
from tracemalloc import start
import nasdaqdatalink
import pandas as pd 
import numpy as np
import numpy_financial as npf
from scipy.stats.mstats import gmean
import yfinance as yf

import seaborn as sns
import matplotlib.pyplot as plt 

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(proj_root)
sys.path.append(proj_root)

from sqlalchemy import create_engine

import nasdaq_data_link as nasdaq
from nasdaq_data_link import Sharadar
from numeric import custom_formatting
import calendar_dates 

cal = calendar_dates.Calendar()


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


        self.data.calendardate = [d.strftime('%Y-%m-%d') for d in self.data.calendardate]

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
                self.df = self.df[self.columns][::-1].set_index('calendardate').pivot(columns = ['ticker']).iloc[-limit:, :]
            
            else:
                self.df = self.df[self.columns][::-1].set_index('calendardate').pivot(columns = ['ticker'])
        
        return self


    def for_js(self):
        
        df = self.df.divide(1000000).T.reset_index(level=[0,1]).reset_index(drop=False).drop(columns = ['ticker', 'index'])
        
        df.columns.name = None
        
        return df

        

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
        ''' A percent change function for representing negative numbers intuitively was opposed to absolute change.
        https://github.com/pandas-dev/pandas/issues/22596
        '''
        self.pct_chg = self.df.copy()
        
        for c in self.pct_chg.columns:
            self.pct_chg[c] =  self.pct_chg[c].pct_change()*np.sign(self.pct_chg[c].shift(periods=1))
        
        # self.pct_chg.dropna(how = 'all', axis=0, inplace = True) 
        
        return self


    def describe(self):
        self.desc = self.pct_chg.describe().loc[['mean', 'std','25%','50%','75%'], :]
        
        return self


    def delta(self):
        ''' Change from previous quarter; Quarter over Quarter change, Same Quarter last year.'''
        print('Change Since: ')

        sub = self.df.iloc[[-5, -2, -1], :].dropna(how='any', axis=1)
        
        sub = sub.iloc[-1].squeeze() - sub.iloc[:-1]
        
        sub.index.name = 'Change Since'

        self.delta = sub
        
        return self

    
    def get_peers(self):
        tick = nasdaq.Tickers()
        tick.full_export(curl=False)
        
        industry = tick.get_industry(self.ticker[0])

        self.industry = industry
        print(industry)

        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///C:\data\industry_fundamentals.db', echo=False)
        
        self.cnxn = engine.connect()
        
        base = pd.read_sql(f"select * from CompFunBase where industry == '{industry}'", self.cnxn)

        base = base[base.industry == industry].sort_values(by = 'revenue', ascending = False)
        
        peers = base.ticker.unique().tolist()
        
        index = peers.index(self.ticker[0])
        
        if index == 0:
            search = 0, [1, 2, 3]
        
        elif index == 1:
            search = [1, 0, 2, 3]
        
        elif index == 2:
            search = [2, 0, 1, 3]
        
        else:
            search = [index, index-3, index-2, index-1]

        return [peers[i] for i in search]



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


    def plot_box_plot(self, cols):
        engine = create_engine('sqlite:///C:\data\industry_fundamentals.db', echo=False)
        
        cnxn = engine.connect()
        
        self.get_peers()

        date = cal.prior_quarter_end().strftime("%Y-%m-%d")
        
        df = pd.read_sql(f"select * from CompFunBase where industry = '{self.industry}' and calendardate = '{date}' ", cnxn)
        
        df = df[['ticker'] + cols]
        
        melt = df.melt(id_vars = 'ticker').dropna()

        def annotate(data, **kws):
            n = np.round(data.value.loc[data.ticker==self.ticker[0]].values[0],3)
            ax = plt.gca()
            ax.text(.15, n, f"{n}")
            ax.scatter(.1, n, color = 'orange')
                
        g = sns.FacetGrid(melt, col="variable", sharey=False,  col_wrap=7, height=5)
        
        g.map_dataframe(sns.boxplot, y="value", showfliers=False)
        
        g.map_dataframe(annotate)
        return g


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

        engine = create_engine('sqlite:///C:\data\industry_fundamentals.db', echo=False)
        self.cnxn = engine.connect()


    def get_ranks(self):
        ''' Returns ranks for all periods and one ticker '''
        df =  pd.read_sql(f"select * from CompFunRanks where ticker == '{self.ticker}'", self.cnxn)
        self.rank_pivot =  df.pivot(index = ['calendardate'], columns = ['variable'], values= ['value'])
        return self


    def style_jupyter(self, cols, units = '%'):

        r = self.rank_pivot.iloc[:, self.rank_pivot.columns.get_level_values(1).isin(cols)].round(2)
        
        new_cols = r.columns.reindex(cols, level=1)
        
        r =r.reindex(columns=new_cols[0])
        
        r = r.T.droplevel(0).iloc[:, -5:].T
        
        r.columns.name = None        

        
        return r.T.multiply(100).style      \
            .format("{:,.0f}%") \
            .applymap(lambda x: f"color: {'red' if x < 0 else 'black'}") \



    def plot_dual_axis_ranks(self, fun_obj, cols):
        ''' Plots a time series of fundamental values for an individual metric and company on one axis; With the % rank vs peer group on the second axis
        '''
        engine = create_engine('sqlite:///C:\data\industry_fundamentals.db', echo=False)

        cnxn = engine.connect()

        fun_obj.get_peers()

        # Base value
        df = pd.read_sql(f"select * from CompFunBase where ticker = '{fun_obj.ticker[0]}' ", cnxn)
        df = df[cols]
        base = df.melt(id_vars = ['ticker','calendardate']).dropna()

        # Peer ranks
        df = pd.read_sql(f"select * from CompFunRanks where ticker = '{fun_obj.ticker[0]}' ", cnxn)
        rank = df[df.variable.isin(cols)][['ticker', 'calendardate', 'variable','value']]

        # Chart data
        data = base.merge(rank, how = 'outer', on = ['calendardate','variable'])
        data.rename(columns = { 'value_x':'base_val'}, inplace = True)
        data.rename(columns = { 'value_y':'rank_val'}, inplace = True)


        def facetgrid_two_axes(*args, **kwargs):
            data = kwargs.pop('data')
            dual_axis = kwargs.pop('dual_axis')
            alpha = kwargs.pop('alpha', 0.8)
            kwargs.pop('color')
            ax = plt.gca()
            if dual_axis:
                ax2 = ax.twinx()
                ax2.set_ylabel('Percentile Rank')

            ax.plot(data['calendardate'],data['base_val'], **kwargs, color='red',alpha=alpha, label = 'Reported Value')
        #     ax.legend(loc=1)

            if dual_axis:
                ax2.plot(data['calendardate'],data['rank_val'], **kwargs, color='blue',alpha=alpha, label = 'Percentile Rank')
        #         ax2.legend(loc=4)

        sns.set_style('whitegrid') 
        g = sns.FacetGrid(data, col='variable', size=5, col_wrap = 7, sharex = False, sharey=False)
        g.map_dataframe(facetgrid_two_axes, dual_axis=True).set_axis_labels("Period", "Reported Value")

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        g.set_xticklabels(rotation=60)
        # g.add_legend()
        plt.show()
        return g



class DCF:

    def __init__(self, ticker,  REV_GROWTH=0.05):
        self.ticker = ticker
        
        self.cf = Fundamentals(ticker).get( columns = Columns.DCF.value + ['sharesbas'], limit = 5 )
        self.inc = Fundamentals(ticker).get( columns = Columns.INCOME.value, limit = 5 )
        self.bal = Fundamentals(ticker).get( columns = Columns.BALANCE.value + ['revenue', 'depamor', 'intexp', 'taxrate'], limit = 5 )

        self.FORECAST_PERIODS = 5
        self.REV_GROWTH = REV_GROWTH  

        self.forecast_as_percent_of_revenue(type = 'INCOME')
        self.forecast_as_percent_of_revenue(type = 'BALANCE')
        self.forecast_as_percent_of_revenue(type = 'CF')

        # x = dcf.style_jupyter(dcf.bal_forecast)
        # y = dcf.style_jupyter(dcf.inc_forecast)

        # fun.get( columns = Columns.INCOME.value, limit = 8 ).percent_change().style_jupyter(fun.pct_chg, units='%')
        # x = fun.get( columns = Columns.INCOME.value, limit = 8).describe().style_jupyter(fun.desc, units = '%')

        # fun.get( columns = Columns.CASHFLOW.value, limit = 8 ).percent_change().style_jupyter(fun.pct_chg, units='%')
        # y = fun.get( columns = Columns.INCOME.value, limit = 8).describe().style_jupyter(fun.desc, units = '%')

        # display_side_by_side(y,x)


    def forecast_as_percent_of_revenue(self, type = None):
        '''Grow revenue by a projected amount. Calculate other items as a percentage of revenue and forecast them as a percent of the grown revenue. 
        '''
        # TODO Pass Fundamental() object of Income Statement and use isinstance instead of string?
        if type == 'INCOME':
            statement = self.inc.df.copy()
        if type == 'BALANCE':
            statement = self.bal.df.copy()
        if type == 'CF':
            statement = self.cf.df.copy()

        pctrev = statement.divide(statement.revenue, axis=0) # statement fields as a percent of revenue
        pctrevavg = pctrev.median() # average percent of revenue for each metric over n periods provided

        # Project future revenue based on REV_GROWTH
        starting_rev = statement['revenue'].iloc[-1]
        forecast_rev = starting_rev
        arr = []
        for i in range(self.FORECAST_PERIODS):
            forecast_rev = forecast_rev * (1 + self.REV_GROWTH)
            arr.append(forecast_rev)

        forecast = pd.DataFrame(columns = statement.columns.tolist())
        forecast['revenue'] = arr

        # Project other metrics based on the % of 'Revenue' using the projected future revenues from above
        for i in range(self.FORECAST_PERIODS):
            for c in [c for c in forecast.columns if c not in ['revenue']]:
                forecast[c].iloc[i] = forecast['revenue'].iloc[i] * pctrevavg[c]

        forecast.index = [f'T+{i+1}' for i in range(0, self.FORECAST_PERIODS)]

        forecast = pd.concat([statement, forecast], axis = 0)

        for c in forecast.columns:
            forecast[c] = pd.to_numeric(forecast[c])

        if type == 'INCOME':
            self.inc_forecast = forecast
        if type == 'BALANCE':
            self.bal_forecast = forecast
        if type == 'CF':
            self.cf_forecast = forecast

        return pctrevavg


    def forecast_cf_from_opperations(self):
        bal = self.bal_forecast[['receivables','payables','inventory','depamor', 'equity', 'debt', 'intexp', 'taxrate']]
        inc = self.cf_forecast[['netinc', 'ncfo', 'capex', 'fcf']]
        
        df = pd.concat([bal, inc], axis=1)
        
        self.cf_from_opp = df

        return self.cf_from_opp


    def beta(self):
        df = yf.download(f"SPY {self.ticker[0].upper()}", start="2017-01-01", end="2017-04-30")['Adj Close']
        
        covariance = df.cov().iloc[0,1]
        
        benchmark_variance = df.SPY.var()

        beta = covariance / benchmark_variance 
        
        print('beta: ', beta)

        return beta


    def discount(self, ERM = 0.8, RFR=0):
        ''' CAPM defines cost of equity as beta; Discount future cash flows back to present value using WACC.
        '''
        rf = RFR  # risk free rate
        Erm = ERM  # exoected return on market

        df = self.cf_from_opp

        df['bv_equity'] = df['equity']
        
        df['cost_of_equity']  = rf  + (self.beta() * (Erm - rf))
        
        df['cost_of_debt'] = df['intexp'] / df['debt'].iloc[0:5].mean() # average debt over past 5 periods
        
        df['bv_debt']  = df['debt']
        
        tc = df['taxrate']

        eq = (df['bv_equity'] / (df['bv_equity'] + df['bv_debt'])) * df['cost_of_equity']
        
        dbt = (df['bv_debt'] / (df['bv_equity'] + df['bv_debt'])) * df['cost_of_debt']
        
        df['wacc'] =  (eq + dbt)  * ( 1 - df['taxrate'] / 100)

        wacc = pd.to_numeric(df['wacc'].iloc[-1])

        print('wacc: ', wacc)

        fcf = pd.to_numeric(df['fcf'].iloc[-self.FORECAST_PERIODS:]).values.tolist()

        self.npv = npf.npv(wacc,fcf)

        return self.npv
        

    def terminal_value(self, TERMINAL_GROWTH=0.03):
        self.TERMINAL_GROWTH = TERMINAL_GROWTH

        if self.cf_from_opp['wacc'].iloc[-1] > self.TERMINAL_GROWTH:
            terminal_value = (self.cf_from_opp['fcf'].iloc[-1] * (1+ self.TERMINAL_GROWTH)) / (self.cf_from_opp['wacc'].iloc[-1] )
        else:
            terminal_value = (self.cf_from_opp['fcf'].iloc[-1] * (1+ self.TERMINAL_GROWTH)) / (self.cf_from_opp['wacc'].iloc[-1])
        
        self.terminal_value_discounted = terminal_value/(1+self.cf_from_opp['wacc'].iloc[-1] )**4
        
        return self.terminal_value_discounted
    

    def estimate_price_per_share(self):
        print('npv: ', self.npv)
        print('terminal value: ', self.terminal_value_discounted)
        
        value = self.npv + self.terminal_value_discounted

        price = value / self.cf.df['sharesbas'].iloc[-1]
        
        print('pv future cash flows: ', price)

        return  None
        

    def style_jupyter(self, df):
        df = df.divide(1000000).round(2).T

        return df.style      \
            .format("${:,}") \
            .applymap(lambda x: f"color: {'red' if x < 0 else 'black'}") \
            .set_properties(**{'border': '1px solid lightgrey'})      \
            .set_table_styles(
            [
            {"selector": "td, th", "props": [("border", "1px solid lightgrey !important")]},
            ]
        )



class Columns(Enum):
    ''' Column views for financial statements
    '''
    ID = ['ticker', 'calendardate' ]
    
    CASHFLOW = ID +  ['cashneq', 'netinc', 'depamor', 'opex', 'receivables', 'payables', 'inventory', 'ncfo', 'ncfbus', 'ncfi', 'ncfinv',  'ncfdiv',  'ncfx', 'ncff', 'fcf', 'ncf']
    
    INCOME = ID +  ['revenue', 'cogs','gp', 'opex','opinc','ebt','netinc','ebitda', 'depamor'] 
    
    BALANCE = ID + ['assetsc', 'assetsnc', 'receivables', 'inventory', 'assets', 'liabilitiesc','liabilitiesnc', 'payables', 'debt','equity','retearn']
    
    PEERS = ID +  ['divyield', 'grossmargin', 'netmargin', 'fcfmargin', 'oppmargin','roe', 'roic', 'ros', 'roc']
    
    EXP = ID + ['retentionratio', 'roe', 'retearn','expnetincgrow', 'exproegrow', 'eqreinvestrate', 'expebitgrow','expgrowthrate']
    
    DCF = ID + ['netinc', 'revenue','cogs','gp', 'rnd','sgna','ebit', 'payables','receivables', 'inventory', 'depamor','ebitda','capex', 'fcf', 'ncfo']
    
    RANKS = list(set(ID + CASHFLOW + INCOME + BALANCE + PEERS + EXP + DCF + ['name', 'industry', 'sector', 'famaindustry', 'famasector', 'scalemarketcap','scalerevenue'] + ['pe','eps']))

    CASHFLOW_ = ID +  ['cashneq', 'depamor','retearn', 'opex', 'capex', 'ncfo', 'ncfdiv', 'fcf', 'ncf']
