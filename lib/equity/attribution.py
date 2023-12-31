# Famma French; Brinson; Contribution of n stocks to S&P500 return
import pandas as pd
import pandas_datareader
import numpy as np
import scipy.stats
from functools import reduce
import pandas_datareader as web
import requests
import re
import json
from bs4 import BeautifulSoup
import yfinance as yf
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from calendar_dates import Calendar
cal = Calendar()

class Attribution:

    def __init__(self):
        self.START_DATE = '2021-01-01'


    def get_holdings(self, ticker):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"
        }

        def main(url, tickers):
            with requests.Session() as req:
                req.headers.update(headers)    
                frames = []
                for key in tickers:
                    r = req.get(url.format(key))
                    print(f"Extracting: {r.url}")      
                    rows = []
                    for line in r.text.splitlines():
                        if not line.startswith('etf_holdings.formatted_data'):
                            continue
                        data = json.loads(line[30:-1])
                        for holding in data:
                            goal = re.search(r'etf/([^"]*)', holding[1])
                            if goal:
                                rows.append([goal.group(1), *holding[2:5]])
                    df = pd.DataFrame(rows, columns = ['Symbol', 'Shares', 'Weight', '52 week change'])
                    for err in ['NaN', 'NA', 'nan']:
                        df["Weight"] = df['Weight'].apply(lambda x : x.replace(err, ''))
                    
                    df.dropna(how='any', axis=0, inplace=True)
                    df = df[['Symbol', 'Shares', 'Weight']]
                    df['Shares'] = pd.to_numeric([x.replace(',','') for x in df['Shares']])
                    df['Weight'] = pd.to_numeric(df['Weight'])
                    df.dropna(how='any', axis=0, inplace=True)
                    frames.append(df)
            return frames
                    
        portfolio = main(url = "https://www.zacks.com/funds/etf/{}/holding", tickers = [ ticker ])
        # print(portfolio)

        self.symbols = portfolio[0].Symbol.tolist()
        self.weights = portfolio[0].Weight.tolist()
        
        return portfolio
    

    def get_portfolio_returns(self):
        ''' calculate weighted returns of a portfolio'''
        frames, invalid = [], []

        for ix, symbol in enumerate(self.symbols):
            
            returns = pd.DataFrame(yf.download(symbol, self.START_DATE, cal.today(), progress=False)['Adj Close'].pct_change().dropna())
            
            returns.rename(columns={'Adj Close': symbol}, inplace=True)

            frames.append(returns) if len(returns) > 0 else invalid.append(ix) # if no data is returned, note the index of the symbol so the weight can be removed
        
        if len(invalid) > 0:

            for i in invalid:
            
                del self.weights[i]
                
        df = reduce(lambda x, y: pd.merge(x, y, on='Date', how = 'outer'), frames) # Each underlying holding may have a different date range available, so outer join and fillna returns as 0

        df.fillna(0, inplace=True)
        
        df['portf_rtn'] = pd.DataFrame(df.dot(self.weights)) # calculate weighted returns via matrix multiplication

        df = df[[x for x in df.columns if x not in self.symbols]] # drop individual asset returns and keep only weighted portfolio returns

        df.index = pd.to_datetime(df.index)

        self.portf_rets = df
        
        return df




class FammaFrench(Attribution):

    def __init__(self, model='ThreeFactor', portf_rets = None):
        super().__init__()
    
        if model == 'ThreeFactors':
            factors = self.get_ff_three_factor()
            df = portf_rets.merge(factors,  left_index=True, right_index=True, how='inner')
            self.summary, self.results = self.three_factor_model(df)

        elif model == 'IndustryFactors':
            factors = self.get_ff_industry_factors()
            df = portf_rets.merge(factors,  left_index=True, right_index=True, how='inner')
            self.summary, self.results = self.industry_factor_model(df)


    def get_ff_three_factor(self):

        df = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=self.START_DATE)[0]
        
        df = df.div(100)
        
        df.index = df.index.strftime('%Y-%m-%d')
        
        df.index.name = 'Date'
        
        df.index = pd.to_datetime(df.index)
        
        return df


    def get_ff_industry_factors(self):
        
        df = web.DataReader('10_Industry_Portfolios', 'famafrench', 
                                        start=self.START_DATE)[0]
        
        df = df.div(100)
        
        df.index = df.index.strftime('%Y-%m-%d')
        
        df.index.name = 'Date'
        
        df.index = pd.to_datetime(df.index)
        return df


    def rolling_factor_model(self, input_data, formula, window_size):
        '''
        Function for estimating the Fama-French (n-factor) model using a rolling window of fixed size.
        Parameters
        ------------
        input_data : pd.DataFrame
            A DataFrame containing the factors and asset/portfolio returns
        formula : str
            `statsmodels` compatible formula representing the OLS regression  
        window_size : int
            Rolling window length.
        Returns
        -----------
        coeffs_df : pd.DataFrame
            DataFrame containing the intercept and the three factors for each iteration.
        '''
        coeffs = []

        for start_index in range(len(input_data) - window_size + 1):
            end_index = start_index + window_size

            # define and fit the regression model
            ff_model = smf.ols(
                formula=formula,
                data=input_data[start_index:end_index]
            ).fit()

            # store coefficients
            coeffs.append(ff_model.params)

        coeffs_df = pd.DataFrame(
            coeffs,
            index=input_data.index[window_size - 1:]
        )

        return coeffs_df


    def three_factor_model(self, df):

        ff_data = df

        ff_data.columns = ['portf_rtn', 'mkt', 'smb', 'hml', 'rf']
        
        ff_data['portf_ex_rtn'] = ff_data.portf_rtn - ff_data.rf

        ff_model = smf.ols(formula='portf_ex_rtn ~ mkt + smb + hml',
                           data=ff_data).fit()
        
        # print(ff_model.summary())

        for c in ff_data.columns:
            ff_data[c] = pd.to_numeric(ff_data[c])

        MODEL_FORMULA = 'portf_ex_rtn ~ mkt + smb + hml'
        
        results_df = self.rolling_factor_model(ff_data,
                                               MODEL_FORMULA,
                                               window_size=3)
        
        return ff_model.summary(), results_df


    def industry_factor_model(self, df):

        ff_data = df

        ff_data.columns = ['portf_rtn', 'NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth', 'Utils', 'Other']
        
        ff_model = smf.ols(formula='portf_rtn ~ NoDur + Durbl + Manuf + Enrgy + HiTec + Telcm + Shops + Hlth + Utils + Other', data=ff_data).fit()
        
        for c in ff_data.columns:
            ff_data[c] = pd.to_numeric(ff_data[c])

        MODEL_FORMULA = 'portf_rtn ~ NoDur + Durbl + Manuf + Enrgy + HiTec + Telcm + Shops + Hlth + Utils + Other'
        
        results_df = self.rolling_factor_model(ff_data, 
                                          MODEL_FORMULA, 
                                          window_size=3)

        return ff_model.summary(), results_df


    def beta(self):
        ff_data = self.data

        covariance = ff_data[['mkt','portf_ex_rtn']].cov().iloc[0,1]
        benchmark_variance = ff_data.mkt.var()
        return covariance / benchmark_variance # beta


    def plot(self):
        plt.rcParams["figure.figsize"] = (20,7)
        self.results.plot(title=f'Rolling Fama-French Industry Factor model')
        plt.legend(bbox_to_anchor=(1.1, 1.05))

    def explain(self):
        '''Construct a sentence in english explaining how much of the SPY return over the past 30 days is attributable to each factor '''
        pass



class Brinson:

    def __init__(self):
        pass