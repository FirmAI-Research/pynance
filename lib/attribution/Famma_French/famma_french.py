
import sys
import os
import urllib.request
import zipfile
import pandas as pd
from functools import reduce
import statsmodels.api as smf
import matplotlib.pyplot as plt
import yfinance as yf
from os import listdir
from os.path import isfile, join

from lib.calendar import Calendar
cal = Calendar()

class FammaFrench:
    '''
    Regress excess returns against various risk factors

    Excess Return = Return of investment - risk free rate of return
        Run a regression on this excess return against various famma french factors
    
    The famma factors are coefficients ( beta )
        You can say a fund performs good or bed when certain factors are performing in certain ways
        Positive coeficients indicate excess returns of investment strategy; however if the p value is too small, the excess returns (Coef. of Intercept) may not be statistically significan

    Take all stocks; rank them by the market value of their equity of compute median mkt value of equity
        All stocks below are small and all above are big
        SMB --> Small Cap stock excess return minus Large stocks excess return

    Rank all stocks according to book value of equity and market value of equtiy    
        Split based on 30%/70% marks to define Growth/Value
        HML --> Value minus Growth --> HML refers to high M/B as "Growth" and low M/B as "Value"

    Coefficient --> "exposure" to various risk factors
    Y Intercept --> Coefficient of Y Intercept == "Alpha": Check p value to see if it's statistically significant

    Assume a fund:      
        calculate annualized mean return & standard deviation
        calculate sharpe ratio



    ***models***
    1 - Regress Excess return of SPY against 5 famma french factors to attribute broad market movements
    
    https://blogs.cfainstitute.org/investor/2022/01/10/fama-and-french-the-five-factor-model-revisited/
    https://www.quantconnect.com/tutorials/introduction-to-financial-python/fama-french-multi-factor-models
    '''

    def __init__(self, symbols: list, weights: list, iodir:str):
        ''' 
        :param: symbols: list of tickers to retrieve historical prices for for factor attribution against the FF 5 factors
        '''
        self.symbols = symbols
        self.weights = weights
        self.iodir = iodir
        self.data_dir = os.path.join(self.iodir, 'data')
        # key = file name; values = skiprows, zip file download path
        self.parse_dict = {'F-F_Momentum_Factor_daily_CSV': (13, 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'),
                           'F-F_Research_Data_Factors_daily_CSV': (4, 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'),
                           'F-F_ST_Reversal_Factor_daily_CSV': (13, 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_ST_Reversal_Factor_daily_CSV.zip'),
                           'F-F_LT_Reversal_Factor_daily_CSV': (13,  'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_LT_Reversal_Factor_daily_CSV.zip'), }


    def get_data(self, fname):

        skiprows, url = self.parse_dict.get(fname)
        print(f"Parsing {fname}: SkipRows:{skiprows}")

        ''' retrieve & extract .zip data '''
        localp_zip = f'{self.iodir}/data/{fname}.zip'
        urllib.request.urlretrieve(url, localp_zip)
        zip_file = zipfile.ZipFile(localp_zip, 'r')
        zip_file.extractall(self.data_dir)
        zip_file.close()
        
        csv_name = fname[:-4]  # drop the _CSV included in zip name

        # linux filepaths are case sensitive, windows are insensitive
        onlyfiles = [f for f in listdir(f'{self.iodir}/data/') if isfile(join(self.iodir, 'data',f))]
        for f in onlyfiles: 
            if '.zip' not in f:
                os.rename( join(self.iodir, 'data',f), join(self.iodir, 'data',f).replace('CSV', 'csv'))

        ''' read local data '''
        localp_csv = f'{self.iodir}/data/{csv_name}.csv'
        ff_factors = pd.read_csv(localp_csv, skiprows=skiprows, index_col=0).reset_index().dropna().rename(columns={'index': 'date'})
        ff_factors['date'] = pd.to_datetime(
            ff_factors['date'], format='%Y%m%d')
        ff_factors = ff_factors.set_index('date').dropna()

        def fx(x):
            return x/100  # convert values from percentages to decimals

        for c in ff_factors.columns:
            ff_factors[c] = pd.to_numeric(ff_factors[c]).apply(fx)

        return ff_factors

    # TODO
    def populate_db_table(self):
        pass

    def build_df(self):
        frames = []
        for k, v in self.parse_dict.items():
            _ = self.get_data(k)
            frames.append(_)
        df = reduce(lambda x, y: pd.merge(x, y, on='date'), frames)
        df.to_csv(f'{self.iodir}/ff_factors.csv')
        return df


    def get_returns(self):
        ''' calculate weighted returns of a portfolio'''
        frames = []
        print(len(self.symbols))
        if isinstance(self.symbols, list) and len(self.symbols) > 1:
            print('Using weighting scheme')
            for symbol in self.symbols:
                returns = pd.DataFrame(yf.download(symbol,'1975-01-01')['Adj Close'].pct_change().dropna())
                returns.rename(columns = {'Adj Close':symbol}, inplace = True)
                frames.append(returns)
            from functools import reduce
            df = reduce(lambda x, y: pd.merge(x, y, on = 'Date'), frames)
            print(df)

            df['weighted_returns'] = pd.DataFrame(df.dot(self.weights)) # calculate weighted returns via matrix multiplication
        else:
            print('Not using weighting scheme')
            symbol = self.symbols[0]
            returns = pd.DataFrame(yf.download(symbol,'1975-01-01')['Adj Close'].pct_change().dropna())
            returns.rename(columns = {'Adj Close':'weighted_returns'}, inplace = True)
            df = returns

        return df


    def merge_factors_and_portfolio(self, download_ff_data=True):
        ''' set download_ff_data to skip the download and use the most recent csv file written. the file is writen after each get-data call and data only changes daily '''
        if download_ff_data == True:
            factors = self.build_df()
        else:
            factors = pd.read_csv(f'{self.iodir}/ff_factors.csv')
            factors.date = pd.to_datetime(factors['date'], format='%Y-%m-%d')
            factors = factors.set_index('date')

        portfolio = self.get_returns()
        portfolio.index.names = ['date']
        
        df = factors.merge(portfolio, left_index=True,
                           right_index=True, how='inner')
        df.rename(columns={"Mkt-RF": "mkt_excess",
                  "Mom   ": "Mom"}, inplace=True)
        print(df)
        df['port_excess'] = df['weighted_returns'] - df['RF']
        self.df = df


    ''' multiple regression models '''

    def three_factor(self):
        self.model = smf.formula.ols(
            formula="port_excess ~ mkt_excess + SMB + HML", data=self.df).fit()

    def carhar_four_factor(self):
        self.model = smf.formula.ols(
            formula="port_excess ~ mkt_excess + SMB + HML + Mom", data=self.df).fit()

    def five_factor(self):
        self.model = smf.formula.ols(
            formula="port_excess ~ mkt_excess + SMB + HML + ST_Rev + LT_Rev + Mom", data=self.df).fit()

    def print_summary(self):
        print(self.df.tail())
        print(self.model.summary())
        print('Parameters: ', self.model.params)
        print('R2: ', self.model.rsquared)

    def plot(self):
        ((self.df + 1).cumprod()).plot(figsize=(15, 7))
        plt.title(f"Famma French Factors", fontsize=16)
        plt.ylabel('Portfolio Returns', fontsize=14)
        plt.xlabel('Year', fontsize=14)
        plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.yscale('log')
        plt.show()

    def plot_text(self):
        plt.text(0.01, 0.05, str(self.model.summary()), {
                 'fontsize': 10}, fontproperties='monospace')
