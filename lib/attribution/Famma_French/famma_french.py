
import sys,os 
import urllib.request
import zipfile
import pandas as pd 
from functools import reduce
import statsmodels.api as smf
import matplotlib.pyplot as plt

from model.performance.returns import Returns
from controller.calendar import Calendar

class FammaFrench:
    '''
    https://blogs.cfainstitute.org/investor/2022/01/10/fama-and-french-the-five-factor-model-revisited/
    https://www.quantconnect.com/tutorials/introduction-to-financial-python/fama-french-multi-factor-models
    '''

    def __init__(self, symbols:str, weights:str):
        ''' 
        :param: symbols: list of tickers to retrieve historical prices for for factor attribution against the FF 5 factors
        '''
        self.symbols = symbols
        self.weights = weights
        self.iodir = './vendors/output/famma_french'
        self.data_dir= os.path.join(self.iodir, 'data')       
        # key = file name; values = skiprows, zip file download path
        self.parse_dict = {'F-F_Momentum_Factor_daily_CSV':(13, 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'),
                    'F-F_Research_Data_Factors_daily_CSV':(4, 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'),
                    'F-F_ST_Reversal_Factor_daily_CSV':(13, 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_ST_Reversal_Factor_daily_CSV.zip'),
                    'F-F_LT_Reversal_Factor_daily_CSV':(13,  'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_LT_Reversal_Factor_daily_CSV.zip'),
                    }

        

    def get_data(self, fname):

        skiprows, url = self.parse_dict.get(fname)
        print(f"Parsing {fname}: SkipRows:{skiprows}")

        ''' retrieve & extract .zip data '''
        localp_zip = f'{self.iodir}/data/{fname}.zip'
        urllib.request.urlretrieve(url,localp_zip)
        zip_file = zipfile.ZipFile(localp_zip, 'r')
        zip_file.extractall(self.data_dir)
        zip_file.close()
        csv_name = fname[:-4] #drop the _CSV included in zip name

        ''' read local data '''
        localp_csv = f'{self.iodir}/data/{csv_name}.CSV'
        ff_factors = pd.read_csv(localp_csv, skiprows = skiprows, index_col = 0).reset_index().dropna().rename(columns={'index':'date'})
        ff_factors['date']= pd.to_datetime(ff_factors['date'], format= '%Y%m%d')
        ff_factors = ff_factors.set_index('date').dropna()

        def fx(x):
            return x/100 # convert values from percentages to decimals

        for c in ff_factors.columns:
            ff_factors[c] = pd.to_numeric(ff_factors[c]).apply(fx)

        return ff_factors


    def build_df(self):
        frames = []
        for k,v in self.parse_dict.items():
            _ = self.get_data(k)
            frames.append(_)
        df = reduce(lambda x, y: pd.merge(x, y, on = 'date'), frames)
        df.to_csv(f'{self.iodir}/ff_factors.csv')
        return df 


    def get_returns(self):
        return Returns(symbols=self.symbols, weights=self.weights, start_date='1975-01-01', end_date=Calendar().today()).weighted_returns()


    def merge_factors_and_portfolio(self, download_ff_data=True):
        ''' set download_ff_data to skip the download and use the most recent csv file written. the file is writen after each get-data call and data only changes daily
        '''
        if download_ff_data == True:
            factors = self.build_df()
        else:
            factors = pd.read_csv(f'{self.iodir}/ff_factors.csv')
            factors.date = pd.to_datetime(factors['date'], format= '%Y-%m-%d')
            factors = factors.set_index('date')

        portfolio = self.get_returns()
        portfolio.index.names = ['date']

        df = factors.merge(portfolio, left_index = True, right_index = True, how = 'inner')
        df.rename(columns={"Mkt-RF":"mkt_excess", "Mom   ":"Mom"}, inplace=True)
        df['port_excess'] = df['weighted_returns'] - df['RF']
        self.df = df


    def three_factor(self):
        ''' multiple regression model '''
        self.model  = smf.formula.ols(formula = "port_excess ~ mkt_excess + SMB + HML", data = self.df).fit()

    def carhar_four_factor(self):
        self.model  = smf.formula.ols(formula = "port_excess ~ mkt_excess + SMB + HML + Mom", data = self.df).fit()
    
    def five_factor(self):
        self.model = smf.formula.ols(formula = "port_excess ~ mkt_excess + SMB + HML + ST_Rev + LT_Rev + Mom", data = self.df).fit()
        

    def print_summary(self):
        print(self.df.tail())
        print(self.model.summary())
        print('Parameters: ', self.model.params)
        print('R2: ', self.model.rsquared)


    def plot(self):
        ((self.df +1).cumprod()).plot(figsize=(15, 7))
        plt.title(f"Famma French Factors", fontsize=16)
        plt.ylabel('Portfolio Returns', fontsize=14)
        plt.xlabel('Year', fontsize=14)
        plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.yscale('log')
        plt.show()


    def plot_text(self):
        plt.text(0.01, 0.05, str(self.model.summary()), {'fontsize': 10}, fontproperties = 'monospace')  