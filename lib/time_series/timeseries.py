'''
https://machinelearningmastery.com/time-series-data-stationary-python/
Machine Learning for Algorithmic Trading
https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/blob/master/09_time_series_models/01_tsa_and_stationarity.ipynb
'''
import pandas as pd 
import numpy as np
import sys,os
from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from matplotlib import pyplot
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import os, pathlib

class TimeSeries():
    ''' Apply time series analysis to a series of data
    '''
    def __init__(self, data, column:str=None) -> None:
        
        
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError('[ERROR] Column string is required if a datafrme is passed to the constructor')
            self.data = data[column]
            self.data_fmt = pd.DataFrame

        elif isinstance(data, pd.Series):
            self.data = data
            self.data_fmt = pd.Series

        elif type(data) == str:
            if column is None:
                raise ValueError('[ERROR] Column string is required if a datafrme is passed to the constructor')
            if '.csv' in data:
                self.data = pd.read_csv(data, index_col=0)
            elif '.xls' in data: 
                self.data = pd.read_excel(data, index_col=0)
            print(self.data)
            self.data = pd.DataFrame(self.data[column])
            self.data_fmt = pathlib.Path

        if not isinstance(self.data.index, pd.DatetimeIndex):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except:
                raise ValueError('[ERROR] Index must be a datetime index')

        self.column = column

        if self.data.isnull().values.any():
            print('[WARNING] NaN values found in dataframe')
            try:
                self.data = self.interpolate_na( method='time' ).dropna()
                print('[INFO] NaN values interpolated')
            except: 
                self.data.dropna(inplace=True)


        print(self.__dict__)

    
    def __str__(self):
        return 'TimeSeries Object'


    def decomposition(self, model:str = None, plot=True):
        ''' model = 'additive' or 'multiplicative'
        '''
        from statsmodels.tsa.seasonal import seasonal_decompose
        components = seasonal_decompose(self.data, model=model, period=1) # freq is required else error thrown
        if plot == True:
            ts = (self.data
                .assign(Trend=components.trend)
                .assign(Seasonality=components.seasonal)
                .assign(Residual=components.resid))
            with sns.axes_style('white'):
                ts.plot(subplots=True, figsize=(14, 8), title=['Original Series', 'Trend Component', 'Seasonal Component','Residuals'], legend=False)
                plt.suptitle('Seasonal Decomposition', fontsize=14)
                sns.despine()
                plt.tight_layout()
                plt.subplots_adjust(top=.91)
        
            # alternative method:
            # components.plot() \
            #              .suptitle(f'{model}  decomposition', 
            #                        fontsize=14)

        plt.show()


    def check_stationarity(self, plot=True):
        X = self.data.values
        split = round(len(X) / 2)
        X1, X2 = X[0:split], X[split:]
        mean1, mean2 = X1.mean(), X2.mean()
        var1, var2 = X1.var(), X2.var()
        print('mean1=%f, mean2=%f' % (mean1, mean2))
        print('variance1=%f, variance2=%f' % (var1, var2))
        self.data.hist()

        if plot == True:
            X = np.log(X) #log transform
            plt.hist(X)
            plt.show()

    def auto_correlation(self, lags=24):
        from statsmodels.graphics import tsaplots
        fig = tsaplots.plot_acf(self.data)
        plt.show()


    def interpolate_na(self, col_name=None, method='time'):
        self.data = self.data.interpolate(method, axis = 0) #linear or time
        return self.data
    

    def prophet_forecast(self):
        from fbprophet import Prophet
        
        data = self.data.reset_index()
        data.rename(columns={'Date': 'ds', self.column: 'y'}, inplace=True)

        df_train = data.iloc[ : int(len(data)/2), :] # fixme use sklearn
        df_test = data.iloc[ int(len(data)/2):, :]

        model_prophet = Prophet(seasonality_mode='additive')
        model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model_prophet.fit(df_train)

        df_future = model_prophet.make_future_dataframe(periods=365)
        df_pred = model_prophet.predict(df_future)
        model_prophet.plot(df_pred)
        plt.show()
        

plt.tight_layout()
#plt.savefig('images/ch3_im3.png')
plt.show()

# @Test 
ts = TimeSeries(data='/Users/michaelsands/data/stock_prices.csv', column='IAU')
# ts.decomposition(model='multiplicative')
# ts.check_stationarity()
# ts.auto_correlation( lags=24 )
ts.prophet_forecast()



############################################################################################

# tickers = ['GOOG','AMZN','MSFT','AAPL', 'BLK','JPM','IAU']
# import pandas_datareader as data
# import pandas as pd
# data_source = 'yahoo'
# start_date = '2000-01-01'
# end_date = '2022-04-30'
# frames = []
# for ticker in tickers:
#     stockdata = data.DataReader(ticker, data_source, start_date, end_date)['Close']
#     stockdata =  pd.DataFrame(stockdata).rename(columns={'Close':ticker})
#     frames.append(stockdata)
# df = pd.concat(frames, axis=1)
# df.to_csv('/Users/michaelsands/data/stock_prices.csv')

# import datetime
# start = datetime.datetime (2000, 1, 1)
# end = datetime.datetime (2022, 4, 30)
# seriesid = ['PAYEMS', 'GDP', 'UNRATE', 'CPILFESL', 'T10YIE', 'DGS10', 'DGS30']
# frames = []
# for ticker in seriesid:
#     stockdata = data.DataReader(ticker, 'fred', start_date, end_date)
#     frames.append(stockdata)
# df = pd.concat(frames, axis=1)
# df.to_csv('/Users/michaelsands/data/fred_prices.csv')
