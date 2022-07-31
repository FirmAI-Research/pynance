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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from statsmodels.tsa.stattools import adfuller

import seaborn as sns 
import os, pathlib

cwd = os.getcwd()

class TimeSeries():
    ''' Apply time series analysis to a series of data
    '''
    def __init__(self, data, column:str=None) -> None:
        
        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError('[ERROR] Column string is required if a datafrme is passed to the constructor')
            self.data = data[column].dropna()
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


        # print(self.__dict__)

    
    def __str__(self):
        return 'TimeSeries Object'


    def decomposition(self, model:str = None, plot=True):
        ''' model = 'additive' or 'multiplicative'
        '''
        from statsmodels.tsa.seasonal import seasonal_decompose
        components = seasonal_decompose(self.data, model=model, period=30) # freq is required else error thrown
        if plot == True:

            ts = (pd.DataFrame(self.data)
                .assign(Trend=components.trend)
                .assign(Seasonality=components.seasonal)
                .assign(Residual=components.resid))

            with sns.axes_style('white'):
                ts.plot(subplots=True, figsize=(14, 8), title=['Original Series', 'Trend Component', 'Seasonal Component','Residuals'], legend=False)
                plt.suptitle('Seasonal Decomposition', fontsize=14)
                sns.despine()
                # fig.set_size_inches(5, 8)
                plt.tight_layout()
                plt.subplots_adjust(top=.91)
                plt.show()
             

    def check_stationarity(self, difference = False):
        if difference == True:
            X = self.difference(self.data.values)
        else:
            X = self.data.values
        split = round(len(X) / 2)
        X1, X2 = X[0:split], X[split:]
        mean1, mean2 = X1.mean(), X2.mean()
        var1, var2 = X1.var(), X2.var()
        print('mean1=%f, mean2=%f' % (mean1, mean2))
        print('variance1=%f, variance2=%f' % (var1, var2))


    def test_stationarity(self):
        '''
        Null Hypothesis: The series has a unit root (value of a =1)

        Alternate Hypothesis: The series has no unit root.
        If the null hypothesis is not rejected, the series is said to be non-stationary. The series can be linear or difference stationary as a result of this.

        The series becomes stationary if both the mean and standard deviation are flat lines (constant mean and constant variance).

        The increasing mean and standard deviation may be seen in the graph above, indicating that our series isn’t stationary.

        We can’t rule out the Null hypothesis because the p-value is bigger than 0.05. Additionally, the test statistics exceed the critical values. As a result, the data is nonlinear.
        '''
        timeseries = self.data
        #Determing rolling statistics
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()
        #Plot rolling statistics:
        plt.plot(timeseries, color='blue',label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean and Standard Deviation')
        plt.show(block=False)
        print("Results of dickey fuller test")
        adft = adfuller(timeseries,autolag='AIC')
        # output for dft will give us without defining what the values are.
        #hence we manually write what values does it explains using a for loop
        output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
        for key,values in adft[4].items():
            output['critical value (%s)'%key] =  values
        return output


    def remove_nonstationary_trend(self):
        from pylab import rcParams
        rcParams['figure.figsize'] = 10, 6
        df_log = np.log(self.data)
        moving_avg = df_log.rolling(12).mean()
        std_dev = df_log.rolling(12).std()
        plt.legend(loc='best')
        plt.title('Moving Average')
        plt.plot(std_dev, color ="black", label = "Standard Deviation")
        plt.plot(moving_avg, color="red", label = "Mean")
        plt.legend()
        plt.show()
        self.df_log = df_log


    def t_t_split(self, plot = True):
        if plot:
            train_data, test_data = self.df_log[3:int(len(self.df_log)*0.9)], self.df_log[int(len(self.df_log)*0.9):]
            plt.figure(figsize=(10,6))
            plt.grid(True)
            plt.xlabel('Dates')
            plt.ylabel('Closing Prices')
            plt.plot(self.df_log, 'green', label='Train data')
            plt.plot(test_data, 'blue', label='Test data')
            plt.legend()
        self.train_data = train_data
        self.test_data = test_data


    def difference(self, interval=5):
        # # if dataset is nonstationary

        self.data = self.data.diff()

        # TODO: must drop infinite values after differencing
        if np.inf in self.data:
            self.data = self.data.loc[self.data != np.inf] # FIXME
        return self.data

        # Manual implementation
        # diff = list()
        # for i in range(interval, len(dataset)):
        #     value = dataset[i] - dataset[i - interval]
        #     diff.append(value)
        # return Series(diff)


    def auto_correlation(self, lags=24):
        from statsmodels.graphics import tsaplots
        tsaplots.plot_acf(self.data)
        plt.show()


    def interpolate_na(self, col_name=None, method='time'):
        self.data = self.data.interpolate(method, axis = 0) #linear or time
        return self.data
    

    def prophet_forecast(self): # FIXME
        from sys import platform
        if platform == "linux" or platform == "linux2":
            from prophet import Prophet # linux
        elif platform == "darwin":
            from fbprophet import Prophet # OS X
        elif platform == "win32":
            from fbprophet import Prophet # Windows...
            
        data = self.data.reset_index()
        data.rename(columns={'Date': 'ds', self.column: 'y'}, inplace=True)

        # FIXME and take user input
        size = len(data)
        df_train = data.iloc[ int(-size*.10): , :] 
        df_test = data.iloc[   int(-size*.20): ,  :]

        model_prophet = Prophet(seasonality_mode='additive')
        model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model_prophet.fit(df_train)

        df_future = model_prophet.make_future_dataframe(periods=365)
        df_pred = model_prophet.predict(df_future)
        model_prophet.plot(df_pred)
        plt.tight_layout()
        plt.title('Prophet Forecast')
        # plt.savefig(os.path.join(img_dirp, f'img/prophet_forecast.png'))

        model_prophet.plot_components(df_pred)
        plt.tight_layout()
        # plt.savefig(os.path.join(img_dirp, 'img/components.png'))

        # merge test set with predicted data and plot accuracy of model's predictions
        selected_columns = ['ds', 'yhat_lower', 'yhat_upper', 'yhat']

        df_pred = df_pred.loc[:, selected_columns].reset_index(drop=True)
        df_test = df_test.merge(df_pred, on=['ds'], how='left')
        df_test.ds = pd.to_datetime(df_test.ds)
        df_test.set_index('ds', inplace=True)
        fig, ax = plt.subplots(1, 1)
        ax = sns.lineplot(data=df_test[['y', 'yhat_lower', 
                                        'yhat_upper', 'yhat']])
        ax.fill_between(df_test.index,
                        df_test.yhat_lower,
                        df_test.yhat_upper,
                        alpha=0.3)
        ax.set(title=f'{self.column} - actual vs. predicted',
            xlabel='Date',
            ylabel='{self.column}')

        plt.tight_layout()
        # plt.savefig(os.path.join(img_dirp, 'img/actual_v_predicted.png'))

        # plt.show()

        self.prophet_df_test_pred = df_test




# @Test 
# ts = TimeSeries(data='/Users/michaelsands/data/stock_prices.csv', column='AMZN')
# ts.decomposition(model='additive')
# ts.check_stationarity()
# ts.difference()
# ts.auto_correlation( lags=24 )
# ts.prophet_forecast()


