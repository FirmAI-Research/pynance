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
    '''

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
                self.data = pd.read_csv(data)
            elif '.xls' in data: 
                self.data = pd.read_excel(data)
            self.data = self.data[column]
            self.data_fmt = pathlib.Path

        else:
            raise ValueError('[ERROR] Invalid data format was supplied by the user. Please pass a path to an existing file, and Series, or a DataFrame')

        print(self.__dict__)

    
    def __str__(self):
        return 'TimeSeries Object'


    def decomposition(self, data, col_name=None, plot=True):
        from statsmodels.tsa.seasonal import seasonal_decompose
        components = seasonal_decompose(data[col_name], model='additive', freq=30) # freq is required else error thrown

        if plot == True:

            ts = (data[col_name].to_frame('Close')
                .assign(Trend=components.trend)
                .assign(Seasonality=components.seasonal)
                .assign(Residual=components.resid))
            with sns.axes_style('white'):
                ts.plot(subplots=True, figsize=(14, 8), title=['Original Series', 'Trend Component', 'Seasonal Component','Residuals'], legend=False)
                plt.suptitle('Seasonal Decomposition', fontsize=14)
                sns.despine()
                plt.tight_layout()
                plt.subplots_adjust(top=.91);
        plt.show()


    def check_stationarity(self, data=None, col_name=None,plot=True):
        X = self.df[col_name].values
        split = round(len(X) / 2)
        X1, X2 = X[0:split], X[split:]
        mean1, mean2 = X1.mean(), X2.mean()
        var1, var2 = X1.var(), X2.var()
        print('mean1=%f, mean2=%f' % (mean1, mean2))
        print('variance1=%f, variance2=%f' % (var1, var2))
        self.df[col_name].hist()

        if plot == True:
            X = np.log(X) #log transform
            plt.hist(X)
            plt.show()

    def auto_correlation(self, data=None, col_name=None,):
        from statsmodels.graphics import tsaplots
        fig = tsaplots.plot_acf(self.data[col_name], lags=24)
        plt.show()


    def interpolate_na(self, col_name=None, method='time'):
        self.data[col_name] = self.data[col_name].interpolate(method, axis = 0) #linear or time
        print(self.data.head())
        return self
    

    # FIXME...
    def plot_series():
        with sns.axes_style('dark'):
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 8))

            nasdaq.plot(ax=axes[0][0],
                        title='NASDAQ  Composite Index')
            axes[0][0].text(x=.03,
                            y=.85,
                            s=f'ADF: {tsa.adfuller(nasdaq.dropna())[1]:.4f}',
                            transform=axes[0][0].transAxes)
            axes[0][0].set_ylabel('Index')

            nasdaq_log.plot(ax=axes[1][0],
                            sharex=axes[0][0])
            axes[1][0].text(x=.03, y=.85,
                            s=f'ADFl: {tsa.adfuller(nasdaq_log.dropna())[1]:.4f}',
                            transform=axes[1][0].transAxes)
            axes[1][0].set_ylabel('Log')

            nasdaq_log_diff.plot(ax=axes[2][0],
                                sharex=axes[0][0])
            axes[2][0].text(x=.03, y=.85,
                            s=f'ADF: {tsa.adfuller(nasdaq_log_diff.dropna())[1]:.4f}',
                            transform=axes[2][0].transAxes)
            axes[2][0].set_ylabel('Log, Diff')

            industrial_production.plot(ax=axes[0][1],
                                    title='Industrial Production: Manufacturing')
            axes[0][1].text(x=.03, y=.85,
                            s=f'ADF: {tsa.adfuller(industrial_production)[1]:.4f}',
                            transform=axes[0][1].transAxes)
            axes[0][1].set_ylabel('Index')

            industrial_production_log.plot(ax=axes[1][1],
                                        sharex=axes[0][1])
            axes[1][1].text(x=.03, y=.85,
                            s=f'ADF: {tsa.adfuller(industrial_production_log.dropna())[1]:.4f}',
                            transform=axes[1][1].transAxes)
            axes[1][1].set_ylabel('Log')

            industrial_production_log_diff.plot(ax=axes[2][1],
                                                sharex=axes[0][1])
            axes[2][1].text(x=.83, y=.85,
                            s=f'ADF: {tsa.adfuller(industrial_production_log_diff.dropna())[1]:.4f}',
                            transform=axes[2][1].transAxes)
            axes[2][1].set_ylabel('Log, Seasonal Diff')
            sns.despine()
            fig.tight_layout()
            fig.align_ylabels(axes)
            