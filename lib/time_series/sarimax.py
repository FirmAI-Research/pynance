import warnings
warnings.filterwarnings('ignore')
import sys, os

import seaborn as sns
import pandas_datareader as web
import statsmodels.tsa.api as tsa
import pandas as pd
import numpy as np 
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, acf, plot_pacf, pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
import statsmodels.api as sm
from scipy.stats import probplot, moment
from sklearn.metrics import mean_squared_error
from datetime import date 
'''
@ original source from:
-pakt publishing ML for algorithmic trading; Stefan Jansen-
https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/blob/master/09_time_series_models/02_arima_models.ipynb

'''
iop = os.getcwd() +'/io/'


def plot_model_summary(model_summary, title = None):
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(model_summary), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{str(iop)}{title}.png')


def plot_correlogram(x, lags=None, title=None):    
    lags = min(10, int(len(x)/5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    x.plot(ax=axes[0][0], title='Residuals')
    x.rolling(21).mean().plot(ax=axes[0][0], c='k', lw=1)
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    fig.suptitle(title, fontsize=14)
    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(f'{str(iop)}{title}.png')


def get_data():

    import yfinance as yf 
    time_series = yf.download('X','2000-01-01')['Adj Close'].squeeze().dropna()
    time_series_log = np.log(time_series)
    time_series_log_diff = time_series_log.diff(12).dropna()
    # print(time_series_log.isna().sum())


    ''' industrial production '''
    # time_series = web.DataReader('IPGMFN', 'fred', '1988', '2017-12').squeeze().dropna()
    # time_series_log = np.log(time_series)
    # time_series_log_diff = time_series_log.diff(12).dropna()

    return (time_series, time_series_log, time_series_log_diff)


time_series, time_series_log, time_series_log_diff = get_data()

''' SARIMAX '''

model1 = tsa.statespace.SARIMAX(time_series_log, order=(2,0,2), seasonal_order=(0,1,0,12)).fit()
print(model1.summary())
plot_model_summary(model1.summary(), title = 'ARMA_model_summary_1')


model2 = tsa.statespace.SARIMAX(time_series_log_diff, order=(2,0,2), seasonal_order=(0,0,0,12)).fit()
print(model2.summary())
plot_model_summary(model2.summary(), title = 'SARIMAX_model_summary_1')


print(model1.params.to_frame('SARIMAX').join(model2.params.to_frame('diff')))


best_model = tsa.SARIMAX(endog=time_series_log_diff, order=(2, 0, 3),
                        seasonal_order=(1, 0, 0, 12)).fit()
print(best_model.summary())
plot_model_summary(best_model.summary(), title = 'best_SARIMAX_model_summary')
plot_correlogram(best_model.resid, lags=20, title='Residuals_SARIMAX')
