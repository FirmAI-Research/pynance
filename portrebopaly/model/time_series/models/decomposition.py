'''
A program to: 


@ original source from:
-pakt publishing ML for algorithmic trading; Stefan Jansen-
https://github.com/PacktPublishing/Hands-On-Machine-Learning-for-Algorithmic-Trading/blob/master/Chapter08/01_stationarity_and_arima.ipynb
@ notes
singular value decomposition
use SVD to reduce data into key features to describe dataset
data driven generalization of fourrier transoformation
use autocorrelations to compute correlation coefficients as a function of the lag



'''
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
import datetime


p = os.getcwd() +'/io/'



industrial_production = web.DataReader('IPGMFN','fred','2000','2020-12').squeeze()
print(industrial_production.head())
print(type(industrial_production.head()))
components = tsa.seasonal_decompose(industrial_production, model='additive')

def additive_components(components, the_title=None):
    ts = industrial_production.to_frame('Original').assign(Trend = components.trend).assign(Seasonality=components.seasonal).assign(Residual=components.resid)
    ts.plot(subplots=True, figsize=(14,8))
    fig1 = plt.gcf()

    plt.show()
    fig1.savefig(str(p) +'Additive Components.png')
# additive_components(components)


def differencing():
    industrial_production = web.DataReader('IPGMFN', 'fred', '1988', '2017-12').squeeze().dropna()
    nasdaq = web.DataReader('NASDAQCOM', 'fred', '1990', '2017-12-31').squeeze().dropna()
    (nasdaq == 0).any(), (industrial_production==0).any()


    nasdaq = web.DataReader('NASDAQCOM', 'fred', '1990', '2017-12-31').squeeze().dropna()


    nasdaq_log = np.log(nasdaq)
    industrial_production_log = np.log(industrial_production)


    nasdaq_log_diff = nasdaq_log.diff().dropna()

    # seasonal differencing => yoy instantanteous returns
    industrial_production_log_diff = industrial_production_log.diff(12).dropna()


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
        fig1 = plt.gcf()

        fig1.savefig(str(p)  +'Differencing.png')

# differencing()



def autocorrelations(data, the_title=None):
    # industrial_production = web.DataReader('IPGMFN', 'fred', '1988', '2017-12').squeeze().dropna()
    industrial_production_log = np.log(data)
    industrial_production_log_diff = industrial_production_log.diff(12).dropna() # seasonal differencing => yoy instantanteous returns


    from statsmodels.graphics.tsaplots import plot_acf, acf, plot_pacf, pacf
    from statsmodels.tsa.stattools import acf, q_stat, adfuller
    import statsmodels.api as sm
    from scipy.stats import probplot, moment
    from sklearn.metrics import mean_squared_error
    def plot_correlogram(x, lags=None, title=None):    
        lags = min(10, int(len(x)/5)) if lags is None else lags
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
        x.plot(ax=axes[0][0])
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
        fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(top=.9)

    plot_correlogram(industrial_production_log_diff, title=f'{the_title} AutoCorrelation')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(str(p)  +'AutoCorrelation.png')
# autocorrelations()


