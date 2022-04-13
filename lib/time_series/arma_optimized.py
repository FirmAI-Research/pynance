
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

iop = os.getcwd() +'/io/'
'''
@ original source from:
-pakt publishing ML for algorithmic trading; Stefan Jansen-
https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/blob/master/09_time_series_models/02_arima_models.ipynb

'''

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


def get_data(d):
    series_name = 'X'

    import yfinance as yf
    time_series = yf.download(series_name,'2019-12-01')['Adj Close'].squeeze().dropna()
    #time_series.index = time_series.index.to_period('M')

    # time_series = web.DataReader('IPGMFN', 'fred', '1988', '2017-12').squeeze().dropna() # industrial production

    time_series_log = np.log(time_series)
    time_series_log_diff = time_series_log.diff(d).dropna() # differencing of raw observations to make series stationary ("I" Integrated); represents d in ARIMA order (p,d,q)
    # print(time_series_log.isna().sum())

    return (series_name, time_series, time_series_log, time_series_log_diff)


def univariate_time_series_model():
    
    ''' ___ @ set params ___ '''
    initial_p, initial_q = (1,4)
    pq_iterations = 5
    # p_values = [0, 1, 2, 4, 6, 8, 10] # Autoregressive order ---> use the maximum statisticaly significant lag from partial auto correlation plot
    # d_values = range(0, 3) # integration order
    # q_values = range(0, 3) # moving average order --> use the maximum statistically significant lag from auto correlation plot
    d = 12 # d = 0 if series is stationary; use dicky fuller test to determine --> see decomposition.py
    ''' ___________________  '''

    series_name, time_series, time_series_log, time_series_log_diff = get_data(d)


    ''' ARMA ''' # but we are using the x_log_diff time series data to fit the ARMA model --> primitive AR"I"MA
    model = tsa.ARMA(endog=time_series_log_diff, order=(initial_p, initial_q)).fit() # endogenous variable; order(p,q) ---> ARIMA order is really (p,d,q) ; p=autoregressive, q=movingaverage
    print(model.summary())
    plot_model_summary(model.summary(), title = f'ARMA_Model_Summary_{initial_p}_{initial_q}_{series_name}')
    plot_correlogram(model.resid, title=f'ARMA_Residuals_Correlogram_{series_name}')


    '''
    Find optimal ARMA lags "We iterate over various (p, q) lag combinations 
    & collect diagnostic statistics to compare the result" 
    '''

    train_size = 120
    test_results = {}
    y_true = time_series_log_diff.iloc[train_size:]
    for p in range(pq_iterations):
        for q in range(pq_iterations):
            aic, bic = [], []
            if p == 0 and q == 0:
                continue
            print(p, q)
            convergence_error = stationarity_error = 0
            y_pred = []
            for T in range(train_size, len(time_series_log_diff)):
                train_set = time_series_log_diff.iloc[T-train_size:T] # split data into test train to prevent overfitting when predicting
                try:
                    model = tsa.ARMA(endog=train_set, order=(p, q)).fit() # fit model by iterating through p,q values
                except LinAlgError:
                    convergence_error += 1
                except ValueError:
                    stationarity_error += 1

                forecast, _, _ = model.forecast(steps=1)
                y_pred.append(forecast[0])
                aic.append(model.aic)
                bic.append(model.bic)

            result = (pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}) # collect results on this instance of the iteration
                    .replace(np.inf, np.nan)
                    .dropna())

            rmse = np.sqrt(mean_squared_error(
                y_true=result.y_true, y_pred=result.y_pred)) # calculate prediction error

            test_results[(p, q)] = [rmse,
                                    np.mean(aic),
                                    np.mean(bic),
                                    convergence_error,
                                    stationarity_error] # aggregate results of each p,q iteration


    test_results = pd.DataFrame(test_results).T
    test_results.columns = ['RMSE', 'AIC', 'BIC', 'convergence', 'stationarity']
    test_results.index.names = ['p', 'q']
    test_results.info()
    test_results.dropna()

    print(test_results.nsmallest(5, columns=['RMSE']))
    print(test_results.nsmallest(5, columns=['BIC']))

    # '''Root mean squared error'''
    sns.heatmap(test_results.RMSE.unstack().mul(10), fmt='.2', annot=True, cmap='Blues_r')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(f'{str(iop)}{series_name}_RMSE_heatmap.png')

    # '''Bayesian Information Criterion'''
    sns.heatmap(test_results.BIC.unstack(), fmt='.2f', annot=True, cmap='Blues_r')
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig(f'{str(iop)}{series_name}_BIC_heatmap.png')

    #''' use optimized ARMA lags to refit model '''
    best_p, best_q = test_results.rank().loc[:, ['RMSE', 'BIC']].mean(1).idxmin()  # utilize best p,q values as determined by lowest RMSE,BIC
    best_arma_model = tsa.ARMA(endog=time_series_log_diff, order=(best_p, best_q)).fit()
    print(best_arma_model.summary())
    plot_model_summary(best_arma_model.summary(), title = f'ARMA_Opt_Model_Summary_Opt_{best_p}_{best_q}_{series_name}')
    plot_correlogram(best_arma_model.resid, lags=20, title=f'ARMA_Opt_Residuals_Correlogram_{best_p}_{best_q}_{series_name}')
 


univariate_time_series_model()